"""Post-processing: DFL decoding, anchor generation, NMS, polygon decoding.

Converts raw head outputs to final detections:
    1. Decode DFL box distributions to (l, t, r, b) offsets.
    2. Apply per-level anchor grids to produce xyxy boxes.
    3. Normalize to [0, 1] in yxyx format.
    4. Apply sigmoid to class logits.
    5. Apply top-1 class masking (each anchor keeps only its highest-scoring class).
    6. Run greedy NMS (score_threshold=0.05, iou_threshold=0.65) — either
       independently per class or once class-agnostically (``nms_class_mode``).
    7. Merge survivors, sort by score, keep top-max_boxes.
    8. Apply polygon activations: softplus(dist), sigmoid(angle), sigmoid(conf).
    9. Decode distance: exp(log_dist), clamped to [min_distance, max_distance].

Classes:
    YoloV8Layer: Wraps post-processing as a callable layer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

_LEVEL_STRIDES = {"3": 8, "4": 16, "5": 32}


class YoloV8Layer:
    """Post-processing layer converting raw head output to detections.

    Top-1 class masking zeroes out all classes except the argmax per anchor;
    score_threshold=0.05 filters boxes before NMS. The suppression scope is
    selected by ``nms_class_mode``:

    - ``per_class``: NMS runs independently per class — two overlapping boxes
      with different argmax classes never suppress each other.
    - ``agnostic``: ONE NMS over all boxes regardless of class — at each
      location only the highest-scored box survives, so cross-class
      duplicates are removed.

    Output predictions schema:
        bbox:           float32 [batch, max_boxes, 4]     yxyx normalized [0,1]
        classes:        int64   [batch, max_boxes]
        confidence:     float32 [batch, max_boxes]
        num_detections: int32   [batch]
        polygons:       float32 [batch, max_boxes, 24, 3] (conf,dist,angle) activated
        distance:       float32 [batch, max_boxes]        metres
    """

    def __init__(
        self,
        input_image_size: List[int],
        num_classes: int = 39,
        max_boxes: int = 300,
        nms_thresh: float = 0.65,
        score_thresh: float = 0.05,
        nms_class_mode: str = "per_class",
        reg_max: int = 16,
        output_poly_size: int = 24,
        min_distance: float = 0.5,
        max_distance: float = 10.0,
    ):
        if nms_class_mode not in ("per_class", "agnostic"):
            raise ValueError(
                f"nms_class_mode must be 'per_class' or 'agnostic', got "
                f"'{nms_class_mode}'."
            )
        self.input_image_size = input_image_size[:2]   # [H, W]
        self.num_classes      = num_classes
        self.max_boxes        = max_boxes
        self.nms_thresh       = nms_thresh
        self.score_thresh     = score_thresh
        self.nms_class_mode   = nms_class_mode
        self.reg_max          = reg_max
        self.output_poly_size = output_poly_size
        self.min_distance     = min_distance
        self.max_distance     = max_distance

        # Bin indices for DFL decoding: [0, 1, ..., reg_max-1]
        self._bins = tf.cast(tf.range(reg_max), tf.float32)   # [reg_max]

    # ------------------------------------------------------------------
    # Anchor grid
    # ------------------------------------------------------------------

    def _generate_anchor_grid(
        self, feature_shape: List[int], stride: int
    ) -> tf.Tensor:
        """Generate anchor center coordinates for a single FPN level.

        Returns:
            float32 [H*W, 2]  (cx, cy) in input-image pixels, offset by 0.5
        """
        H, W = feature_shape[0], feature_shape[1]
        ys = (tf.cast(tf.range(H), tf.float32) + 0.5) * stride   # [H]
        xs = (tf.cast(tf.range(W), tf.float32) + 0.5) * stride   # [W]
        grid_x, grid_y = tf.meshgrid(xs, ys)                      # [H, W] each
        cx = tf.reshape(grid_x, [-1])                              # [H*W]
        cy = tf.reshape(grid_y, [-1])                              # [H*W]
        return tf.stack([cx, cy], axis=-1)                         # [H*W, 2]

    # ------------------------------------------------------------------
    # DFL decoding
    # ------------------------------------------------------------------

    def _decode_dfl(self, box_logits: tf.Tensor) -> tf.Tensor:
        """Convert DFL distribution logits to (l, t, r, b) offsets.

        Args:
            box_logits: float32 [B, H, W, 4*reg_max]
        Returns:
            float32 [B, H, W, 4]  unnormalized pixel offsets
        """
        B = tf.shape(box_logits)[0]
        H = tf.shape(box_logits)[1]
        W = tf.shape(box_logits)[2]
        logits_4 = tf.reshape(box_logits, [B, H, W, 4, self.reg_max])
        probs = tf.nn.softmax(logits_4, axis=-1)                   # [B, H, W, 4, reg_max]
        offsets = tf.reduce_sum(probs * self._bins, axis=-1)        # [B, H, W, 4]
        return offsets

    # ------------------------------------------------------------------
    # Per-image NMS
    # ------------------------------------------------------------------

    def _nms_single(
        self,
        boxes: tf.Tensor,       # [N, 4] yxyx normalized
        scores: tf.Tensor,      # [N, num_classes]
        poly_angle: Optional[tf.Tensor],   # [N, 24] or None
        poly_dist:  Optional[tf.Tensor],
        poly_conf:  Optional[tf.Tensor],
        distance:   Optional[tf.Tensor],   # [N] or None
    ) -> Tuple[tf.Tensor, ...]:
        """NMS with top-1 masking for one image.

        Steps:
            1. Top-1 masking: zero out all classes except argmax per anchor.
            2. NMS with score_threshold=self.score_thresh — per class
               (``per_class``) or once over all boxes (``agnostic``).
            3. Merge survivors, sort by score, pad to max_boxes.
        """
        # Top-1 class masking
        top_class     = tf.argmax(scores, axis=-1)                         # [N]
        one_hot       = tf.one_hot(top_class, self.num_classes, dtype=tf.float32)  # [N, nc]
        scores_masked = scores * one_hot                                    # [N, nc]

        if self.nms_class_mode == "agnostic":
            # ONE NMS over all boxes: each anchor competes with its argmax-class
            # score; overlapping boxes suppress each other regardless of class.
            cs      = tf.reduce_max(scores_masked, axis=-1)   # [N] argmax-class score
            nms_idx = tf.image.non_max_suppression(
                boxes, cs,
                max_output_size=self.max_boxes,
                iou_threshold=self.nms_thresh,
                score_threshold=self.score_thresh,
            )
            m_boxes   = tf.gather(boxes, nms_idx)             # [?, 4]
            m_scores  = tf.gather(cs, nms_idx)                # [?]
            m_classes = tf.gather(top_class, nms_idx)         # [?]
            c_pa = [tf.gather(poly_angle, nms_idx)] if poly_angle is not None else []
            c_pd = [tf.gather(poly_dist,  nms_idx)] if poly_angle is not None else []
            c_pc = [tf.gather(poly_conf,  nms_idx)] if poly_angle is not None else []
            c_di = [tf.gather(distance,   nms_idx)] if distance   is not None else []
        else:
            # Per-class NMS
            c_boxes, c_scores, c_classes = [], [], []
            c_pa, c_pd, c_pc, c_di = [], [], [], []

            for c in range(self.num_classes):
                cs      = scores_masked[:, c]   # [N]
                nms_idx = tf.image.non_max_suppression(
                    boxes, cs,
                    max_output_size=self.max_boxes,
                    iou_threshold=self.nms_thresh,
                    score_threshold=self.score_thresh,
                )
                c_boxes.append(tf.gather(boxes, nms_idx))
                c_scores.append(tf.gather(cs, nms_idx))
                c_classes.append(tf.fill([tf.shape(nms_idx)[0]], tf.cast(c, tf.int64)))
                if poly_angle is not None:
                    c_pa.append(tf.gather(poly_angle, nms_idx))
                    c_pd.append(tf.gather(poly_dist,  nms_idx))
                    c_pc.append(tf.gather(poly_conf,  nms_idx))
                if distance is not None:
                    c_di.append(tf.gather(distance, nms_idx))

            # Merge survivors from all classes
            m_boxes   = tf.concat(c_boxes,   axis=0)   # [?, 4]
            m_scores  = tf.concat(c_scores,  axis=0)   # [?]
            m_classes = tf.concat(c_classes, axis=0)   # [?]

        # Sort by score descending, keep top-max_boxes
        sort_idx = tf.argsort(m_scores, direction='DESCENDING')
        k   = tf.minimum(tf.shape(m_scores)[0], self.max_boxes)
        top = sort_idx[:k]
        pad = self.max_boxes - k

        sel_boxes   = tf.ensure_shape(
            tf.pad(tf.gather(m_boxes,   top), [[0, pad], [0, 0]]), [self.max_boxes, 4])
        sel_scores  = tf.ensure_shape(
            tf.pad(tf.gather(m_scores,  top), [[0, pad]]),          [self.max_boxes])
        sel_classes = tf.ensure_shape(
            tf.pad(tf.gather(m_classes, top), [[0, pad]]),          [self.max_boxes])

        # Polygon: apply activations to detected boxes, then pad with zeros
        if poly_angle is not None:
            m_pa = tf.concat(c_pa, axis=0)
            m_pd = tf.concat(c_pd, axis=0)
            m_pc = tf.concat(c_pc, axis=0)
            sel_pa = tf.ensure_shape(
                tf.pad(tf.math.sigmoid(tf.gather(m_pa, top)),  [[0, pad], [0, 0]]),
                [self.max_boxes, self.output_poly_size])
            sel_pd = tf.ensure_shape(
                tf.pad(tf.math.softplus(tf.gather(m_pd, top)), [[0, pad], [0, 0]]),
                [self.max_boxes, self.output_poly_size])
            sel_pc = tf.ensure_shape(
                tf.pad(tf.math.sigmoid(tf.gather(m_pc, top)),  [[0, pad], [0, 0]]),
                [self.max_boxes, self.output_poly_size])
        else:
            sel_pa = tf.zeros([self.max_boxes, self.output_poly_size])
            sel_pd = tf.zeros([self.max_boxes, self.output_poly_size])
            sel_pc = tf.zeros([self.max_boxes, self.output_poly_size])

        # Distance: exp + clamp, then pad
        if distance is not None:
            m_di     = tf.concat(c_di, axis=0)
            # Clamp in LOG space *before* exp. The head emits log-distance; a few
            # very large logits would overflow exp() to +inf, and clip(inf, ...)
            # silently yields max_distance with no NaN/inf signal. Clamping the log
            # first bounds the exp input to [log(min), log(max)] — identical result
            # for in-range values, but no transient inf. min/max_distance are
            # guaranteed positive (range [0.5, 10.0]), so the logs are finite.
            log_di   = tf.clip_by_value(
                tf.gather(m_di, top),
                math.log(self.min_distance),
                math.log(self.max_distance),
            )
            di_exp   = tf.exp(log_di)
            sel_dist = tf.ensure_shape(tf.pad(di_exp, [[0, pad]]), [self.max_boxes])
        else:
            sel_dist = tf.zeros([self.max_boxes])

        return sel_boxes, sel_scores, sel_classes, tf.cast(k, tf.int32), sel_pa, sel_pd, sel_pc, sel_dist

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        raw_outputs: Dict[str, Dict[str, tf.Tensor]],
        image_info: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Decode and NMS the raw head outputs."""
        H_img, W_img = self.input_image_size
        H_f = tf.cast(H_img, tf.float32)
        W_f = tf.cast(W_img, tf.float32)

        all_boxes      = []
        all_cls        = []
        all_poly_angle = []
        all_poly_dist  = []
        all_poly_conf  = []
        all_dist       = []

        has_poly = "poly_angle" in raw_outputs
        has_dist = "dist" in raw_outputs

        for level in ["3", "4", "5"]:
            stride = _LEVEL_STRIDES[level]
            # Cast to float32 up front: under a mixed_bfloat16 policy the head
            # emits bfloat16, which would otherwise clash with the float32
            # constants/grids below and the float32 fn_output_signature. No-op
            # under the default float32 policy.
            box_raw = tf.cast(raw_outputs["box"][level], tf.float32)  # [B, H, W, 4*reg_max]
            cls_raw = tf.cast(raw_outputs["cls"][level], tf.float32)  # [B, H, W, num_classes]

            B = tf.shape(box_raw)[0]
            fH = tf.shape(box_raw)[1]
            fW = tf.shape(box_raw)[2]

            # Decode DFL offsets (in feature-map pixel units)
            ltrb = self._decode_dfl(box_raw)            # [B, H, W, 4]
            # Scale to input-image pixel units
            ltrb_scaled = ltrb * tf.cast(stride, tf.float32)

            # Anchor centres
            anchors = self._generate_anchor_grid([fH, fW], stride)  # [H*W, 2] cx,cy
            cx = tf.reshape(anchors[:, 0], [1, -1])    # [1, H*W]
            cy = tf.reshape(anchors[:, 1], [1, -1])    # [1, H*W]

            ltrb_flat = tf.reshape(ltrb_scaled, [B, -1, 4])          # [B, H*W, 4]
            l, t, r, b = (ltrb_flat[..., i] for i in range(4))

            # xyxy → yxyx normalized
            x1 = cx - l;  y1 = cy - t
            x2 = cx + r;  y2 = cy + b
            yxyx = tf.stack([y1 / H_f, x1 / W_f, y2 / H_f, x2 / W_f], axis=-1)  # [B, H*W, 4]
            all_boxes.append(yxyx)

            cls_flat = tf.sigmoid(tf.reshape(cls_raw, [B, -1, tf.shape(cls_raw)[-1]]))
            all_cls.append(cls_flat)

            if has_poly:
                all_poly_angle.append(tf.cast(tf.reshape(raw_outputs["poly_angle"][level], [B, -1, self.output_poly_size]), tf.float32))
                all_poly_dist.append( tf.cast(tf.reshape(raw_outputs["poly_dist"][level],  [B, -1, self.output_poly_size]), tf.float32))
                all_poly_conf.append( tf.cast(tf.reshape(raw_outputs["poly_conf"][level],  [B, -1, self.output_poly_size]), tf.float32))

            if has_dist:
                all_dist.append(tf.cast(tf.reshape(raw_outputs["dist"][level][:, :, :, 0], [B, -1]), tf.float32))

        boxes  = tf.concat(all_boxes, axis=1)   # [B, N_anchors, 4]
        scores = tf.concat(all_cls,   axis=1)   # [B, N_anchors, nc]
        poly_a = tf.concat(all_poly_angle, axis=1) if has_poly else None
        poly_d = tf.concat(all_poly_dist,  axis=1) if has_poly else None
        poly_c = tf.concat(all_poly_conf,  axis=1) if has_poly else None
        dist   = tf.concat(all_dist, axis=1)         if has_dist else None

        # Per-image NMS via tf.map_fn
        def _map_fn(args):
            b, s = args[0], args[1]
            pa = args[2] if has_poly else None
            pd = args[3] if has_poly else None
            pc = args[4] if has_poly else None
            di = args[5] if has_dist else None
            return self._nms_single(b, s, pa, pd, pc, di)

        elems = [boxes, scores]
        if has_poly:
            elems += [poly_a, poly_d, poly_c]
        else:
            elems += [tf.zeros_like(boxes[:, :, :1]),
                      tf.zeros_like(boxes[:, :, :1]),
                      tf.zeros_like(boxes[:, :, :1])]
        if has_dist:
            elems.append(dist)
        else:
            elems.append(tf.zeros_like(boxes[:, :, 0]))

        (out_boxes, out_scores, out_classes, out_n,
         out_pa, out_pd, out_pc, out_dist) = tf.map_fn(
            _map_fn,
            elems=elems,
            fn_output_signature=(
                tf.TensorSpec(shape=[self.max_boxes, 4],                     dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_boxes],                        dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_boxes],                        dtype=tf.int64),
                tf.TensorSpec(shape=[],                                      dtype=tf.int32),
                tf.TensorSpec(shape=[self.max_boxes, self.output_poly_size], dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_boxes, self.output_poly_size], dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_boxes, self.output_poly_size], dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_boxes],                        dtype=tf.float32),
            ),
        )

        # Stack polygon channels: (conf, dist, angle) — all activated
        poly_out = tf.stack([out_pc, out_pd, out_pa], axis=-1)   # [B, max_boxes, 24, 3]

        # Clip final boxes to the image: the DFL decode can place edges beyond
        # the borders (cx − l < 0 etc.), which draws boxes outside the frame in
        # overlays and slightly penalizes IoU against edge-clipped GT in eval.
        # Clipping after NMS (here) leaves suppression behavior unchanged.
        out_boxes = tf.clip_by_value(out_boxes, 0.0, 1.0)

        return {
            "bbox":           out_boxes,        # [B, max_boxes, 4] yxyx normalized
            "classes":        out_classes,       # [B, max_boxes]
            "confidence":     out_scores,        # [B, max_boxes]
            "num_detections": out_n,             # [B]
            "polygons":       poly_out,          # [B, max_boxes, 24, 3] (conf,dist,angle) activated
            "distance":       out_dist,          # [B, max_boxes]
        }
