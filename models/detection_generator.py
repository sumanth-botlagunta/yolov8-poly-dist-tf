"""Post-processing: DFL decoding, anchor generation, NMS, polygon decoding.

Converts raw head outputs to final detections:
    1. Decode DFL box distributions to (l, t, r, b) offsets.
    2. Apply per-level anchor grids to produce xyxy boxes.
    3. Normalize to [0, 1] in yxyx format.
    4. Apply sigmoid to class logits.
    5. Run per-image greedy NMS (class-agnostic) via tf.map_fn.
    6. Gather polygon and distance predictions for surviving boxes.
    7. Decode distance: exp(log_dist), clamped to [min_distance, max_distance].

Classes:
    YoloV8Layer: Wraps post-processing as a callable layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tensorflow as tf

_LEVEL_STRIDES = {"3": 8, "4": 16, "5": 32}


class YoloV8Layer:
    """Post-processing layer converting raw head output to detections.

    Output predictions schema:
        bbox:           float32 [batch, max_boxes, 4]     yxyx normalized [0,1]
        classes:        int64   [batch, max_boxes]
        confidence:     float32 [batch, max_boxes]
        num_detections: int32   [batch]
        polygons:       float32 [batch, max_boxes, 24, 3] (conf,dy,dx) per vertex
        distance:       float32 [batch, max_boxes]        metres
    """

    def __init__(
        self,
        input_image_size: List[int],
        max_boxes: int = 300,
        nms_thresh: float = 0.65,
        iou_thresh: float = 0.001,
        pre_nms_points: int = 30000,
        nms_type: str = "greedy",
        reg_max: int = 16,
        output_poly_size: int = 24,
        min_distance: float = 0.5,
        max_distance: float = 10.0,
    ):
        self.input_image_size = input_image_size[:2]   # [H, W]
        self.max_boxes        = max_boxes
        self.nms_thresh       = nms_thresh
        self.iou_thresh       = iou_thresh
        self.pre_nms_points   = pre_nms_points
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
        # Centres at (i+0.5)*stride for i in 0..H-1 (standard YOLOv8 convention)
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
        # Reshape to [B, H, W, 4, reg_max]
        logits_4 = tf.reshape(box_logits, [B, H, W, 4, self.reg_max])
        # Softmax over bin dimension
        probs = tf.nn.softmax(logits_4, axis=-1)                   # [B, H, W, 4, reg_max]
        # Expected bin value
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
        """Greedy class-agnostic NMS for one image."""
        max_scores  = tf.reduce_max(scores, axis=-1)        # [N]
        top_classes = tf.argmax(scores, axis=-1)            # [N] int64

        # Pre-filter by score threshold before NMS for speed
        keep_mask = max_scores >= self.iou_thresh
        boxes_f   = tf.boolean_mask(boxes, keep_mask)
        scores_f  = tf.boolean_mask(max_scores, keep_mask)
        classes_f = tf.boolean_mask(top_classes, keep_mask)

        # Gather optional tensors
        def _mask(t):
            return tf.boolean_mask(t, keep_mask) if t is not None else None

        pa_f = _mask(poly_angle)
        pd_f = _mask(poly_dist)
        pc_f = _mask(poly_conf)
        di_f = _mask(distance)

        # Greedy NMS (class-agnostic)
        nms_idx = tf.image.non_max_suppression(
            boxes_f,
            scores_f,
            max_output_size=self.max_boxes,
            iou_threshold=self.nms_thresh,
            score_threshold=float("-inf"),
        )                                                   # [k]

        k   = tf.shape(nms_idx)[0]
        pad = self.max_boxes - k

        def _gather_pad(t, default_val=0.0):
            if t is None:
                return None
            g = tf.gather(t, nms_idx)
            # Pad along axis 0 to max_boxes
            pads = [[0, pad]] + [[0, 0]] * (len(t.shape) - 1)
            return tf.pad(g, pads, constant_values=default_val)

        sel_boxes   = _gather_pad(boxes_f)                  # [max_boxes, 4]
        sel_scores  = _gather_pad(scores_f)                 # [max_boxes]
        sel_classes = tf.pad(tf.gather(classes_f, nms_idx),
                             [[0, pad]])                    # [max_boxes]

        sel_boxes   = tf.ensure_shape(sel_boxes,  [self.max_boxes, 4])
        sel_scores  = tf.ensure_shape(sel_scores, [self.max_boxes])
        sel_classes = tf.ensure_shape(sel_classes,[self.max_boxes])

        sel_pa = _gather_pad(pa_f) if pa_f is not None else tf.zeros([self.max_boxes, self.output_poly_size])
        sel_pd = _gather_pad(pd_f) if pd_f is not None else tf.zeros([self.max_boxes, self.output_poly_size])
        sel_pc = _gather_pad(pc_f) if pc_f is not None else tf.zeros([self.max_boxes, self.output_poly_size])

        if di_f is not None:
            di_g = tf.gather(di_f, nms_idx)
            di_g = tf.clip_by_value(tf.exp(di_g), self.min_distance, self.max_distance)
            sel_dist = tf.pad(di_g, [[0, pad]])
        else:
            sel_dist = tf.zeros([self.max_boxes])

        sel_pa = tf.ensure_shape(sel_pa, [self.max_boxes, self.output_poly_size])
        sel_pd = tf.ensure_shape(sel_pd, [self.max_boxes, self.output_poly_size])
        sel_pc = tf.ensure_shape(sel_pc, [self.max_boxes, self.output_poly_size])
        sel_dist = tf.ensure_shape(sel_dist, [self.max_boxes])

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
            box_raw = raw_outputs["box"][level]         # [B, H, W, 4*reg_max]
            cls_raw = raw_outputs["cls"][level]         # [B, H, W, num_classes]

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
                all_poly_angle.append(tf.reshape(raw_outputs["poly_angle"][level], [B, -1, self.output_poly_size]))
                all_poly_dist.append( tf.reshape(raw_outputs["poly_dist"][level],  [B, -1, self.output_poly_size]))
                all_poly_conf.append( tf.reshape(raw_outputs["poly_conf"][level],  [B, -1, self.output_poly_size]))

            if has_dist:
                all_dist.append(tf.reshape(raw_outputs["dist"][level][:, :, :, 0], [B, -1]))

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

        # Stack polygon channels into [B, max_boxes, 24, 3] (conf, dy, dx)
        poly_out = tf.stack([out_pc, out_pd, out_pa], axis=-1)   # [B, max_boxes, 24, 3]

        return {
            "bbox":           out_boxes,        # [B, max_boxes, 4] yxyx normalized
            "classes":        out_classes,       # [B, max_boxes]
            "confidence":     out_scores,        # [B, max_boxes]
            "num_detections": out_n,             # [B]
            "polygons":       poly_out,          # [B, max_boxes, 24, 3]
            "distance":       out_dist,          # [B, max_boxes]
        }
