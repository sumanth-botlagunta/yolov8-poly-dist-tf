"""Copy-Paste augmentation for instance-level data augmentation.

Applied per tile inside the mosaic stage with probability prob_copy_n_paste.
Object sources come from a separate RGBA TFDS (cleaner_copy_paste:1.0.0); the
alpha channel is the compositing mask, and polygon vertices are transformed with
the same placement applied to the object crop.

Classes:
    CopyAndPasteModule: wraps the augmentation as a callable pipeline stage.
"""

from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf


class CopyAndPasteModule:
    """Copy-Paste augmentation using RGBA object crops.

    Process:
        1. Randomly resize the object within [min_resize_ratio, max_resize_ratio].
        2. Randomly place object in the background within the height_limit region.
        3. Composite using the alpha mask channel.
        4. Append updated bounding boxes, classes, and polygon vertices.

    A drawn placement is DISCARDED (background returned unchanged) when the
    pasted object's box would cover more than ``max_occlusion_frac`` of any
    pre-existing GT box: the composite overwrites those pixels but the old
    label is kept, so a heavily-occluded object's box/class/polygon would
    point at content that is no longer there.
    """

    def __init__(
        self,
        prob: float = 0.2,
        min_height: float = 60,
        min_width: float = 100,
        max_resize_ratio: float = 1.5,
        min_resize_ratio: float = 0.2,
        height_limit: float = 0.6,
        max_occlusion_frac: float = 0.5,
    ):
        self._prob = prob
        self._min_height = min_height
        self._min_width = min_width
        self._max_resize_ratio = max_resize_ratio
        self._min_resize_ratio = min_resize_ratio
        self._height_limit = height_limit
        # Occlusion gate threshold: skip the paste when it would cover more
        # than this fraction of any existing GT box's area. 0.5 mirrors the
        # candidate-filter convention (a label is kept iff >= 50% of the
        # object stays visible). None disables the gate.
        self._max_occlusion_frac = max_occlusion_frac

    # ------------------------------------------------------------------
    # Core compositing
    # ------------------------------------------------------------------

    def _copy_and_paste(
        self,
        bg_data: Dict[str, tf.Tensor],
        obj_data: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Draw an adaptive resize ratio, then composite.

        The ratio bounds are constructed in ORIGINAL background pixel units
        (like the reference implementation) so the pasted object meets the
        ``min_height x min_width`` floor whenever feasible under the
        containment and ``max_resize_ratio`` caps (containment wins on
        conflict, same as the reference), never exceeds ``height_limit`` of
        the background height or the full background width, and the paste
        always happens once the probability gate fired:

            max_ratio = min(bg_h*height_limit/obj_h, bg_w/obj_w, max_resize_ratio)
            min_ratio = max(min_height/obj_h, min_width/obj_w, min_resize_ratio)
            min_ratio = min(min_ratio, max_ratio)
            ratio ~ U(min_ratio, max_ratio)

        Containment and the size floor are guaranteed by construction; there
        is no size-based accept/reject gate. The only post-draw rejection is
        the occlusion gate in ``_paste`` (see ``max_occlusion_frac``), so the
        realized paste rate can fall slightly below ``prob`` on crowded
        backgrounds.
        """
        obj_shape = tf.shape(obj_data['image'])
        obj_h_f = tf.cast(obj_shape[0], tf.float32)
        obj_w_f = tf.cast(obj_shape[1], tf.float32)

        # Original background dims (the pipeline pre-resizes to the model input
        # size before copy-paste; height/width carry the capture dims). The
        # bounds are defined in original units, matching the resolution the
        # min_height/min_width floor and obj crop dims are expressed in.
        bg_shape = tf.shape(bg_data['image'])
        orig_h_f = tf.cast(bg_data.get('height', bg_shape[0]), tf.float32)
        orig_w_f = tf.cast(bg_data.get('width', bg_shape[1]), tf.float32)

        max_ratio = tf.reduce_min(tf.stack([
            orig_h_f * self._height_limit / obj_h_f,
            orig_w_f / obj_w_f,
            self._max_resize_ratio,
        ]))
        min_ratio = tf.reduce_max(tf.stack([
            self._min_height / obj_h_f,
            self._min_width / obj_w_f,
            self._min_resize_ratio,
        ]))
        min_ratio = tf.minimum(min_ratio, max_ratio)
        resize_ratio = tf.random.uniform([], min_ratio, max_ratio)
        return self._paste(bg_data, obj_data, resize_ratio)

    def _paste(
        self,
        bg_data: Dict[str, tf.Tensor],
        obj_data: Dict[str, tf.Tensor],
        resize_ratio: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Composite one object onto the background and update annotations.

        Args:
            bg_data: decoded background example.
                image:               uint8 [H, W, 3]
                groundtruth_boxes:   float32 [N, 4] yxyx
                groundtruth_classes: int64 [N]
                groundtruth_polygons: float32 [N, max_v]
                groundtruth_is_crowd: bool [N]
                groundtruth_area:    float32 [N]
                groundtruth_dontcare: int64 [N]

            obj_data: decoded copy-paste example.
                image:    uint8 [H_o, W_o, 4] RGBA
                orig_bbox: float32 [4] yxyx normalised in object image
                label:    int64 scalar
                points:   float32 [max_v] flat xy, normalised in object image

        Returns:
            bg_data dict with the pasted object appended to annotations, or
            the input unchanged when the occlusion gate rejects the placement.
        """
        orig_data = bg_data  # untouched input, returned if the gate rejects
        bg_img = tf.cast(bg_data['image'], tf.float32)      # [H, W, 3]
        H = tf.shape(bg_img)[0]
        W = tf.shape(bg_img)[1]
        H_f = tf.cast(H, tf.float32)
        W_f = tf.cast(W, tf.float32)

        obj_rgba = tf.cast(obj_data['image'], tf.float32)  # [H_o, W_o, 4]
        obj_rgb  = obj_rgba[:, :, :3]                       # [H_o, W_o, 3]
        alpha    = obj_rgba[:, :, 3:4] / 255.0              # [H_o, W_o, 1]

        # Resolution correction: the background is pre-resized to the model input
        # size before copy-paste, but the object's target size is defined relative
        # to the original background dims, so scale by (current/original) per axis.
        # This commutes with the background resize (pixel-equivalent to compositing
        # at full resolution and resizing afterwards). Unresized background:
        # height/width == current dims, correction is 1.
        orig_h_f = tf.cast(bg_data.get('height', H), tf.float32)
        orig_w_f = tf.cast(bg_data.get('width', W), tf.float32)
        corr_h = H_f / tf.maximum(orig_h_f, 1.0)
        corr_w = W_f / tf.maximum(orig_w_f, 1.0)

        # resize_ratio is drawn by _copy_and_paste with adaptive bounds that
        # guarantee the min-size floor and full containment in original units.
        obj_h_f = tf.cast(tf.shape(obj_rgb)[0], tf.float32)
        obj_w_f = tf.cast(tf.shape(obj_rgb)[1], tf.float32)
        new_h = tf.maximum(tf.cast(tf.round(obj_h_f * resize_ratio * corr_h), tf.int32), 1)
        new_w = tf.maximum(tf.cast(tf.round(obj_w_f * resize_ratio * corr_w), tf.int32), 1)
        new_h = tf.minimum(new_h, H)
        new_w = tf.minimum(new_w, W)

        # Resize obj and alpha. The alpha mask resizes with 'nearest' (the
        # reference behavior): the binary mask stays exactly binary through the
        # resize instead of picking up bilinear edge blending.
        obj_rgb_r = tf.image.resize(obj_rgb, [new_h, new_w], method='bilinear')
        alpha_r   = tf.image.resize(alpha,   [new_h, new_w], method='nearest')
        alpha_r   = tf.clip_by_value(alpha_r, 0.0, 1.0)

        # Random placement, reference formulation. Vertical: the object's top
        # edge lands in the LOWER band of the frame,
        #     offset_h_min = H*(1 - height_limit)
        #     offset_h_max = H - new_h
        #     offset_h = U(0.1, 0.9)*(max - min) + min
        # (floor-facing camera: pasted objects belong near the floor, not at
        # the top of the frame). Horizontal: offset_w = U(0.1, 0.9)*(W - new_w).
        # Both clipped to keep the object fully inside the canvas.
        new_h_f = tf.cast(new_h, tf.float32)
        new_w_f = tf.cast(new_w, tf.float32)
        off_h_min = H_f * (1.0 - self._height_limit)
        off_h_max = H_f - new_h_f
        off_h = tf.random.uniform([], 0.1, 0.9) * (off_h_max - off_h_min) + off_h_min
        off_w = tf.random.uniform([], 0.1, 0.9) * (W_f - new_w_f)

        paste_y = tf.clip_by_value(tf.cast(off_h, tf.int32), 0, H - new_h)
        paste_x = tf.clip_by_value(tf.cast(off_w, tf.int32), 0, W - new_w)

        # Build full-canvas alpha mask (0 everywhere except paste region).
        pad_top    = paste_y
        pad_bottom = H - paste_y - new_h
        pad_left   = paste_x
        pad_right  = W - paste_x - new_w
        pad_bottom = tf.maximum(pad_bottom, 0)
        pad_right  = tf.maximum(pad_right, 0)
        new_h_clipped = tf.minimum(new_h, H - paste_y)
        new_w_clipped = tf.minimum(new_w, W - paste_x)
        obj_rgb_r_c = obj_rgb_r[:new_h_clipped, :new_w_clipped, :]
        alpha_r_c   = alpha_r[:new_h_clipped, :new_w_clipped, :]

        alpha_canvas = tf.pad(
            alpha_r_c,
            [[pad_top, H - paste_y - new_h_clipped],
             [pad_left, W - paste_x - new_w_clipped],
             [0, 0]],
        )  # [H, W, 1]

        obj_canvas = tf.pad(
            obj_rgb_r_c,
            [[pad_top, H - paste_y - new_h_clipped],
             [pad_left, W - paste_x - new_w_clipped],
             [0, 0]],
        )  # [H, W, 3]

        # Hard-mask composite: use alpha > 0.5 as binary mask.
        hard_mask = alpha_canvas > 0.5  # [H, W, 1] bool, broadcasts over channels
        blended = tf.where(hard_mask, obj_canvas, bg_img)
        bg_data = dict(bg_data)
        bg_data['image'] = tf.cast(blended, tf.uint8)

        # --- Update annotations ---
        # Compute the new bbox for the pasted object in background normalised coords.
        orig_box = obj_data.get('orig_bbox', tf.constant([0.0, 0.0, 1.0, 1.0]))
        # orig_box: yxyx normalised in obj image.
        paste_y_f = tf.cast(paste_y, tf.float32)
        paste_x_f = tf.cast(paste_x, tf.float32)
        new_h_f   = tf.cast(new_h, tf.float32)
        new_w_f   = tf.cast(new_w, tf.float32)

        new_ymin = (paste_y_f + orig_box[0] * new_h_f) / H_f
        new_xmin = (paste_x_f + orig_box[1] * new_w_f) / W_f
        new_ymax = (paste_y_f + orig_box[2] * new_h_f) / H_f
        new_xmax = (paste_x_f + orig_box[3] * new_w_f) / W_f
        new_box  = tf.clip_by_value(
            tf.reshape(tf.stack([new_ymin, new_xmin, new_ymax, new_xmax]), [1, 4]),
            0.0, 1.0,
        )  # [1, 4]

        # Append box and class.
        bg_data['groundtruth_boxes'] = tf.concat(
            [bg_data['groundtruth_boxes'], new_box], axis=0
        )
        new_cls = tf.reshape(obj_data.get('label', tf.constant(0, tf.int64)), [1])
        bg_data['groundtruth_classes'] = tf.concat(
            [bg_data['groundtruth_classes'], new_cls], axis=0
        )

        # Append is_crowd, area, dontcare with default values.
        bg_data['groundtruth_is_crowd'] = tf.concat(
            [bg_data['groundtruth_is_crowd'], tf.constant([False])], axis=0
        )
        # Area from the CLIPPED box (new_box), not the raw pre-clip extents; a
        # pasted object overhanging the canvas edge otherwise overstates its
        # visible area.
        box_area = (new_box[0, 2] - new_box[0, 0]) * (new_box[0, 3] - new_box[0, 1])
        bg_data['groundtruth_area'] = tf.concat(
            [bg_data['groundtruth_area'], tf.reshape(box_area, [1])], axis=0
        )
        bg_data['groundtruth_dontcare'] = tf.concat(
            [bg_data['groundtruth_dontcare'], tf.constant([0], tf.int64)], axis=0
        )
        # Pasted object has no distance measurement; append the sentinel.
        if 'groundtruth_dists' in bg_data:
            bg_data['groundtruth_dists'] = tf.concat(
                [bg_data['groundtruth_dists'], tf.constant([-1.0])], axis=0
            )

        # Append polygon (transform from obj-normalised to bg-normalised).
        obj_pts = obj_data.get('points', tf.constant([], tf.float32))
        # Use the STATIC shape (a Python int, e.g. 3972 from CopyPasteDecoder), not
        # tf.shape(...) which returns a symbolic Tensor. This branch runs inside the
        # tf.cond lambda in process_fn(); AutoGraph does not convert it, so a Python
        # `if` on a symbolic Tensor raises OperatorNotAllowedInGraphError at .map()
        # trace time. A static int keeps `max_v == 0` a real Python bool.
        max_v = obj_pts.shape[0] if obj_pts.shape.rank and obj_pts.shape[0] is not None \
            else tf.shape(obj_pts)[0]

        if isinstance(max_v, int) and max_v == 0:
            # No polygon points; use an empty padded polygon.
            n_poly_cols = tf.shape(bg_data['groundtruth_polygons'])[1]
            new_poly = tf.fill([1, n_poly_cols], -1.0)
        else:
            n_pairs = max_v // 2
            pts = tf.reshape(obj_pts, [n_pairs, 2])       # [n_pairs, (x, y)]
            valid = pts[:, 0] > -1.0                      # [n_pairs]; reserved sentinel is exactly -1.0

            # Transform: x_bg = (paste_x + x_obj * new_w) / W
            x_bg = (paste_x_f + pts[:, 0] * new_w_f) / W_f
            y_bg = (paste_y_f + pts[:, 1] * new_h_f) / H_f
            # A valid vertex that lands outside the background image is invalidated
            # (-1 sentinel), UNLIKE mosaic (transform_boxes_polygons / random_perspective)
            # which clips out-of-frame vertices to the edge for box-GT consistency.
            # Clipping here would pin the vertex to the edge and inject a wrong radial
            # distance into the pasted object's GT, so it is dropped instead.
            in_bounds = tf.logical_and(
                tf.logical_and(x_bg >= 0.0, x_bg <= 1.0),
                tf.logical_and(y_bg >= 0.0, y_bg <= 1.0),
            )
            keep = tf.logical_and(valid, in_bounds)
            neg1 = tf.fill(tf.shape(x_bg), -1.0)
            x_bg = tf.where(keep, x_bg, neg1)
            y_bg = tf.where(keep, y_bg, neg1)

            new_pts = tf.reshape(tf.stack([x_bg, y_bg], axis=-1), [1, max_v])

            # Fit to the bg polygon column count. The copy-paste source decoder does
            # not resample, so an object can carry more vertices than the background's
            # (possibly resampled) width. Evenly resample the valid vertices to the
            # column budget; slicing the first n_poly_cols would keep only a
            # contiguous arc of the contour and corrupt the PolyYOLO radial target.
            from data_pipeline.augmentations import resample_polygons
            n_poly_cols = tf.shape(bg_data['groundtruth_polygons'])[1]
            cur_cols = tf.shape(new_pts)[1]
            new_pts = tf.cond(
                cur_cols >= n_poly_cols,
                # compact=True: the in-bounds invalidation above (tf.where -> -1)
                # can leave scattered sentinels, so the valid vertices are NOT a
                # prefix here and must be compacted before the even-spaced resample.
                lambda: resample_polygons(new_pts, n_poly_cols // 2, compact=True),
                lambda: tf.pad(
                    new_pts,
                    [[0, 0], [0, n_poly_cols - cur_cols]],
                    constant_values=-1.0,
                ),
            )
            new_poly = new_pts  # [1, n_poly_cols]

        bg_data['groundtruth_polygons'] = tf.concat(
            [bg_data['groundtruth_polygons'], new_poly], axis=0
        )

        if self._max_occlusion_frac is None:
            return bg_data

        # Occlusion gate: fraction of each PRE-EXISTING GT box covered by the
        # pasted object's box (intersection / existing-box area). The hard-mask
        # composite replaces those pixels while the old label is kept unchanged,
        # so a heavily-covered object would train the model on a box/class that
        # no longer matches the image. Skip the paste in that case. The paste
        # box is a conservative proxy for the alpha mask (mask area <= box area).
        orig_boxes = orig_data['groundtruth_boxes']  # [N, 4] yxyx normalized
        inter_ymin = tf.maximum(orig_boxes[:, 0], new_box[0, 0])
        inter_xmin = tf.maximum(orig_boxes[:, 1], new_box[0, 1])
        inter_ymax = tf.minimum(orig_boxes[:, 2], new_box[0, 2])
        inter_xmax = tf.minimum(orig_boxes[:, 3], new_box[0, 3])
        inter_area = (
            tf.maximum(inter_ymax - inter_ymin, 0.0)
            * tf.maximum(inter_xmax - inter_xmin, 0.0)
        )
        box_areas = tf.maximum(
            (orig_boxes[:, 2] - orig_boxes[:, 0])
            * (orig_boxes[:, 3] - orig_boxes[:, 1]),
            1e-9,
        )
        # [0.0] guard keeps the reduce_max defined when there is no existing GT.
        max_occlusion = tf.reduce_max(
            tf.concat([inter_area / box_areas, [0.0]], axis=0)
        )
        return tf.cond(
            max_occlusion > self._max_occlusion_frac,
            lambda: orig_data,
            lambda: bg_data,
        )

    # ------------------------------------------------------------------
    # Dataset-map interface
    # ------------------------------------------------------------------

    def process_fn(self, is_training: bool = True) -> Callable:
        """Return a function suitable for use in tf.data.Dataset.map().

        The returned function accepts (bg_data, obj_data) and returns bg_data
        updated with the pasted object.
        """
        prob = self._prob

        def _fn(bg_data: Dict, obj_data: Dict) -> Dict:
            do_paste = tf.random.uniform([]) < prob
            if not is_training:
                return bg_data
            return tf.cond(
                do_paste,
                lambda: self._copy_and_paste(bg_data, obj_data),
                lambda: bg_data,
            )

        return _fn
