"""Copy-Paste augmentation for instance-level data augmentation.

Applied before Mosaic in the pipeline with probability prob_copy_n_paste=0.2.
Object sources come from a separate RGBA TFDS (cleaner_copy_paste:1.0.0).
The alpha channel is used as a compositing mask; polygons are transformed
with the same affine applied to the object crop.

Classes:
    CopyAndPasteModule: Wraps the augmentation as a callable pipeline stage.
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
    """

    def __init__(
        self,
        prob: float = 0.2,
        min_height: float = 60,
        min_width: float = 100,
        max_resize_ratio: float = 1.5,
        min_resize_ratio: float = 0.2,
        height_limit: float = 0.6,
    ):
        self._prob = prob
        self._min_height = min_height
        self._min_width = min_width
        self._max_resize_ratio = max_resize_ratio
        self._min_resize_ratio = min_resize_ratio
        self._height_limit = height_limit

    # ------------------------------------------------------------------
    # Core compositing
    # ------------------------------------------------------------------

    def _copy_and_paste(
        self,
        bg_data: Dict[str, tf.Tensor],
        obj_data: Dict[str, tf.Tensor],
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
            bg_data dict with the pasted object appended to annotations.
        """
        bg_img = tf.cast(bg_data['image'], tf.float32)      # [H, W, 3]
        H = tf.shape(bg_img)[0]
        W = tf.shape(bg_img)[1]
        H_f = tf.cast(H, tf.float32)
        W_f = tf.cast(W, tf.float32)

        obj_rgba = tf.cast(obj_data['image'], tf.float32)  # [H_o, W_o, 4]
        obj_rgb  = obj_rgba[:, :, :3]                       # [H_o, W_o, 3]
        alpha    = obj_rgba[:, :, 3:4] / 255.0              # [H_o, W_o, 1]

        # Resize object by a ratio applied directly to its own dimensions.
        #
        # Resolution correction: the background may already have been resized
        # (the pipeline pre-resizes to the model input size BEFORE copy-paste so
        # the composite runs on 672² pixels, not full resolution). The object's
        # target size is defined relative to the ORIGINAL background dims — so
        # scale it by (current/original) per axis. This commutes exactly with
        # the background resize: compositing here then is pixel-equivalent in
        # geometry to compositing at full resolution and resizing afterwards.
        # When the background is unresized, height/width == current dims and
        # the correction is 1 (fully backward compatible).
        orig_h_f = tf.cast(bg_data.get('height', H), tf.float32)
        orig_w_f = tf.cast(bg_data.get('width', W), tf.float32)
        corr_h = H_f / tf.maximum(orig_h_f, 1.0)
        corr_w = W_f / tf.maximum(orig_w_f, 1.0)

        resize_ratio = tf.random.uniform([], self._min_resize_ratio, self._max_resize_ratio)
        obj_h_f = tf.cast(tf.shape(obj_rgb)[0], tf.float32)
        obj_w_f = tf.cast(tf.shape(obj_rgb)[1], tf.float32)
        new_h = tf.maximum(tf.cast(tf.round(obj_h_f * resize_ratio * corr_h), tf.int32), 1)
        new_w = tf.maximum(tf.cast(tf.round(obj_w_f * resize_ratio * corr_w), tf.int32), 1)

        # Resize obj and alpha
        obj_rgb_r = tf.image.resize(obj_rgb, [new_h, new_w], method='bilinear')
        alpha_r   = tf.image.resize(alpha,   [new_h, new_w], method='bilinear')
        alpha_r   = tf.clip_by_value(alpha_r, 0.0, 1.0)

        # Random placement in [10%, height_limit] × [10%, 90%] of background.
        _margin = 0.1
        min_y = tf.cast(H_f * _margin, tf.int32)
        min_x = tf.cast(W_f * _margin, tf.int32)
        max_y = tf.maximum(tf.cast(H_f * self._height_limit, tf.int32) - new_h, min_y)
        max_x = tf.maximum(tf.cast(W_f * (1.0 - _margin), tf.int32) - new_w, min_x)

        paste_y = tf.random.uniform([], min_y, max_y + 1, dtype=tf.int32)
        paste_x = tf.random.uniform([], min_x, max_x + 1, dtype=tf.int32)

        # Build full-canvas alpha mask (0 everywhere except paste region)
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

        # Hard-mask composite: use alpha > 0.5 as binary mask (matches old codebase).
        hard_mask = alpha_canvas > 0.5  # [H, W, 1] bool, broadcasts over channels
        blended = tf.where(hard_mask, obj_canvas, bg_img)
        bg_data = dict(bg_data)
        bg_data['image'] = tf.cast(blended, tf.uint8)

        # --- Update annotations ---
        # Compute the new bbox for the pasted object in background normalised coords.
        orig_box = obj_data.get('orig_bbox', tf.constant([0.0, 0.0, 1.0, 1.0]))
        # orig_box: yxyx normalised in obj image
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

        # Append box, class
        bg_data['groundtruth_boxes'] = tf.concat(
            [bg_data['groundtruth_boxes'], new_box], axis=0
        )
        new_cls = tf.reshape(obj_data.get('label', tf.constant(0, tf.int64)), [1])
        bg_data['groundtruth_classes'] = tf.concat(
            [bg_data['groundtruth_classes'], new_cls], axis=0
        )

        # Append is_crowd, area, dontcare with default values
        bg_data['groundtruth_is_crowd'] = tf.concat(
            [bg_data['groundtruth_is_crowd'], tf.constant([False])], axis=0
        )
        box_area = (new_ymax - new_ymin) * (new_xmax - new_xmin)
        bg_data['groundtruth_area'] = tf.concat(
            [bg_data['groundtruth_area'], tf.reshape(box_area, [1])], axis=0
        )
        bg_data['groundtruth_dontcare'] = tf.concat(
            [bg_data['groundtruth_dontcare'], tf.constant([0], tf.int64)], axis=0
        )
        # Pasted object has no distance measurement — append sentinel
        if 'groundtruth_dists' in bg_data:
            bg_data['groundtruth_dists'] = tf.concat(
                [bg_data['groundtruth_dists'], tf.constant([-1.0])], axis=0
            )

        # Append polygon (transform from obj-normalised to bg-normalised)
        obj_pts = obj_data.get('points', tf.constant([], tf.float32))
        # Use the STATIC shape (a Python int, e.g. 3972 from CopyPasteDecoder), not
        # tf.shape(...) which returns a symbolic Tensor. This branch runs inside the
        # tf.cond lambda in process_fn(); AutoGraph does not convert it, so a Python
        # `if` on a symbolic Tensor raises OperatorNotAllowedInGraphError at .map()
        # trace time. A static int keeps `max_v == 0` a real Python bool.
        max_v = obj_pts.shape[0] if obj_pts.shape.rank and obj_pts.shape[0] is not None \
            else tf.shape(obj_pts)[0]

        if isinstance(max_v, int) and max_v == 0:
            # No polygon points — use empty padded polygon
            n_poly_cols = tf.shape(bg_data['groundtruth_polygons'])[1]
            new_poly = tf.fill([1, n_poly_cols], -1.0)
        else:
            n_pairs = max_v // 2
            pts = tf.reshape(obj_pts, [n_pairs, 2])       # [n_pairs, (x, y)]
            valid = pts[:, 0] > -1.0                      # [n_pairs] — sentinel is -1.0 (design_register entry 10)

            # Transform: x_bg = (paste_x + x_obj * new_w) / W
            x_bg = (paste_x_f + pts[:, 0] * new_w_f) / W_f
            y_bg = (paste_y_f + pts[:, 1] * new_h_f) / H_f
            # A valid vertex that lands outside the background image is invalidated
            # (-1 sentinel), UNLIKE mosaic (transform_boxes_polygons / random_perspective)
            # which CLIPS out-of-frame vertices to the edge for box-GT consistency
            # (design_register entry 9). Here clipping would pin the vertex to the edge
            # and inject a wrong radial distance into the pasted object's GT, so we drop
            # it instead.
            in_bounds = tf.logical_and(
                tf.logical_and(x_bg >= 0.0, x_bg <= 1.0),
                tf.logical_and(y_bg >= 0.0, y_bg <= 1.0),
            )
            keep = tf.logical_and(valid, in_bounds)
            neg1 = tf.fill(tf.shape(x_bg), -1.0)
            x_bg = tf.where(keep, x_bg, neg1)
            y_bg = tf.where(keep, y_bg, neg1)

            new_pts = tf.reshape(tf.stack([x_bg, y_bg], axis=-1), [1, max_v])

            # Fit to the bg polygon column count. The copy-paste source decoder
            # does NOT resample, so an object can carry far more vertices than the
            # background's (possibly resampled) width. When it does, EVENLY RESAMPLE
            # the valid vertices to the column budget rather than taking the first
            # n_poly_cols raw vertices — the latter keeps only a contiguous arc of
            # the contour, which corrupts the PolyYOLO radial target (it discards
            # the far side of the polygon entirely). resample_polygons preserves the
            # per-bin max radius to within sampling resolution, identical to the
            # decode-time resample the rest of the pipeline already relies on.
            from data_pipeline.augmentations import resample_polygons
            n_poly_cols = tf.shape(bg_data['groundtruth_polygons'])[1]
            cur_cols = tf.shape(new_pts)[1]
            new_pts = tf.cond(
                cur_cols >= n_poly_cols,
                lambda: resample_polygons(new_pts, n_poly_cols // 2),
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

        return bg_data

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
