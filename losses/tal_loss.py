"""Task-Aligned Learning loss with polygon and distance extensions.

Loss components and gains (from experiment_config.yaml):
    iou_gain:        7.5   CIoU box loss
    dfl_gain:        1.5   Distribution Focal Loss (DFL) for box regression
    cls_gain:        0.5   BCE classification loss
    dist_gain:       1.0   L1 distance loss (on valid samples only)
    poly_dist_gain:  0.45  PolyYOLO radial distance regression
    poly_angle_gain: 0.4   PolyYOLO angle classification
    poly_conf_gain:  0.2   PolyYOLO vertex confidence

TAL assignment parameters:
    tal_alpha: 0.5   (score exponent in alignment metric)
    tal_beta:  6.0   (IoU exponent in alignment metric)
    topk:      10    (top-k candidates per GT)

ignore_bg logic:
    ignore_bg=0: standard BCE on all anchors (detection data)
    ignore_bg=1: class loss masked to foreground only (distance data)

Classes:
    TaskAlignedLossExtended: Full loss computation with TAL assignment.
"""

import math
from typing import Dict, Optional, Tuple

import tensorflow as tf

from losses.distance_loss import (
    INVALID_DISTANCE_SENTINEL,
    distance_l1_loss,
)
from losses.polygon_loss import (
    polygon_angle_loss,
    polygon_conf_loss,
    polygon_dist_loss,
)
from losses.tal_assigner import TaskAlignedAssigner


def _replica_sum(x: tf.Tensor) -> tf.Tensor:
    """Sum a scalar across replicas (no-op under a single replica).

    The loss normalizers (num_objs, target_scores_sum) must be the GLOBAL counts
    so that, with MirroredStrategy summing per-replica gradients, the result equals
    the single-device gradient. Under one replica this returns ``x`` unchanged, so
    single-device training is numerically identical.
    """
    ctx = tf.distribute.get_replica_context()
    if ctx is None or ctx.num_replicas_in_sync == 1:
        return x
    return ctx.all_reduce(tf.distribute.ReduceOp.SUM, x)

_LEVEL_STRIDES = {"3": 8, "4": 16, "5": 32}


# ---------------------------------------------------------------------------
# CIoU helper (module-level to avoid closure overhead)
# ---------------------------------------------------------------------------

def _ciou_loss(b1: tf.Tensor, b2: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """Complete IoU loss between two sets of xyxy boxes.

    Args:
        b1, b2: float32 [..., 4]  xyxy in the same coordinate system.

    Returns:
        float32 [...]  per-box CIoU loss in [0, 2].
    """
    ix1 = tf.maximum(b1[..., 0], b2[..., 0])
    iy1 = tf.maximum(b1[..., 1], b2[..., 1])
    ix2 = tf.minimum(b1[..., 2], b2[..., 2])
    iy2 = tf.minimum(b1[..., 3], b2[..., 3])
    inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)

    a1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    a2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = a1 + a2 - inter + eps
    iou = inter / union

    # Center distance squared
    cx1 = (b1[..., 0] + b1[..., 2]) * 0.5
    cy1 = (b1[..., 1] + b1[..., 3]) * 0.5
    cx2 = (b2[..., 0] + b2[..., 2]) * 0.5
    cy2 = (b2[..., 1] + b2[..., 3]) * 0.5
    rho2 = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)

    # Enclosing box diagonal squared
    ex1 = tf.minimum(b1[..., 0], b2[..., 0])
    ey1 = tf.minimum(b1[..., 1], b2[..., 1])
    ex2 = tf.maximum(b1[..., 2], b2[..., 2])
    ey2 = tf.maximum(b1[..., 3], b2[..., 3])
    c2  = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps

    # Aspect-ratio consistency penalty
    w1 = b1[..., 2] - b1[..., 0]
    h1 = b1[..., 3] - b1[..., 1]
    w2 = b2[..., 2] - b2[..., 0]
    h2 = b2[..., 3] - b2[..., 1]
    v      = (4.0 / (math.pi ** 2)) * tf.square(
        tf.math.atan2(w2, h2 + eps) - tf.math.atan2(w1, h1 + eps)
    )
    alpha_v = v / (1.0 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_v * v
    return 1.0 - ciou


# ---------------------------------------------------------------------------

class TaskAlignedLossExtended:
    """TAL-based loss for box, cls, polygon, and distance prediction."""

    def __init__(
        self,
        num_classes: int = 39,
        iou_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        dist_gain: float = 1.0,
        poly_dist_gain: float = 0.45,
        poly_conf_gain: float = 0.2,
        poly_angle_gain: float = 0.4,
        poly_gain: float = 0.5,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
        topk: int = 10,
        reg_max: int = 16,
        with_polygons: bool = True,
        with_distance: bool = True,
        angle_step: int = 15,
        use_acsl: bool = False,
    ):
        self.num_classes    = num_classes
        self.iou_gain       = iou_gain
        self.cls_gain       = cls_gain
        self.dfl_gain       = dfl_gain
        self.dist_gain      = dist_gain
        self.poly_dist_gain = poly_dist_gain
        self.poly_conf_gain = poly_conf_gain
        self.poly_angle_gain= poly_angle_gain
        self.poly_gain      = poly_gain
        self.with_polygons  = with_polygons
        self.with_distance  = with_distance
        self.reg_max        = reg_max
        self.angle_step     = angle_step
        self.use_acsl       = use_acsl
        self.num_vertices   = 360 // angle_step  # = 24

        # ACSL (Adaptive Class Suppression Loss) is parsed from config (AcslConfig)
        # but the weighting math is NOT implemented here. The flag used to be a
        # silent no-op: a user could set `use_acsl: true` in YAML and train as if
        # it had taken effect. Fail loud instead so the dead knob can never lie.
        # See docs/design_register.md entry "ACSL config knob is not implemented".
        if use_acsl:
            raise NotImplementedError(
                "use_acsl=True is not supported: the ACSL class-suppression "
                "weighting is parsed from config (AcslConfig) but not implemented "
                "in TaskAlignedLossExtended._class_loss. Set acsl.use_acsl=false "
                "(the default) until the weighting is implemented. See "
                "docs/design_register.md."
            )

        self._assigner_fn = TaskAlignedAssigner(
            topk=topk, alpha=tal_alpha, beta=tal_beta, angle_step=angle_step
        )
        # DFL bin indices: [0, 1, ..., reg_max-1]
        self._dfl_bins = tf.cast(tf.range(reg_max), tf.float32)

    # ------------------------------------------------------------------

    def _assigner(
        self,
        pd_scores: tf.Tensor,
        pd_bboxes: tf.Tensor,
        anc_points: tf.Tensor,
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
        mask_gt: tf.Tensor,
        gt_polys: Optional[tf.Tensor] = None,
        gt_dists: Optional[tf.Tensor] = None,
    ) -> Tuple:
        """Task-Aligned label assignment.

        Assignment steps:
            1. Compute alignment metric: score^alpha * IoU^beta.
            2. Select top-k candidates per GT.
            3. Apply spatial constraint (anchor center inside GT box).
            4. Handle duplicate assignments (keep max-IoU GT).

        Returns:
            target_labels, target_bboxes, target_scores,
            target_polygons, target_dists, fg_mask, target_gt_idx
        """
        return self._assigner_fn(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=gt_polys, gt_dists=gt_dists,
        )

    # ------------------------------------------------------------------

    def _box_loss(
        self,
        pd_bboxes: tf.Tensor,
        target_bboxes: tf.Tensor,
        target_scores: tf.Tensor,   # [B, A, C] — per-anchor weighting (Ultralytics)
        target_scores_sum: tf.Tensor,
        fg_mask: tf.Tensor,
        pd_box_raw: tf.Tensor,
        anc_strides: tf.Tensor,     # [A, 1]  — added for DFL target normalisation
        anc_points: tf.Tensor,      # [A, 2]  — added to build LTRB targets
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """CIoU loss + DFL loss for foreground anchors.

        Both terms are weighted per-anchor by ``sum(target_scores, -1)`` so that
        better-aligned anchors dominate the box gradient, matching the reference
        Ultralytics YOLOv8 recipe (``(loss * weight).sum() / target_scores_sum``).

        Returns:
            (ciou_loss, dfl_loss) both scalar tensors.
        """
        fg_float = tf.cast(fg_mask, tf.float32)           # [B, A]
        weight   = tf.reduce_sum(target_scores, axis=-1)  # [B, A]

        # ── CIoU ──────────────────────────────────────────────────────
        ciou = _ciou_loss(pd_bboxes, target_bboxes)        # [B, A]
        ciou_loss = tf.reduce_sum(ciou * weight * fg_float) / target_scores_sum

        # ── DFL ───────────────────────────────────────────────────────
        # Target LTRB offsets in feature-map units
        cx = anc_points[:, 0]   # [A]
        cy = anc_points[:, 1]   # [A]

        tgt_l = cx[tf.newaxis] - target_bboxes[..., 0]   # [B, A]
        tgt_t = cy[tf.newaxis] - target_bboxes[..., 1]
        tgt_r = target_bboxes[..., 2] - cx[tf.newaxis]
        tgt_b = target_bboxes[..., 3] - cy[tf.newaxis]

        tgt_ltrb_px = tf.stack([tgt_l, tgt_t, tgt_r, tgt_b], axis=-1)   # [B, A, 4]
        tgt_ltrb_fm = tgt_ltrb_px / anc_strides[tf.newaxis]              # [B, A, 4]
        tgt_ltrb_fm = tf.clip_by_value(
            tgt_ltrb_fm, 0.0, float(self.reg_max) - 1.001
        )

        tgt_floor        = tf.floor(tgt_ltrb_fm)                         # [B, A, 4]
        weight_right     = tgt_ltrb_fm - tgt_floor
        weight_left      = 1.0 - weight_right

        B_val = tf.shape(pd_box_raw)[0]
        A_val = tf.shape(pd_box_raw)[1]
        pd_logits = tf.reshape(pd_box_raw, [B_val, A_val, 4, self.reg_max])
        pd_log_softmax = tf.nn.log_softmax(pd_logits, axis=-1)           # [B, A, 4, R]

        fl_idx = tf.cast(tgt_floor, tf.int32)                            # [B, A, 4]
        cl_idx = tf.minimum(fl_idx + 1, self.reg_max - 1)

        # Gather log-probs via one-hot matmul
        fl_oh = tf.one_hot(fl_idx, self.reg_max)                         # [B, A, 4, R]
        cl_oh = tf.one_hot(cl_idx, self.reg_max)

        log_p_fl = tf.reduce_sum(pd_log_softmax * fl_oh, axis=-1)        # [B, A, 4]
        log_p_cl = tf.reduce_sum(pd_log_softmax * cl_oh, axis=-1)

        dfl_raw  = -(weight_left * log_p_fl + weight_right * log_p_cl)   # [B, A, 4]
        dfl_mean = tf.reduce_mean(dfl_raw, axis=-1)                       # [B, A]
        dfl_loss = tf.reduce_sum(dfl_mean * weight * fg_float) / target_scores_sum

        return ciou_loss, dfl_loss

    # ------------------------------------------------------------------

    def _class_loss(
        self,
        pred_scores: tf.Tensor,
        target_scores: tf.Tensor,
        target_scores_sum: tf.Tensor,
        fg_mask: tf.Tensor,
        ignore_bg: tf.Tensor,
    ) -> tf.Tensor:
        """BCE classification loss, with ignore_bg and optional ACSL weighting."""
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_scores, logits=pred_scores
        )  # [B, A, C]
        bce_sum = tf.reduce_sum(bce, axis=-1)   # [B, A]

        # ignore_bg=1 → apply loss only on foreground anchors for that image
        ignore_bg_f = tf.cast(ignore_bg, tf.float32)                   # [B]
        fg_float    = tf.cast(fg_mask, tf.float32)                     # [B, A]
        # mask = 1.0 when ignore_bg=0; mask = fg when ignore_bg=1
        mask = (
            (1.0 - ignore_bg_f[:, tf.newaxis]) +
            ignore_bg_f[:, tf.newaxis] * fg_float
        )  # [B, A]

        return tf.reduce_sum(bce_sum * mask) / target_scores_sum

    # ------------------------------------------------------------------

    def _distance_loss(
        self,
        pd_dist: tf.Tensor,
        target_dist: tf.Tensor,
        fg_mask: tf.Tensor,
        num_objs: tf.Tensor,
    ) -> tf.Tensor:
        """L1 loss on log-scale distances, masked to valid GT entries (> -10.0).

        Normalized by ``num_objs`` (total GT object count in the batch), matching
        the old-codebase convention. dist_gain is calibrated to this scale.
        """
        return distance_l1_loss(pd_dist, target_dist, fg_mask, num_objs)

    # ------------------------------------------------------------------

    def _polygon_loss(
        self,
        pd_poly_angle: tf.Tensor,
        pd_poly_dist: tf.Tensor,
        pd_poly_conf: tf.Tensor,
        target_polygons: tf.Tensor,
        fg_mask: tf.Tensor,
        num_objs: tf.Tensor,
        ignore_bg: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Combined PolyYOLO polygon loss (angle + dist + conf).

        target_polygons layout: [dist0, angle0, conf0, dist1, angle1, conf1, ...]
            dist:  radial distance from box center (pre-computed by parser).
            angle: sub-bin angular offset (vertex_angle - bin_start)/angle_step
                   in [0, 1) on bins that hold a vertex, 0.0 elsewhere.
            conf:  1.0 if a valid vertex was assigned to this bin (the validity
                   mask used by the angle/dist losses).

        ``ignore_bg`` ([B] int) marks distance-stream images (ignore_bg=1) that
        carry NO polygon GT — their ``target_polygons`` is all-zero, so every
        conf bin reads 0. The conf loss trains on ALL bins (negative signal on
        empty bins), so without this guard a distance-stream foreground anchor
        (a real object) would be trained to emit conf≈0 on every vertex, which
        is wrong: those rows have no polygon label, not an empty polygon. We
        therefore zero the polygon loss on ignore_bg=1 rows entirely (angle/dist
        already contribute zero there since their vertex mask is empty, but conf
        would not). This mirrors the ignore_bg guard in ``_class_loss``.

        All three normalize by num_objs. Angle and dist average over the VALID
        vertices only (masked by conf). Conf averages over ALL bins to provide a
        negative signal that suppresses confidence on empty bins (it is NOT
        conf-masked, unlike angle/dist).

        Returns:
            (poly_total, angle_loss, dist_loss_val, conf_loss_val)
            poly_total:    weighted sum (poly_gain * (angle_gain*angle + dist_gain*dist + conf_gain*conf))
            angle_loss:    raw pre-gain polygon angle loss
            dist_loss_val: raw polygon distance loss
            conf_loss_val: raw polygon confidence loss
        """
        target_dist  = target_polygons[:, :, 0::3]   # [B, A, 24]
        target_angle = target_polygons[:, :, 1::3]   # [B, A, 24] — sub-bin offset
        conf         = target_polygons[:, :, 2::3]   # [B, A, 24] — per-bin validity
        vertex_mask  = conf                          # valid-vertex mask for angle/dist

        # ignore_bg=1 rows (distance stream) carry no polygon GT: drop them from
        # the foreground mask so the all-bins conf loss does not push their real
        # objects' conf to 0. ignore_bg=0 rows keep their full fg_mask.
        keep_row    = 1.0 - tf.cast(ignore_bg, tf.float32)[:, tf.newaxis]  # [B, 1]
        poly_fg     = tf.cast(fg_mask, tf.float32) * keep_row              # [B, A] float
        poly_fg_b   = poly_fg > 0.5                                        # [B, A] bool

        angle_loss = polygon_angle_loss(
            pd_poly_angle, target_angle, vertex_mask, poly_fg_b, num_objs
        )
        dist_loss_val = polygon_dist_loss(
            pd_poly_dist, target_dist, vertex_mask, poly_fg_b, num_objs
        )
        conf_loss_val = polygon_conf_loss(
            pd_poly_conf, conf, vertex_mask, poly_fg_b, num_objs
        )

        poly_total = self.poly_gain * (
            self.poly_angle_gain * angle_loss +
            self.poly_dist_gain  * dist_loss_val +
            self.poly_conf_gain  * conf_loss_val
        )

        return poly_total, angle_loss, dist_loss_val, conf_loss_val

    # ------------------------------------------------------------------

    def __call__(
        self,
        feats: Dict[str, Dict[str, tf.Tensor]],
        batch: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, ...]:
        """Compute total loss.

        Returns:
            9-tuple: (total_loss, box_loss, dfl_loss, cls_loss, dist_loss,
                      poly_loss, poly_angle_loss, poly_dist_loss, poly_conf_loss)
            The last three are raw (pre-gain-weighted) polygon sub-losses;
            poly_loss is the fully gain-weighted combined polygon term.
        """
        # ── 1. Flatten FPN outputs and build anchor grid ──────────────
        box_raw_list, cls_list = [], []
        poly_a_list, poly_d_list, poly_c_list, dist_list = [], [], [], []
        anc_list, stride_list = [], []

        for level_str, stride_val in _LEVEL_STRIDES.items():
            box_lvl = feats["box"][level_str]                # [B, H, W, 64]
            B_val   = tf.shape(box_lvl)[0]
            fH      = tf.shape(box_lvl)[1]
            fW      = tf.shape(box_lvl)[2]
            A_lvl   = fH * fW

            box_raw_list.append(tf.reshape(box_lvl, [B_val, A_lvl, -1]))
            cls_list.append(
                tf.reshape(feats["cls"][level_str], [B_val, A_lvl, -1])
            )

            if self.with_polygons:
                poly_a_list.append(
                    tf.reshape(feats["poly_angle"][level_str], [B_val, A_lvl, -1])
                )
                poly_d_list.append(
                    tf.reshape(feats["poly_dist"][level_str], [B_val, A_lvl, -1])
                )
                poly_c_list.append(
                    tf.reshape(feats["poly_conf"][level_str], [B_val, A_lvl, -1])
                )

            if self.with_distance:
                dist_list.append(
                    tf.reshape(feats["dist"][level_str], [B_val, A_lvl, 1])
                )

            # Anchor grid for this FPN level
            ys = (tf.cast(tf.range(fH), tf.float32) + 0.5) * stride_val
            xs = (tf.cast(tf.range(fW), tf.float32) + 0.5) * stride_val
            grid_x, grid_y = tf.meshgrid(xs, ys)
            cx_flat = tf.reshape(grid_x, [-1])
            cy_flat = tf.reshape(grid_y, [-1])
            anc_list.append(tf.stack([cx_flat, cy_flat], axis=-1))     # [H*W, 2]
            stride_list.append(
                tf.fill([A_lvl, 1], tf.cast(stride_val, tf.float32))
            )

        pd_box_raw  = tf.concat(box_raw_list, axis=1)   # [B, A, 64]
        pd_cls      = tf.concat(cls_list,     axis=1)   # [B, A, C]
        anc_points  = tf.concat(anc_list,     axis=0)   # [A, 2]
        anc_strides = tf.concat(stride_list,  axis=0)   # [A, 1]

        pd_poly_angle = tf.concat(poly_a_list, axis=1) if self.with_polygons else None
        pd_poly_dist  = tf.concat(poly_d_list, axis=1) if self.with_polygons else None
        pd_poly_conf  = tf.concat(poly_c_list, axis=1) if self.with_polygons else None
        pd_dist       = tf.concat(dist_list,   axis=1) if self.with_distance else None

        # ── 2. Decode DFL → xyxy pixel boxes ──────────────────────────
        B_val = tf.shape(pd_box_raw)[0]
        A_val = tf.shape(pd_box_raw)[1]
        logits_4 = tf.reshape(pd_box_raw, [B_val, A_val, 4, self.reg_max])
        probs    = tf.nn.softmax(logits_4, axis=-1)
        ltrb     = tf.reduce_sum(probs * self._dfl_bins, axis=-1)         # [B, A, 4]
        ltrb_px  = ltrb * anc_strides[tf.newaxis]                         # [B, A, 4]

        acx = anc_points[:, 0]   # [A]
        acy = anc_points[:, 1]
        pd_bboxes = tf.stack(
            [
                acx[tf.newaxis] - ltrb_px[..., 0],   # x1
                acy[tf.newaxis] - ltrb_px[..., 1],   # y1
                acx[tf.newaxis] + ltrb_px[..., 2],   # x2
                acy[tf.newaxis] + ltrb_px[..., 3],   # y2
            ],
            axis=-1,
        )  # [B, A, 4] xyxy pixels

        # ── 3. Prepare GT tensors ──────────────────────────────────────
        gt_bboxes_norm = batch["bbox"]       # [B, M, 4] yxyx normalized [0, 1]
        gt_labels      = batch["classes"]    # [B, M] int64
        n_gt           = batch["n_gt"]       # [B] int64

        # Image size inferred from level-3 feature map (stride 8)
        img_H = tf.cast(tf.shape(feats["box"]["3"])[1] * 8, tf.float32)
        img_W = tf.cast(tf.shape(feats["box"]["3"])[2] * 8, tf.float32)

        # Convert yxyx normalized → xyxy pixel space
        gt_bboxes_px = tf.stack(
            [
                gt_bboxes_norm[..., 1] * img_W,   # x1
                gt_bboxes_norm[..., 0] * img_H,   # y1
                gt_bboxes_norm[..., 3] * img_W,   # x2
                gt_bboxes_norm[..., 2] * img_H,   # y2
            ],
            axis=-1,
        )  # [B, M, 4] xyxy pixels

        # Trim GT tensors to the actual max GT count in this batch.
        # max_num_instances pads to M=300, but typical batches have M_eff<<300.
        # Each [B, A, M] intermediate tensor in the assigner shrinks by M/M_eff
        # (typically 15-60x), which is the dominant memory cost.
        M_eff = tf.maximum(tf.cast(tf.reduce_max(n_gt), tf.int32), 1)

        gt_labels    = gt_labels[:, :M_eff]
        gt_bboxes_px = gt_bboxes_px[:, :M_eff, :]
        mask_gt      = tf.sequence_mask(n_gt, maxlen=M_eff)   # [B, M_eff] bool
        # Global GT count across replicas (no-op single-device) so the per-object
        # normalization matches the single-device gradient under MirroredStrategy.
        num_objs     = tf.maximum(_replica_sum(tf.reduce_sum(tf.cast(mask_gt, tf.float32))), 1.0)

        gt_polys_raw = batch.get("polygons")
        gt_dists_raw = batch.get("log_distance")

        gt_polys = gt_polys_raw[:, :M_eff, :] if (self.with_polygons and gt_polys_raw is not None) else None
        gt_dists = gt_dists_raw[:, :M_eff]    if (self.with_distance and gt_dists_raw is not None) else None

        # ── 4. TAL assignment (stop-gradient) ─────────────────────────
        (
            target_labels,
            target_bboxes,
            target_scores,
            target_polygons,
            target_dists,
            fg_mask,
        ) = self._assigner(
            tf.stop_gradient(tf.sigmoid(pd_cls)),
            tf.stop_gradient(pd_bboxes),
            anc_points,
            gt_labels,
            gt_bboxes_px,
            mask_gt,
            gt_polys=gt_polys,
            gt_dists=gt_dists,
        )

        # Global alignment-score sum across replicas (no-op single-device).
        target_scores_sum = tf.maximum(_replica_sum(tf.reduce_sum(target_scores)), 1.0)

        # ── 5. Component losses ────────────────────────────────────────
        ciou_loss, dfl_loss = self._box_loss(
            pd_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask,
            pd_box_raw, anc_strides, anc_points,
        )

        ignore_bg = batch.get("ignore_bg", tf.zeros([B_val], dtype=tf.int64))
        cls_loss = self._class_loss(
            pd_cls, target_scores, target_scores_sum, fg_mask, ignore_bg,
        )

        dist_loss_val = tf.constant(0.0)
        if self.with_distance and pd_dist is not None:
            dist_loss_val = self._distance_loss(
                pd_dist, target_dists, fg_mask, num_objs
            )

        poly_loss_val = tf.constant(0.0)
        poly_angle_l  = tf.constant(0.0)
        poly_dist_l   = tf.constant(0.0)
        poly_conf_l   = tf.constant(0.0)
        if self.with_polygons and pd_poly_angle is not None:
            poly_loss_val, poly_angle_l, poly_dist_l, poly_conf_l = self._polygon_loss(
                pd_poly_angle, pd_poly_dist, pd_poly_conf,
                target_polygons, fg_mask, num_objs, ignore_bg,
            )

        # ── 6. Apply gains and aggregate ──────────────────────────────
        box_loss_w  = self.iou_gain  * ciou_loss
        dfl_loss_w  = self.dfl_gain  * dfl_loss
        cls_loss_w  = self.cls_gain  * cls_loss
        dist_loss_w = self.dist_gain * dist_loss_val
        # poly_loss_val already includes poly_gain and component gains (applied
        # inside _polygon_loss); poly_angle_l/dist_l/conf_l are raw pre-gain values.

        total_loss = box_loss_w + dfl_loss_w + cls_loss_w + dist_loss_w + poly_loss_val

        return (
            total_loss, box_loss_w, dfl_loss_w, cls_loss_w, dist_loss_w,
            poly_loss_val, poly_angle_l, poly_dist_l, poly_conf_l,
        )
