"""Task-Aligned Learning loss with polygon and distance extensions.

Loss components and gains (from the experiment YAML, e.g.
configs/experiments/yolo/yolov8_poly_dist.yaml):
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

def _bbox_iou_loss(b1: tf.Tensor, b2: tf.Tensor, iou_type: str = "ciou",
                   eps: float = 1e-7) -> tf.Tensor:
    """IoU-family box loss between two sets of xyxy boxes.

    ``iou_type`` selects the penalty added to plain IoU:
        iou   — 1 - IoU
        giou  — Generalized IoU (enclosing-area penalty)
        diou  — Distance IoU (center-distance penalty)
        ciou  — Complete IoU (center distance + aspect-ratio); default
        eiou  — Efficient IoU (center + width + height penalties)
        siou  — SCYLLA IoU (angle + distance + shape cost)

    Args:
        b1, b2: float32 [..., 4]  xyxy in the same coordinate system.

    Returns:
        float32 [...]  per-box loss (0 = perfect overlap).
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

    if iou_type == "iou":
        return 1.0 - iou

    # Center coordinates / distance² and the enclosing box (shared by the rest).
    cx1 = (b1[..., 0] + b1[..., 2]) * 0.5
    cy1 = (b1[..., 1] + b1[..., 3]) * 0.5
    cx2 = (b2[..., 0] + b2[..., 2]) * 0.5
    cy2 = (b2[..., 1] + b2[..., 3]) * 0.5
    rho2 = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)

    ex1 = tf.minimum(b1[..., 0], b2[..., 0])
    ey1 = tf.minimum(b1[..., 1], b2[..., 1])
    ex2 = tf.maximum(b1[..., 2], b2[..., 2])
    ey2 = tf.maximum(b1[..., 3], b2[..., 3])

    if iou_type == "giou":
        enclose_area = (ex2 - ex1) * (ey2 - ey1) + eps
        giou = iou - (enclose_area - union) / enclose_area
        return 1.0 - giou

    c2 = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps

    if iou_type == "diou":
        return 1.0 - (iou - rho2 / c2)

    w1 = b1[..., 2] - b1[..., 0]
    h1 = b1[..., 3] - b1[..., 1]
    w2 = b2[..., 2] - b2[..., 0]
    h2 = b2[..., 3] - b2[..., 1]

    if iou_type == "eiou":
        cw2 = tf.square(ex2 - ex1) + eps
        ch2 = tf.square(ey2 - ey1) + eps
        eiou = iou - rho2 / c2 - tf.square(w1 - w2) / cw2 - tf.square(h1 - h2) / ch2
        return 1.0 - eiou

    if iou_type == "siou":
        # Angle cost: prefer aligning along the shorter center offset.
        s_cw = cx2 - cx1
        s_ch = cy2 - cy1
        sigma = tf.sqrt(tf.square(s_cw) + tf.square(s_ch)) + eps
        sin_a = tf.abs(s_ch) / sigma
        sin_b = tf.abs(s_cw) / sigma
        thr = tf.sqrt(2.0) / 2.0
        sin_alpha = tf.where(sin_a > thr, sin_b, sin_a)
        angle_cost = tf.cos(2.0 * tf.asin(tf.clip_by_value(sin_alpha, -1.0, 1.0)) - math.pi / 2.0)
        # Distance cost (gamma weighted by the angle cost).
        cw = ex2 - ex1 + eps
        ch = ey2 - ey1 + eps
        rho_x = tf.square(s_cw / cw)
        rho_y = tf.square(s_ch / ch)
        gamma = 2.0 - angle_cost
        dist_cost = (1.0 - tf.exp(-gamma * rho_x)) + (1.0 - tf.exp(-gamma * rho_y))
        # Shape cost.
        omega_w = tf.abs(w1 - w2) / (tf.maximum(w1, w2) + eps)
        omega_h = tf.abs(h1 - h2) / (tf.maximum(h1, h2) + eps)
        shape_cost = tf.pow(1.0 - tf.exp(-omega_w), 4.0) + tf.pow(1.0 - tf.exp(-omega_h), 4.0)
        siou = iou - 0.5 * (dist_cost + shape_cost)
        return 1.0 - siou

    # Default: ciou.
    v = (4.0 / (math.pi ** 2)) * tf.square(
        tf.math.atan2(w2, h2 + eps) - tf.math.atan2(w1, h1 + eps)
    )
    # alpha is a constant weighting coefficient, not a differentiated term (the
    # reference recipe computes it under no-grad). Without the stop, the
    # aspect-penalty gradient is inflated up to ~2x and a spurious positive
    # gradient path opens through alpha's IoU dependence, braking IoU
    # improvement whenever the aspect ratios disagree. Forward value unchanged.
    alpha_v = tf.stop_gradient(v / (1.0 - iou + v + eps))
    ciou = iou - rho2 / c2 - alpha_v * v
    return 1.0 - ciou


def _ciou_loss(b1: tf.Tensor, b2: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """Complete IoU loss (back-compat alias for ``_bbox_iou_loss(..., 'ciou')``)."""
    return _bbox_iou_loss(b1, b2, "ciou", eps)


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
        box_iou_type: str = "ciou",
        cls_loss_type: str = "bce",
        weighting: str = "soft",
        label_smoothing: float = 0.0,
        focal_gamma: float = 1.5,
        focal_alpha: float = 0.25,
    ):
        if weighting not in ("soft", "legacy_hard"):
            raise ValueError(
                f"losses.weighting must be 'soft' or 'legacy_hard', got {weighting!r}")
        self.num_classes    = num_classes
        self.box_iou_type   = box_iou_type
        self.cls_loss_type  = cls_loss_type
        self.weighting      = weighting
        self.label_smoothing = label_smoothing
        self.focal_gamma    = focal_gamma
        self.focal_alpha    = focal_alpha
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
        # but its weighting math is not implemented here; fail loud rather than
        # silently training as if the flag took effect.
        if use_acsl:
            raise NotImplementedError(
                "use_acsl=True is not supported: the ACSL class-suppression "
                "weighting is parsed from config (AcslConfig) but not implemented "
                "in TaskAlignedLossExtended._class_loss. Set acsl.use_acsl=false "
                "(the default) until the weighting is implemented."
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
        num_objs: Optional[tf.Tensor] = None,   # required for weighting=legacy_hard
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """CIoU loss + DFL loss for foreground anchors.

        Both terms are weighted per-anchor by ``sum(target_scores, -1)`` so that
        better-aligned anchors dominate the box gradient, matching the reference
        Ultralytics YOLOv8 recipe (``(loss * weight).sum() / target_scores_sum``).

        Returns:
            (ciou_loss, dfl_loss) both scalar tensors.
        """
        fg_float = tf.cast(fg_mask, tf.float32)           # [B, A]
        if self.weighting == "legacy_hard":
            # Binary weight, per-object normalizer: every foreground anchor
            # contributes equally regardless of its current alignment score.
            weight = fg_float
            denom  = num_objs
        else:
            weight = tf.reduce_sum(target_scores, axis=-1)  # [B, A]
            denom  = target_scores_sum

        # ── Box IoU loss (ciou by default; giou/diou/eiou/siou selectable) ──
        ciou = _bbox_iou_loss(pd_bboxes, target_bboxes, self.box_iou_type)   # [B, A]
        ciou_loss = tf.reduce_sum(ciou * weight * fg_float) / denom

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
            tgt_ltrb_fm, 0.0, float(self.reg_max) - 1.0 - 0.01
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

        # Gather the floor/ceil bin log-probs directly (indices already clamped to
        # [0, reg_max-1]); avoids materializing two [B, A, 4, reg_max] one-hots.
        log_p_fl = tf.gather(pd_log_softmax, fl_idx, axis=-1, batch_dims=3)  # [B, A, 4]
        log_p_cl = tf.gather(pd_log_softmax, cl_idx, axis=-1, batch_dims=3)

        dfl_raw  = -(weight_left * log_p_fl + weight_right * log_p_cl)   # [B, A, 4]
        dfl_mean = tf.reduce_mean(dfl_raw, axis=-1)                       # [B, A]
        dfl_loss = tf.reduce_sum(dfl_mean * weight * fg_float) / denom

        return ciou_loss, dfl_loss

    # ------------------------------------------------------------------

    def _class_loss(
        self,
        pred_scores: tf.Tensor,
        target_scores: tf.Tensor,
        target_scores_sum: tf.Tensor,
        fg_mask: tf.Tensor,
        ignore_bg: tf.Tensor,
        target_labels: Optional[tf.Tensor] = None,   # required for legacy_hard
        num_objs: Optional[tf.Tensor] = None,        # required for legacy_hard
    ) -> tf.Tensor:
        """Classification loss, with ignore_bg masking.

        ``cls_loss_type`` selects the per-element loss (default ``bce``): ``focal``
        adds the focal modulating factor, ``varifocal`` (VFL) weights positives by
        their soft target and negatives by ``alpha · p^gamma``. ``label_smoothing``
        (0 = off) softens the BCE targets. ``weighting == legacy_hard`` replaces
        the soft alignment-scaled targets with one-hot targets (positives pushed
        toward score 1.0 regardless of current box quality) and normalizes by
        ``num_objs`` instead of ``target_scores_sum``.
        """
        if self.weighting == "legacy_hard":
            fg_f = tf.cast(fg_mask, tf.float32)
            target = (
                tf.one_hot(target_labels, self.num_classes, dtype=tf.float32)
                * fg_f[:, :, tf.newaxis]
            )
        else:
            target = target_scores
        if self.label_smoothing > 0.0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=pred_scores
        )  # [B, A, C]

        if self.cls_loss_type == "focal":
            p = tf.sigmoid(pred_scores)
            p_t = target * p + (1.0 - target) * (1.0 - p)
            alpha_factor = target * self.focal_alpha + (1.0 - target) * (1.0 - self.focal_alpha)
            bce = alpha_factor * tf.pow(1.0 - p_t, self.focal_gamma) * bce
        elif self.cls_loss_type == "varifocal":
            p = tf.sigmoid(pred_scores)
            weight = tf.where(target > 0.0, target,
                              self.focal_alpha * tf.pow(p, self.focal_gamma))
            bce = bce * weight
        # else "bce": default

        bce_sum = tf.reduce_sum(bce, axis=-1)   # [B, A]

        # ignore_bg=1 → apply loss only on foreground anchors for that image
        ignore_bg_f = tf.cast(ignore_bg, tf.float32)                   # [B]
        fg_float    = tf.cast(fg_mask, tf.float32)                     # [B, A]
        # mask = 1.0 when ignore_bg=0; mask = fg when ignore_bg=1
        mask = (
            (1.0 - ignore_bg_f[:, tf.newaxis]) +
            ignore_bg_f[:, tf.newaxis] * fg_float
        )  # [B, A]

        denom = num_objs if self.weighting == "legacy_hard" else target_scores_sum
        return tf.reduce_sum(bce_sum * mask) / denom

    # ------------------------------------------------------------------

    def _distance_loss(
        self,
        pd_dist: tf.Tensor,
        target_dist: tf.Tensor,
        fg_mask: tf.Tensor,
        num_objs: tf.Tensor,
    ) -> tf.Tensor:
        """L1 loss on log-scale distances, masked to valid GT entries (> -10.0).

        Normalized by ``num_objs`` (total GT object count in the batch); dist_gain
        is calibrated to this scale.
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
        anc_strides: tf.Tensor,   # [A, 1] — per-anchor stride for dist units
        img_size: tf.Tensor,      # scalar float — input image side (square)
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Combined PolyYOLO polygon loss (angle + dist + conf).

        target_polygons layout: [dist0, angle0, conf0, dist1, angle1, conf1, ...]
            dist:  radial distance from box center in NORMALIZED image units
                   (pre-computed by parser); converted below to the assigned
                   anchor's GRID units (× img_size / stride — the reference
                   convention, DFL-style per-level normalization) before the
                   softplus regression.
            angle: sub-bin angular offset (vertex_angle - bin_start)/angle_step
                   in [0, 1) on bins that hold a vertex, 0.0 elsewhere.
            conf:  1.0 if a valid vertex was assigned to this bin; also the
                   validity mask used by the angle/dist losses.

        ``ignore_bg`` ([B] int) marks distance-stream images that carry no polygon
        GT (all-zero ``target_polygons``). Since conf trains on all bins, their
        real foreground objects would otherwise be pushed to conf≈0 on every
        vertex, so the whole polygon loss is zeroed on ignore_bg=1 rows (angle/dist
        already contribute zero there via the empty vertex mask; conf would not).

        All three normalize by num_objs. Angle and dist average over the valid
        vertices only (masked by conf); conf averages over all bins to supply a
        negative signal on empty bins.

        Returns:
            (poly_total, angle_loss, dist_loss_val, conf_loss_val); poly_total is
            the gain-weighted sum, the other three are raw pre-gain sub-losses.
        """
        target_dist  = target_polygons[:, :, 0::3]   # [B, A, 24] normalized units
        target_angle = target_polygons[:, :, 1::3]   # [B, A, 24] — sub-bin offset
        conf         = target_polygons[:, :, 2::3]   # [B, A, 24] — per-bin validity
        vertex_mask  = conf                          # valid-vertex mask for angle/dist

        # Normalized image units → the assigned anchor's grid units.
        dist_scale  = img_size / anc_strides[:, 0]                    # [A]
        target_dist = target_dist * dist_scale[tf.newaxis, :, tf.newaxis]

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

        # Diagnostic only (not part of the loss): |sigmoid(pred) − target| MAE over
        # valid vertices of fg anchors. Stashed on self (not returned) so the public
        # 9-tuple contract is unchanged.
        from losses.polygon_loss import polygon_angle_mae
        self.poly_angle_mae_diag = polygon_angle_mae(
            pd_poly_angle, target_angle, vertex_mask, poly_fg_b
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
            # Heads pin their outputs to float32 even under mixed-bfloat16, so these
            # casts are no-ops today; they keep every loss term in full precision if
            # that pinning is ever removed (a bf16 loss here would degrade DFL/BCE
            # numerics).
            box_lvl = tf.cast(feats["box"][level_str], tf.float32)   # [B, H, W, 64]
            B_val   = tf.shape(box_lvl)[0]
            fH      = tf.shape(box_lvl)[1]
            fW      = tf.shape(box_lvl)[2]
            A_lvl   = fH * fW

            box_raw_list.append(tf.reshape(box_lvl, [B_val, A_lvl, -1]))
            cls_list.append(
                tf.reshape(tf.cast(feats["cls"][level_str], tf.float32),
                           [B_val, A_lvl, -1])
            )

            if self.with_polygons:
                poly_a_list.append(
                    tf.reshape(tf.cast(feats["poly_angle"][level_str], tf.float32),
                               [B_val, A_lvl, -1])
                )
                poly_d_list.append(
                    tf.reshape(tf.cast(feats["poly_dist"][level_str], tf.float32),
                               [B_val, A_lvl, -1])
                )
                poly_c_list.append(
                    tf.reshape(tf.cast(feats["poly_conf"][level_str], tf.float32),
                               [B_val, A_lvl, -1])
                )

            if self.with_distance:
                dist_list.append(
                    tf.reshape(tf.cast(feats["dist"][level_str], tf.float32),
                               [B_val, A_lvl, 1])
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
            pd_box_raw, anc_strides, anc_points, num_objs=num_objs,
        )

        ignore_bg = batch.get("ignore_bg", tf.zeros([B_val], dtype=tf.int64))
        cls_loss = self._class_loss(
            pd_cls, target_scores, target_scores_sum, fg_mask, ignore_bg,
            target_labels=target_labels, num_objs=num_objs,
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
        # Diagnostic default; overwritten inside _polygon_loss when polygons are on.
        self.poly_angle_mae_diag = tf.constant(0.0)
        if self.with_polygons and pd_poly_angle is not None:
            poly_loss_val, poly_angle_l, poly_dist_l, poly_conf_l = self._polygon_loss(
                pd_poly_angle, pd_poly_dist, pd_poly_conf,
                target_polygons, fg_mask, num_objs, ignore_bg,
                anc_strides=anc_strides, img_size=img_H,
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
