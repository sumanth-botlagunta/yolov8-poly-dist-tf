"""Task-Aligned label assignment (stop-gradient).

Implements the TAL assignment algorithm from YOLOv8:
    alignment_metric = pred_score^alpha * IoU^beta
    top-k candidates per GT, filtered by spatial constraint,
    duplicates resolved by max-IoU.

Classes:
    TaskAlignedAssigner: Pure assignment — no gradient flows through this module.
"""

from typing import Optional, Tuple

import tensorflow as tf


class TaskAlignedAssigner:
    """Stop-gradient TAL label assignment.

    All tensor operations use tf.stop_gradient on inputs before any computation
    so the assignment never contributes to the training gradient.
    """

    def __init__(
        self,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
        angle_step: int = 15,
    ):
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        # Polygon target width = (360 // angle_step) bins × 3 channels
        # (dist, angle, conf). Used only in the no-GT fallback path below; the
        # gather path infers width from gt_polys. Derived from angle_step so a
        # non-15° config (e.g. 10° → 36 bins → 108) gets the right zero-target
        # shape instead of a hardcoded 72.
        self.angle_step = angle_step
        self.poly_size  = (360 // angle_step) * 3

    # ------------------------------------------------------------------

    def __call__(
        self,
        pd_scores: tf.Tensor,     # [B, A, C]  post-sigmoid, stop-grad applied by caller
        pd_bboxes: tf.Tensor,     # [B, A, 4]  xyxy pixels, stop-grad applied by caller
        anc_points: tf.Tensor,    # [A, 2]     cx, cy in image pixels
        gt_labels: tf.Tensor,     # [B, M]     int64
        gt_bboxes: tf.Tensor,     # [B, M, 4]  xyxy pixels
        mask_gt: tf.Tensor,       # [B, M]     bool — False for padded GT rows
        gt_polys: Optional[tf.Tensor] = None,   # [B, M, 72]
        gt_dists: Optional[tf.Tensor] = None,   # [B, M]
    ) -> Tuple[tf.Tensor, ...]:
        """Assign each anchor to a GT or background.

        Returns:
            target_labels    int64   [B, A]
            target_bboxes    float32 [B, A, 4]
            target_scores    float32 [B, A, C]  soft, alignment-weighted one-hot
            target_polygons  float32 [B, A, 72]
            target_dists     float32 [B, A, 1]
            fg_mask          bool    [B, A]
        """
        B = tf.shape(pd_scores)[0]
        A = tf.shape(pd_scores)[1]
        C = tf.shape(pd_scores)[2]
        M = tf.shape(gt_labels)[1]

        # ── 1. IoU between predictions and GTs ──────────────────────────
        # pd_bboxes [B, A, 4] × gt_bboxes [B, M, 4] → [B, A, M]
        pd_exp = pd_bboxes[:, :, tf.newaxis, :]   # [B, A, 1, 4]
        gt_exp = gt_bboxes[:, tf.newaxis, :, :]   # [B, 1, M, 4]

        ix1 = tf.maximum(pd_exp[..., 0], gt_exp[..., 0])
        iy1 = tf.maximum(pd_exp[..., 1], gt_exp[..., 1])
        ix2 = tf.minimum(pd_exp[..., 2], gt_exp[..., 2])
        iy2 = tf.minimum(pd_exp[..., 3], gt_exp[..., 3])
        inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)  # [B, A, M]

        area_pd = (
            (pd_bboxes[..., 2] - pd_bboxes[..., 0]) *
            (pd_bboxes[..., 3] - pd_bboxes[..., 1])
        )  # [B, A]
        area_gt = (
            (gt_bboxes[..., 2] - gt_bboxes[..., 0]) *
            (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        )  # [B, M]
        union = (
            area_pd[:, :, tf.newaxis] + area_gt[:, tf.newaxis, :] - inter + self.eps
        )  # [B, A, M]
        iou = inter / union  # [B, A, M]

        # ── 2. Predicted score for each GT class ─────────────────────────
        # Efficient gather: [B, C, A] → gather at gt_labels [B, M] → [B, M, A]
        pd_scores_tc = tf.transpose(pd_scores, [0, 2, 1])          # [B, C, A]
        pd_scores_gt = tf.gather(pd_scores_tc, gt_labels, batch_dims=1)  # [B, M, A]
        pd_scores_gt = tf.transpose(pd_scores_gt, [0, 2, 1])       # [B, A, M]

        # ── 3. Alignment metric: score^alpha × IoU^beta ──────────────────
        # Use log-space pow to avoid underflow when beta=6.0
        align_metric = (
            tf.exp(self.alpha * tf.math.log(pd_scores_gt + self.eps)) *
            tf.exp(self.beta  * tf.math.log(iou           + self.eps))
        )  # [B, A, M]

        # Zero out invalid GT entries
        mask_gt_exp = tf.cast(mask_gt[:, tf.newaxis, :], tf.float32)  # [B, 1, M]
        align_metric = align_metric * mask_gt_exp                      # [B, A, M]

        # ── 4. Spatial mask: anchor center inside GT box ─────────────────
        cx = anc_points[:, 0]   # [A]
        cy = anc_points[:, 1]   # [A]

        cx_exp = cx[tf.newaxis, :, tf.newaxis]    # [1, A, 1]
        cy_exp = cy[tf.newaxis, :, tf.newaxis]
        gt_x1  = gt_bboxes[:, tf.newaxis, :, 0]  # [B, 1, M]
        gt_y1  = gt_bboxes[:, tf.newaxis, :, 1]
        gt_x2  = gt_bboxes[:, tf.newaxis, :, 2]
        gt_y2  = gt_bboxes[:, tf.newaxis, :, 3]

        spatial_mask = (
            (cx_exp >= gt_x1) & (cx_exp <= gt_x2) &
            (cy_exp >= gt_y1) & (cy_exp <= gt_y2)
        )  # [B, A, M]
        spatial_mask = spatial_mask & tf.broadcast_to(
            mask_gt[:, tf.newaxis, :], tf.shape(spatial_mask)
        )

        # ── 5. Top-k per GT along the anchor dimension ───────────────────
        align_spatial = align_metric * tf.cast(spatial_mask, tf.float32)  # [B, A, M]
        align_t = tf.transpose(align_spatial, [0, 2, 1])  # [B, M, A]

        k = tf.minimum(self.topk, A)
        topk_vals, _ = tf.math.top_k(align_t, k=k)        # [B, M, k]
        topk_thresh  = topk_vals[:, :, -1:]                # [B, M, 1]
        topk_mask    = align_t >= topk_thresh              # [B, M, A]
        topk_mask    = tf.transpose(topk_mask, [0, 2, 1]) # [B, A, M]

        # ── 6. Combined candidate mask + duplicate resolution ────────────
        candidate_mask = topk_mask & spatial_mask          # [B, A, M]

        iou_cand      = iou * tf.cast(candidate_mask, tf.float32)  # [B, A, M]
        target_gt_idx = tf.argmax(iou_cand, axis=-1, output_type=tf.int32)  # [B, A]
        fg_mask       = tf.reduce_any(candidate_mask, axis=-1)     # [B, A] bool

        # ── 7. Gather assigned GT attributes ─────────────────────────────
        # INVARIANT: for BACKGROUND anchors (fg_mask == False) target_gt_idx is 0
        # (argmax over an all-zero row), so every target_* below holds GT-0's
        # values, NOT a meaningful assignment. Consumers MUST mask by fg_mask.
        # Do NOT "clean this up" by zeroing the background targets: TaskAlignedLoss
        # computes CIoU on ALL anchors before weighting by fg_mask, and CIoU on a
        # zeroed [0,0,0,0] box hits atan(0/0)=NaN — then NaN*0 (the bg weight)
        # poisons the whole box loss. GT-0 is a real, finite box, which keeps that
        # masked-out term finite. The GT-0 gather is therefore load-bearing for
        # NaN-safety, not a bug. target_scores below IS zeroed for background
        # (via assigned_align * fg_mask), which is what the cls loss reads.
        target_labels = tf.gather(gt_labels, target_gt_idx, batch_dims=1)   # [B, A]
        target_bboxes = tf.gather(gt_bboxes, target_gt_idx, batch_dims=1)   # [B, A, 4]

        # Polygon targets
        poly_size = self.poly_size
        if gt_polys is not None:
            target_polygons = tf.gather(gt_polys, target_gt_idx, batch_dims=1)  # [B, A, 72]
        else:
            target_polygons = tf.zeros(
                [B, A, poly_size], dtype=tf.float32
            )

        # Distance targets
        if gt_dists is not None:
            target_dists = tf.gather(gt_dists, target_gt_idx, batch_dims=1)  # [B, A]
            target_dists = tf.expand_dims(target_dists, axis=-1)              # [B, A, 1]
        else:
            target_dists = tf.zeros([B, A, 1], dtype=tf.float32)

        # ── 8. Soft target_scores: one-hot × normalized alignment ────────
        # Matches Ultralytics YOLOv8: the soft target is scaled by the GT's
        # localization quality (pos_overlaps = per-GT max IoU over candidates), so
        # well-localized objects get higher classification targets. Omitting
        # pos_overlaps would flatten every assigned anchor's target to a max of 1.0
        # and drop the IoU-quality weighting of the classification loss.
        align_max_per_gt = tf.reduce_max(
            align_spatial, axis=1, keepdims=True
        )  # [B, 1, M]
        pos_overlaps = tf.reduce_max(
            iou * tf.cast(candidate_mask, tf.float32), axis=1, keepdims=True
        )  # [B, 1, M]
        align_norm = (
            align_spatial * pos_overlaps / (align_max_per_gt + self.eps)
        )  # [B, A, M]

        gt_idx_oh    = tf.one_hot(target_gt_idx, M)                           # [B, A, M]
        assigned_align = tf.reduce_sum(align_norm * gt_idx_oh, axis=-1)       # [B, A]
        assigned_align = assigned_align * tf.cast(fg_mask, tf.float32)

        target_scores = (
            tf.one_hot(target_labels, C, dtype=tf.float32) *
            assigned_align[:, :, tf.newaxis]
        )  # [B, A, C]

        return (
            target_labels,
            target_bboxes,
            target_scores,
            target_polygons,
            target_dists,
            fg_mask,
        )
