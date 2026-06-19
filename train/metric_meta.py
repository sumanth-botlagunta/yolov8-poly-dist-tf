"""Human-readable names + formulae for every scalar written to TensorBoard.

Each TensorBoard scalar gets a markdown ``description`` (shown in the UI tooltip)
built from this registry, so a reader doesn't have to remember what a short tag
like ``cls_loss`` or ``dist_absrel_far`` means. ``describe()`` also handles the
per-category tags ``cls/<NN>_<name>/<metric>`` by composing the class name with
the per-category metric's description.

Keys here are the SHORT metric keys (no ``train/`` / ``val/`` prefix).
"""

from typing import Optional

# Short key → "Full Name — formula / definition".
METRIC_DESCRIPTIONS = {
    # ---- Losses (train/val; *_loss are gain-weighted unless noted "raw") ----
    "total_loss": (
        "**Total Loss** — gain-weighted sum of all heads: "
        "`iou·CIoU + dfl·DFL + cls·BCE + dist·L1 + poly_gain·(poly_angle·angle + poly_dist·dist + poly_conf·conf)`."
    ),
    "box_loss": (
        "**Box CIoU Loss** (gain-weighted, gain=7.5) — `1 − CIoU(pred_box, gt_box)`, "
        "weighted per-anchor by `Σ target_scores` and divided by `target_scores_sum`."
    ),
    "dfl_loss": (
        "**Distribution Focal Loss** (gain-weighted, gain=1.5) — cross-entropy of the "
        "predicted box-edge distribution over `reg_max=16` bins vs the soft LTRB target, "
        "÷ `target_scores_sum`."
    ),
    "cls_loss": (
        "**Classification Loss** (gain-weighted, gain=0.5) — "
        "`BCE(sigmoid(pred_cls), one_hot · align_norm · pos_overlap)` summed over classes, "
        "÷ `target_scores_sum`."
    ),
    "dist_loss": (
        "**Distance Loss** (gain-weighted, gain=1.0) — `L1(pred, gt)` on log-scale distance, "
        "masked to valid samples (`gt > −10`), ÷ `num_objs`."
    ),
    "poly_loss": (
        "**Polygon Loss** (gain-weighted total) — "
        "`poly_gain · (poly_angle·angle_loss + poly_dist·dist_loss + poly_conf·conf_loss)`."
    ),
    "poly_angle_loss": (
        "**Polygon Angle Loss** (raw, pre-gain) — `BCE(sigmoid(pred), sub-bin offset)` where "
        "offset = `(vertex_angle − bin_start)/angle_step ∈ [0,1)`; mean over **valid** vertices, ÷ `num_objs`."
    ),
    "poly_dist_loss": (
        "**Polygon Distance Loss** (raw, pre-gain) — `(softplus(pred) − target_radius)²`; "
        "mean over **valid** vertices, ÷ `num_objs`."
    ),
    "poly_conf_loss": (
        "**Polygon Confidence Loss** (raw, pre-gain) — `BCE(sigmoid(pred), per-bin validity)`; "
        "mean over **ALL 24 bins** (occupied → 1, empty → 0 — empty bins need "
        "the negative gradient), ÷ `num_objs`."
    ),

    # ---- COCO detection metrics (val) ----
    "mAP": "**Mean Average Precision @IoU=0.50:0.95** — primary COCO detection metric (averaged over 10 IoU thresholds).",
    "mAP50": "**Average Precision @IoU=0.50** — PASCAL-VOC-style AP at a single 0.50 IoU.",
    "AR100": "**Average Recall @IoU=0.50:0.95** with up to 100 detections per image.",
    "F1score50": "**Peak F1 @IoU=0.50** — `max_conf 2·P·R/(P+R)`, macro-averaged over classes.",
    "best_conf_thresh": "**Best confidence threshold** — the confidence at which F1@50 peaks (macro-mean over classes).",

    # ---- Distance metrics (val), meter space ----
    "dist_mae":         "**Distance MAE** (meters) — `mean(|pred − gt|)` over all valid samples.",
    "dist_rmse":        "**Distance RMSE** (meters) — `sqrt(mean((pred − gt)²))`.",
    "dist_absrel":      "**Distance Absolute-Relative Error** — `mean(|pred − gt| / gt)`.",
    "dist_abs_near":    "**Distance MAE, near** (meters) — MAE for objects with `gt < 5 m`.",
    "dist_absrel_near": "**Distance AbsRel, near** — relative MAE for `gt < 5 m`.",
    "dist_abs_far":     "**Distance MAE, far** (meters) — MAE for objects with `gt ≥ 5 m`.",
    "dist_absrel_far":  "**Distance AbsRel, far** — relative MAE for `gt ≥ 5 m`.",

    # ---- Polygon metrics (val) ----
    "poly_mIoU":     "**Polygon mean mask IoU** — mean `|pred∩gt| / |pred∪gt|` over matched (pred, GT) pairs (bbox IoU ≥ 0.5).",
    "poly_recall50": "**Polygon Recall@50** — fraction of GT polygons matched at bbox IoU ≥ 0.5 (recall, not AP).",

    # ---- Optimizer / runtime (train/epoch/system) ----
    "lr":                   "**Learning Rate** — current value (cosine decay from 0.01, α=0.01, after linear warmup).",
    "momentum":             "**SGD Momentum** — Nesterov momentum, linearly warmed 0.8 → 0.937.",
    "ema_decay":            "**EMA Decay** — current weight-averaging decay `min(average_decay, (1+step)/(10+step))` (average_decay=0.9999 by default).",
    "step_time_ms":         "**Step Time** (ms) — GPU compute per training step "
                            "(`_compiled_train_step` only; excludes waiting for data).",
    "data_wait_ms":         "**Data Wait** (ms) — time per step spent waiting for the input "
                            "pipeline to produce the next batch. ≈0 when the pipeline keeps "
                            "up; step wall-clock = step_time + data_wait.",
    "throughput_img_per_s": "**Throughput** — training images per second of WALL CLOCK "
                            "(merged batch ÷ (step_time + data_wait)) — the honest number "
                            "for epoch-time projections.",
    "gpu_mem_gb":           "**GPU Memory** (GB) — current device allocation.",
    "gpu_mem_peak_gb":      "**GPU Memory, peak** (GB) — peak device allocation.",
    "time_s":               "**Epoch Time** (s) — wall-clock for the epoch (train + validation).",
    "train_time_s":         "**Epoch Train Time** (s) — wall-clock for the training portion.",
    "val_time_s":           "**Epoch Validation Time** (s).",
    "eta_s":                "**ETA** (s) — estimated time to finish remaining epochs.",
    "best_checkpoint_epoch": "**Best-checkpoint epoch** — epoch at which the best validation metric was last improved.",
}

# Per-category metric descriptions (the trailing segment of a ``cls/<NN>_<name>/<m>`` tag).
_PER_CATEGORY = {
    "ap":    "AP @IoU=0.50:0.95",
    "ap50":  "AP @IoU=0.50",
    "ap75":  "AP @IoU=0.75",
    "ap_s":  "AP, small objects (area < 32²)",
    "ap_m":  "AP, medium objects (32²–96²)",
    "ap_l":  "AP, large objects (area > 96²)",
    "ar1":   "AR with ≤1 detection/image",
    "ar10":  "AR with ≤10 detections/image",
    "ar100": "AR with ≤100 detections/image",
    "ar_s":  "AR, small objects",
    "ar_m":  "AR, medium objects",
    "ar_l":  "AR, large objects",
}


def describe(key: str) -> Optional[str]:
    """Return a markdown description for a short metric key, or None if unknown.

    Handles per-category tags of the form ``cls/<NN>_<name>/<metric>``.
    """
    if key.startswith("cls/"):
        parts = key.split("/")
        if len(parts) == 3:
            class_label, metric = parts[1], parts[2]
            base = _PER_CATEGORY.get(metric, metric)
            return f"**Per-class {base}** for class `{class_label}`."
        return None
    # Exact registry entries win over the generic best_ composition — otherwise
    # registered keys like `best_conf_thresh` / `best_checkpoint_epoch` were
    # shadowed by the prefix strip (describe('conf_thresh') → None → no tooltip).
    if key in METRIC_DESCRIPTIONS:
        return METRIC_DESCRIPTIONS[key]
    if key.startswith("best_"):
        inner = describe(key[len("best_"):])
        return f"Best-so-far (max over epochs) of — {inner}" if inner else None
    return None
