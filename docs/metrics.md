# Metrics Glossary

What each metric printed by `python -m utils.eval` (and logged to TensorBoard `val/...` during
training) means. Computed by `eval/coco_metrics.py` + `eval/coco_eval_custom.py` (the custom
COCOeval that adds the F1 confidence sweep and don't-care handling), `eval/polygon_metrics.py`, and
`eval/distance_metrics.py`.

## Detection (COCO)

| Metric | Meaning |
|--------|---------|
| `mAP` | Mean Average Precision averaged over IoU thresholds **0.50:0.95** (the primary COCO metric). |
| `mAP50` | Average Precision at a single IoU threshold of **0.5** (looser, more forgiving). |
| `AR100` | Average Recall with up to **100 detections** per image. |
| `F1score50` | **Macro-averaged best F1** at IoU 0.5, **maxDets=10**: `eval/coco_eval_custom.py:COCOevalCustom` sweeps a confidence threshold grid (`np.arange(0.1, 1.0, 0.05)`) on the cumulative precision/recall (with a hallucination-GT recall correction), takes each class's best `2pr/(p+r)`, then averages over classes with a valid PR point. This is the **best-checkpoint selection metric** (`trainer.best_checkpoint_eval_metric`). |
| `precision50` | Mean precision at each class's peak-F1 operating point (the same point `F1score50` uses). |
| `recall50` | Mean recall at each class's peak-F1 operating point. |

**Per-category** metrics (`--per_category`, and each epoch's entry in `val_history.jsonl`,
extractable with `utils/reports/val_history.py`) give the peak F1 / precision / recall **per class**,
plus the confidence threshold at that peak — useful for spotting a few weak classes dragging
down the macro average.

## Polygon (`with_polygons`)

Predictions are matched to GT by **bbox IoU > 0.5**, then polygon **mask** IoU is computed
(`cv2.fillPoly` rasterization, conf-gated to occupied bins).

| Metric | Meaning |
|--------|---------|
| `poly_mIoU` | Mean polygon **mask** IoU over matched (bbox-IoU>0.5) pairs. Measures segmentation quality, not detection. |
| `poly_recall50` | Fraction of GT objects matched by a prediction at bbox IoU 0.5 (polygon detection recall). |

## Distance (`with_distance`, when the eval split carries distance GT)

Each GT is matched to its highest-IoU detection (≥0.5); errors are in **meters** (predictions are
exp'd from log-space first). Near/far split at **5 m** (`_NEAR_FAR_THRESHOLD`).

| Metric | Meaning |
|--------|---------|
| `dist_mae` | Mean absolute error `mean(|pred − gt|)`, meters. |
| `dist_rmse` | Root mean squared error, meters (penalizes large misses more). |
| `dist_absrel` | Mean **relative** error `mean(|pred − gt| / gt)` — error as a fraction of true distance. |
| `dist_abs_near` / `dist_absrel_near` | `dist_mae` / `dist_absrel` restricted to **near** objects (gt < 5 m). |
| `dist_abs_far` / `dist_absrel_far` | Same, restricted to **far** objects (gt ≥ 5 m). |

> Distance is currently scored only when the eval split has distance labels. The shipped distance
> dataset is training-only, so distance is **not** evaluated during normal validation.
