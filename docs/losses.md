# Losses

All under `losses/`. The total loss is a Task-Aligned Learning (TAL) detection loss plus
polygon and distance terms. Entry point: `tal_loss.py:TaskAlignedLossExtended.__call__`,
which returns `(total, box, dfl, cls, dist, poly)`.

## Task-Aligned assignment — `tal_assigner.py:TaskAlignedAssigner`
Pure (stop-gradient) label assignment per the Ultralytics YOLOv8 recipe:
1. IoU between predicted and GT boxes (`xyxy` pixels).
2. **Alignment metric** `align = score^alpha · IoU^beta` (config `tal_alpha`, `tal_beta`), in
   log-space for stability.
3. **Top-k** candidates per GT along the anchor axis, AND a **spatial mask** (anchor center
   inside the GT box).
4. **Duplicate resolution**: an anchor matched to multiple GTs takes the max-IoU GT.
5. **Soft `target_scores`** = `one_hot(label) · (align_norm · pos_overlaps)`, where
   `align_norm = align / max_align_per_gt` and `pos_overlaps = per-GT max IoU`. The
   `pos_overlaps` factor scales the classification target by localization quality (matches
   reference; omitting it inflates cls targets).

## Box loss — `tal_loss.py:_box_loss`
- **CIoU** (`_ciou_loss`): `1 − (IoU − ρ²/c² − α·v)`.
- **DFL**: LTRB targets in feature-map units, clipped to `[0, reg_max−1.001]`, floor/ceil
  log-softmax interpolation, mean over the 4 sides.
- Both are **weighted per-anchor by `sum(target_scores, -1)`** and divided by
  `target_scores_sum = max(sum(target_scores), 1)` — so well-aligned anchors dominate the box
  gradient (reference behavior).

## Classification — `tal_loss.py:_class_loss`
BCE-with-logits summed over classes, divided by `target_scores_sum`. `ignore_bg=1` (distance-only
samples) masks the class loss to foreground anchors only.

## Polygon — `polygon_loss.py`
Three per-vertex components over the 24 bins:
- `polygon_angle_loss` — sigmoid-CE over angle bins, **averaged** over vertices (`reduce_mean`).
- `polygon_dist_loss` — L1 on radial distance, **summed** over vertices (`reduce_sum`).
- `polygon_conf_loss` — BCE on vertex validity, **summed** over vertices (`reduce_sum`).
Combined in `tal_loss.py:_polygon_loss` with the component gains, then multiplied by the overall
`poly_gain`.

## Distance — `distance_loss.py:distance_l1_loss`
L1 on log-scale distance, masked to valid foreground entries (`target > -10.0` sentinel),
normalized by the **count of valid entries** (`n_valid`) — a per-valid-object mean. Valid range
`[0.5, 10.0]` m.

## Gains (from the experiment YAML)
`iou=7.5, cls=0.5, dfl=1.5, dist=1.0, poly_dist=0.45, poly_angle=0.4, poly_conf=0.2`, plus an
overall `poly_gain=0.5`. The detection gains are the Ultralytics defaults and are calibrated for
the weighted formulation above.

## Normalization conventions — important
Not all terms share the same denominator/reduction. **These are intentional and documented**, but
mean the gains are not directly comparable across heads:

| Term | Denominator | Vertex reduction |
|------|-------------|------------------|
| box CIoU / DFL | `target_scores_sum`, per-anchor weighted | — |
| cls | `target_scores_sum` | — |
| distance | `n_valid` (count) | — |
| polygon angle | `target_scores_sum` | **mean** over 24 |
| polygon dist / conf | `target_scores_sum` | **sum** over 24 |

Consequences:
- `dist_gain` is on a different scale than the detection gains (per-valid-object mean).
- poly dist/conf are ~24× poly angle before gains; the poly gains absorb that ×24 factor. **Do not
  change the vertex count without re-checking the gains.**

These are pinned by `tests/test_polygon_loss_conventions.py`. Unifying them (e.g. all on
`target_scores_sum` + `reduce_mean`) is deferred to a future gain sweep, because it changes loss
magnitudes and forces re-tuning every gain + a re-validation run.

## Tests
- `tests/test_loss_reference_parity.py` — pins the `pos_overlaps` scaling and the per-anchor
  box/DFL weighting against the reference formulas.
- `tests/test_loss_computation.py`, `tests/test_distance_loss.py`,
  `tests/test_polygon_loss_conventions.py`, `tests/unit/test_tal_assigner.py`.
