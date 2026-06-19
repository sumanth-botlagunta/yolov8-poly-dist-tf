# Losses

All under `losses/`. The total loss is a Task-Aligned Learning (TAL) detection loss plus
polygon and distance terms. Entry point: `tal_loss.py:TaskAlignedLossExtended.__call__`,
which returns a **9-tuple**:
`(total, box, dfl, cls, dist, poly, poly_angle_raw, poly_dist_raw, poly_conf_raw)`.
The last three are raw pre-gain polygon sub-losses for TensorBoard logging only — they do
not re-enter `total_loss`.

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
Three per-vertex components over the 24 bins. The `conf` channel of the target is the per-bin
validity mask (`vertex_mask`); all normalize by `num_objs`:
- `polygon_angle_loss` — `BCE(sigmoid(pred), sub-bin offset)`, offset =
  `(vertex_angle − bin_start)/angle_step ∈ [0,1)`; **mean over valid vertices**, ÷ `num_objs`.
- `polygon_dist_loss` — `(target_radius − softplus(pred))²`; **mean over valid vertices**, ÷ `num_objs`.
- `polygon_conf_loss` — BCE on per-bin vertex validity; **mean over ALL 24 bins** (occupied → 1,
  empty → 0), ÷ `num_objs`. Conf is the decode gate and must see negatives: the earlier masked
  form (valid-bins-only) gave empty bins zero gradient ever, so their conf drifted above
  the 0.4 decode/viz threshold while their dist stayed untrained — the star/spiky polygon
  artifacts seen in val overlays. The masked form is preserved verbatim in
  `polygon_conf_loss`'s docstring as a one-line swap. Angle/dist remain masked because their
  regression targets are undefined on empty bins.
Combined in `tal_loss.py:_polygon_loss` with the component gains; the overall `poly_gain`
multiplier is applied inside `_polygon_loss`.

## Distance — `distance_loss.py:distance_l1_loss`
L1 on log-scale distance, masked to valid foreground entries (`target > -10.0` sentinel),
normalized by **total GT object count** (`num_objs` = all GTs in the batch, including
detection-stream GTs with sentinel distance). Valid range `[0.5, 10.0]` m.

## Gains (from the experiment YAML)
`iou=7.5, cls=0.5, dfl=1.5, dist=1.0, poly_dist=0.45, poly_angle=0.4, poly_conf=0.2`, plus an
overall `poly_gain=0.5`. The detection gains are the Ultralytics defaults and are calibrated for
the weighted formulation above.

## Normalization conventions — important
Not all terms share the same denominator/reduction. **These are intentional**, but mean the gains
are not directly comparable across heads:

| Term | Denominator | Vertex reduction |
|------|-------------|------------------|
| box CIoU / DFL | `target_scores_sum`, per-anchor weighted | — |
| cls | `target_scores_sum` | — |
| distance | `num_objs` (total batch GT count) | — |
| polygon angle | `num_objs` | **mean over valid vertices** |
| polygon dist | `num_objs` | **mean over valid vertices** |
| polygon conf | `num_objs` | **mean over ALL 24 bins** |

Consequences:
- `dist_gain` and the poly gains divide by `num_objs` (not `target_scores_sum`), so they are on a
  different scale than the detection gains.
- Angle/dist average over the **valid** vertex count per anchor (empty bins do not dilute the
  mean); conf averages over all 24 bins so empty bins receive a negative signal (see the
  Polygon section). The conf magnitude reflects that choice (negatives dominate the
  24-bin mean early in training); `poly_conf_gain=0.2` was deliberately kept. **Re-tune the poly
  gains if you change the masking or vertex count.**

These conventions are pinned by `tests/test_polygon_loss_conventions.py`.

## Tests
- `tests/test_loss_reference_parity.py` — pins the `pos_overlaps` scaling and the per-anchor
  box/DFL weighting against the reference formulas.
- `tests/test_loss_computation.py`, `tests/test_distance_loss.py`,
  `tests/test_polygon_loss_conventions.py`, `tests/unit/test_tal_assigner.py`.
