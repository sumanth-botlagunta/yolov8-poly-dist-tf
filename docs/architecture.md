# Architecture

A TensorFlow reimplementation of YOLOv8 with two extensions: PolyYOLO radial polygon
segmentation and per-object distance estimation. Input is `672×672×3`, 39 classes,
3 FPN levels (strides 8/16/32).

```
image → backbone (CSPDarkNetV8-S) → decoder (FPN-PAN, C2f) → head (6 branches) → detection_generator (NMS)
```

Assembled by `models/yolo_v8.py:build_yolov8`.

## Backbone - `models/backbone.py`
CSPDarkNetV8 with C2f blocks and SPPF. The `-S` size uses `depth_scale=0.33`,
`width_scale=0.5`. When a config YAML sets `depth_scale: 1.0 / width_scale: 1.0`, the
`model_id: cspdarknetv8s` takes precedence and the model is still the `-S` variant. Emits
feature maps at levels `"3"`, `"4"`, `"5"`.

## Decoder - `models/decoder.py`
FPN-PAN with C2f stacks (top-down + bottom-up), producing fused features per level for the
head. Activation defaults to ReLU and is config-selectable via
`task.model.norm_activation.activation` (`relu`, `silu`/`swish`, `gelu`, `leaky_relu`,
`mish`, `hardswish`).

## Heads - `models/head.py:YoloV8Head`
Six per-pixel branches, each emitted at all 3 FPN levels:

| Head | Channels | Meaning |
|------|----------|---------|
| `box` | 64 | DFL distribution = 4 sides x 16 bins |
| `cls` | 39 | per-class logits |
| `poly_angle` | 24 | per-vertex sub-bin angle offset; sigmoid maps to `[0,1)` |
| `poly_dist` | 24 | per-vertex radial distance |
| `poly_conf` | 24 | per-vertex validity logit |
| `dist` | 1 | log-scale object distance |

Smart bias init: class bias = `log(5 / num_classes / (input_size/stride)^2)` with the live
input size (672); box bias = `1.0`.
Init checkpoint (transfer-init): only the selected modules (default backbone + decoder) are
loaded; the head is randomly initialized. For same-task fine-tuning of a trained model use
`task.finetune_from`, which loads the full EMA/deployed weights into a fresh optimizer. See
[guides/finetuning.md](guides/finetuning.md).

## Anchors & strides
Anchor-free (1 anchor/cell). Anchor points are cell centers: `(i+0.5)·stride`, `(j+0.5)·stride`
for strides 8/16/32, built inside the loss (`losses/tal_loss.py`) and the detection generator.

## Detection generator - `models/detection_generator.py`
Post-processing for inference (`deploy=True`): DFL decode to xyxy boxes, greedy NMS
(`max_boxes=300`, `nms_thresh=0.65`, `score_thresh=0.05`), and decode of polygon + distance
outputs. The NMS suppression scope is config-selectable (`detection_generator.nms_class_mode`):
`per_class` (default) filters each class independently, with no cross-class suppression;
`agnostic` runs one NMS over all boxes, suppressing cross-class duplicates at the same location.
This is eval-time post-processing only and has no effect on training. Distance is decoded from
log scale via `exp` and clipped to `[min_distance, max_distance]` (`[0.5, 10.0]` m).
Polygon outputs `(conf, dist, angle)` are all sigmoid/softmax-activated; `conf` values are not
raw logits.

## Polygon representation (PolyYOLO radial)
24 vertices in 15-degree bins. The distance head predicts the radial distance per bin; the
angle head predicts a sub-bin offset, so the exact vertex angle is `θᵢ = (i + offset)·2π/24`
rather than the bin center; the confidence head gates which bins hold a vertex (absent bins
encode distance 0, offset 0). See [data_pipeline.md](data_pipeline.md) for the pipeline-side
tensor formats.

### Polygon formats across the stack

| Stage | Format | Notes |
|-------|--------|-------|
| TFDS input | `[N, 3972]` xy interleaved, `-1` padded | raw dataset (`objects/points`); `-1.0` is the reserved sentinel |
| Training GT (PolyYOLO target) | `[N, 72]` = `[dist, angle, conf] × 24` interleaved | origin implicit (box center `cx, cy`); `angle` = sub-bin offset in `[0,1)`; built in `losses/tal_loss.py:_polygon_loss` |
| Prediction output | `[B, max_det, 24, 3]` = `(conf, dist, angle)`, all activated | from `detection_generator`; `conf` is already sigmoid, so thresholds apply directly |
| Cartesian (transient) | `[K, 2]` pixel `(x, y)`, `K ≤ 24` | reconstructed only at IoU time (`eval/polygon_metrics.py:_radial_to_cartesian`), conf-gated to occupied bins; never persisted |
| Eval GT | `[N, 72]` radial (same as training GT) | GT stays radial through eval; it is not converted to Cartesian |

Decode of a predicted vertex: keep bin `i` when `conf_i ≥ 0.4`
(`eval/polygon_metrics.DEFAULT_POLY_CONF_THRESH`), then
`vertex_angle = (i + angle_i)·angle_step`, `radius = dist_i`, placed relative to the box center.
