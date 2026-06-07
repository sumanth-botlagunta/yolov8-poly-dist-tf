# Architecture

A TensorFlow reimplementation of YOLOv8 with two extra capabilities: **PolyYOLO radial
polygon segmentation** and **per-object distance estimation**. Input is `672×672×3`,
39 classes, 3 FPN levels (strides 8/16/32).

```
image → backbone (CSPDarkNetV8-S) → decoder (FPN-PAN, C2f) → head (6 branches) → detection_generator (NMS)
```

Assembled by `models/yolo_v8.py:build_yolov8`.

## Backbone — `models/backbone.py`
CSPDarkNetV8 with C2f blocks and SPPF. The `-S` size uses `depth_scale=0.33`,
`width_scale=0.5`. Note: even when a config YAML sets `depth_scale: 1.0 / width_scale: 1.0`,
the `model_id: cspdarknetv8s` takes precedence (the model is small). Emits feature maps at
levels `"3"`, `"4"`, `"5"`.

## Decoder — `models/decoder.py`
FPN-PAN with C2f stacks (top-down + bottom-up), producing fused features per level for the
head. Activation is ReLU throughout.

## Heads — `models/head.py:YoloV8Head`
Six per-pixel branches, each emitted at all 3 FPN levels:

| Head | Channels | Meaning |
|------|----------|---------|
| `box` | 64 | DFL distribution = 4 sides × 16 bins |
| `cls` | 39 | per-class logits |
| `poly_angle` | 24 | per-vertex angle-bin logits |
| `poly_dist` | 24 | per-vertex radial distance |
| `poly_conf` | 24 | per-vertex validity logit |
| `dist` | 1 | log-scale object distance |

**Smart bias init**: class bias = `log(5 / num_classes / (640/stride)^2)`, box bias = `1.0`.
**Init checkpoint**: only backbone + decoder weights are loaded; the head is randomly initialized.

## Anchors & strides
Anchor-free (1 anchor/cell). Anchor points are cell centers: `(i+0.5)·stride`, `(j+0.5)·stride`
for strides 8/16/32 — built inside the loss (`losses/tal_loss.py`) and the detection generator.

## Detection generator — `models/detection_generator.py`
Post-processing for inference (`deploy=True`): DFL decode → xyxy boxes, class-agnostic greedy
NMS (`max_boxes=300`, `nms_thresh=0.65`), and decode of polygon + distance outputs. Distance is
`exp`'d from log-scale and clipped to `[min_distance, max_distance]` (`[0.5, 10.0]` m).

## Polygon representation (PolyYOLO radial)
24 vertices at fixed 15° steps (`θᵢ = i·2π/24`). The distance head predicts the radial distance
at each angle; the confidence head gates which vertices exist; absent vertices encode distance 0.
See [data_pipeline.md](data_pipeline.md) for the exact tensor formats.
