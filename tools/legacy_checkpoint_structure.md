# Legacy checkpoint structure (ckpt-319992, 39 classes)

Extracted from the old-codebase checkpoint variable dump
(`tmp_reference_docs/old_codebase_checkpoint_names/*`). Cached here so the OCR /
manual inspection does not need to be repeated. All variable keys end with
`/.ATTRIBUTES/VARIABLE_VALUE` (omitted below). Roles: `conv/kernel`,
`bn/{beta,gamma,moving_mean,moving_variance}`.

This is the authoritative source for `tools/checkpoint_weight_map.py`. The
mapping is by **architecture position**, not shape (same shapes are many-to-one).

## Counts
- backbone: 135 vars, `layer_with_weights-0 .. -9` (10 blocks)
- decoder: 90 vars, `layer_with_weights-0 .. -5` (6 blocks)
- head: 111 vars, `_head/{3,4,5}` (3 levels)

## Backbone block sequence (legacy ordinal -> new block, by shape/role)
| lww | type | new block | notes |
|-----|------|-----------|-------|
| 0 | plain ConvBnAct | stem_conv1 | conv (3,3,3,32) |
| 1 | plain ConvBnAct | stem_conv2 | conv (3,3,32,64) |
| 2 | C2f n=1 | stem_c2f | hidden 32, out 64 |
| 3 | plain (downsample) | down1 | conv (3,3,64,128) |
| 4 | C2f n=2 | c2f_p3 | hidden 64, out 128 |
| 5 | plain (downsample) | down2 | conv (3,3,128,256) |
| 6 | C2f n=2 | c2f_p4 | hidden 128, out 256 |
| 7 | plain (downsample) | down3 | conv (3,3,256,512) |
| 8 | C2f n=1 | c2f_p5_pre | hidden 256, out 512 |
| 9 | SPPF | sppf | cv1 (1,1,512,256), cv2 (1,1,1024,512) |

## Decoder block sequence
| lww | type | new block |
|-----|------|-----------|
| 0 | C2f n=1 | fpn_c2f_p4 |
| 1 | C2f n=1 | fpn_c2f_p3 |
| 2 | plain (downsample) | pan_down_p3 |
| 3 | C2f n=1 | pan_c2f_p4 |
| 4 | plain (downsample) | pan_down_p4 |
| 5 | C2f n=1 | pan_c2f_p5 |

## Sub-block name mapping (legacy -> new), C2f / SPPF / plain
The legacy C2f names are architecturally "inverted" vs the new names — the
identity is fixed by **data-flow role + input-channel shape**, NOT by the digit
in the name:

| legacy sub-path | new sub-block | role in C2f |
|-----------------|---------------|-------------|
| `_route/_conv2`            | `cv1`        | input/split conv (c_in -> c_out) |
| `_connect/_conv1`          | `cv2`        | output conv (concat -> c_out), input = (2+n)*hidden |
| `_model_to_wrap/{i}/_conv1`| `bn{i}/cv1`  | bottleneck i, first conv |
| `_model_to_wrap/{i}/_conv2`| `bn{i}/cv2`  | bottleneck i, second conv |
| `_conv1` (SPPF top-level)  | `cv1`        | SPPF input conv |
| `_conv2` (SPPF top-level)  | `cv2`        | SPPF output conv |
| *(none — direct `conv`/`bn`)* | *(block itself)* | plain ConvBnAct |

Verified example (stem_c2f, lww-2):
```
_route/_conv2/conv/kernel        (1,1,64,64)   -> cv1
_connect/_conv1/conv/kernel      (1,1,96,64)   -> cv2   (96 = (2+1)*32 concat)
_model_to_wrap/0/_conv1/conv/kernel (3,3,32,32)-> bn0/cv1
_model_to_wrap/0/_conv2/conv/kernel (3,3,32,32)-> bn0/cv2
```

## Head (per level L in {3,4,5}): `head/_head/{L}/...`
| legacy sub-path | new attribute | shape (L=3) |
|-----------------|---------------|-------------|
| `cv2feat/layer_with_weights-0` | `cv2feat_s1_{L}` | conv (3,3,128,136) |
| `cv2feat/layer_with_weights-1` | `cv2feat_s2_{L}` | conv (3,3,136,136) |
| `box/conv`                     | `box_pred_{L}`   | (1,1,136,64) + bias |
| `cv3/layer_with_weights-0`     | `cls_s1_{L}`     | conv (3,3,128,128) |
| `cv3/layer_with_weights-1`     | `cls_s2_{L}`     | conv (3,3,128,128) |
| `cv3/layer_with_weights-2/conv`| `cls_pred_{L}`   | (1,1,128,39) + bias |
| `cv4/layer_with_weights-0`     | `dist_s0_{L}`    | conv (3,3,128,128) |
| `cv4/layer_with_weights-1/conv`| `dist_pred_{L}`  | (1,1,128,1) + bias |
| `poly_angle/conv`              | `pa_pred_{L}`    | (1,1,136,24) + bias |
| `poly_dist/conv`               | `pd_pred_{L}`    | (1,1,136,24) + bias |
| `poly_conf/conv`               | `pc_pred_{L}`    | (1,1,136,24) + bias |

Stems (`cv2feat_*`, `cls_*`, `dist_s0`) are `_ConvBnAct` (conv/kernel +
bn/{gamma,beta,moving_mean,moving_variance}). Predictors (`*_pred`) are plain
Conv2D (kernel + bias). 11 conv units/level x 3 levels = 33 sub-blocks, 111 vars.
