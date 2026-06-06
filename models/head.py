"""Multi-head output layer for YOLOv8 with polygon and distance branches.

Six output heads per FPN level (strides 8, 16, 32):
    box:        float32 [batch, H, W, 64]   DFL distribution (4 × 16 bins)
    cls:        float32 [batch, H, W, 39]   class logits
    poly_angle: float32 [batch, H, W, 24]   per-vertex angle classification
    poly_dist:  float32 [batch, H, W, 24]   per-vertex radial distance
    poly_conf:  float32 [batch, H, W, 24]   per-vertex confidence
    dist:       float32 [batch, H, W,  1]   object distance (log-scale)

Smart bias initialization is applied when smart_bias=True:
    class head bias: log(5 / num_classes / (640/stride)^2)
    box head bias:   1.0

Classes:
    YoloV8Head: Builds and applies all six heads across all FPN levels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import tensorflow as tf

from configs.registry import HEADS
from models.backbone import _ConvBnAct

_STRIDES = {"3": 8, "4": 16, "5": 32}
_HIDDEN  = 256   # hidden channels for all branch stems


@HEADS.register("yolov8_head")
class YoloV8Head(tf.keras.layers.Layer):
    """Multi-branch detection head for YOLOv8.

    Per-level branch weights (not shared across levels) for all heads.
    Smart bias is initialised via initialize_biases() after the first forward pass.
    """

    def __init__(
        self,
        num_classes: int = 39,
        output_poly_size: int = 24,
        output_dist_size: int = 1,
        num_dist_block: int = 1,
        reg_max: int = 16,
        smart_bias: bool = True,
        with_polygons: bool = True,
        with_distance: bool = True,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes    = num_classes
        self.output_poly_size = output_poly_size
        self.output_dist_size = output_dist_size
        self.num_dist_block = num_dist_block
        self.reg_max        = reg_max
        self.smart_bias     = smart_bias
        self.with_polygons  = with_polygons
        self.with_distance  = with_distance

        self._norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )
        self._levels: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------

    def build(self, input_shape: Dict) -> None:
        """Build per-level branch layers.

        Args:
            input_shape: dict like {'3': TensorShape([None,84,84,128]), ...}
        """
        levels = sorted(input_shape.keys())
        self._levels = levels

        for level in levels:
            nk = self._norm_kw

            # ---- box branch: 2×Conv(256,3×3) + Conv(4*reg_max, 1×1) ----
            setattr(self, f"box_s1_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
            setattr(self, f"box_s2_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
            setattr(self, f"box_pred_{level}",
                    tf.keras.layers.Conv2D(4 * self.reg_max, 1, use_bias=True,
                                           padding="same", name=f"box_pred_{level}"))

            # ---- cls branch: 2×Conv(256,3×3) + Conv(num_classes, 1×1) ----
            setattr(self, f"cls_s1_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
            setattr(self, f"cls_s2_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
            setattr(self, f"cls_pred_{level}",
                    tf.keras.layers.Conv2D(self.num_classes, 1, use_bias=True,
                                           padding="same", name=f"cls_pred_{level}"))

            if self.with_polygons:
                # ---- poly_angle branch ----
                setattr(self, f"pa_s1_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pa_s2_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pa_pred_{level}",
                        tf.keras.layers.Conv2D(self.output_poly_size, 1, use_bias=True,
                                               padding="same", name=f"pa_pred_{level}"))

                # ---- poly_dist branch ----
                setattr(self, f"pd_s1_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pd_s2_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pd_pred_{level}",
                        tf.keras.layers.Conv2D(self.output_poly_size, 1, use_bias=True,
                                               padding="same", name=f"pd_pred_{level}"))

                # ---- poly_conf branch ----
                setattr(self, f"pc_s1_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pc_s2_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"pc_pred_{level}",
                        tf.keras.layers.Conv2D(self.output_poly_size, 1, use_bias=True,
                                               padding="same", name=f"pc_pred_{level}"))

            if self.with_distance:
                # ---- dist branch: num_dist_block×Conv(256,3×3) + Conv(1, 1×1) ----
                for bi in range(self.num_dist_block):
                    setattr(self, f"dist_s{bi}_{level}", _ConvBnAct(_HIDDEN, 3, **nk))
                setattr(self, f"dist_pred_{level}",
                        tf.keras.layers.Conv2D(self.output_dist_size, 1, use_bias=True,
                                               padding="same", name=f"dist_pred_{level}"))

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        features: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        """Apply all heads to each FPN level.

        Returns:
            Nested dict keyed by head name → level:
            {
                'box':        {'3': ..., '4': ..., '5': ...},
                'cls':        {'3': ..., '4': ..., '5': ...},
                'poly_angle': {'3': ..., '4': ..., '5': ...},  # if with_polygons
                'poly_dist':  {'3': ..., '4': ..., '5': ...},  # if with_polygons
                'poly_conf':  {'3': ..., '4': ..., '5': ...},  # if with_polygons
                'dist':       {'3': ..., '4': ..., '5': ...},  # if with_distance
            }
        """
        out_box = {}
        out_cls = {}
        out_pa  = {}
        out_pd  = {}
        out_pc  = {}
        out_dist = {}

        for level in self._levels:
            x = features[level]

            # box
            h = getattr(self, f"box_s1_{level}")(x, training=training)
            h = getattr(self, f"box_s2_{level}")(h, training=training)
            out_box[level] = getattr(self, f"box_pred_{level}")(h)

            # cls
            h = getattr(self, f"cls_s1_{level}")(x, training=training)
            h = getattr(self, f"cls_s2_{level}")(h, training=training)
            out_cls[level] = getattr(self, f"cls_pred_{level}")(h)

            if self.with_polygons:
                # poly_angle
                h = getattr(self, f"pa_s1_{level}")(x, training=training)
                h = getattr(self, f"pa_s2_{level}")(h, training=training)
                out_pa[level] = getattr(self, f"pa_pred_{level}")(h)

                # poly_dist
                h = getattr(self, f"pd_s1_{level}")(x, training=training)
                h = getattr(self, f"pd_s2_{level}")(h, training=training)
                out_pd[level] = getattr(self, f"pd_pred_{level}")(h)

                # poly_conf
                h = getattr(self, f"pc_s1_{level}")(x, training=training)
                h = getattr(self, f"pc_s2_{level}")(h, training=training)
                out_pc[level] = getattr(self, f"pc_pred_{level}")(h)

            if self.with_distance:
                h = x
                for bi in range(self.num_dist_block):
                    h = getattr(self, f"dist_s{bi}_{level}")(h, training=training)
                out_dist[level] = getattr(self, f"dist_pred_{level}")(h)

        result: Dict[str, Dict[str, tf.Tensor]] = {
            "box": out_box,
            "cls": out_cls,
        }
        if self.with_polygons:
            result["poly_angle"] = out_pa
            result["poly_dist"]  = out_pd
            result["poly_conf"]  = out_pc
        if self.with_distance:
            result["dist"] = out_dist

        return result

    # ------------------------------------------------------------------
    # Smart bias initialisation
    # ------------------------------------------------------------------

    def initialize_biases(self, input_size: int = 672) -> None:
        """Set class and box prediction biases after first forward pass.

        class bias = log(5 / num_classes / (input_size / stride)^2)
        box bias   = 1.0
        """
        if not self.built:
            raise RuntimeError("Call the head once before initializing biases.")

        for level in self._levels:
            stride = _STRIDES[level]

            cls_pred = getattr(self, f"cls_pred_{level}")
            cls_val  = math.log(5.0 / self.num_classes / (input_size / stride) ** 2)
            cls_pred.bias.assign(
                tf.fill(cls_pred.bias.shape, tf.cast(cls_val, tf.float32))
            )

            box_pred = getattr(self, f"box_pred_{level}")
            box_pred.bias.assign(tf.ones_like(box_pred.bias))
