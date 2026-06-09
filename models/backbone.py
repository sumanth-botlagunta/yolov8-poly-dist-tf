"""CSPDarkNetV8 backbone for YOLOv8.

YOLOv8-S configuration (model_id='cspdarknetv8s'):
    depth_scale: 0.33   (number of bottleneck repetitions per stage)
    width_scale: 0.50   (channel multiplier)

Architecture stages:
    Stem: Conv(32, 3x3, s=2) → Conv(64, 3x3, s=2) → C2f(64, n=1)
    P3:   Conv(128, 3x3, s=2) → C2f(128, n=2)          stride 8  → level '3'
    P4:   Conv(256, 3x3, s=2) → C2f(256, n=2)          stride 16 → level '4'
    P5:   Conv(512, 3x3, s=2) → C2f(512, n=1) → SPPF   stride 32 → level '5'

All convolutions use BN + activation (default: relu per experiment config).

Classes:
    _ConvBnAct: Conv2D + BatchNormalization + Activation (shared across models/).
    C2fBottleneck: Single bottleneck block used inside C2f.
    C2f: Cross-Stage-Partial block with n bottleneck repetitions.
    SPPF: Spatial Pyramid Pooling - Fast.
    CSPDarkNetV8: Full backbone returning multi-scale feature maps.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import tensorflow as tf

from configs.registry import BACKBONES


# ---------------------------------------------------------------------------
# Shared building block (imported by decoder.py and head.py)
# ---------------------------------------------------------------------------

class _ConvBnAct(tf.keras.layers.Layer):
    """Conv2D (no bias, SAME padding) + BatchNormalization + Activation."""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 1,
        strides: int = 1,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
        )
        # Keras 3 / TF 2.16 removed tf.keras.layers.experimental.SyncBatchNormalization;
        # synchronized BN is now a flag on BatchNormalization.
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=norm_momentum, epsilon=norm_epsilon, synchronized=use_sync_bn
        )
        self.act = tf.keras.layers.Activation(activation)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.act(self.bn(self.conv(x), training=training))


# ---------------------------------------------------------------------------
# C2f building blocks
# ---------------------------------------------------------------------------

class C2fBottleneck(tf.keras.layers.Layer):
    """Single bottleneck block (shortcut + two 3×3 convs)."""

    def __init__(
        self,
        filters: int,
        shortcut: bool = True,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shortcut = shortcut
        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )
        self.cv1 = _ConvBnAct(filters, 3, **norm_kw, name="cv1")
        self.cv2 = _ConvBnAct(filters, 3, **norm_kw, name="cv2")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.cv2(self.cv1(x, training=training), training=training)
        return x + h if self._shortcut else h


class C2f(tf.keras.layers.Layer):
    """Cross-Stage-Partial block with n C2fBottleneck repetitions.

    Forward:
        cv1(1×1 → filters) → split into two halves of size filters//2
        → pass second half through n bottlenecks
        → concat all chunks → cv2(1×1 → filters)
    """

    def __init__(
        self,
        filters: int,
        n: int = 1,
        shortcut: bool = True,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._c = filters // 2
        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )
        self.cv1 = _ConvBnAct(filters, 1, **norm_kw, name="cv1")
        self.cv2 = _ConvBnAct(filters, 1, **norm_kw, name="cv2")
        self.bottlenecks = [
            C2fBottleneck(self._c, shortcut=shortcut, **norm_kw, name=f"bn{i}")
            for i in range(n)
        ]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        y = self.cv1(x, training=training)      # [B, H, W, filters]
        c = self._c
        chunks = [y[..., :c], y[..., c:]]       # two halves of size c
        for bn in self.bottlenecks:
            chunks.append(bn(chunks[-1], training=training))
        return self.cv2(tf.concat(chunks, axis=-1), training=training)


# ---------------------------------------------------------------------------
# SPPF
# ---------------------------------------------------------------------------

class SPPF(tf.keras.layers.Layer):
    """Spatial Pyramid Pooling - Fast.

    cv1(1×1 → filters//2) → three sequential max-pools →
    concat([original, pool1, pool2, pool3]) → cv2(1×1 → filters)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 5,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        hidden = filters // 2
        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )
        self.cv1 = _ConvBnAct(hidden, 1, **norm_kw, name="cv1")
        self.cv2 = _ConvBnAct(filters, 1, **norm_kw, name="cv2")
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=kernel_size, strides=1, padding="same"
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.cv1(x, training=training)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(tf.concat([x, y1, y2, y3], axis=-1), training=training)


# ---------------------------------------------------------------------------
# Backbone configs — model_id takes precedence over constructor width/depth
# ---------------------------------------------------------------------------

_BACKBONE_CONFIGS: Dict[str, Dict] = {
    "cspdarknetv8n": {"width": 0.25, "depth": 0.33},
    "cspdarknetv8s": {"width": 0.50, "depth": 0.33},
    "cspdarknetv8m": {"width": 0.75, "depth": 0.67},
    "cspdarknetv8l": {"width": 1.00, "depth": 1.00},
    "cspdarknetv8x": {"width": 1.25, "depth": 1.00},
}

# Base channel counts at width_scale = 1.0
_BASE_CHANNELS = [64, 128, 256, 512, 1024]
# Base bottleneck counts at depth_scale = 1.0
_BASE_DEPTHS = [3, 6, 6, 3]


def _make_div8(v: float) -> int:
    return max(8, int(v + 4) // 8 * 8)


def _scale_ch(base: int, width: float) -> int:
    return _make_div8(base * width)


def _scale_n(base: int, depth: float) -> int:
    return max(1, round(base * depth))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

@BACKBONES.register("cspdarknetv8s")
class CSPDarkNetV8(tf.keras.Model):
    """CSPDarkNet backbone returning P3/P4/P5 feature maps.

    Returns:
        dict with keys '3', '4', '5':
            '3': float32 [batch, H/8,  W/8,  C3]
            '4': float32 [batch, H/16, W/16, C4]
            '5': float32 [batch, H/32, W/32, C5]

    For cspdarknetv8s with 672×672 input:
        '3': [B, 84, 84, 128]
        '4': [B, 42, 42, 256]
        '5': [B, 21, 21, 512]
    """

    def __init__(
        self,
        model_id: str = "cspdarknetv8s",
        input_specs: Optional[tf.keras.layers.InputSpec] = None,
        min_level: int = 3,
        max_level: int = 5,
        width_scale: float = 0.5,
        depth_scale: float = 0.33,
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # model_id overrides constructor width/depth
        cfg = _BACKBONE_CONFIGS.get(model_id, {})
        width = cfg.get("width", width_scale)
        depth = cfg.get("depth", depth_scale)

        ch = [_scale_ch(c, width) for c in _BASE_CHANNELS]
        ns = [_scale_n(n, depth) for n in _BASE_DEPTHS]

        # Store channel counts for output_specs
        self._c3, self._c4, self._c5 = ch[2], ch[3], ch[4]
        self._min_level = min_level
        self._max_level = max_level

        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )

        # Stem (stride 4 total: two stride-2 convs + C2f)
        self.stem_conv1 = _ConvBnAct(ch[0], 3, strides=2, **norm_kw, name="stem_conv1")
        self.stem_conv2 = _ConvBnAct(ch[1], 3, strides=2, **norm_kw, name="stem_conv2")
        self.stem_c2f   = C2f(ch[1], n=ns[0], shortcut=True, **norm_kw, name="stem_c2f")

        # P3 (stride 8)
        self.down1  = _ConvBnAct(ch[2], 3, strides=2, **norm_kw, name="down1")
        self.c2f_p3 = C2f(ch[2], n=ns[1], shortcut=True, **norm_kw, name="c2f_p3")

        # P4 (stride 16)
        self.down2  = _ConvBnAct(ch[3], 3, strides=2, **norm_kw, name="down2")
        self.c2f_p4 = C2f(ch[3], n=ns[2], shortcut=True, **norm_kw, name="c2f_p4")

        # P5 (stride 32)
        self.down3      = _ConvBnAct(ch[4], 3, strides=2, **norm_kw, name="down3")
        self.c2f_p5_pre = C2f(ch[4], n=ns[3], shortcut=True, **norm_kw, name="c2f_p5_pre")
        self.sppf       = SPPF(ch[4], kernel_size=5, **norm_kw, name="sppf")

    @property
    def output_specs(self) -> Dict[str, int]:
        """Channel counts of each output level (ints, not TensorShapes)."""
        return {"3": self._c3, "4": self._c4, "5": self._c5}

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Dict[str, tf.Tensor]:
        x = self.stem_conv1(inputs, training=training)
        x = self.stem_conv2(x, training=training)
        x = self.stem_c2f(x, training=training)

        x  = self.down1(x, training=training)
        p3 = self.c2f_p3(x, training=training)

        x  = self.down2(p3, training=training)
        p4 = self.c2f_p4(x, training=training)

        x  = self.down3(p4, training=training)
        x  = self.c2f_p5_pre(x, training=training)
        p5 = self.sppf(x, training=training)

        return {"3": p3, "4": p4, "5": p5}
