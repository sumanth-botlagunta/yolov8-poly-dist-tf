"""YOLOv8 FPN-PAN decoder with C2f stacks.

Takes backbone features {3, 4, 5} and produces enriched multi-scale feature
maps via top-down FPN and bottom-up PAN paths, each step using C2f blocks.

Channel flow for cspdarknetv8s (c3=128, c4=256, c5=512):

  FPN top-down:
    P5(512) → upsample → concat(P4=256) → C2f(256) → P4'(256)   [C2f cv1: 768→256]
    P4'(256)→ upsample → concat(P3=128) → C2f(128) → P3'(128)   [C2f cv1: 384→128]

  PAN bottom-up:
    P3'(128)→ conv(s=2,128) → concat(P4'=256)  → C2f(256) → P4''(256)
    P4''(256)→ conv(s=2,256) → concat(P5=512)  → C2f(512) → P5''(512)

Classes:
    YoloDecoder: FPN-PAN decoder returning the same {3, 4, 5} key schema.
"""

from __future__ import annotations

from typing import Dict, Union

import tensorflow as tf

from configs.registry import DECODERS
from models.backbone import C2f, _ConvBnAct


# Bottleneck repetitions per decoder size variant
_DECODER_DEPTH: Dict[str, int] = {
    "s": 1,
    "m": 2,
    "l": 3,
    "x": 3,
}


@DECODERS.register("yolo_decoder")
class YoloDecoder(tf.keras.Model):
    """FPN-PAN neck for YOLOv8.

    FPN (top-down):
        P5 → upsample → concat(P4) → C2f → P4'
        P4'→ upsample → concat(P3) → C2f → P3'

    PAN (bottom-up):
        P3'→ conv(s=2) → concat(P4') → C2f → P4''
        P4''→ conv(s=2) → concat(P5)  → C2f → P5''

    Returns:
        dict with keys '3', '4', '5' containing the enriched feature maps.
    """

    def __init__(
        self,
        input_specs: Dict[str, Union[int, tf.TensorShape]],
        model_id: str = "v8s",
        version: str = "v8",
        activation: str = "relu",
        norm_momentum: float = 0.97,
        norm_epsilon: float = 0.001,
        use_sync_bn: bool = False,
        use_separable_conv: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Extract channel counts — accept both int and TensorShape values
        def _ch(v: Union[int, tf.TensorShape]) -> int:
            return int(v) if isinstance(v, int) else int(v[-1])

        c3 = _ch(input_specs["3"])   # e.g. 128
        c4 = _ch(input_specs["4"])   # e.g. 256
        c5 = _ch(input_specs["5"])   # e.g. 512

        # Number of C2f bottleneck repetitions in the neck
        size_key = model_id.replace("v8", "").lstrip("_") or "s"
        n = _DECODER_DEPTH.get(size_key, 1)

        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )

        # ---- FPN top-down ----
        # P5(c5) upsample and concat directly with P4(c4) → c5+c4 channels into C2f
        self.fpn_c2f_p4 = C2f(c4, n=n, shortcut=False, **norm_kw, name="fpn_c2f_p4")
        # P4'(c4) upsample and concat directly with P3(c3) → c4+c3 channels into C2f
        self.fpn_c2f_p3 = C2f(c3, n=n, shortcut=False, **norm_kw, name="fpn_c2f_p3")

        # ---- PAN bottom-up ----
        # Stride-2 conv to downsample P3'
        self.pan_down_p3 = _ConvBnAct(c3, 3, strides=2, **norm_kw, name="pan_down_p3")
        # After concat: c3 + c4 → C2f → c4
        self.pan_c2f_p4 = C2f(c4, n=n, shortcut=False, **norm_kw, name="pan_c2f_p4")

        # Stride-2 conv to downsample P4''
        self.pan_down_p4 = _ConvBnAct(c4, 3, strides=2, **norm_kw, name="pan_down_p4")
        # After concat: c4 + c5 → C2f → c5
        self.pan_c2f_p5 = C2f(c5, n=n, shortcut=False, **norm_kw, name="pan_c2f_p5")

        # FPN upsample mode. Default DYNAMIC (tf.image.resize to the target level's
        # runtime size): robust — it can never disagree with the level it concatenates
        # to, at any input size or build-vs-run mismatch. The device exporter flips this
        # to True so the upsample size is a compile-time constant (the export is a fixed
        # input size), which removes the Shape→StridedSlice the dynamic form emits — that
        # subgraph is what the SNPE converter's StridedSliceLayerBuilder rejects. The two
        # are numerically identical whenever the static size is the true runtime size,
        # which it is for the fixed-size export.
        self.static_resize = False

    # ------------------------------------------------------------------

    def _upsample(self, src: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
        """Nearest upsample ``src`` to ``ref``'s spatial size.

        Dynamic by default (size read from the runtime ``ref`` → robust to any input
        size / build-vs-run mismatch). When ``static_resize`` is set (device export at a
        fixed input size) the size is a compile-time constant, so the graph carries no
        Shape→StridedSlice for the SNPE converter to reject. Identical output when the
        static size equals the runtime size, which holds for the fixed-size export.
        """
        s = ref.shape
        if self.static_resize and s.rank == 4 and s[1] is not None and s[2] is not None:
            return tf.image.resize(src, [int(s[1]), int(s[2])], method="nearest")
        return tf.image.resize(src, tf.shape(ref)[1:3], method="nearest")

    def c2f_stack(
        self,
        inputs: tf.Tensor,
        filters: int,
        n: int,
        name: str = "",
        training: bool = False,
    ) -> tf.Tensor:
        """Apply a pre-built C2f layer looked up by attribute name."""
        layer = getattr(self, name, None)
        if layer is None:
            raise ValueError(f"YoloDecoder has no C2f layer named '{name}'")
        return layer(inputs, training=training)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        p3 = inputs["3"]   # [B, H/8,  W/8,  c3]
        p4 = inputs["4"]   # [B, H/16, W/16, c4]
        p5 = inputs["5"]   # [B, H/32, W/32, c5]

        # --- FPN: top-down ---
        p5_up  = self._upsample(p5, p4)                                               # upsample P5(c5) → P4 size
        p4_fpn = self.fpn_c2f_p4(tf.concat([p5_up, p4], axis=-1), training=training)  # concat c5+c4 → C2f → c4

        p4_up  = self._upsample(p4_fpn, p3)                                           # upsample P4'(c4) → P3 size
        p3_out = self.fpn_c2f_p3(tf.concat([p4_up, p3], axis=-1), training=training)  # concat c4+c3 → C2f → c3

        # --- PAN: bottom-up ---
        p3_down = self.pan_down_p3(p3_out, training=training)                          # stride-2 → c3
        p4_out  = self.pan_c2f_p4(tf.concat([p3_down, p4_fpn], axis=-1), training=training)  # → c4

        p4_down = self.pan_down_p4(p4_out, training=training)                          # stride-2 → c4
        p5_out  = self.pan_c2f_p5(tf.concat([p4_down, p5], axis=-1), training=training)      # → c5

        return {"3": p3_out, "4": p4_out, "5": p5_out}
