"""YOLOv8 FPN-PAN decoder with C2f stacks.

Takes backbone features {3, 4, 5} and produces enriched multi-scale feature
maps via top-down FPN and bottom-up PAN paths, each using C2f blocks.

Channel flow for cspdarknetv8s (c3=128, c4=256, c5=512):

  FPN top-down:
    P5(512)  -> upsample -> concat(P4=256) -> C2f(256) -> P4'(256)
    P4'(256) -> upsample -> concat(P3=128) -> C2f(128) -> P3'(128)

  PAN bottom-up:
    P3'(128)  -> conv(s=2,128) -> concat(P4'=256) -> C2f(256) -> P4''(256)
    P4''(256) -> conv(s=2,256) -> concat(P5=512)  -> C2f(512) -> P5''(512)
"""

from __future__ import annotations

from typing import Dict, Union

import tensorflow as tf

from configs.registry import DECODERS
from models.backbone import C2f, _ConvBnAct


# Bottleneck repetitions per decoder size variant.
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
        P5   -> upsample -> concat(P4) -> C2f -> P4'
        P4'  -> upsample -> concat(P3) -> C2f -> P3'

    PAN (bottom-up):
        P3'  -> conv(s=2) -> concat(P4') -> C2f -> P4''
        P4'' -> conv(s=2) -> concat(P5)  -> C2f -> P5''

    Returns a dict with keys '3', '4', '5' containing the enriched feature maps.
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
        """Initializes the decoder.

        Args:
            input_specs: Dict keyed by level ('3', '4', '5') giving each backbone
                output's channel count as an int or TensorShape.
            model_id: Size variant (e.g. 'v8s'); selects the C2f repetition count.
            version: Unused; kept for interface compatibility.
            activation: Activation name for all conv blocks.
            norm_momentum: BatchNormalization momentum.
            norm_epsilon: BatchNormalization epsilon.
            use_sync_bn: Whether to use synchronized BatchNormalization.
            use_separable_conv: Unused; kept for interface compatibility.
            **kwargs: Passed to tf.keras.Model.
        """
        super().__init__(**kwargs)

        # Channel counts; accept both int and TensorShape values.
        def _ch(v: Union[int, tf.TensorShape]) -> int:
            return int(v) if isinstance(v, int) else int(v[-1])

        c3 = _ch(input_specs["3"])   # e.g. 128
        c4 = _ch(input_specs["4"])   # e.g. 256
        c5 = _ch(input_specs["5"])   # e.g. 512

        # C2f bottleneck repetitions in the neck.
        size_key = model_id.replace("v8", "").lstrip("_") or "s"
        n = _DECODER_DEPTH.get(size_key, 1)

        norm_kw = dict(
            activation=activation,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
            use_sync_bn=use_sync_bn,
        )

        # FPN top-down: concat(P5 up, P4) -> C2f -> c4; concat(P4' up, P3) -> C2f -> c3.
        self.fpn_c2f_p4 = C2f(c4, n=n, shortcut=False, **norm_kw, name="fpn_c2f_p4")
        self.fpn_c2f_p3 = C2f(c3, n=n, shortcut=False, **norm_kw, name="fpn_c2f_p3")

        # PAN bottom-up: stride-2 downsample -> concat -> C2f, twice.
        self.pan_down_p3 = _ConvBnAct(c3, 3, strides=2, **norm_kw, name="pan_down_p3")
        self.pan_c2f_p4 = C2f(c4, n=n, shortcut=False, **norm_kw, name="pan_c2f_p4")

        self.pan_down_p4 = _ConvBnAct(c4, 3, strides=2, **norm_kw, name="pan_down_p4")
        self.pan_c2f_p5 = C2f(c5, n=n, shortcut=False, **norm_kw, name="pan_c2f_p5")

        # Upsample mode. Default dynamic (tf.image.resize to the target level's runtime
        # size) can never disagree with the level it concatenates to. The device
        # exporter sets static_resize=True at a fixed input size so the upsample size is
        # a compile-time constant, dropping the Shape->StridedSlice the SNPE converter
        # rejects; numerically identical when the static size is the runtime size.
        self.static_resize = False

    # ------------------------------------------------------------------

    def _upsample(self, src: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
        """Nearest-upsamples `src` to `ref`'s spatial size.

        Dynamic by default (size read from the runtime `ref`). When `static_resize`
        is set (device export at a fixed input size) the size is a compile-time
        constant, so the graph carries no Shape->StridedSlice; the output is
        identical when the static size equals the runtime size.
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
        """Applies a pre-built C2f layer looked up by attribute name.

        Raises:
            ValueError: If no C2f layer with the given name exists.
        """
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
        p5_up  = self._upsample(p5, p4)                                               # upsample P5(c5) -> P4 size
        p4_fpn = self.fpn_c2f_p4(tf.concat([p5_up, p4], axis=-1), training=training)  # concat c5+c4 -> C2f -> c4

        p4_up  = self._upsample(p4_fpn, p3)                                           # upsample P4'(c4) -> P3 size
        p3_out = self.fpn_c2f_p3(tf.concat([p4_up, p3], axis=-1), training=training)  # concat c4+c3 -> C2f -> c3

        # --- PAN: bottom-up ---
        p3_down = self.pan_down_p3(p3_out, training=training)                          # stride-2 -> c3
        p4_out  = self.pan_c2f_p4(tf.concat([p3_down, p4_fpn], axis=-1), training=training)  # -> c4

        p4_down = self.pan_down_p4(p4_out, training=training)                          # stride-2 -> c4
        p5_out  = self.pan_c2f_p5(tf.concat([p4_down, p5], axis=-1), training=training)      # -> c5

        return {"3": p3_out, "4": p4_out, "5": p5_out}
