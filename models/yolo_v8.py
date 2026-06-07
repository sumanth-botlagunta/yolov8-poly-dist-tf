"""Top-level YOLOv8 model class assembling backbone, decoder, head, and generator.

During training  → returns raw head outputs (dict of dicts).
During inference → passes raw outputs through YoloV8Layer for decoded detections.

The deploy flag (experiment_config: deploy=True) switches between the two modes
so that the same model class handles both cases.

Classes:
    YoloV8: Full YOLOv8 model with polygon and distance branches.

Factory function:
    build_yolov8(config) → YoloV8 assembled from a ModelConfig.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import tensorflow as tf

from configs.model_config import ModelConfig
from configs.registry import BACKBONES, DECODERS, HEADS
from models.backbone import CSPDarkNetV8
from models.decoder import YoloDecoder
from models.detection_generator import YoloV8Layer
from models.head import YoloV8Head


class YoloV8(tf.keras.Model):
    """YOLOv8 object detector with polygon segmentation and distance estimation.

    Training mode  (deploy=False): returns raw head output dict for loss computation.
    Inference mode (deploy=True):  returns decoded + NMS-filtered detections.

    Call build_and_init() once before training to build all sub-layers and
    run smart bias initialisation.
    """

    def __init__(
        self,
        backbone: CSPDarkNetV8,
        decoder: YoloDecoder,
        head: YoloV8Head,
        detection_generator: Optional[YoloV8Layer] = None,
        deploy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone           = backbone
        self.decoder            = decoder
        self.head               = head
        self.detection_generator = detection_generator
        self.deploy             = deploy
        self._biases_initialized = False

    # ------------------------------------------------------------------

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Union[Dict[str, Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
        """Forward pass.

        Training mode (deploy=False):
            Returns raw head output dict:
            {
                'box':        {'3': [B,H3,W3,64], '4': ..., '5': ...},
                'cls':        {'3': [B,H3,W3,39], ...},
                'poly_angle': {...},   # if with_polygons
                'poly_dist':  {...},   # if with_polygons
                'poly_conf':  {...},   # if with_polygons
                'dist':       {...},   # if with_distance
            }

        Inference mode (deploy=True):
            Returns decoded detections dict from YoloV8Layer.
        """
        feats      = self.backbone(inputs, training=training)
        decoded    = self.decoder(feats, training=training)
        raw_output = self.head(decoded, training=training)

        if self.deploy and self.detection_generator is not None:
            return self.detection_generator(raw_output)
        return raw_output

    # ------------------------------------------------------------------

    def build_and_init(self, input_size: Optional[List[int]] = None) -> None:
        """Build all sub-layers and initialize smart biases.

        Call this once after model instantiation and before training.

        Args:
            input_size: [H, W, C] or [H, W]. Defaults to [672, 672, 3].
        """
        if input_size is None:
            input_size = [672, 672, 3]
        h, w = input_size[0], input_size[1]
        dummy = tf.zeros([1, h, w, 3])
        self(dummy, training=False)   # triggers lazy build of all sub-layers

        if self.head.smart_bias and not self._biases_initialized:
            self.head.initialize_biases(input_size=h)
            self._biases_initialized = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_yolov8(config: ModelConfig) -> YoloV8:
    """Assemble a YoloV8 from a ModelConfig dataclass.

    Backbone is selected via BACKBONES registry using config.backbone.model_id.
    Decoder is selected via DECODERS registry using config.decoder.type.
    Head is always 'yolov8_head' (only one registered).
    """
    na = config.norm_activation

    # --- Backbone ---
    backbone_cls = BACKBONES.get(config.backbone.model_id)
    backbone = backbone_cls(
        model_id     = config.backbone.model_id,
        min_level    = config.backbone.min_level,
        max_level    = config.backbone.max_level,
        activation   = na.activation,
        norm_momentum= na.norm_momentum,
        norm_epsilon = na.norm_epsilon,
        use_sync_bn  = na.use_sync_bn,
        name         = "backbone",
    )

    # --- Decoder ---
    # Derive input specs (channel counts) from backbone config
    decoder_cls = DECODERS.get(config.decoder.type)
    # Build backbone + decoder on a dummy input to get output channel counts
    dummy = tf.zeros([1] + config.input_size)
    bb_out = backbone(dummy, training=False)
    input_specs = {k: int(v.shape[-1]) for k, v in bb_out.items()}

    # Determine decoder size variant (s/m/l/x) from config
    model_id = getattr(config.decoder, "model_type", "s") or "s"
    dec_activation = na.activation if config.decoder.activation == "same" else config.decoder.activation

    decoder = decoder_cls(
        input_specs      = input_specs,
        model_id         = model_id,
        version          = config.decoder.version,
        activation       = dec_activation,
        norm_momentum    = na.norm_momentum,
        norm_epsilon     = na.norm_epsilon,
        use_sync_bn      = na.use_sync_bn,
        use_separable_conv = config.decoder.use_separable_conv,
        name             = "decoder",
    )

    # --- Head ---
    head_cls = HEADS.get("yolov8_head")
    head = head_cls(
        num_classes      = config.num_classes,
        output_poly_size = config.output_poly_size,
        output_dist_size = config.output_dist_size,
        num_dist_block   = config.num_dist_block,
        reg_max          = 16,
        smart_bias       = config.head.smart_bias,
        with_polygons    = config.with_polygons,
        with_distance    = config.with_distance,
        activation       = na.activation,
        norm_momentum    = na.norm_momentum,
        norm_epsilon     = na.norm_epsilon,
        use_sync_bn      = na.use_sync_bn,
        name             = "head",
    )

    # --- Detection generator (inference only) ---
    dg_cfg = config.detection_generator
    detection_generator = YoloV8Layer(
        input_image_size = config.input_size[:2],
        max_boxes        = dg_cfg.max_boxes,
        nms_thresh       = dg_cfg.nms_thresh,
        iou_thresh       = dg_cfg.iou_thresh,
        pre_nms_points   = dg_cfg.pre_nms_points,
        nms_type         = dg_cfg.nms_type,
        reg_max          = 16,
        output_poly_size = config.output_poly_size,
    )

    model = YoloV8(
        backbone           = backbone,
        decoder            = decoder,
        head               = head,
        detection_generator= detection_generator,
        deploy             = config.deploy,
        name               = "yolo_v8",
    )

    return model
