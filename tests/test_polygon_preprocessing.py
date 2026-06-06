"""Tests for polygon format conversion (_preprocess_polygons_v2).

Validates:
    - PolyYOLO output shape is [N, 360/angle_step * 3].
    - Origin (first two values) equals the box center.
    - dx, dy are relative to origin and not normalized beyond [-1, 1].
    - conf is 1.0 for valid vertices and 0.0 for absent ones.
"""

import tensorflow as tf
import unittest

from data_pipeline.yolo_parser import V8ParserExtended


class TestPreprocessPolygonsV2(unittest.TestCase):
    def test_output_shape(self):
        raise NotImplementedError

    def test_origin_is_box_center(self):
        raise NotImplementedError

    def test_conf_binary(self):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
