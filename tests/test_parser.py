"""Tests for V8ParserExtended and V8DistanceParser output formats.

Validates:
    - Image is float32 normalized to [0, 1].
    - labels['bbox'] shape is [max_instances, 4].
    - labels['n_gt'] matches the number of actual ground-truth boxes.
    - Eval parser does not apply random flip.
    - Distance parser sets ignore_bg=1.
"""

import tensorflow as tf
import unittest

from data_pipeline.yolo_parser import V8ParserExtended
from data_pipeline.distance_parser import V8DistanceParser


class TestV8ParserExtended(unittest.TestCase):
    def test_image_range(self):
        raise NotImplementedError

    def test_label_shapes(self):
        raise NotImplementedError

    def test_n_gt_correct(self):
        raise NotImplementedError


class TestV8DistanceParser(unittest.TestCase):
    def test_ignore_bg_set(self):
        raise NotImplementedError

    def test_log_distance_encoding(self):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
