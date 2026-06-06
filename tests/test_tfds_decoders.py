"""Tests for TFDS decoder output formats.

Validates:
    - Output tensor dtypes and shapes match the expected schema.
    - Polygon coordinates are in [0, 1] normalized range.
    - Padding sentinel (-1) is present on short polygon arrays.
    - ServingBotDetDecoder includes groundtruth_dists.
"""

import tensorflow as tf
import unittest

from data_pipeline.tfds_decoders import (
    CopyPasteDecoder,
    PolygonDecoder,
    ServingBotDetDecoder,
)


class TestPolygonDecoder(unittest.TestCase):
    def test_output_keys(self):
        raise NotImplementedError

    def test_image_dtype(self):
        raise NotImplementedError

    def test_polygon_range(self):
        raise NotImplementedError


class TestServingBotDetDecoder(unittest.TestCase):
    def test_distance_field_present(self):
        raise NotImplementedError


class TestCopyPasteDecoder(unittest.TestCase):
    def test_rgba_image(self):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
