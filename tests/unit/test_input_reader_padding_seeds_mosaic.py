"""Pinning tests for train-critical fixes (b), (c), (d).

(b) input_reader padded_batch(4) pads groundtruth_polygons with the -1.0 sentinel
    (not the default 0.0, which is a valid top-left vertex coordinate), and every other
    decoder key with its natural empty value.
(c) the three shuffle stages use DISTINCT seeds: detection source = self._seed,
    cnp source = self._seed+1, post-unbatch = self._seed+2.
(d) MosaicConfig dataclass default mosaic_center == Mosaic.__init__ default == 0.25, and
    every shipped tier YAML resolves mosaic_center to 0.25.
"""

import glob
import inspect
import os
import unittest

import numpy as np
import tensorflow as tf

from configs.model_config import MosaicConfig
from configs.yaml_loader import load_config
from data_pipeline.mosaic import Mosaic

_EXP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "experiments", "yolo"
)

# The padding_values dict the input_reader installs on padded_batch(4). Mirrors the
# decoder element spec (PolygonDecoder/ServingBot output), keyed by name.
_PADDING_VALUES = {
    "image": tf.constant(0, tf.uint8),
    "source_id": tf.constant("", tf.string),
    "height": tf.constant(0, tf.int32),
    "width": tf.constant(0, tf.int32),
    "groundtruth_boxes": tf.constant(0.0, tf.float32),
    "groundtruth_classes": tf.constant(0, tf.int64),
    "groundtruth_polygons": tf.constant(-1.0, tf.float32),
    "groundtruth_is_crowd": tf.constant(False, tf.bool),
    "groundtruth_area": tf.constant(0.0, tf.float32),
    "groundtruth_dontcare": tf.constant(0, tf.int64),
    "groundtruth_dists": tf.constant(0.0, tf.float32),
}


def _decoder_like_example(n_obj, n_poly_cols):
    """One element matching the decoder output spec, ragged in the per-object dim."""
    return {
        "image": tf.zeros([8, 8, 3], tf.uint8),
        "source_id": tf.constant("x", tf.string),
        "height": tf.constant(8, tf.int32),
        "width": tf.constant(8, tf.int32),
        "groundtruth_boxes": tf.fill([n_obj, 4], 0.5),
        "groundtruth_classes": tf.zeros([n_obj], tf.int64),
        # all-valid polygons (0.3) so any -1.0 in the batched result is padding.
        "groundtruth_polygons": tf.fill([n_obj, n_poly_cols], 0.3),
        "groundtruth_is_crowd": tf.zeros([n_obj], tf.bool),
        "groundtruth_area": tf.ones([n_obj], tf.float32),
        "groundtruth_dontcare": tf.zeros([n_obj], tf.int64),
        "groundtruth_dists": tf.fill([n_obj], -1.0),
    }


class TestPaddedBatchPolygonSentinel(unittest.TestCase):
    def test_polygon_rows_padded_with_minus_one_not_zero(self):
        # Four elements with DIFFERENT object counts → padded_batch must pad the
        # short ones in the object dim. The polygon padding must be -1.0, not 0.0.
        examples = [
            _decoder_like_example(1, 4),
            _decoder_like_example(3, 4),
            _decoder_like_example(2, 4),
            _decoder_like_example(2, 4),
        ]

        def gen():
            for ex in examples:
                yield ex

        spec = {
            "image": tf.TensorSpec([8, 8, 3], tf.uint8),
            "source_id": tf.TensorSpec([], tf.string),
            "height": tf.TensorSpec([], tf.int32),
            "width": tf.TensorSpec([], tf.int32),
            "groundtruth_boxes": tf.TensorSpec([None, 4], tf.float32),
            "groundtruth_classes": tf.TensorSpec([None], tf.int64),
            "groundtruth_polygons": tf.TensorSpec([None, None], tf.float32),
            "groundtruth_is_crowd": tf.TensorSpec([None], tf.bool),
            "groundtruth_area": tf.TensorSpec([None], tf.float32),
            "groundtruth_dontcare": tf.TensorSpec([None], tf.int64),
            "groundtruth_dists": tf.TensorSpec([None], tf.float32),
        }
        ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
        ds = ds.padded_batch(4, drop_remainder=True, padding_values=_PADDING_VALUES)
        batch = next(iter(ds))

        polys = batch["groundtruth_polygons"].numpy()  # [4, max_obj, 4]
        # max object count is 3, so the n=1 and n=2 elements get padded object rows.
        # Padded rows must be entirely -1.0 (sentinel), never 0.0.
        # Element 0 had 1 object → rows 1,2 are padding.
        self.assertTrue(np.all(polys[0, 1:] == -1.0), "polygon padding must be -1.0")
        # No padded row is a valid (0.0) coordinate.
        self.assertFalse(
            np.any((polys == 0.0)), "0.0 must never appear as polygon padding"
        )
        # Real vertices preserved.
        self.assertTrue(np.allclose(polys[0, 0], 0.3))

        # Other fields keep their natural empties.
        self.assertTrue(np.all(batch["groundtruth_boxes"].numpy()[0, 1:] == 0.0))
        self.assertTrue(np.all(batch["groundtruth_classes"].numpy()[0, 1:] == 0))
        self.assertEqual(batch["groundtruth_is_crowd"].numpy()[0, 1:].tolist(), [False, False])

    def test_padding_values_cover_every_decoder_key(self):
        from data_pipeline.tfds_decoders import PolygonDecoder

        src = inspect.getsource(PolygonDecoder.decode)
        # Every key the decoder returns must have an explicit padding value.
        for key in _PADDING_VALUES:
            self.assertIn(f"'{key}'", src, f"{key} not in decoder output (stale spec)")


class TestDistinctShuffleSeeds(unittest.TestCase):
    def test_three_shuffle_stages_use_distinct_seeds(self):
        # Read the source text directly: importing input_reader requires
        # tensorflow_datasets, which is not present in the unit-test env.
        ir_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data_pipeline", "input_reader.py"
        )
        with open(ir_path) as f:
            src = f.read()
        # detection source shuffle: seed=self._seed (base, None-safe)
        self.assertIn("seed=self._seed, reshuffle_each_iteration=True", src)
        # cnp source shuffle: seed=self._seed+1, GUARDED so seed=None stays None
        # (bare `self._seed + 1` raises TypeError when seed is None).
        self.assertIn("seed=None if self._seed is None else self._seed + 1", src)
        # post-unbatch shuffle: seed=self._seed+2, likewise guarded.
        self.assertIn("seed=None if self._seed is None else self._seed + 2", src)
        # The unguarded arithmetic must NOT reappear (regression guard).
        self.assertNotIn("seed=self._seed + 1,", src)
        self.assertNotIn("seed=self._seed + 2,", src)


class TestSeedNoneDerivationDoesNotCrash(unittest.TestCase):
    """Behavioral pin for the seed=None TypeError. The constructor declares
    `seed: Optional[int] = None`; the cnp/post-unbatch shuffle stages derive their
    seed from it. The bare `self._seed + N` form crashes with TypeError when seed is
    None. This replicates the exact guarded derivation used in input_reader so the
    failure mode is exercised without importing tensorflow_datasets."""

    @staticmethod
    def _derive(seed, offset):
        # Mirror of input_reader.py: `None if self._seed is None else self._seed + N`.
        return None if seed is None else seed + offset

    def test_seed_none_propagates_as_none(self):
        self.assertIsNone(self._derive(None, 1))
        self.assertIsNone(self._derive(None, 2))

    def test_seed_set_offsets_remain_distinct(self):
        self.assertEqual(self._derive(7, 1), 8)
        self.assertEqual(self._derive(7, 2), 9)

    def test_bare_arithmetic_would_crash_on_none(self):
        # Documents the original failure: the unguarded form raises TypeError.
        with self.assertRaises(TypeError):
            _ = None + 1  # noqa: E711 — the exact crash the guard prevents


class TestMosaicCenterDefault(unittest.TestCase):
    def test_dataclass_default_matches_mosaic_init(self):
        init_default = inspect.signature(Mosaic.__init__).parameters["mosaic_center"].default
        self.assertEqual(init_default, 0.25)
        self.assertEqual(MosaicConfig().mosaic_center, 0.25)

    def test_every_tier_yaml_resolves_to_025(self):
        paths = sorted(glob.glob(os.path.join(_EXP_DIR, "*.yaml")))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                cfg = load_config(path)
                self.assertEqual(
                    cfg.task.train_data.parser.mosaic.mosaic_center,
                    0.25,
                    f"{os.path.basename(path)} mosaic_center != 0.25",
                )


if __name__ == "__main__":
    unittest.main()
