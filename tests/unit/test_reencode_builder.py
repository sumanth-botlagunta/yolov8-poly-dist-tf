"""Unit test for tools/reencode_tfds_672.reencode_one.

Runs entirely on a tiny SYNTHETIC TFDS dataset built in a tmpdir via the same
``tfds.dataset_builders.TfDataBuilder`` API the tool uses -- no real dataset
files required. We:

  1. write a fake ``cleaner_polygon2026:2.0.0`` source (8 examples, variable
     image sizes, nested ``objects`` with the project schema incl. points[N,3972]),
  2. run ``reencode_one`` against the tmpdir,
  3. assert the produced ``cleaner_polygon2026_672`` dataset: images are
     (size,size,3) uint8, orig_height/orig_width == synthetic originals,
     objects/points + labels pass through bit-exact, example count preserved,
  4. assert PolygonDecoder.decode picks up the ORIGINAL dims (not 672) from the
     stored orig_height/orig_width -- the value the copy-paste correction consumes.

Fast (<60s), no special markers.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

# Make the repo root importable (tools/, data_pipeline/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tfds = pytest.importorskip("tensorflow_datasets")

from tools.reencode_tfds_672 import reencode_one  # noqa: E402

_SIZE = 64                      # small target -> fast JPEG re-encode
_N = 8
_POINTS_WIDTH = 3972            # cleaner_polygon2026 schema width
_VERSION = "2.0.0"
_SRC_NAME = "cleaner_polygon2026"


def _synthetic_originals():
    """Deterministic per-example (height, width) for the 8 synthetic images."""
    dims = []
    for i in range(_N):
        if i % 2 == 0:
            dims.append((100, 160))
        else:
            dims.append((90, 120))
    return dims


def _build_synthetic_source(data_dir: str):
    """Write a fake cleaner_polygon2026:2.0.0 dataset; return the truth tables."""
    dims = _synthetic_originals()

    features = tfds.features.FeaturesDict({
        "image": tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
        "image/filename": tfds.features.Text(),
        "image/id": tfds.features.Scalar(dtype=np.int64),
        "objects": tfds.features.Sequence({
            "area": tfds.features.Scalar(dtype=np.int64),
            "bbox": tfds.features.BBoxFeature(),
            "id": tfds.features.Scalar(dtype=np.int64),
            "is_crowd": tfds.features.Scalar(dtype=np.bool_),
            "is_dontcare": tfds.features.Scalar(dtype=np.bool_),
            "label": tfds.features.ClassLabel(num_classes=39),
            "points": tfds.features.Tensor(shape=(_POINTS_WIDTH,), dtype=np.float32),
        }),
    })

    # Build the ground-truth tables we will later assert pass-through on.
    truth = {}
    for i, (h, w) in enumerate(dims):
        rng = np.random.RandomState(i)
        labels = np.array([1 + (i % 5), 2 + (i % 7)], np.int64)
        points = (rng.rand(2, _POINTS_WIDTH).astype(np.float32))
        truth[i] = {
            "orig_height": h,
            "orig_width": w,
            "labels": labels,
            "points": points,
        }

    def gen():
        for i, (h, w) in enumerate(dims):
            t = truth[i]
            rng = np.random.RandomState(1000 + i)
            img = (rng.rand(h, w, 3) * 255).astype(np.uint8)
            yield {
                "image": img,
                "image/filename": f"img_{i}.jpg",
                "image/id": i,
                "objects": {
                    "area": np.array([h * w // 4, h * w // 8], np.int64),
                    "bbox": np.array(
                        [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]], np.float32
                    ),
                    "id": np.array([i * 2, i * 2 + 1], np.int64),
                    "is_crowd": np.array([False, True], np.bool_),
                    "is_dontcare": np.array([False, False], np.bool_),
                    "label": t["labels"],
                    "points": t["points"],
                },
            }

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "image": tf.TensorSpec((None, None, 3), tf.uint8),
            "image/filename": tf.TensorSpec((), tf.string),
            "image/id": tf.TensorSpec((), tf.int64),
            "objects": {
                "area": tf.TensorSpec((None,), tf.int64),
                "bbox": tf.TensorSpec((None, 4), tf.float32),
                "id": tf.TensorSpec((None,), tf.int64),
                "is_crowd": tf.TensorSpec((None,), tf.bool),
                "is_dontcare": tf.TensorSpec((None,), tf.bool),
                "label": tf.TensorSpec((None,), tf.int64),
                "points": tf.TensorSpec((None, _POINTS_WIDTH), tf.float32),
            },
        },
    )

    builder = tfds.dataset_builders.TfDataBuilder(
        name=_SRC_NAME,
        version=_VERSION,
        features=features,
        split_datasets={"train": ds},
        data_dir=data_dir,
        description="synthetic source for reencode test",
    )
    builder.download_and_prepare()
    return truth


def test_reencode_builder_end_to_end(tmp_path):
    data_dir = str(tmp_path)
    truth = _build_synthetic_source(data_dir)

    out_name = reencode_one(
        name=f"{_SRC_NAME}:{_VERSION}",
        data_dir=data_dir,
        size=_SIZE,
        splits="all",
    )
    assert out_name == f"{_SRC_NAME}_672"

    # Loadable via tfds with the tmp data_dir.
    out_ds = tfds.load(
        f"{out_name}:{_VERSION}", data_dir=data_dir, split="train",
        shuffle_files=False,
    )

    # image/id -> example map; iterate in id order is not guaranteed, so key by id.
    seen_ids = set()
    count = 0
    for ex in out_ds:
        count += 1
        ex_id = int(ex["image/id"].numpy())
        seen_ids.add(ex_id)
        t = truth[ex_id]

        # Image is (size,size,3) uint8.
        img = ex["image"].numpy()
        assert img.shape == (_SIZE, _SIZE, 3), img.shape
        assert img.dtype == np.uint8

        # Original dims preserved as int64 scalars.
        assert int(ex["orig_height"].numpy()) == t["orig_height"]
        assert int(ex["orig_width"].numpy()) == t["orig_width"]

        # Labels + points pass through bit-exact.
        np.testing.assert_array_equal(ex["objects"]["label"].numpy(), t["labels"])
        np.testing.assert_array_equal(ex["objects"]["points"].numpy(), t["points"])

    # Example count preserved; all ids present.
    assert count == _N
    assert seen_ids == set(range(_N))


def test_reencode_decoder_uses_original_dims(tmp_path):
    """PolygonDecoder.decode must report ORIGINAL dims (from orig_height/width)."""
    from data_pipeline.tfds_decoders import PolygonDecoder

    data_dir = str(tmp_path)
    truth = _build_synthetic_source(data_dir)
    reencode_one(
        name=f"{_SRC_NAME}:{_VERSION}", data_dir=data_dir, size=_SIZE, splits="all",
    )

    out_ds = tfds.load(
        f"{_SRC_NAME}_672:{_VERSION}", data_dir=data_dir, split="train",
        shuffle_files=False,
    )

    decoder = PolygonDecoder(max_vertices=_POINTS_WIDTH - 2, num_classes=39)
    checked = 0
    for ex in out_ds:
        ex_id = int(ex["image/id"].numpy())
        t = truth[ex_id]
        decoded = decoder.decode(ex)
        # height/width come from orig_height/orig_width -> the ORIGINAL capture
        # size, NOT the stored 672/_SIZE image. This is what copy-paste consumes.
        assert int(decoded["height"].numpy()) == t["orig_height"]
        assert int(decoded["width"].numpy()) == t["orig_width"]
        assert decoded["height"].numpy() != _SIZE or decoded["width"].numpy() != _SIZE
        checked += 1
    assert checked == _N
