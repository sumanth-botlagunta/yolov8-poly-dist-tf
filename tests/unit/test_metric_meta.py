"""Pins train/metric_meta.describe() resolution.

Regression tests for the `best_` prefix shadowing bug: registered keys
(`best_conf_thresh`, `best_checkpoint_epoch`) were unreachable because the
prefix-strip branch ran before the direct registry lookup, so their TensorBoard
tooltips silently disappeared.
"""

from train.metric_meta import METRIC_DESCRIPTIONS, describe


def test_every_registered_key_resolves_to_its_own_entry():
    for key, expected in METRIC_DESCRIPTIONS.items():
        assert describe(key) == expected, (
            f"registered key {key!r} did not resolve to its registry entry"
        )


def test_best_prefix_composition_still_works_for_unregistered_keys():
    # best_<metric> with no direct entry composes from the inner metric.
    d = describe("best_F1score50")
    assert d is not None and "Best-so-far" in d and "F1" in d


def test_per_category_tags_resolve():
    d = describe("cls/35_label_35/ap50")
    assert d is not None and "ap" in d.lower()


def test_unknown_key_returns_none():
    assert describe("no_such_metric") is None
    assert describe("best_no_such_metric") is None


def test_poly_conf_description_matches_all_bins_convention():
    # The conf loss averages over ALL 24 bins since 2026-06-11 (decode gate
    # needs negative gradient on empty bins) — the tooltip must not claim the
    # old masked form.
    d = describe("poly_conf_loss")
    assert "ALL 24 bins" in d
