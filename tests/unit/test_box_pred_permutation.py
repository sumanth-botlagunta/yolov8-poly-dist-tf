"""Pins the legacy box-head DFL-group permutation used by checkpoint migration.

The legacy codebase stores box distances y-first [top, left, bottom, right]
(paired with (y,x) anchors); this codebase's detection_generator decodes x-first
[left, top, right, bottom] with (x,y) anchors. Migration reorders the 4 DFL groups
by [1,0,3,2] so migrated boxes decode correctly. A regression here reintroduces the
zero-F1 / flooded-NMS bug on migrated legacy checkpoints.
"""

import numpy as np

from tools.checkpoint_migration import (
    _permute_legacy_box_pred,
    _LEGACY_BOX_ORDER_PERM,
)

REG_MAX = 16
CIN = 136  # cv2feat hidden -> box_pred input channels


def _tagged_channels():
    """64 values encoding group*100 + bin so each group/bin is identifiable."""
    return np.array([g * 100 + b for g in range(4) for b in range(REG_MAX)],
                    dtype=np.float32)


def test_group_order_matches_perm():
    bias = _tagged_channels()
    out = _permute_legacy_box_pred(bias)
    lead_group_id = (out.reshape(4, REG_MAX)[:, 0] // 100).astype(int)
    assert list(lead_group_id) == list(_LEGACY_BOX_ORDER_PERM) == [1, 0, 3, 2]


def test_within_group_bins_untouched():
    bias = _tagged_channels()
    out = _permute_legacy_box_pred(bias).reshape(4, REG_MAX)
    for gi in range(4):
        assert list((out[gi] % 100).astype(int)) == list(range(REG_MAX))


def test_involution_bias_and_kernel():
    bias = _tagged_channels()
    kernel = np.tile(bias, (1, 1, CIN, 1)).reshape(1, 1, CIN, 4 * REG_MAX)
    # Applying the permutation twice must return the original (self-inverse).
    np.testing.assert_array_equal(
        _permute_legacy_box_pred(_permute_legacy_box_pred(bias)), bias)
    np.testing.assert_array_equal(
        _permute_legacy_box_pred(_permute_legacy_box_pred(kernel)), kernel)


def test_element_map_new_from_legacy():
    # new[0]=left<-legacy[1], new[1]=top<-legacy[0], new[2]=right<-legacy[3],
    # new[3]=bottom<-legacy[2].
    g = _tagged_channels().reshape(4, REG_MAX)
    pg = _permute_legacy_box_pred(_tagged_channels()).reshape(4, REG_MAX)
    np.testing.assert_array_equal(pg[0], g[1])
    np.testing.assert_array_equal(pg[1], g[0])
    np.testing.assert_array_equal(pg[2], g[3])
    np.testing.assert_array_equal(pg[3], g[2])


def test_kernel_shape_preserved():
    kernel = np.zeros((1, 1, CIN, 4 * REG_MAX), dtype=np.float32)
    assert _permute_legacy_box_pred(kernel).shape == kernel.shape
    bias = np.zeros((4 * REG_MAX,), dtype=np.float32)
    assert _permute_legacy_box_pred(bias).shape == bias.shape
