"""Pure-numpy tests for the confusion-matrix matching + accumulation.

Imports only the numpy helpers from utils.confusion_matrix (no TensorFlow, no
model, no dataset). Boxes are yxyx; IoU is scale-free so unit-square coords are
used throughout.
"""

import numpy as np

from utils.confusion_matrix import ConfusionMatrix, iou_matrix, match_detections


def _box(y1, x1, y2, x2):
    return [y1, x1, y2, x2]


def test_iou_matrix_identity_and_disjoint():
    a = np.array([_box(0, 0, 1, 1)])
    b = np.array([_box(0, 0, 1, 1), _box(2, 2, 3, 3)])
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == 1.0
    assert iou[0, 1] == 0.0


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.array([_box(0, 0, 1, 1)])).shape == (0, 1)
    assert iou_matrix(np.array([_box(0, 0, 1, 1)]), np.zeros((0, 4))).shape == (1, 0)


def test_match_greedy_highest_score_first():
    # Two predictions overlap the same GT; the higher-score one must win it.
    gt = np.array([_box(0, 0, 1, 1)])
    preds = np.array([_box(0, 0, 1, 1), _box(0, 0, 0.9, 0.9)])
    scores = np.array([0.4, 0.9])   # pred index 1 is higher-scored
    m = match_detections(preds, scores, gt, iou_thresh=0.5)
    assert m[1] == 0        # higher score claims the GT
    assert m[0] == -1       # loser is left unmatched


def test_match_below_iou_threshold_is_unmatched():
    gt = np.array([_box(0, 0, 1, 1)])
    preds = np.array([_box(0, 0, 0.3, 0.3)])   # IoU 0.09 with the GT
    m = match_detections(preds, np.array([0.9]), gt, iou_thresh=0.5)
    assert m[0] == -1


def test_perfect_predictions_give_clean_diagonal():
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1)
    boxes = np.array([_box(0, 0, 1, 1), _box(2, 2, 3, 3)])
    classes = np.array([0, 2])
    cm.update(pred_boxes=boxes, pred_classes=classes, pred_scores=np.array([0.9, 0.8]),
              gt_boxes=boxes, gt_classes=classes)
    mat = cm.as_array()
    assert mat[0, 0] == 1
    assert mat[2, 2] == 1
    # Nothing anywhere else (no background row/col, no off-diagonal).
    assert mat.sum() == 2


def test_misclassification_is_off_diagonal():
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1)
    box = np.array([_box(0, 0, 1, 1)])
    cm.update(pred_boxes=box, pred_classes=np.array([1]), pred_scores=np.array([0.9]),
              gt_boxes=box, gt_classes=np.array([0]))
    mat = cm.as_array()
    assert mat[1, 0] == 1       # predicted class 1, actual class 0
    assert mat.sum() == 1


def test_false_positive_lands_in_background_column():
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1)
    # A prediction with no overlapping GT -> background GT column.
    cm.update(pred_boxes=np.array([_box(0, 0, 1, 1)]), pred_classes=np.array([2]),
              pred_scores=np.array([0.9]),
              gt_boxes=np.zeros((0, 4)), gt_classes=np.zeros((0,), dtype=int))
    mat = cm.as_array()
    assert mat[2, cm.bg] == 1   # predicted 2, background truth
    assert mat.sum() == 1


def test_false_negative_lands_in_background_row():
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1)
    # A GT with no prediction -> background prediction row.
    cm.update(pred_boxes=np.zeros((0, 4)), pred_classes=np.zeros((0,), dtype=int),
              pred_scores=np.zeros((0,)),
              gt_boxes=np.array([_box(0, 0, 1, 1)]), gt_classes=np.array([1]))
    mat = cm.as_array()
    assert mat[cm.bg, 1] == 1   # background prediction, truth class 1
    assert mat.sum() == 1


def test_conf_threshold_drops_low_score_prediction():
    box = np.array([_box(0, 0, 1, 1)])
    # Below threshold: the prediction is dropped -> the GT becomes a false negative.
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.5)
    cm.update(pred_boxes=box, pred_classes=np.array([0]), pred_scores=np.array([0.4]),
              gt_boxes=box, gt_classes=np.array([0]))
    mat = cm.as_array()
    assert mat[0, 0] == 0
    assert mat[cm.bg, 0] == 1
    assert mat.sum() == 1

    # Above threshold: the same prediction now matches -> diagonal.
    cm2 = ConfusionMatrix(num_classes=3, conf_thresh=0.5)
    cm2.update(pred_boxes=box, pred_classes=np.array([0]), pred_scores=np.array([0.6]),
               gt_boxes=box, gt_classes=np.array([0]))
    assert cm2.as_array()[0, 0] == 1


def test_crowd_gt_excluded_when_ignore_iscrowds():
    box = np.array([_box(0, 0, 1, 1)])
    # class 6 is in the default crowd-label set; a crowd GT of that class is skipped.
    cm = ConfusionMatrix(num_classes=39, conf_thresh=0.1, ignore_iscrowds=True)
    cm.update(pred_boxes=np.zeros((0, 4)), pred_classes=np.zeros((0,), dtype=int),
              pred_scores=np.zeros((0,)),
              gt_boxes=box, gt_classes=np.array([6]),
              gt_is_crowd=np.array([True]))
    # Skipped entirely: not counted as a missed detection anywhere.
    assert cm.as_array().sum() == 0

    # With ignore_iscrowds off, the same crowd GT IS scored (a false negative here).
    cm2 = ConfusionMatrix(num_classes=39, conf_thresh=0.1, ignore_iscrowds=False)
    cm2.update(pred_boxes=np.zeros((0, 4)), pred_classes=np.zeros((0,), dtype=int),
               pred_scores=np.zeros((0,)),
               gt_boxes=box, gt_classes=np.array([6]),
               gt_is_crowd=np.array([True]))
    assert cm2.as_array()[cm2.bg, 6] == 1


def test_dontcare_absorbs_overlapping_prediction():
    box = np.array([_box(0, 0, 1, 1)])
    # A prediction overlapping a dontcare GT is neither TP nor FP, and the
    # dontcare GT is not a false negative -> the matrix stays empty.
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1, ignore_dontcare=True)
    cm.update(pred_boxes=box, pred_classes=np.array([1]), pred_scores=np.array([0.9]),
              gt_boxes=box, gt_classes=np.array([1]),
              gt_is_dontcare=np.array([True]))
    assert cm.as_array().sum() == 0

    # A non-overlapping prediction is NOT absorbed -> still a false positive.
    cm2 = ConfusionMatrix(num_classes=3, conf_thresh=0.1, ignore_dontcare=True)
    cm2.update(pred_boxes=np.array([_box(5, 5, 6, 6)]), pred_classes=np.array([1]),
               pred_scores=np.array([0.9]),
               gt_boxes=box, gt_classes=np.array([1]),
               gt_is_dontcare=np.array([True]))
    assert cm2.as_array()[1, cm2.bg] == 1
    assert cm2.as_array().sum() == 1


def test_dontcare_scored_when_ignore_dontcare_off():
    box = np.array([_box(0, 0, 1, 1)])
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1, ignore_dontcare=False)
    cm.update(pred_boxes=box, pred_classes=np.array([1]), pred_scores=np.array([0.9]),
              gt_boxes=box, gt_classes=np.array([1]),
              gt_is_dontcare=np.array([True]))
    # Now the dontcare GT is an ordinary target -> a clean diagonal match.
    assert cm.as_array()[1, 1] == 1


def test_matrix_shape_and_top_confusions():
    cm = ConfusionMatrix(num_classes=3, conf_thresh=0.1)
    assert cm.as_array().shape == (4, 4)
    box = np.array([_box(0, 0, 1, 1)])
    for _ in range(3):
        cm.update(pred_boxes=box, pred_classes=np.array([1]), pred_scores=np.array([0.9]),
                  gt_boxes=box, gt_classes=np.array([0]))
    top = cm.top_confusions(k=5)
    assert top[0][2] == 3            # three class-1-for-class-0 confusions
    assert cm.format_table()        # renders without error
