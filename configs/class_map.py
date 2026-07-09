"""Class name mapping and dataset-specific class remaps.

DETECTION_CLASSES: 39-class taxonomy used in training (placeholder names).
SERVINGBOT_CLASS_REMAP: Maps ServingBot's single foreground class (id=0) to
    class 35 in the main 39-class taxonomy.
"""

_NUM_CLASSES = 39

# Placeholder names — replace with real class names when confirmed.
DETECTION_CLASSES: dict[int, str] = {i: f'label_{i}' for i in range(_NUM_CLASSES)}

# index == category_id is assumed throughout eval (coco_metrics) and logging;
# a size drift mislabels every per-category metric.
assert len(DETECTION_CLASSES) == _NUM_CLASSES, (
    f"DETECTION_CLASSES must have {_NUM_CLASSES} entries, got {len(DETECTION_CLASSES)}"
)

# ServingBot's single foreground class (id=0) maps to class 35 in the taxonomy.
SERVINGBOT_CLASS_REMAP: dict[int, int] = {0: 35}

# The remap target must be a valid class index in the taxonomy.
assert all(0 <= v < _NUM_CLASSES for v in SERVINGBOT_CLASS_REMAP.values()), (
    "SERVINGBOT_CLASS_REMAP targets a class id outside the 39-class taxonomy"
)
