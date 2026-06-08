"""Class name mapping and dataset-specific class remaps.

DETECTION_CLASSES: 39-class taxonomy used in training (placeholder names).
SERVINGBOT_CLASS_REMAP: Maps ServingBot's single foreground class (id=0) to
    class 35 in the main 39-class taxonomy.
"""

# Placeholder names — replace with real class names when confirmed.
DETECTION_CLASSES: dict[int, str] = {i: f'label_{i}' for i in range(39)}

# ServingBot dataset has a single foreground class (id=0).
# It corresponds to class 35 ("label_35") in the main 39-class taxonomy.
# Hardcoded here because this mapping does not change with dataset versions.
SERVINGBOT_CLASS_REMAP: dict[int, int] = {0: 35}
