"""
gesture_labels.py — Gesture Class Definitions
===============================================
Defines the 13 gesture classes used by the CNN classifier.
Provides label ↔ index mapping utilities.
"""

# ----- Ordered list of all gesture classes -----
# The CNN output layer has one neuron per class, in this order.
GESTURE_LABELS = [
    "up",            # 0  — hand pointing up
    "down",          # 1  — hand pointing down
    "left",          # 2  — hand pointing left
    "right",         # 3  — hand pointing right
    "forward",       # 4  — hand pushing forward
    "backward",      # 5  — hand pulling back
    "rotate",        # 6  — circular hand motion
    "flip",          # 7  — quick flick gesture
    "open_palm",     # 8  — all fingers spread open
    "fist",          # 9  — closed fist
    "thumb_up",      # 10 — thumbs-up sign
    "two_fingers",   # 11 — peace / two-finger sign
    "five_fingers",  # 12 — all five fingers raised
]

# Number of classes (used by the CNN output layer)
NUM_CLASSES = len(GESTURE_LABELS)

# ----- Convenience mappings -----

# label string → integer index
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(GESTURE_LABELS)}

# integer index → label string
INDEX_TO_LABEL = {idx: label for idx, label in enumerate(GESTURE_LABELS)}


def get_label(index):
    """Return the gesture label for a given class index, or 'unknown'."""
    return INDEX_TO_LABEL.get(index, "unknown")


def get_index(label):
    """Return the class index for a given gesture label, or -1."""
    return LABEL_TO_INDEX.get(label, -1)
