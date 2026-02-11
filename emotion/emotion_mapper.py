"""
emotion_mapper.py — Gesture and Voice to Emotion Mapping
==========================================================
Maps specific gestures and voice intents to emotion names.
Only non-directional gestures (open_palm, fist, thumb_up, etc.)
trigger emotions. Voice commands for emotions take priority.
"""

from emotion.emotion_patterns import VALID_EMOTIONS


# ----- Gesture → Emotion mapping -----
# Only expressive (non-directional) gestures map to emotions.
# Directional gestures (up, down, left, etc.) go to motion — never here.

GESTURE_TO_EMOTION = {
    "open_palm":    "happy",      # open palm → joy
    "fist":         "confident",  # fist → confidence / determination
    "thumb_up":     "excited",    # thumbs up → excitement
    "two_fingers":  "curious",    # peace sign → curiosity
    "five_fingers": "alert",      # all five up → alert / attention
}

# Set of gestures that map to emotions (for quick lookup)
EMOTION_GESTURES = set(GESTURE_TO_EMOTION.keys())


# ----- Voice → Emotion mapping -----
# These map voice intent keywords to emotion names.

VOICE_TO_EMOTION = {
    "happy":     "happy",
    "excited":   "excited",
    "curious":   "curious",
    "bored":     "bored",
    "sad":       "sad",
    "calm":      "calm",
    "relax":     "calm",       # alias
    "alert":     "alert",
    "tired":     "tired",
    "confident": "confident",
    "confused":  "confused",
}


def gesture_to_emotion(gesture_label):
    """
    Map a gesture label to an emotion name, or None if not an emotion gesture.

    Args:
        gesture_label: string (e.g. "open_palm")

    Returns:
        emotion name string or None
    """
    return GESTURE_TO_EMOTION.get(gesture_label, None)


def voice_to_emotion(keyword):
    """
    Map a voice keyword to an emotion name, or None.

    Args:
        keyword: extracted keyword from voice intent (e.g. "happy", "relax")

    Returns:
        emotion name string or None
    """
    return VOICE_TO_EMOTION.get(keyword.lower(), None)


def is_emotion_gesture(gesture_label):
    """Return True if this gesture should trigger an emotion."""
    return gesture_label in EMOTION_GESTURES
