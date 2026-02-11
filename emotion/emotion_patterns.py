"""
emotion_patterns.py — Emotion Motion Patterns
================================================
Defines 10 emotion patterns, each expressed as a sequence of
small, non-translational drone motions (wiggles, nods, spins, tilts).
All patterns are time-bounded and do not move the drone across space.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EmotionPattern:
    """
    A single emotion expressed as a sequence of micro-motions.

    Attributes:
        name:        emotion name (e.g. "happy")
        description: human-readable description of the expression
        steps:       list of (method_name, kwargs) tuples to call on VirtualDrone
        duration:    total estimated duration in seconds (for UI pacing)
        cancelable:  whether this emotion can be interrupted
    """
    name: str
    description: str
    steps: List[Tuple[str, dict]] = field(default_factory=list)
    duration: float = 1.0
    cancelable: bool = True


# ---------- 10 emotion patterns ----------

EMOTION_PATTERNS = {
    "happy": EmotionPattern(
        name="happy",
        description="Joyful bounce — quick nods and a wiggle",
        steps=[
            ("nod", {"amplitude": 4, "count": 3}),
            ("wiggle", {"amplitude": 3, "count": 2}),
        ],
        duration=1.5,
    ),

    "excited": EmotionPattern(
        name="excited",
        description="Rapid wiggle and a celebratory spin",
        steps=[
            ("wiggle", {"amplitude": 5, "count": 4}),
            ("spin", {"degrees": 360}),
        ],
        duration=2.0,
    ),

    "curious": EmotionPattern(
        name="curious",
        description="Slow tilt side to side, like peeking",
        steps=[
            ("tilt", {"angle": 20}),
            ("tilt", {"angle": -20}),
            ("nod", {"amplitude": 2, "count": 1}),
        ],
        duration=1.5,
    ),

    "bored": EmotionPattern(
        name="bored",
        description="Lazy, slow droop and sigh-nod",
        steps=[
            ("nod", {"amplitude": 1, "count": 1}),
        ],
        duration=2.0,
    ),

    "sad": EmotionPattern(
        name="sad",
        description="Slow downward nod, like drooping",
        steps=[
            ("nod", {"amplitude": 3, "count": 1}),
            ("tilt", {"angle": -10}),
        ],
        duration=2.5,
    ),

    "calm": EmotionPattern(
        name="calm",
        description="Gentle, slow nod — serene hover",
        steps=[
            ("nod", {"amplitude": 1, "count": 2}),
        ],
        duration=2.0,
    ),

    "alert": EmotionPattern(
        name="alert",
        description="Quick snap-look left and right",
        steps=[
            ("tilt", {"angle": 25}),
            ("tilt", {"angle": -25}),
            ("nod", {"amplitude": 2, "count": 1}),
        ],
        duration=1.0,
    ),

    "tired": EmotionPattern(
        name="tired",
        description="Heavy, slow droop with minimal movement",
        steps=[
            ("nod", {"amplitude": 2, "count": 1}),
        ],
        duration=3.0,
    ),

    "confident": EmotionPattern(
        name="confident",
        description="Proud spin and firm nod",
        steps=[
            ("spin", {"degrees": 180}),
            ("nod", {"amplitude": 3, "count": 2}),
        ],
        duration=1.5,
    ),

    "confused": EmotionPattern(
        name="confused",
        description="Rapid tilts and a small wiggle — 'huh?'",
        steps=[
            ("tilt", {"angle": 15}),
            ("tilt", {"angle": -15}),
            ("wiggle", {"amplitude": 2, "count": 2}),
        ],
        duration=1.5,
    ),
}


# Quick lookup: set of all valid emotion names
VALID_EMOTIONS = set(EMOTION_PATTERNS.keys())


def get_pattern(emotion_name):
    """
    Return the EmotionPattern for a given emotion name, or None.
    """
    return EMOTION_PATTERNS.get(emotion_name, None)
