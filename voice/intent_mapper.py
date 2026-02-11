"""
intent_mapper.py — Voice Command Intent Parser
=================================================
Parses raw voice text for "ATLAS" prefix commands and maps them
to either a MotionCommand or an emotion name.

Voice commands have HIGHER PRIORITY than gestures.

Supported formats:
    "ATLAS go up"          → motion: MOVE_UP
    "ATLAS move forward"   → motion: MOVE_FORWARD
    "ATLAS rotate"         → motion: ROTATE
    "ATLAS stop"           → motion: EMERGENCY_STOP
    "ATLAS hover"          → motion: HOVER
    "ATLAS be happy"       → emotion: happy
    "ATLAS relax"          → emotion: calm
    "ATLAS stay calm"      → emotion: calm
    "ATLAS emergency stop" → motion: EMERGENCY_STOP
"""

from motion.motion_commands import MotionCommand
from emotion.emotion_mapper import voice_to_emotion


class IntentMapper:
    """
    Parses voice text and returns structured intents.
    Returns (intent_type, intent_value):
        - ("motion", MotionCommand)
        - ("emotion", emotion_name)
        - ("reset_emergency", None)
        - ("unknown", raw_text)
    """

    # Keywords → MotionCommand
    MOTION_KEYWORDS = {
        "go up":         MotionCommand.MOVE_UP,
        "move up":       MotionCommand.MOVE_UP,
        "up":            MotionCommand.MOVE_UP,
        "go down":       MotionCommand.MOVE_DOWN,
        "move down":     MotionCommand.MOVE_DOWN,
        "down":          MotionCommand.MOVE_DOWN,
        "go left":       MotionCommand.MOVE_LEFT,
        "move left":     MotionCommand.MOVE_LEFT,
        "left":          MotionCommand.MOVE_LEFT,
        "go right":      MotionCommand.MOVE_RIGHT,
        "move right":    MotionCommand.MOVE_RIGHT,
        "right":         MotionCommand.MOVE_RIGHT,
        "go forward":    MotionCommand.MOVE_FORWARD,
        "move forward":  MotionCommand.MOVE_FORWARD,
        "forward":       MotionCommand.MOVE_FORWARD,
        "go backward":   MotionCommand.MOVE_BACKWARD,
        "move backward": MotionCommand.MOVE_BACKWARD,
        "go back":       MotionCommand.MOVE_BACKWARD,
        "backward":      MotionCommand.MOVE_BACKWARD,
        "rotate":        MotionCommand.ROTATE,
        "spin":          MotionCommand.ROTATE,
        "flip":          MotionCommand.FLIP,
        "hover":         MotionCommand.HOVER,
        "stay":          MotionCommand.HOVER,
        "stop":          MotionCommand.EMERGENCY_STOP,
        "emergency stop": MotionCommand.EMERGENCY_STOP,
        "emergency":     MotionCommand.EMERGENCY_STOP,
        "halt":          MotionCommand.EMERGENCY_STOP,
        "slow":          MotionCommand.SLOW_MODE,
        "slow mode":     MotionCommand.SLOW_MODE,
        "fast":          MotionCommand.FAST_MODE,
        "fast mode":     MotionCommand.FAST_MODE,
    }

    # Keywords → emotion name
    EMOTION_KEYWORDS = {
        "be happy":     "happy",
        "happy":        "happy",
        "be excited":   "excited",
        "excited":      "excited",
        "be curious":   "curious",
        "curious":      "curious",
        "be bored":     "bored",
        "bored":        "bored",
        "be sad":       "sad",
        "sad":          "sad",
        "stay calm":    "calm",
        "be calm":      "calm",
        "calm":         "calm",
        "relax":        "calm",
        "chill":        "calm",
        "be alert":     "alert",
        "alert":        "alert",
        "attention":    "alert",
        "be tired":     "tired",
        "tired":        "tired",
        "be confident": "confident",
        "confident":    "confident",
        "be confused":  "confused",
        "confused":     "confused",
    }

    # Special commands
    SPECIAL_KEYWORDS = {
        "reset": "reset_emergency",
        "resume": "reset_emergency",
        "clear emergency": "reset_emergency",
    }

    def parse(self, raw_text):
        """
        Parse raw voice text and return an intent.

        Args:
            raw_text: full recognized text string

        Returns:
            (intent_type, intent_value) tuple:
                ("motion", MotionCommand)
                ("emotion", str)
                ("reset_emergency", None)
                ("unknown", str)
        """
        if not raw_text:
            return ("unknown", "")

        text = raw_text.lower().strip()

        # Remove the "atlas" prefix if present
        if text.startswith("atlas"):
            text = text[len("atlas"):].strip()

        # Empty after stripping prefix
        if not text:
            return ("unknown", raw_text)

        # Check special commands first
        for keyword, intent in self.SPECIAL_KEYWORDS.items():
            if keyword in text:
                return (intent, None)

        # Check motion keywords (longest match first)
        for keyword in sorted(self.MOTION_KEYWORDS.keys(),
                              key=len, reverse=True):
            if keyword in text:
                return ("motion", self.MOTION_KEYWORDS[keyword])

        # Check emotion keywords (longest match first)
        for keyword in sorted(self.EMOTION_KEYWORDS.keys(),
                              key=len, reverse=True):
            if keyword in text:
                return ("emotion", self.EMOTION_KEYWORDS[keyword])

        return ("unknown", raw_text)
