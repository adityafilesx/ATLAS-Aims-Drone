"""
motion_commands.py — Motion Command Definitions
=================================================
Defines the 12 functional motion commands as an enum.
Maps gesture labels to their corresponding motion commands.
"""

from enum import Enum


class MotionCommand(Enum):
    """
    All functional motion commands the drone can execute.
    Each command maps to a discrete, interruptible drone action.
    """
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    ROTATE = "rotate"
    FLIP = "flip"
    HOVER = "hover"
    SLOW_MODE = "slow_mode"
    FAST_MODE = "fast_mode"
    EMERGENCY_STOP = "emergency_stop"


# ----- Gesture → Motion mapping -----
# Only directional/action gestures map to motion commands.
# Gestures like open_palm, fist, etc. are reserved for emotions.

GESTURE_TO_MOTION = {
    "up":       MotionCommand.MOVE_UP,
    "down":     MotionCommand.MOVE_DOWN,
    "left":     MotionCommand.MOVE_LEFT,
    "right":    MotionCommand.MOVE_RIGHT,
    "forward":  MotionCommand.MOVE_FORWARD,
    "backward": MotionCommand.MOVE_BACKWARD,
    "rotate":   MotionCommand.ROTATE,
    "flip":     MotionCommand.FLIP,
}

# Gestures that map to motion (for quick lookup)
MOTION_GESTURES = set(GESTURE_TO_MOTION.keys())


def gesture_to_motion(gesture_label):
    """
    Convert a gesture label to a MotionCommand, or None if not a motion gesture.

    Args:
        gesture_label: string gesture name (e.g. "up", "rotate")

    Returns:
        MotionCommand or None
    """
    return GESTURE_TO_MOTION.get(gesture_label, None)
