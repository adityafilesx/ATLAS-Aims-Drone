"""
mode_manager.py — Finite State Machine for Drone Modes
=========================================================
Manages drone states: IDLE, MOVING, EMOTING, EMERGENCY_STOPPED.
Enforces the core separation between motion and emotion:
  • Gesture ≠ Navigation intent ≠ Emotion
  • Motion commands are short, discrete, interruptible
  • Emotions cannot override emergency stop
  • Voice has higher priority than gestures
  • Emergency stop overrides everything
"""

from enum import Enum

from motion.motion_commands import MotionCommand, gesture_to_motion
from motion.motion_controller import MotionController
from emotion.emotion_mapper import gesture_to_emotion, is_emotion_gesture
from emotion.emote_engine import EmoteEngine
from voice.intent_mapper import IntentMapper


class DroneState(Enum):
    """All possible states of the drone FSM."""
    IDLE = "idle"
    MOVING = "moving"
    EMOTING = "emoting"
    EMERGENCY_STOPPED = "emergency_stopped"


class ModeManager:
    """
    Finite State Machine that routes inputs (gestures, voice) to the
    correct subsystem (motion or emotion) while enforcing safety rules.
    """

    # Minimum confidence to accept a gesture prediction
    GESTURE_CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, drone=None, motion_controller=None, emote_engine=None):
        """
        Initialize the FSM. Can be used standalone (for testing) or
        with injected dependencies.

        Args:
            drone:             VirtualDrone instance
            motion_controller: MotionController instance
            emote_engine:      EmoteEngine instance
        """
        self.state = DroneState.IDLE
        self.drone = drone
        self.motion_controller = motion_controller
        self.emote_engine = emote_engine
        self.intent_mapper = IntentMapper()

        # Track last action for UI display
        self.last_action = "none"
        self.last_source = "none"  # "voice" or "gesture"

    # ---------- voice input (highest priority) ----------

    def handle_voice(self, raw_text):
        """
        Process a voice command. Voice has HIGHER priority than gestures.

        Args:
            raw_text: recognized speech text

        Returns:
            True if a valid command was executed
        """
        intent_type, intent_value = self.intent_mapper.parse(raw_text)

        if intent_type == "motion":
            return self._execute_motion(intent_value, source="voice")

        elif intent_type == "emotion":
            return self._execute_emotion(intent_value, source="voice")

        elif intent_type == "reset_emergency":
            return self._reset_emergency(source="voice")

        else:
            print(f"[ModeManager] Voice not recognized: \"{raw_text}\"")
            return False

    # ---------- gesture input ----------

    def handle_gesture(self, gesture_label, confidence):
        """
        Process a gesture prediction. Gestures trigger motion OR emotion,
        NEVER both.

        Args:
            gesture_label: string label from the CNN (e.g. "up", "fist")
            confidence:    float 0.0–1.0 from the CNN softmax output

        Returns:
            True if a valid command was executed
        """
        # Reject low-confidence predictions
        if confidence < self.GESTURE_CONFIDENCE_THRESHOLD:
            return False

        # Check if drone is emergency-stopped (only voice can reset it)
        if self.state == DroneState.EMERGENCY_STOPPED:
            print("[ModeManager] Gesture ignored — drone is emergency-stopped. "
                  "Use voice command to reset.")
            return False

        # Determine if this gesture maps to motion or emotion
        if is_emotion_gesture(gesture_label):
            emotion = gesture_to_emotion(gesture_label)
            if emotion:
                return self._execute_emotion(emotion, source="gesture")
        else:
            motion_cmd = gesture_to_motion(gesture_label)
            if motion_cmd:
                return self._execute_motion(motion_cmd, source="gesture")

        return False

    # ---------- internal execution ----------

    def _execute_motion(self, command, source="unknown"):
        """
        Execute a motion command and update FSM state.

        Args:
            command: MotionCommand enum value
            source:  "voice" or "gesture"
        """
        # Emergency stop always works, from any state
        if command == MotionCommand.EMERGENCY_STOP:
            self.state = DroneState.EMERGENCY_STOPPED
            if self.emote_engine and self.emote_engine.is_emoting:
                self.emote_engine.cancel()
            if self.motion_controller:
                self.motion_controller.execute(command)
            self.last_action = "EMERGENCY STOP"
            self.last_source = source
            return True

        # Can't do motion while emergency-stopped
        if self.state == DroneState.EMERGENCY_STOPPED:
            print("[ModeManager] Motion blocked — emergency stop active.")
            return False

        # Cancel any ongoing emotion before executing motion
        if self.state == DroneState.EMOTING:
            if self.emote_engine:
                self.emote_engine.cancel()

        # Execute the motion
        self.state = DroneState.MOVING
        if self.motion_controller:
            self.motion_controller.execute(command)

        self.last_action = command.value
        self.last_source = source

        # Return to IDLE after discrete command completes
        if self.state != DroneState.EMERGENCY_STOPPED:
            self.state = DroneState.IDLE

        return True

    def _execute_emotion(self, emotion_name, source="unknown"):
        """
        Execute an emotion expression and update FSM state.

        Args:
            emotion_name: string (e.g. "happy")
            source:       "voice" or "gesture"
        """
        # Can't emote while emergency-stopped
        if self.state == DroneState.EMERGENCY_STOPPED:
            print("[ModeManager] Emotion blocked — emergency stop active.")
            return False

        # Play the emotion
        self.state = DroneState.EMOTING
        if self.emote_engine:
            self.emote_engine.play(emotion_name)

        self.last_action = f"emotion:{emotion_name}"
        self.last_source = source

        # Return to IDLE after emotion completes
        if self.state != DroneState.EMERGENCY_STOPPED:
            self.state = DroneState.IDLE

        return True

    def _reset_emergency(self, source="unknown"):
        """Clear emergency stop and return to IDLE."""
        if self.state == DroneState.EMERGENCY_STOPPED:
            if self.motion_controller:
                self.motion_controller.reset_emergency()
            self.state = DroneState.IDLE
            self.last_action = "emergency_reset"
            self.last_source = source
            print("[ModeManager] ✅ Emergency stop cleared. Drone ready.")
            return True
        else:
            print("[ModeManager] No emergency to reset.")
            return False

    # ---------- status ----------

    def get_status(self):
        """Return current FSM status dict."""
        return {
            "state": self.state.value,
            "last_action": self.last_action,
            "last_source": self.last_source,
        }

    def __repr__(self):
        return (f"ModeManager(state={self.state.value}, "
                f"last={self.last_action} via {self.last_source})")
