"""
emote_engine.py — Emotion Expression Engine
==============================================
Plays emotion patterns on the VirtualDrone using small,
non-translational micro-motions. All emotions are:
  • Time-bounded
  • Cancelable
  • Cannot override emergency stop
  • No spatial translation
"""

import time

from emotion.emotion_patterns import get_pattern, VALID_EMOTIONS


class EmoteEngine:
    """
    Executes emotion patterns on a VirtualDrone.
    Emotions are short animation sequences composed of
    wiggles, nods, spins, and tilts — no translation across space.
    """

    def __init__(self, drone):
        """
        Args:
            drone: a VirtualDrone instance
        """
        self.drone = drone
        self.is_emoting = False      # True while an emotion is playing
        self.current_emotion = None  # name of current emotion
        self._cancelled = False      # flag to cancel mid-emotion

    def play(self, emotion_name):
        """
        Play an emotion pattern on the drone.

        Args:
            emotion_name: string (e.g. "happy", "curious")

        Returns:
            True if played successfully, False if blocked or invalid
        """
        # Block if drone is emergency-stopped
        if self.drone.is_emergency_stopped:
            print("[EmoteEngine] Blocked — drone is emergency-stopped.")
            return False

        # Validate emotion name
        pattern = get_pattern(emotion_name)
        if pattern is None:
            print(f"[EmoteEngine] Unknown emotion: '{emotion_name}'")
            print(f"  Valid emotions: {sorted(VALID_EMOTIONS)}")
            return False

        # Cancel any current emotion before starting a new one
        if self.is_emoting:
            self.cancel()

        self.is_emoting = True
        self.current_emotion = emotion_name
        self._cancelled = False

        print(f"[EmoteEngine] 🎭 Playing: {emotion_name} "
              f"— {pattern.description}")

        # Record position before emotion (to ensure no drift)
        original_x = self.drone.x
        original_y = self.drone.y
        original_alt = self.drone.altitude

        # Execute each step in the pattern
        step_delay = pattern.duration / max(len(pattern.steps), 1)

        for step_name, kwargs in pattern.steps:
            # Check for cancellation or emergency stop
            if self._cancelled or self.drone.is_emergency_stopped:
                print(f"[EmoteEngine] ⚠️  {emotion_name} interrupted!")
                break

            # Call the drone's micro-motion method
            method = getattr(self.drone, step_name, None)
            if method is not None:
                method(**kwargs)
            else:
                print(f"[EmoteEngine] Warning: drone has no method "
                      f"'{step_name}'")

            # Small delay between steps (for pacing)
            time.sleep(step_delay)

        # Restore exact position (safety — no drift from emotions)
        self.drone.x = original_x
        self.drone.y = original_y
        self.drone.altitude = original_alt

        self.is_emoting = False
        self.current_emotion = None
        print(f"[EmoteEngine] ✅ {emotion_name} complete.")
        return True

    def cancel(self):
        """Cancel the currently playing emotion."""
        if self.is_emoting:
            print(f"[EmoteEngine] Cancelling: {self.current_emotion}")
            self._cancelled = True

    def get_status(self):
        """Return current emote engine status."""
        return {
            "is_emoting": self.is_emoting,
            "current_emotion": self.current_emotion,
        }
