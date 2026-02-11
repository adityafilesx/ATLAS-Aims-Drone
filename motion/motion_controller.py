"""
motion_controller.py — Motion Execution Controller
=====================================================
Takes a MotionCommand and dispatches it to the VirtualDrone.
Checks emergency stop before executing any command.
All commands are short, discrete, and interruptible.
"""

from motion.motion_commands import MotionCommand


class MotionController:
    """
    Controller that translates MotionCommands into VirtualDrone actions.
    """

    def __init__(self, drone):
        """
        Args:
            drone: a VirtualDrone instance
        """
        self.drone = drone

        # Map each command to its drone method
        self._dispatch = {
            MotionCommand.MOVE_UP:       self.drone.move_up,
            MotionCommand.MOVE_DOWN:     self.drone.move_down,
            MotionCommand.MOVE_LEFT:     self.drone.move_left,
            MotionCommand.MOVE_RIGHT:    self.drone.move_right,
            MotionCommand.MOVE_FORWARD:  self.drone.move_forward,
            MotionCommand.MOVE_BACKWARD: self.drone.move_backward,
            MotionCommand.ROTATE:        self.drone.rotate,
            MotionCommand.FLIP:          self.drone.flip,
            MotionCommand.HOVER:         self.drone.hover,
            MotionCommand.SLOW_MODE:     self.drone.slow_mode,
            MotionCommand.FAST_MODE:     self.drone.fast_mode,
            MotionCommand.EMERGENCY_STOP: self.drone.emergency_stop,
        }

    def execute(self, command):
        """
        Execute a single motion command on the drone.

        Args:
            command: a MotionCommand enum value

        Returns:
            True if executed, False if blocked (emergency stop or invalid)
        """
        # Emergency stop always goes through, even if already stopped
        if command == MotionCommand.EMERGENCY_STOP:
            self.drone.emergency_stop()
            print("[MotionController] ⛔ EMERGENCY STOP activated!")
            return True

        # Block all other commands if drone is emergency-stopped
        if self.drone.is_emergency_stopped:
            print("[MotionController] Blocked — drone is emergency-stopped. "
                  "Reset emergency first.")
            return False

        # Look up and call the drone method
        action = self._dispatch.get(command)
        if action is None:
            print(f"[MotionController] Unknown command: {command}")
            return False

        action()
        print(f"[MotionController] Executed: {command.value}")
        return True

    def reset_emergency(self):
        """Clear the emergency stop so the drone can fly again."""
        self.drone.reset_emergency()
        print("[MotionController] Emergency stop cleared.")
