"""
virtual_drone.py — 2D Virtual Drone Simulation
================================================
Simulates a drone in a 2D space with position, altitude, heading, and speed.
All movement is discrete, bounded, and interruptible.
No hardware assumptions — purely virtual.
"""

import time


class VirtualDrone:
    """
    A virtual 2D drone with position (x, y), altitude, heading, and speed.
    Supports functional motion commands and small emotion-driven motions.
    """

    # ---------- boundaries ----------
    MIN_POS = -500      # minimum x or y coordinate
    MAX_POS = 500       # maximum x or y coordinate
    MIN_ALT = 0         # ground level
    MAX_ALT = 200       # maximum altitude
    STEP_NORMAL = 10    # default movement step size
    STEP_SLOW = 4       # slow-mode step size
    STEP_FAST = 20      # fast-mode step size
    ROTATION_STEP = 45  # degrees per rotation command

    def __init__(self):
        """Initialize drone at the center, on the ground, facing north."""
        self.x = 0.0            # horizontal position
        self.y = 0.0            # depth position (forward/backward)
        self.altitude = 0.0     # vertical height
        self.heading = 0.0      # degrees, 0 = north
        self.speed_mode = "normal"  # "slow", "normal", or "fast"
        self.is_emergency_stopped = False
        self.is_flipping = False
        self.state_log = []     # recent action history

    # ---------- helpers ----------

    def _step(self):
        """Return the current step size based on speed mode."""
        if self.speed_mode == "slow":
            return self.STEP_SLOW
        elif self.speed_mode == "fast":
            return self.STEP_FAST
        return self.STEP_NORMAL

    def _clamp(self, value, low, high):
        """Clamp a value between low and high."""
        return max(low, min(high, value))

    def _log(self, action):
        """Record an action to the state log."""
        entry = {"action": action, "time": time.time(),
                 "pos": (self.x, self.y, self.altitude),
                 "heading": self.heading}
        self.state_log.append(entry)
        # Keep log manageable
        if len(self.state_log) > 100:
            self.state_log = self.state_log[-50:]

    def _check_emergency(self):
        """Return True if drone is emergency-stopped (blocks all motion)."""
        return self.is_emergency_stopped

    # ---------- functional motion commands ----------

    def move_up(self):
        """Increase altitude by one step."""
        if self._check_emergency():
            return
        self.altitude = self._clamp(self.altitude + self._step(),
                                    self.MIN_ALT, self.MAX_ALT)
        self._log("move_up")

    def move_down(self):
        """Decrease altitude by one step."""
        if self._check_emergency():
            return
        self.altitude = self._clamp(self.altitude - self._step(),
                                    self.MIN_ALT, self.MAX_ALT)
        self._log("move_down")

    def move_left(self):
        """Move left (negative x) by one step."""
        if self._check_emergency():
            return
        self.x = self._clamp(self.x - self._step(),
                             self.MIN_POS, self.MAX_POS)
        self._log("move_left")

    def move_right(self):
        """Move right (positive x) by one step."""
        if self._check_emergency():
            return
        self.x = self._clamp(self.x + self._step(),
                             self.MIN_POS, self.MAX_POS)
        self._log("move_right")

    def move_forward(self):
        """Move forward (positive y) by one step."""
        if self._check_emergency():
            return
        self.y = self._clamp(self.y + self._step(),
                             self.MIN_POS, self.MAX_POS)
        self._log("move_forward")

    def move_backward(self):
        """Move backward (negative y) by one step."""
        if self._check_emergency():
            return
        self.y = self._clamp(self.y - self._step(),
                             self.MIN_POS, self.MAX_POS)
        self._log("move_backward")

    def rotate(self, degrees=None):
        """Rotate the drone by ROTATION_STEP degrees clockwise."""
        if self._check_emergency():
            return
        step = degrees if degrees is not None else self.ROTATION_STEP
        self.heading = (self.heading + step) % 360
        self._log(f"rotate({step})")

    def flip(self):
        """Perform a flip animation (no spatial change)."""
        if self._check_emergency():
            return
        self.is_flipping = True
        self._log("flip")
        # Flip is instantaneous in this simulation
        self.is_flipping = False

    # ---------- state commands ----------

    def hover(self):
        """Hold current position (no-op, logs the intent)."""
        if self._check_emergency():
            return
        self._log("hover")

    def slow_mode(self):
        """Switch to slow movement speed."""
        if self._check_emergency():
            return
        self.speed_mode = "slow"
        self._log("slow_mode")

    def fast_mode(self):
        """Switch to fast movement speed."""
        if self._check_emergency():
            return
        self.speed_mode = "fast"
        self._log("fast_mode")

    def emergency_stop(self):
        """Immediately stop all motion. Overrides everything."""
        self.is_emergency_stopped = True
        self.speed_mode = "normal"
        self._log("EMERGENCY_STOP")

    def reset_emergency(self):
        """Clear emergency stop so the drone can move again."""
        self.is_emergency_stopped = False
        self._log("emergency_reset")

    # ---------- small emotion motions (non-translational) ----------

    def wiggle(self, amplitude=2, count=3):
        """Small left-right oscillation in place (for emotions)."""
        if self._check_emergency():
            return
        original_x = self.x
        for _ in range(count):
            self.x = original_x + amplitude
            self.x = original_x - amplitude
        self.x = original_x  # return to exact position
        self._log(f"wiggle(amp={amplitude}, n={count})")

    def nod(self, amplitude=3, count=2):
        """Small up-down oscillation in place (for emotions)."""
        if self._check_emergency():
            return
        original_alt = self.altitude
        for _ in range(count):
            self.altitude = self._clamp(original_alt + amplitude,
                                        self.MIN_ALT, self.MAX_ALT)
            self.altitude = self._clamp(original_alt - amplitude,
                                        self.MIN_ALT, self.MAX_ALT)
        self.altitude = original_alt
        self._log(f"nod(amp={amplitude}, n={count})")

    def spin(self, degrees=360):
        """Full or partial spin in place (for emotions). No translation."""
        if self._check_emergency():
            return
        self.heading = (self.heading + degrees) % 360
        self._log(f"spin({degrees})")

    def tilt(self, angle=15):
        """Simulate a tilt effect (heading micro-adjustment, then return)."""
        if self._check_emergency():
            return
        original = self.heading
        self.heading = (self.heading + angle) % 360
        self.heading = original  # snap back
        self._log(f"tilt({angle})")

    # ---------- status ----------

    def get_status(self):
        """Return a dict summarizing current drone state."""
        return {
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "heading": self.heading,
            "speed_mode": self.speed_mode,
            "emergency_stopped": self.is_emergency_stopped,
        }

    def __repr__(self):
        return (f"VirtualDrone(x={self.x}, y={self.y}, alt={self.altitude}, "
                f"hdg={self.heading}°, speed={self.speed_mode}, "
                f"e-stop={self.is_emergency_stopped})")
