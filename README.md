# ATLAS — Autonomous Terrain-Learning and Analysis System (Mode-1)

**Status:** Mode-1 (Human-in-the-Loop) implemented. Mode-2 (Machine-in-the-Loop) is planned but not yet functional.

ATLAS explores safe, explainable human–robot interaction. The core principle: *an intelligent system must know when to act — and when to stop.*

---

## Modes

- **Mode-1: Human-in-the-Loop** (implemented)
  - Voice (primary authority)
  - Hand gestures (secondary)
  - Emergency stop overrides everything
  - Emotions are non-navigational and cancelable
- **Mode-2: Machine-in-the-Loop** (planned)
  - Autonomous terrain analysis and energy-aware path planning (A*, risk/uncertainty/energy cost)
  - Strict separation from human control

---

## Architecture (Mode-1)

Sensors (mic, webcam) → Perception (ASR, Gesture CNN) → Mode Manager (FSM) → Motion/Emotion Controllers → Virtual Drone (Pygame sim) → Logging/HUD

**Principles**
1. Perception ≠ Decision ≠ Control
2. Strict mode separation
3. Uncertainty is a signal
4. Self-preservation over aggression
5. Energy is a strategic resource (planned for Mode-2)

---

## Features (Mode-1)

- **Voice commands (primary):** “ATLAS go up”, “rotate”, “hover”, “be happy”, “emergency stop”
- **Hand gestures (secondary):** up/down/left/right/forward/backward/rotate/flip/open_palm/fist/thumb_up/two_fingers/five_fingers
- **Motion commands:** move_* , rotate, flip, hover, slow_mode, fast_mode, emergency_stop
- **Emotions (non-navigational):** happy, excited, curious, bored, sad, calm, alert, tired, confident, confused
- **Safety:** Emergency stop trumps all; gestures ignored when stopped; voice can reset
- **Stability improvements:** Gesture smoothing (window/majority), per-command cooldowns, non-blocking timed emotions, centralized guards, basic logging

---

## Requirements

See `requirements.txt`. Key deps:
- Python 3.9+ recommended
- OpenCV, MediaPipe (hand tracking), PyTorch (gesture CNN), pygame (sim), vosk/sounddevice (offline ASR), numpy

Install:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Running

### Pygame sim + gestures + voice
```bash
python main.py
```

### Disable components
```bash
python main.py --no-voice       # no microphone/ASR
python main.py --no-gesture     # no webcam/gesture
python main.py --no-gui         # terminal mode only
```

---

## Controls (Mode-1)

- **Voice (examples):** “ATLAS go up”, “ATLAS rotate”, “ATLAS hover”, “ATLAS be happy”, “ATLAS emergency stop”
- **Keyboard (Pygame window):**
  - Movement: W/A/S/D, ↑/↓, Q (rotate), SPACE (hover), F (flip)
  - Speed: 1=slow, 2=fast, 3=normal
  - Safety: X (emergency stop), R (reset emergency)
  - Exit: ESC
- **Gestures:** Mapped to motion/emotion via the Gesture CNN (confidence threshold + smoothing)

---

## Project Structure (key files)

- `main.py` — entry point; spawns voice thread, gesture process, Pygame renderer; smoothing/cooldowns in main loop
- `core/mode_manager.py` — FSM; enforces emergency stop, cooldowns, non-blocking emotions, state/status
- `motion/` — motion commands and controller
- `gesture/` — gesture model, MediaPipe hand landmarking, CNN inference
- `voice/` — offline ASR listener and intent mapping
- `emotion/` — emotion mapping and non-blocking emote engine
- `drone/` — virtual drone model for simulation

---

## Recent improvements (Mode-1)

- Gesture smoothing (windowed confidence-weighted majority) and freshness drop
- Per-command cooldowns (debounce gesture/voice spam)
- Non-blocking, time-bounded emotions with periodic tick
- Centralized emergency and guards in ModeManager
- Basic logging with reduced dependency noise
- Removed arbitrary sleeps in gesture loop for better responsiveness

---

## Known limitations

- Mode-2 autonomy/path-planning is not implemented yet.
- Uses a single webcam/microphone; no multi-sensor fusion.
- Offline ASR quality depends on the installed model and mic setup.
- Gesture model quality depends on provided training data.

---

## Roadmap (planned)

- Implement Mode-2: terrain CNN, uncertainty/risk analysis, energy-aware A* planner
- Expand explainability/logging for decisions and safety triggers
- Add tests for ModeManager, motion/emotion handlers, and smoothing logic

---

## License

MIT (if not specified otherwise). Check repository license file if added.
