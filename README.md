# ATLAS v1 — Motion-Only Gesture-Controlled Drone

A clean, minimal drone control system using hand-gesture recognition.

## Architecture

```
┌──────────────────────────┐     Queue(1)     ┌─────────────────────────┐
│    Perception Process    │ ──────────────→  │      Main Process       │
│  (gesture_process.py)    │  (label, conf)   │       (main.py)         │
│                          │                  │                         │
│  Webcam → MediaPipe      │                  │  ModeManager (2-state)  │
│  → 42D features          │                  │  → MotionController     │
│  → MLP (42→64→32→7)      │                  │  → VirtualDrone         │
│  → EMA smooth            │                  │  → Pygame renderer      │
│  → send-on-change        │                  │                         │
└──────────────────────────┘                  └─────────────────────────┘
```

## 7 Gesture Classes

| Gesture        | Drone Action      |
|:---------------|:------------------|
| `MOVE_UP`      | Gain altitude     |
| `MOVE_DOWN`    | Lose altitude     |
| `MOVE_LEFT`    | Strafe left       |
| `MOVE_RIGHT`   | Strafe right      |
| `MOVE_FORWARD` | Move forward      |
| `MOVE_BACKWARD`| Move backward     |
| `HOVER`        | Stop (zero vel)   |

## Project Structure

```
ATLAS-Aims-drone/
├── config.py                  # Central constants
├── main.py                    # Pygame renderer + main loop
├── drone/
│   └── virtual_drone.py       # Velocity-based drone model
├── motion/
│   ├── motion_commands.py     # 7 MotionCommand enum
│   └── motion_controller.py   # Command → drone dispatch
├── core/
│   └── mode_manager.py        # 2-state FSM (IDLE / MOVING)
├── gesture/
│   ├── gesture_process.py     # Perception process (camera + MLP)
│   ├── feature_engineering.py # Landmark → 42D vector
│   ├── mlp_model.py           # GestureMLP + GestureClassifier
│   ├── train_mlp.py           # Training script
│   ├── gesture_labels.py      # Label utilities
│   └── hand_landmarker.task   # MediaPipe model file
├── models/
│   └── gesture_mlp.pth        # Trained MLP weights
└── dataset/
    └── landmarks_train.npy    # Training data (after collection)
```

## Quick Start

```bash
# Run with gesture control
python main.py

# Keyboard only
python main.py --no-gesture
```

### Keyboard Controls
- **WASD** — move (forward/back/left/right)
- **↑ / ↓** — altitude up/down
- **SPACE** — hover (stop)
- **ESC** — quit

