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

## Design Decisions

### Why MLP over CNN?
The CNN required full hand-crop images (128×128 grayscale), which were sensitive to cropping quality, lighting, and hand orientation. The MLP takes **21 landmark coordinates** directly from MediaPipe — already normalised, rotation-invariant, and lighting-independent. The 42D feature vector (wrist-relative, span-normalised) makes the model fast, stable, and easy to train with small datasets.

### Why Velocity-Based?
The previous system used discrete step commands — each gesture triggered a single position update. The new model sets a **velocity vector** that persists until overridden. This creates smooth, continuous motion that naturally maps to "hold gesture = keep moving, release = stop."

### Why Send-on-Change Queue?
The old system spammed the queue every frame, flooding it with redundant `NO_GESTURE` messages. The new system only sends when:
1. The gesture **changes** to a different class
2. Confidence is **≥ threshold** (0.6)
3. Hand disappears → **single** `HOVER` event

This eliminates queue spam and makes the system deterministic.

### Axis Mirroring Resolution
The webcam frame is flipped horizontally (`cv2.flip(frame, 1)`) so the user sees a mirror view. The landmark coordinates are already in the flipped frame, so **no additional axis inversion** is needed. `MOVE_LEFT` in the camera corresponds to `MOVE_LEFT` on screen.

## Training the MLP

1. Collect landmark data (store as `.npy` files in `dataset/`)
2. Train:
```bash
python -m gesture.train_mlp --epochs 30 --lr 0.001
```
3. Weights saved to `models/gesture_mlp.pth`
