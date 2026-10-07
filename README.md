# Cybernetic Vision Matrix // Hand, Face & Full-Body AI Suite

Real-time, multi-subject Computer Vision applications powered by **MediaPipe Tasks API (Vision 1.x)** and **OpenCV** with Cyberpunk / Sci-Fi HUD interfaces.

---

## 1. Full Body Tracking & Biomechanics: `full_body.py`

Tracks the full human body using **33 3D skeletal landmarks** with real-time joint angle analytics, posture evaluation, and exercise repetition tracking.

### Features
- **33-Point Full-Body Landmark Tracking**: Shoulders, elbows, wrists, hips, knees, ankles, feet, spine, and face anchor points.
- **Glowing Dual-Layer Skeleton**: Outer neon bloom underlay + crisp laser bone lines.
- **Cybernetic Torso Armor**: Translucent polygon chest and torso armor with spine alignment axis.
- **Real-Time Joint Angle Arcs**:
  - Live callouts directly at elbow and knee joints (e.g., `165°`, `90°`).
  - Shoulder angle, hip angle, and shoulder tilt.
- **Biomechanical Posture Assessment**:
  - Detects `OPTIMAL POSTURE`, `SHOULDER TILT`, or `UNALIGNED SPINE`.
- **Pose & Action Classification**:
  - `STANDING`
  - `SQUAT` / `HALF SQUAT`
  - `HANDS UP // VICTORY`
  - `T-POSE`
  - `ARMS CROSSED`
  - `RIGHT HAND RAISED` / `LEFT HAND RAISED`
  - `LEANING`
- **Exercise Repetition Counter**: Automatic live squat repetition counter (`SQUAT REPS: 0`). Press `[R]` to reset.
- **Multi-Subject Tracking**: Detects up to 2 bodies simultaneously with individual bounding reticles and telemetry.

### Run `full_body.py`
```powershell
.\.venv\Scripts\python full_body.py
```

### Keyboard Controls (`full_body.py`)
| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Quit application |
| **`S`** | Toggle Full-Body Skeleton on / off |
| **`A`** | Toggle Joint Angle Badges at elbows and knees |
| **`T`** | Toggle Cybernetic Torso Armor polygon |
| **`E`** | Toggle Biomechanics Telemetry sidebar |
| **`R`** | Reset Squat Repetition counter |
| **`B`** | Toggle Sci-Fi Target Bounding Box |
| **`C`** | Cycle 4 color themes (Cyberpunk, Matrix, Synthwave, Deep Ice) |
| **`SPACE`** | Save high-resolution snapshot to `captures/` |

---

## 2. Face Tracking & 478-Point Dense Mesh: `face.py`

Tracks and plots **every point of the human face** in real time with 52 facial blendshapes and expression telemetry.

### Run `face.py`
```powershell
.\.venv\Scripts\python face.py
```

### Keyboard Controls (`face.py`)
| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Quit application |
| **`M`** | Toggle 2,556-line wireframe tessellation mesh |
| **`P`** | Toggle all 478 individual landmark points |
| **`C`** | Toggle facial contours (Lips, Eyes, Eyebrows, Irises) |
| **`E`** | Toggle blendshape telemetry dashboard |
| **`B`** | Toggle sci-fi bounding box reticle |
| **`T`** | Cycle 4 color themes |
| **`SPACE`** | Save high-resolution snapshot to `captures/` |

---

## 3. Hand Tracking & Gesture Recognition: `hand.py`

Tracks up to 4 hands with full skeleton and multi-gesture recognition.

### Run `hand.py`
```powershell
.\.venv\Scripts\python hand.py
```

### Keyboard Controls (`hand.py`)
| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Quit application |
| **`C`** | Cycle through 4 futuristic color palettes |
| **`S`** | Toggle Hand Skeleton on / off |
| **`B`** | Toggle Sci-Fi Bounding Boxes on / off |
| **`F`** | Toggle Finger Extension Meters `[T][I][M][R][P]` |
| **`H`** | Toggle HUD display on / off |
| **`SPACE`** | Take a high-resolution snapshot (saved to `captures/`) |
