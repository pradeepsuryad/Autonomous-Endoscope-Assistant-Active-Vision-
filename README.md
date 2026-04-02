# Autonomous Endoscope Assistant (Active Vision)

A Reinforcement Learning system that trains an agent to autonomously control
an active-vision endoscope camera — keeping a moving surgical instrument tip
centred inside a crop window panning over a full surgical video frame.

Built with **Gymnasium**, **Stable-Baselines3 (SAC)**, and the
**MICCAI EndoVis** stereo surgical video dataset.

---

## Architecture Overview

Two environment modes are available:

### Mode 1 — State-based (MlpPolicy)
```
┌──────────────────────────────────────────────────────────────┐
│                     RL Training Loop                          │
│                                                               │
│   ┌─────────────┐    action [Δx, Δy]     ┌────────────────┐  │
│   │  SAC Agent  │ ──────────────────────▶│ EndoscopeEnv   │  │
│   │  MlpPolicy  │                         │                │  │
│   │             │ ◀────────────────────── │  crop pan +    │  │
│   └─────────────┘  obs (dx,dy) + reward   │  track tool    │  │
│                                           └───────┬────────┘  │
│                                  MOG2 trajectory  │            │
│                                  (from video.mp4) │            │
└──────────────────────────────────────────────────────────────┘
```

### Mode 2 — Visual (CnnPolicy)
Same loop but the agent receives an **84×84 RGB image** — the actual surgical
video crop — instead of a pre-computed offset. Requires a CNN policy.

| Component | State Mode | Visual Mode |
|-----------|-----------|-------------|
| **Policy** | SAC MlpPolicy | SAC CnnPolicy (NatureCNN) |
| **Observation** | `(dx, dy)` offset · shape `(2,)` | 84×84 RGB crop · shape `(84,84,3)` |
| **Action** | `[Δx, Δy]` velocity · shape `(2,)` · range `[-1,1]` | same |
| **Reward** | `-distance` + boundary penalty + smoothness | same |
| **Progress** | tqdm bar with ETA, distance, reward | same |

---

## Dataset — MICCAI EndoVis

The dataset contains stereo surgical video sequences recorded with a da Vinci
robotic system.

**Expected directory layout:**
```
<train_root>/
    case_1/
        1/
            video.mp4          ← stereo surgical video (1280×1024)
            info.yaml          ← resolution, video_stack: "vertical"
            calibration.yaml
        2/ ...
    case_2/ ...
    ...
    case_12/
```

**Stereo convention:** `video_stack: "vertical"` means the left camera occupies
the **top half** (rows 0–511, 1280×512) and the right camera the **bottom half**.
The loader uses the left (top) view by default.

**No pre-labelled annotations are needed.** `MICCAILoader` automatically
extracts the instrument-tip trajectory from raw video using **MOG2 background
subtraction** + contour tracking. Extracted trajectories are cached as
`trajectory_top.npy` next to each video so the first run processes the video
once and every subsequent run is instant.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/pradeepsuryad/Autonomous-Endoscope-Assistant-Active-Vision-.git
cd Autonomous-Endoscope-Assistant-Active-Vision-

# 2. Virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` installs a CPU PyTorch build. For CUDA
> training visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
> and install the matching GPU wheel before running `pip install -r requirements.txt`.

---

## Usage

### Quick start — no dataset needed

```bash
python scripts/train.py
```
Runs on a synthetic Lissajous trajectory. Good for verifying your install.

### State-based training on real data (recommended first step)

```bash
python scripts/train.py \
    --case_dir "C:/path/to/train/case_1/1" \
    --total_timesteps 500000
```

### Visual (CNN) training on real data

```bash
python scripts/train.py --visual \
    --case_dir "C:/path/to/train/case_1/1" \
    --total_timesteps 1000000
```

### Train across multiple sequences (state mode)

```bash
python scripts/train.py \
    --case_dir "C:/path/to/train/case_1/1" \
                "C:/path/to/train/case_1/2" \
                "C:/path/to/train/case_2/1" \
    --total_timesteps 1000000
```

### Progress bar

Training prints a live tqdm bar with ETA, mean tracking distance, and mean reward:

```
SAC Training:  42%|████████        | 210k/500k [03:21<04:35, 1043 step/s] dist=87px  rew=-91.3
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--case_dir` | `None` | One or more sequence dirs containing `video.mp4` |
| `--visual` | `False` | Use `CnnPolicy` + image observations |
| `--stereo_half` | `top` | `top` (left cam) or `bottom` (right cam) |
| `--total_timesteps` | `500000` | Total SAC training steps |
| `--save_dir` | `models/` | Where to save model weights |
| `--eval_freq` | `5000` | Evaluate every N steps |
| `--n_eval_episodes` | `5` | Episodes per evaluation |
| `--learning_rate` | `3e-4` | SAC learning rate |
| `--buffer_size` | `100000` | Replay buffer size |
| `--batch_size` | `256` | Mini-batch size |
| `--seed` | `42` | Random seed |

---

## Loading a Saved Model

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs import EndoscopeEnv
from src.data import MICCAILoader
import numpy as np

loader = MICCAILoader()
trajectory = loader.load_trajectory("path/to/case_1/1")
frame_size = loader.get_frame_size("path/to/case_1/1")

env = DummyVecEnv([lambda: EndoscopeEnv(trajectory, frame_size=frame_size)])
env = VecNormalize.load("models/vecnormalize_state.pkl", env)
env.training = False

model = SAC.load("models/best_model", env=env)
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

---

## Environment Details

### EndoscopeEnv (state-based)

| | |
|---|---|
| **Observation** | `Box(shape=(2,), low=-1, high=1)` — normalised `(dx, dy)` offset |
| **Action** | `Box(shape=(2,), low=-1, high=1)` — velocity command scaled by `max_velocity` |
| **Frame size** | 1280 × 512 px (single stereo view) |
| **Crop window** | 400 × 400 px |

### EndoscopeVisualEnv (visual)

| | |
|---|---|
| **Observation** | `Box(shape=(84,84,3), dtype=uint8)` — RGB crop from the real video |
| **Action** | same as above |
| **Frame loading** | Sequential `cv2.VideoCapture` reads — no full video pre-load |

### Reward Function

```
reward = -euclidean_distance(tool_tip, crop_center)   # dense tracking
       + boundary_penalty   if crop hit the frame edge  # default −10.0
       − velocity_weight × ‖action‖₂                   # smoothness, default 0.01
```

---

## Project Structure

```
active-vision-endoscope-rl/
├── src/
│   ├── data/
│   │   └── miccai_loader.py        # MICCAILoader: MOG2 video extraction + caching
│   └── envs/
│       ├── endoscope_env.py        # State-based env  (MlpPolicy)
│       └── endoscope_visual_env.py # Image-based env  (CnnPolicy)
├── scripts/
│   └── train.py                   # Training entry-point + tqdm progress callback
├── models/                        # Saved weights (git-ignored)
├── requirements.txt
└── README.md
```

---

## License

MIT License
