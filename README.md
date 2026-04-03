# Autonomous Endoscope Assistant (Active Vision)

A Reinforcement Learning system that trains an agent to autonomously control
an active-vision endoscope camera -- keeping a moving surgical instrument tip
centred inside a crop window panning over a full surgical video frame.

Built with **Gymnasium**, **Stable-Baselines3 (SAC)**, and the
**MICCAI EndoVis** stereo surgical video dataset.

---

## Architecture Overview

Two environment modes are available:

### Mode 1 -- State-based (MlpPolicy)

```
+------------------------------------------------------------------+
|                       RL Training Loop                           |
|                                                                  |
|  +-------------+   action [dx, dy]    +----------------------+  |
|  |  SAC Agent  | -------------------> |   EndoscopeEnv       |  |
|  |  MlpPolicy  |                      |                      |  |
|  |             | <------------------- |  crop pan + reward   |  |
|  +-------------+  obs (dx,dy) + rew   +----------+-----------+  |
|                                                  |               |
|                              MOG2 trajectory     |               |
|                              (extracted from     |               |
|                               video.mp4)         |               |
+------------------------------------------------------------------+
```

### Mode 2 -- Visual (CnnPolicy)

Same loop, but the observation is an **84x84 RGB image** -- the actual surgical
video crop centred on the current window position.  The agent learns to detect
the instrument from pixels directly.

| Component | State Mode | Visual Mode |
|-----------|-----------|-------------|
| **Policy** | SAC MlpPolicy | SAC CnnPolicy (NatureCNN) |
| **Observation** | `(dx, dy)` normalised offset -- shape `(2,)`, unbounded | 84x84 RGB crop -- shape `(84, 84, 3)`, uint8 |
| **Action** | `[delta_x, delta_y]` velocity -- shape `(2,)`, range `[-1, 1]` | same |
| **Reward** | `-distance` + boundary penalty + smoothness | same |
| **Obs normalisation** | VecNormalize (running mean/std) | none (NatureCNN divides by 255) |
| **Eval normalisation** | Synced from train VecNormalize stats | same |

---

## Dataset -- MICCAI EndoVis

The dataset contains stereo surgical video sequences recorded with a da Vinci
robotic system.

**Expected directory layout:**
```
<train_root>/
    case_1/
        1/
            video.mp4          <- stereo surgical video, 1280 x 2048 px
            info.yaml          <- resolution: {width: 1280, height: 1024}
                                  video_stack: "vertical"
            calibration.yaml
        2/ ...
    case_2/ ...
    case_12/
```

**Stereo convention:**

The physical video file is **1280 x 2048** (width x height).  The two
camera views are stacked vertically, each occupying **1280 x 1024** px:

```
Row    0 - 1023 : left  camera (stereo_half = "top",    default)
Row 1024 - 2047 : right camera (stereo_half = "bottom")
```

`info.yaml` reports the **single-view** resolution (`1280 x 1024`), not the
full frame height.  `MICCAILoader.get_frame_size()` returns `(1280, 1024)`
directly from the yaml without further division.

**No pre-labelled annotations required.**  `MICCAILoader` extracts the
instrument-tip trajectory automatically using a two-pass MOG2 background
subtraction algorithm:

1. **Pass 1 (warm-up):** Feed the first 30 frames through MOG2 to build the
   background model, then rewind to frame 0.
2. **Pass 2 (extraction):** Read every frame and detect the largest foreground
   contour.  One centroid is recorded per frame, so
   `len(trajectory) == CAP_PROP_FRAME_COUNT` is always guaranteed.

Extracted trajectories are cached as `trajectory_top.npy` (or `_bottom.npy`)
next to each `video.mp4`.  Delete the `.npy` to force re-extraction.

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

> **GPU note:** `requirements.txt` installs a CPU PyTorch build.  For CUDA
> training visit https://pytorch.org/get-started/locally/ and install the
> matching GPU wheel before running `pip install -r requirements.txt`.

---

## Usage

### Quick start -- no dataset needed

```bash
python scripts/train.py
```

Runs on a 1000-frame synthetic Lissajous trajectory.  Good for verifying
your install end-to-end.

### State-based training on real data (recommended first step)

```bash
python scripts/train.py \
    --case_dir "C:/path/to/train/case_1/1" \
    --total_timesteps 500000
```

### Train across multiple sequences (state mode only)

```bash
python scripts/train.py \
    --case_dir "C:/path/to/train/case_1/1" \
               "C:/path/to/train/case_1/2" \
               "C:/path/to/train/case_1/3" \
    --total_timesteps 1000000
```

### Visual (CNN) training -- single sequence only

```bash
python scripts/train.py --visual \
    --case_dir "C:/path/to/train/case_1/1" \
    --total_timesteps 1000000
```

Passing multiple `--case_dir` values with `--visual` is rejected immediately
with an error (one video stream per episode is required).

### Progress bar

```
SAC Training:  42%|################| 210k/500k [03:21<04:35, 1043 step/s] dist=87px  rew=-91.3
```

- `dist` -- mean Euclidean distance (px) between tool tip and crop centre over
  the last 500 steps.  Target: below 50 px.
- `rew` -- mean raw reward.  Approaches 0 as tracking improves.
- Uses ASCII-only characters; safe on Windows cp1252 consoles.

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--case_dir` | `None` | One or more sequence dirs containing `video.mp4` |
| `--visual` | `False` | Use CnnPolicy + 84x84 image observations |
| `--stereo_half` | `top` | `top` (left cam) or `bottom` (right cam) |
| `--total_timesteps` | `500000` | Total SAC training steps |
| `--save_dir` | `models/` | Output directory for weights |
| `--eval_freq` | `5000` | Evaluate every N steps |
| `--n_eval_episodes` | `5` | Episodes per evaluation |
| `--learning_rate` | `3e-4` | SAC learning rate |
| `--buffer_size` | `100000` | Replay buffer size |
| `--batch_size` | `256` | Mini-batch size |
| `--seed` | `42` | Random seed |

---

## Environment Details

### EndoscopeEnv (state-based)

| | |
|---|---|
| **Observation space** | `Box(-inf, +inf, shape=(2,), float32)` -- normalised `(dx, dy)` offset of tool tip from crop centre, divided by half-crop size.  Unbounded so the agent retains full error magnitude when the tool leaves the window; `VecNormalize` handles the scale. |
| **Action space** | `Box(-1, 1, shape=(2,), float32)` -- velocity command scaled by `max_velocity` (default 50 px/step) |
| **Frame size** | 1280 x 1024 px (single stereo view) |
| **Crop window** | 400 x 400 px |

### EndoscopeVisualEnv (visual)

| | |
|---|---|
| **Observation space** | `Box(0, 255, shape=(84, 84, 3), uint8)` -- RGB crop from the actual video |
| **Action space** | same as above |
| **Frame loading** | Sequential `cv2.VideoCapture` reads -- no full video pre-load |
| **Alignment** | `trajectory[i]` corresponds to video frame `i` (guaranteed by two-pass extraction) |

### Reward Function

```
reward = -euclidean_distance(tool_tip, crop_center)   # dense tracking
       + boundary_penalty   if crop hit the frame edge  # default -10.0
       - velocity_weight * ||action||_2                 # smoothness, default 0.01
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

| Test | What it checks |
|------|----------------|
| `test_trajectory_frame_count` | `len(trajectory) == CAP_PROP_FRAME_COUNT` |
| `test_trajectory_frame_alignment` | `trajectory[i]` computed from video frame `i` |
| `test_stereo_half_propagation` | `get_frame_size()` returns `(1280, 1024)` for both halves |
| `test_visual_mode_rejects_multi_case` | `--visual` with multiple `--case_dir` exits with code 1 |
| `test_synthetic_training_smoke` | 200-step SAC training run completes and predicts valid actions |

---

## Project Structure

```
active-vision-endoscope-rl/
+-- src/
|   +-- data/
|   |   +-- miccai_loader.py        # MICCAILoader: two-pass MOG2 extraction + cache
|   +-- envs/
|       +-- endoscope_env.py        # State-based env  (MlpPolicy)
|       +-- endoscope_visual_env.py # Image-based env  (CnnPolicy)
+-- scripts/
|   +-- train.py                   # Training entry-point + tqdm progress callback
+-- tests/
|   +-- test_loader.py             # Automated test suite (5 tests)
+-- models/                        # Saved weights (git-ignored)
+-- requirements.txt
+-- README.md
```

---

## License

MIT License
