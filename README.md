# Autonomous Endoscope Assistant (Active Vision)

A Reinforcement Learning system that trains an agent to autonomously control
an active-vision endoscope camera — keeping a moving surgical instrument tip
centred inside a crop window panning over a full high-resolution video frame.

---

## Architecture Overview

The system implements a classic Active-Vision RL loop:

```
┌──────────────────────────────────────────────────────────┐
│                     RL Training Loop                      │
│                                                           │
│   ┌─────────────┐    action (Δx, Δy)    ┌─────────────┐  │
│   │  SAC Agent  │ ─────────────────────▶│ EndoscopeEnv│  │
│   │  (MlpPolicy)│                        │             │  │
│   │             │ ◀───────────────────── │  crop pan + │  │
│   └─────────────┘  obs (dx,dy) + reward  │  track tool │  │
│                                          └──────┬──────┘  │
│                                                 │          │
│                                    MICCAI trajectory       │
│                                    (instrument tip coords) │
└──────────────────────────────────────────────────────────┘
```

| Component | Details |
|-----------|---------|
| **Policy** | Soft Actor-Critic (SAC) with MLP policy |
| **Observation** | Normalised `(dx, dy)` offset of instrument tip from crop centre — shape `(2,)`, range `[-1, 1]` |
| **Action** | `[delta_x, delta_y]` velocity command — shape `(2,)`, range `[-1, 1]`, scaled by `max_velocity` |
| **Reward** | Dense: `-euclidean_distance` + boundary penalty + smoothness penalty |
| **Env normalisation** | `VecNormalize` (obs + reward) |

---

## Dataset — MICCAI EndoVis

The project is designed for the **MICCAI Robotic Instrument Tracking** challenge
datasets:

| Year | Challenge | Link |
|------|-----------|------|
| 2015 | Instrument Tracking | [grand-challenge.org](https://endovissub-instrument.grand-challenge.org/) |
| 2017 | Robotic Instrument Segmentation | [grand-challenge.org](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) |
| 2018 | Robotic Scene Segmentation | [grand-challenge.org](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) |

### Annotation formats supported by `MICCAILoader`

| Format | Description |
|--------|-------------|
| **CSV centroid** | Columns `frame, x, y` — direct tip coordinates |
| **CSV bounding box** | Columns `frame, xmin, ymin, xmax, ymax` — centroid derived as box midpoint |
| **XML (PASCAL VOC)** | Per-frame `.xml` files with `<bndbox>` elements |
| **JSON** | Single file mapping frame index → `{x, y}` or bounding-box dict |

Place downloaded sequences under `data/raw/<sequence_name>/`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/pradeepsuryad/Autonomous-Endoscope-Assistant-Active-Vision-.git
cd Autonomous-Endoscope-Assistant-Active-Vision-

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **PyTorch note**: the `requirements.txt` installs a CPU build of PyTorch by
> default.  For GPU training follow the instructions at
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and
> install the appropriate CUDA-enabled wheel first.

---

## Usage

### Without real data (synthetic trajectory)

No dataset download required — the script generates a sinusoidal Lissajous
trajectory automatically:

```bash
python scripts/train.py
```

### With MICCAI EndoVis data

```bash
python scripts/train.py \
    --data_path data/raw/seq_01 \
    --total_timesteps 1000000 \
    --save_dir models/
```

### All training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `None` | Path to MICCAI sequence directory |
| `--total_timesteps` | `500000` | Total env steps for SAC |
| `--save_dir` | `models/` | Where to save model & stats |
| `--eval_freq` | `5000` | Evaluate every N steps |
| `--n_eval_episodes` | `5` | Episodes per evaluation |
| `--learning_rate` | `3e-4` | SAC learning rate |
| `--buffer_size` | `100000` | Replay buffer capacity |
| `--batch_size` | `256` | Mini-batch size |
| `--seed` | `42` | Global random seed |

### Loading a saved model

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs import EndoscopeEnv
import numpy as np

trajectory = np.load("my_trajectory.npy")  # shape (N, 2)
env = DummyVecEnv([lambda: EndoscopeEnv(trajectory)])
env = VecNormalize.load("models/vecnormalize.pkl", env)
env.training = False

model = SAC.load("models/best_model", env=env)
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

---

## Environment Details

### State Space

`gymnasium.spaces.Box(shape=(2,), low=-1, high=1, dtype=float32)`

The observation is the normalised offset of the instrument tip from the crop
window centre:

```
obs[0] = (tip_x - crop_cx) / (crop_width  / 2)
obs[1] = (tip_y - crop_cy) / (crop_height / 2)
```

Values of `±1` mean the tip is exactly at the edge of the crop window.
Values beyond `±1` (clipped to `[-1, 1]`) indicate the tip has left the window.

### Action Space

`gymnasium.spaces.Box(shape=(2,), low=-1, high=1, dtype=float32)`

The agent outputs a 2-D velocity command `[delta_x, delta_y]`.  Each component
is multiplied by `max_velocity` (default `50.0` pixels/step) to obtain the
actual pixel displacement applied to the crop centre.

### Reward Function

```
reward = -euclidean_distance(tool_tip, crop_center)   # dense tracking reward
       + boundary_penalty  (if crop hit frame edge)    # default -10.0
       - velocity_penalty_weight * ||action||_2         # smoothness; default 0.01
```

The episode terminates (`terminated=True`) when the last frame of the trajectory
is reached.

---

## Project Structure

```
active-vision-endoscope-rl/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── miccai_loader.py       # MICCAILoader: CSV / XML / JSON parsing
│   └── envs/
│       ├── __init__.py
│       └── endoscope_env.py       # EndoscopeEnv: gymnasium.Env implementation
├── scripts/
│   └── train.py                   # SAC training entry-point
├── models/
│   └── .gitkeep                   # Saved weights land here (git-ignored)
├── data/                          # Raw MICCAI sequences (git-ignored)
│   └── raw/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Citation

If you use this code in your research, please cite the relevant MICCAI EndoVis
challenge papers alongside this repository.

---

## License

This project is released under the MIT License.
