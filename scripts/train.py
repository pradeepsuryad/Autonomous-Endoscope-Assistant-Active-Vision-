"""
Training Script — Autonomous Endoscope Assistant (Active Vision)
================================================================

Train a Soft Actor-Critic (SAC) agent to control an active-vision endoscope
camera crop window so that it tracks a moving surgical instrument tip.

Quick start (no real data needed)
----------------------------------
    python scripts/train.py

With MICCAI EndoVis data
------------------------
    python scripts/train.py --data_path data/raw/seq_01 --total_timesteps 1000000

The script will:
  1. Load (or synthesise) a per-frame instrument-tip trajectory.
  2. Construct an ``EndoscopeEnv`` and wrap it with ``DummyVecEnv`` +
     ``VecNormalize`` for stable training.
  3. Train SAC with the given hyper-parameters and periodic evaluation.
  4. Save the final policy and normalisation statistics to ``save_dir``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.miccai_loader import MICCAILoader
from src.envs.endoscope_env import EndoscopeEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# --------------------------------------------------------------------------- #
# Synthetic trajectory                                                          #
# --------------------------------------------------------------------------- #

def make_synthetic_trajectory(
    n_frames: int = 1000,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> np.ndarray:
    """Generate a sinusoidal synthetic trajectory for smoke-testing.

    The tool tip follows a Lissajous-like figure-8 path centred in the frame.

    Parameters
    ----------
    n_frames : int
        Number of frames (trajectory length).
    frame_w, frame_h : int
        Full video frame dimensions in pixels.

    Returns
    -------
    np.ndarray, shape (n_frames, 2)
        Pixel coordinates ``(x, y)`` for each frame.
    """
    t = np.linspace(0, 4 * math.pi, n_frames)
    amplitude_x = frame_w * 0.35
    amplitude_y = frame_h * 0.30
    cx = frame_w / 2.0 + amplitude_x * np.cos(t)
    cy = frame_h / 2.0 + amplitude_y * np.sin(2.0 * t)
    return np.stack([cx, cy], axis=1)


# --------------------------------------------------------------------------- #
# Argument parser                                                               #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on the EndoscopeEnv active-vision task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to a MICCAI EndoVis sequence directory containing annotation "
            "files (CSV / JSON / XML).  If omitted, a synthetic sinusoidal "
            "trajectory is used so the script works out-of-the-box."
        ),
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500_000,
        help="Total environment steps for SAC training.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/",
        help="Directory for saving the best model and VecNormalize statistics.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5_000,
        help="Evaluate the policy every N environment steps.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of episodes used during each evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="SAC learning rate for all optimisers.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
        help="Size of the SAC replay buffer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for SAC gradient updates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Environment factory                                                           #
# --------------------------------------------------------------------------- #

def make_env(trajectory: np.ndarray, seed: int = 0):
    """Return a thunk that creates and seeds an ``EndoscopeEnv``."""
    def _init():
        env = EndoscopeEnv(
            trajectory=trajectory,
            frame_size=(1920, 1080),
            crop_size=(400, 400),
            max_velocity=50.0,
            boundary_penalty=-10.0,
            velocity_penalty_weight=0.01,
        )
        env.reset(seed=seed)
        return env
    return _init


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load or synthesise trajectory ──────────────────────────────── #
    if args.data_path is not None:
        data_path = Path(args.data_path)
        if not data_path.is_dir():
            print(f"[ERROR] --data_path '{data_path}' is not a directory.")
            sys.exit(1)
        print(f"[INFO] Loading MICCAI trajectory from: {data_path}")
        loader = MICCAILoader()
        trajectory = loader.load_trajectory(data_path)
        print(f"[INFO] Loaded trajectory with {len(trajectory)} frames.")
    else:
        print(
            "[INFO] No --data_path provided.  Using a synthetic sinusoidal "
            "trajectory (1000 frames).  Pass --data_path to use real data."
        )
        trajectory = make_synthetic_trajectory(n_frames=1_000)

    # ── 2. Wrap in DummyVecEnv + VecNormalize ─────────────────────────── #
    train_env = DummyVecEnv([make_env(trajectory, seed=args.seed)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # Separate (un-normalised-reward) eval env so EvalCallback reports raw reward
    eval_env = DummyVecEnv([make_env(trajectory, seed=args.seed + 100)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # keep reward unnormalised for interpretable eval metrics
        training=False,     # do not update running stats during eval
    )

    # ── 3. Build SAC model ────────────────────────────────────────────── #
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    # ── 4. Evaluation callback ─────────────────────────────────────────── #
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ── 5. Train ──────────────────────────────────────────────────────── #
    print(f"[INFO] Starting SAC training for {args.total_timesteps:,} steps ...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )

    # ── 6. Save final model and normalisation stats ────────────────────── #
    final_model_path = save_dir / "sac_endoscope_final"
    model.save(str(final_model_path))
    print(f"[INFO] Final model saved to: {final_model_path}.zip")

    vec_norm_path = save_dir / "vecnormalize.pkl"
    train_env.save(str(vec_norm_path))
    print(f"[INFO] VecNormalize stats saved to: {vec_norm_path}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
