"""
Training Script — Autonomous Endoscope Assistant (Active Vision)
================================================================

Train a Soft Actor-Critic (SAC) agent to control an active-vision endoscope
camera crop window so that it tracks a moving surgical instrument tip.

Quick start (no real data — uses synthetic trajectory)
------------------------------------------------------
    python scripts/train.py

With real MICCAI EndoVis data
-----------------------------
    python scripts/train.py --case_dir "C:/Users/dprad/Downloads/train/train/case_1/1"

    # Train on multiple cases by listing them
    python scripts/train.py --case_dir ".../case_1/1" ".../case_1/2"

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
# Synthetic trajectory fallback                                                #
# --------------------------------------------------------------------------- #

def make_synthetic_trajectory(
    n_frames: int = 1000,
    frame_w: int = 1280,
    frame_h: int = 512,
) -> np.ndarray:
    """Lissajous figure-8 path for smoke-testing without real data."""
    t = np.linspace(0, 4 * math.pi, n_frames)
    cx = frame_w / 2.0 + frame_w * 0.30 * np.cos(t)
    cy = frame_h / 2.0 + frame_h * 0.25 * np.sin(2.0 * t)
    return np.stack([cx, cy], axis=1)


# --------------------------------------------------------------------------- #
# Argument parser                                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on the EndoscopeEnv active-vision task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case_dir",
        type=str,
        nargs="+",
        default=None,
        metavar="DIR",
        help=(
            "One or more MICCAI sequence directories each containing "
            "video.mp4 (e.g. .../train/case_1/1).  Trajectories from all "
            "directories are concatenated.  Omit to use a synthetic trajectory."
        ),
    )
    parser.add_argument(
        "--stereo_half",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Which stereo camera half to use ('top' = left camera).",
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
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Environment factory                                                          #
# --------------------------------------------------------------------------- #

def make_env(trajectory: np.ndarray, frame_size: tuple, seed: int = 0):
    """Return a thunk that creates and seeds an ``EndoscopeEnv``."""
    def _init():
        env = EndoscopeEnv(
            trajectory=trajectory,
            frame_size=frame_size,
            crop_size=(400, 400),
            max_velocity=50.0,
            boundary_penalty=-10.0,
            velocity_penalty_weight=0.01,
        )
        env.reset(seed=seed)
        return env
    return _init


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = MICCAILoader(stereo_half=args.stereo_half, smooth_sigma=3.0)

    # ── 1. Load or synthesise trajectory ──────────────────────────────── #
    if args.case_dir is not None:
        trajectories = []
        frame_size = (MICCAILoader.STEREO_W, MICCAILoader.STEREO_HALF_H)

        for case_path in args.case_dir:
            p = Path(case_path)
            if not p.is_dir():
                print(f"[ERROR] case_dir '{p}' is not a directory.")
                sys.exit(1)

            # Read frame size from info.yaml (falls back to 1280×512)
            frame_size = loader.get_frame_size(p)
            print(f"[INFO] Loading trajectory from: {p}  (frame size {frame_size})")

            traj = loader.load_trajectory(p)
            trajectories.append(traj)
            print(f"[INFO]   → {len(traj)} frames loaded.")

        trajectory = np.concatenate(trajectories, axis=0)
        print(f"[INFO] Total trajectory length: {len(trajectory)} frames.")
    else:
        frame_size = (MICCAILoader.STEREO_W, MICCAILoader.STEREO_HALF_H)
        print(
            "[INFO] No --case_dir provided.  Using synthetic trajectory "
            f"({frame_size[0]}×{frame_size[1]}, 1000 frames).  "
            "Pass --case_dir to use real MICCAI data."
        )
        trajectory = make_synthetic_trajectory(
            n_frames=1_000, frame_w=frame_size[0], frame_h=frame_size[1]
        )

    print(f"[INFO] Frame size for environment: {frame_size}")
    print(f"[INFO] Trajectory shape: {trajectory.shape}")

    # ── 2. Wrap in DummyVecEnv + VecNormalize ─────────────────────────── #
    train_env = DummyVecEnv([make_env(trajectory, frame_size, seed=args.seed)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    eval_env = DummyVecEnv([make_env(trajectory, frame_size, seed=args.seed + 100)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
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
