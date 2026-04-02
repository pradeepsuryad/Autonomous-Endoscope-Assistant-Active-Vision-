"""
Training Script — Autonomous Endoscope Assistant (Active Vision)
================================================================

Train a Soft Actor-Critic (SAC) agent on one of two environment modes:

  MLP mode (default)  — state = normalised (dx, dy) offset, MlpPolicy
  Visual mode         — state = 84×84 RGB crop from real video, CnnPolicy

Quick start (no real data)
--------------------------
    python scripts/train.py
    python scripts/train.py --visual          # CNN on synthetic frames

With real MICCAI EndoVis data
-----------------------------
    python scripts/train.py \\
        --case_dir "C:/Users/dprad/Downloads/train/train/case_1/1"

    python scripts/train.py --visual \\
        --case_dir "C:/Users/dprad/Downloads/train/train/case_1/1"

    # Multiple sequences (MLP mode only — concatenates trajectories)
    python scripts/train.py \\
        --case_dir ".../case_1/1" ".../case_1/2" ".../case_2/1"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm as tqdm_lib

# Allow running from the repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.miccai_loader import MICCAILoader
from src.envs.endoscope_env import EndoscopeEnv
from src.envs.endoscope_visual_env import EndoscopeVisualEnv

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# --------------------------------------------------------------------------- #
# Custom tqdm training progress callback                                       #
# --------------------------------------------------------------------------- #

class TrainingProgressCallback(BaseCallback):
    """Live tqdm progress bar with ETA, mean distance, and mean reward.

    Displays a single bar like::

        SAC Training: 42%|████████        | 210k/500k [03:21<04:35, 1043 step/s]
                      dist=87px  rew=-91.3

    Parameters
    ----------
    total_timesteps : int
        Total training steps (used to size the bar).
    log_interval : int
        How often (in steps) to refresh the postfix metrics.
    """

    def __init__(self, total_timesteps: int, log_interval: int = 500) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._pbar: Optional[tqdm_lib.tqdm] = None
        self._distances: list[float] = []
        self._rewards: list[float] = []

    def _on_training_start(self) -> None:
        self._pbar = tqdm_lib.tqdm(
            total=self.total_timesteps,
            desc="SAC Training",
            unit="step",
            unit_scale=True,
            ncols=110,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            dynamic_ncols=False,
        )

    def _on_step(self) -> bool:
        self._pbar.update(1)

        # Collect distance from info dicts (one per parallel env)
        for info in self.locals.get("infos", []):
            if "distance" in info:
                self._distances.append(float(info["distance"]))

        # Collect raw rewards
        rewards = self.locals.get("rewards")
        if rewards is not None:
            vals = rewards.tolist() if hasattr(rewards, "tolist") else [float(rewards)]
            self._rewards.extend(vals)

        # Refresh postfix every log_interval steps
        if self.num_timesteps % self.log_interval == 0:
            win = self.log_interval
            postfix: dict = {}
            if self._distances:
                postfix["dist"] = f"{np.mean(self._distances[-win:]):.0f}px"
            if self._rewards:
                postfix["rew"] = f"{np.mean(self._rewards[-win:]):.1f}"
            if postfix:
                self._pbar.set_postfix(postfix)

        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            # Final metrics summary
            if self._distances:
                print(
                    f"\n[SUMMARY] Mean distance (last 1k steps): "
                    f"{np.mean(self._distances[-1000:]):.1f} px  |  "
                    f"Mean reward: {np.mean(self._rewards[-1000:]):.2f}"
                )
            self._pbar.close()


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
            "One or more MICCAI sequence directories each containing video.mp4. "
            "In --visual mode only the first directory is used. "
            "Omit to run on a synthetic trajectory."
        ),
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help=(
            "Use EndoscopeVisualEnv (84×84 image observations, CnnPolicy). "
            "Requires --case_dir pointing to a real video. "
            "Default: EndoscopeEnv (2D state, MlpPolicy)."
        ),
    )
    parser.add_argument(
        "--stereo_half",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Which stereo camera half to use.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500_000,
        help="Total SAC training steps.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/",
    )
    parser.add_argument("--eval_freq",    type=int,   default=5_000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--learning_rate",   type=float, default=3e-4)
    parser.add_argument("--buffer_size",     type=int,   default=100_000)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--seed",            type=int,   default=42)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Environment factories                                                        #
# --------------------------------------------------------------------------- #

def make_state_env(trajectory: np.ndarray, frame_size: tuple, seed: int):
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


def make_visual_env(
    video_path: str, trajectory: np.ndarray, frame_size: tuple, seed: int
):
    def _init():
        env = EndoscopeVisualEnv(
            video_path=video_path,
            trajectory=trajectory,
            stereo_half="top",
            frame_size=frame_size,
            crop_size=(400, 400),
            obs_size=84,
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
    frame_size = (MICCAILoader.STEREO_W, MICCAILoader.STEREO_HALF_H)

    # ── 1. Load or synthesise trajectory ──────────────────────────────── #
    video_path: Optional[str] = None

    if args.case_dir is not None:
        trajectories = []
        for i, case_path in enumerate(args.case_dir):
            p = Path(case_path)
            if not p.is_dir():
                print(f"[ERROR] case_dir '{p}' is not a directory.")
                sys.exit(1)
            frame_size = loader.get_frame_size(p)
            print(f"[INFO] Loading: {p}  (single-view {frame_size[0]}×{frame_size[1]})")
            traj = loader.load_trajectory(p)
            trajectories.append(traj)
            print(f"[INFO]   → {len(traj)} frames")
            if i == 0:
                video_path = str(p / "video.mp4")

        trajectory = np.concatenate(trajectories, axis=0)
        print(f"[INFO] Total trajectory: {len(trajectory)} frames")

        if args.visual and len(args.case_dir) > 1:
            print(
                "[WARN] --visual mode uses a single video stream. "
                "Only the first case_dir will be used for the video."
            )
    else:
        mode_str = "visual (synthetic frames)" if args.visual else "state"
        print(
            f"[INFO] No --case_dir provided. Using synthetic trajectory "
            f"({frame_size[0]}×{frame_size[1]}, 1000 frames) in {mode_str} mode."
        )
        trajectory = make_synthetic_trajectory(
            n_frames=1_000, frame_w=frame_size[0], frame_h=frame_size[1]
        )
        if args.visual:
            print("[WARN] --visual requires a real video.mp4 — falling back to MLP mode.")
            args.visual = False

    print(f"[INFO] Mode       : {'Visual (CnnPolicy)' if args.visual else 'State (MlpPolicy)'}")
    print(f"[INFO] Frame size : {frame_size}")
    print(f"[INFO] Trajectory : {trajectory.shape}")

    # ── 2. Build vectorised envs ───────────────────────────────────────── #
    if args.visual:
        policy = "CnnPolicy"
        train_vec = DummyVecEnv([make_visual_env(video_path, trajectory, frame_size, args.seed)])
        # Don't normalise image pixels — NatureCNN divides by 255 internally
        train_env = VecNormalize(
            train_vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99
        )
        eval_vec = DummyVecEnv([make_visual_env(video_path, trajectory, frame_size, args.seed + 100)])
        eval_env = VecNormalize(
            eval_vec, norm_obs=False, norm_reward=False, training=False
        )
    else:
        policy = "MlpPolicy"
        train_vec = DummyVecEnv([make_state_env(trajectory, frame_size, args.seed)])
        train_env = VecNormalize(
            train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99
        )
        eval_vec = DummyVecEnv([make_state_env(trajectory, frame_size, args.seed + 100)])
        eval_env = VecNormalize(
            eval_vec, norm_obs=True, norm_reward=False, training=False
        )

    # ── 3. SAC model ──────────────────────────────────────────────────── #
    # Visual mode needs more warm-up and a larger buffer for image replay
    learning_starts = 5_000 if args.visual else 1_000
    buffer_size = min(args.buffer_size, 50_000) if args.visual else args.buffer_size

    model = SAC(
        policy=policy,
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=buffer_size,
        batch_size=args.batch_size,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=learning_starts,
        verbose=0,          # suppressed — tqdm callback handles progress output
        seed=args.seed,
        device="auto",
    )

    # ── 4. Callbacks ───────────────────────────────────────────────────── #
    progress_cb = TrainingProgressCallback(
        total_timesteps=args.total_timesteps,
        log_interval=500,
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = CallbackList([progress_cb, eval_cb])

    # ── 5. Train ──────────────────────────────────────────────────────── #
    print(f"\n[INFO] Starting SAC training for {args.total_timesteps:,} steps ...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,   # we use our own tqdm bar above
    )

    # ── 6. Save ────────────────────────────────────────────────────────── #
    suffix = "visual" if args.visual else "state"
    final_path = save_dir / f"sac_endoscope_{suffix}_final"
    model.save(str(final_path))
    print(f"[INFO] Model saved     → {final_path}.zip")

    vec_norm_path = save_dir / f"vecnormalize_{suffix}.pkl"
    train_env.save(str(vec_norm_path))
    print(f"[INFO] VecNormalize    → {vec_norm_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
