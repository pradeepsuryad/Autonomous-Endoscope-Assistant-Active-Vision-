"""
MuJoCo ECM Training Script -- Autonomous Endoscope Assistant
=============================================================

Trains a SAC agent to control the 4 joints of a da Vinci ECM
(Endoscope Camera Manipulator) arm to track a moving instrument tip.

Runs entirely on CPU -- no GPU required for physics simulation.
GPU (friend's laptop) recommended only if you add rendered image observations.

Modes
-----
Synthetic trajectory (default, no data needed):
    python scripts/train_mujoco.py

Physics sim trained on MICCAI video trajectories (bridges both components):
    python scripts/train_mujoco.py \
        --case_dir "C:/Users/dprad/Downloads/train/train/case_1/1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm as tqdm_lib

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.envs.mujoco_ecm_env import (
        MuJoCoECMEnv,
        make_synthetic_trajectory_3d,
        miccai_trajectory_to_3d,
    )
    _MUJOCO_OK = True
except ImportError as _e:
    _MUJOCO_OK = False
    _MUJOCO_ERR = str(_e)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_DEFAULT_XML = str(_REPO_ROOT / "assets" / "ecm_mujoco.xml")


# --------------------------------------------------------------------------- #
# tqdm progress callback (same design as train.py)                            #
# --------------------------------------------------------------------------- #

class TrainingProgressCallback(BaseCallback):
    """ASCII-safe tqdm progress bar with ETA, mean distance, and mean reward."""

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
            desc="SAC ECM Training",
            unit="step",
            unit_scale=True,
            ncols=110,
            ascii=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

    def _on_step(self) -> bool:
        self._pbar.update(1)
        for info in self.locals.get("infos", []):
            if "distance_px" in info:
                self._distances.append(float(info["distance_px"]))
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._rewards.extend(
                rewards.tolist() if hasattr(rewards, "tolist") else [float(rewards)]
            )
        if self.num_timesteps % self.log_interval == 0:
            win = self.log_interval
            postfix: dict = {}
            if self._distances:
                postfix["dist"] = f"{np.mean(self._distances[-win:]):.0f}px"
            if self._rewards:
                postfix["rew"] = f"{np.mean(self._rewards[-win:]):.3f}"
            if postfix:
                self._pbar.set_postfix(postfix)
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            if self._distances:
                print(
                    f"\n[SUMMARY] Mean dist (last 1k): "
                    f"{np.mean(self._distances[-1000:]):.1f} px  |  "
                    f"Mean reward: {np.mean(self._rewards[-1000:]):.4f}"
                )
            self._pbar.close()


# --------------------------------------------------------------------------- #
# Argument parser                                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SAC on the MuJoCo ECM active-vision task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--case_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "MICCAI sequence directory containing video.mp4.  "
            "If provided, the 2D pixel trajectory is back-projected to 3D "
            "and used as the instrument tip path in the physics sim.  "
            "Omit to use a synthetic Lissajous trajectory."
        ),
    )
    p.add_argument(
        "--assumed_depth",
        type=float,
        default=0.15,
        help="Assumed instrument depth from camera in metres (for MICCAI bridge).",
    )
    p.add_argument("--stereo_half",      type=str,   default="top", choices=["top","bottom"])
    p.add_argument("--total_timesteps",  type=int,   default=500_000)
    p.add_argument("--save_dir",         type=str,   default="models/mujoco/")
    p.add_argument("--eval_freq",        type=int,   default=5_000)
    p.add_argument("--n_eval_episodes",  type=int,   default=5)
    p.add_argument("--learning_rate",    type=float, default=3e-4)
    p.add_argument("--buffer_size",      type=int,   default=100_000)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--n_substeps",       type=int,   default=5,
                   help="Physics substeps per RL step (controls action frequency).")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Environment factory                                                          #
# --------------------------------------------------------------------------- #

def make_env(trajectory_3d: np.ndarray, n_substeps: int, seed: int):
    def _init():
        env = MuJoCoECMEnv(
            trajectory_3d=trajectory_3d,
            n_substeps=n_substeps,
            boundary_penalty=-5.0,
            velocity_penalty_weight=0.01,
        )
        env.reset(seed=seed)
        return env
    return _init


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    if not _MUJOCO_OK:
        print(f"[ERROR] mujoco is not available: {_MUJOCO_ERR}")
        print("[ERROR] Install with: pip install mujoco>=3.0.0")
        print("[ERROR] If on Python 3.13+, create a Python 3.12 venv first.")
        sys.exit(1)

    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    GAMMA = 0.95

    # ── 1. Build 3D trajectory ─────────────────────────────────────────── #
    if args.case_dir is not None:
        case_path = Path(args.case_dir)
        if not case_path.is_dir():
            print(f"[ERROR] case_dir '{case_path}' is not a directory.")
            sys.exit(1)

        print(f"[INFO] Loading MICCAI trajectory from: {case_path}")
        from src.data.miccai_loader import MICCAILoader
        loader = MICCAILoader(stereo_half=args.stereo_half, smooth_sigma=3.0)
        traj_2d = loader.load_trajectory(case_path)
        print(f"[INFO]   -> {len(traj_2d)} frames (2D pixel trajectory)")

        trajectory = miccai_trajectory_to_3d(
            traj_2d=traj_2d,
            xml_path=_DEFAULT_XML,
            assumed_depth=args.assumed_depth,
        )
        print(
            f"[INFO]   -> back-projected to 3D at depth={args.assumed_depth:.2f} m"
        )
        print(
            f"[INFO]   -> 3D X range: [{trajectory[:,0].min():.4f}, "
            f"{trajectory[:,0].max():.4f}] m"
        )
    else:
        print("[INFO] No --case_dir provided -- using synthetic 3D trajectory.")
        trajectory = make_synthetic_trajectory_3d(
            xml_path=_DEFAULT_XML,
            n_frames=1_000,
        )

    print(f"[INFO] Trajectory : {trajectory.shape}  (N x 3 world coords, metres)")

    # ── 2. Vectorised envs ────────────────────────────────────────────── #
    train_vec = DummyVecEnv([make_env(trajectory, args.n_substeps, args.seed)])
    train_env = VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    eval_vec = DummyVecEnv([make_env(trajectory, args.n_substeps, args.seed + 100)])
    eval_env = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,
        training=False,
        gamma=GAMMA,
    )
    # Sync normalisation stats so eval uses same observation scaling as train
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    # ── 3. SAC model ──────────────────────────────────────────────────── #
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ent_coef=0.01,
        gamma=GAMMA,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
        verbose=0,
        seed=args.seed,
        device="cpu",   # physics sim runs on CPU; force CPU for consistency
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

    # ── 5. Train ──────────────────────────────────────────────────────── #
    print(f"\n[INFO] Starting SAC training for {args.total_timesteps:,} steps (CPU) ...\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([progress_cb, eval_cb]),
        progress_bar=False,
    )

    # ── 6. Save ────────────────────────────────────────────────────────── #
    final_path = save_dir / "sac_ecm_final"
    model.save(str(final_path))
    print(f"[INFO] Model saved      -> {final_path}.zip")

    vec_norm_path = save_dir / "vecnormalize_ecm.pkl"
    train_env.save(str(vec_norm_path))
    print(f"[INFO] VecNormalize     -> {vec_norm_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
