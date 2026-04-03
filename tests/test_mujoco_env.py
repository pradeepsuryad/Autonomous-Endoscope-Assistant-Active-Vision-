"""
Tests for MuJoCoECMEnv.

Skipped automatically when mujoco is not installed (e.g. Python 3.13+).

Run with:
    python -m pytest tests/test_mujoco_env.py -v
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip entire module if mujoco is unavailable
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from src.envs.mujoco_ecm_env import (
    MuJoCoECMEnv,
    make_synthetic_trajectory_3d,
    miccai_trajectory_to_3d,
    _DEFAULT_XML,
    _IMG_W,
    _IMG_H,
)

XML = _DEFAULT_XML


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _simple_traj(n=50):
    """Tiny synthetic trajectory for fast test runs."""
    return make_synthetic_trajectory_3d(xml_path=XML, n_frames=n)


# --------------------------------------------------------------------------- #
# Test 1 -- reset returns correct obs shape and dtype                         #
# --------------------------------------------------------------------------- #

def test_reset_obs_shape():
    traj = _simple_traj()
    env = MuJoCoECMEnv(traj, xml_path=XML)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert "distance_px" in info
    assert "qpos" in info
    env.close()


# --------------------------------------------------------------------------- #
# Test 2 -- step returns correct shapes and finite values                      #
# --------------------------------------------------------------------------- #

def test_step_output_shapes():
    traj = _simple_traj()
    env = MuJoCoECMEnv(traj, xml_path=XML)
    env.reset(seed=1)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs)), "obs contains non-finite values"
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


# --------------------------------------------------------------------------- #
# Test 3 -- episode terminates exactly at trajectory end                      #
# --------------------------------------------------------------------------- #

def test_episode_terminates_at_trajectory_end():
    n = 20
    traj = _simple_traj(n=n)
    env = MuJoCoECMEnv(traj, xml_path=XML)
    env.reset(seed=2)

    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not truncated, "Episode should not be truncated"
        assert steps <= n + 5, "Episode ran longer than trajectory"

    # Should terminate exactly at n-1 steps
    assert steps == n - 1, f"Expected {n-1} steps, got {steps}"
    env.close()


# --------------------------------------------------------------------------- #
# Test 4 -- synthetic trajectory is in front of camera                        #
# --------------------------------------------------------------------------- #

def test_synthetic_trajectory_visible():
    """All points of the synthetic trajectory should project inside the image."""
    traj = make_synthetic_trajectory_3d(xml_path=XML, n_frames=100)
    assert traj.shape == (100, 3)

    env = MuJoCoECMEnv(traj, xml_path=XML)
    env.reset(seed=3)

    for i in range(min(30, len(traj) - 1)):
        obs, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["in_front"], f"Tip not in front of camera at step {i}"
        if terminated:
            break
    env.close()


# --------------------------------------------------------------------------- #
# Test 5 -- MICCAI 2D -> 3D bridge preserves frame count                     #
# --------------------------------------------------------------------------- #

def test_miccai_bridge_frame_count():
    n = 80
    t = np.linspace(0, 4 * math.pi, n)
    traj_2d = np.stack([
        640 + 200 * np.cos(t),
        512 + 150 * np.sin(t),
    ], axis=1)

    traj_3d = miccai_trajectory_to_3d(traj_2d, xml_path=XML)
    assert traj_3d.shape == (n, 3), f"Expected ({n}, 3), got {traj_3d.shape}"
    assert np.all(np.isfinite(traj_3d))


# --------------------------------------------------------------------------- #
# Test 6 -- short SAC smoke test on MuJoCo env (CPU)                         #
# --------------------------------------------------------------------------- #

def test_mujoco_sac_smoke():
    """200-step SAC training on MuJoCoECMEnv must complete and produce valid actions."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    traj = make_synthetic_trajectory_3d(xml_path=XML, n_frames=50)

    def _make():
        env = MuJoCoECMEnv(traj, xml_path=XML)
        env.reset(seed=0)
        return env

    train_env = VecNormalize(
        DummyVecEnv([_make]),
        norm_obs=True, norm_reward=True, gamma=0.95,
    )
    model = SAC(
        "MlpPolicy", train_env,
        learning_starts=10, buffer_size=500, batch_size=16,
        verbose=0, seed=0, device="cpu",
    )
    model.learn(total_timesteps=200)

    obs = train_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 4)
    assert np.all(np.isfinite(action))
    train_env.close()
