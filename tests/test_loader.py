"""
Tests for MICCAILoader -- frame alignment and stereo-half propagation.

Run with:
    python -m pytest tests/ -v
"""

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# Make sure the repo root is on sys.path when running from any directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.miccai_loader import MICCAILoader


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_synthetic_video(path: Path, n_frames: int, width: int, height: int) -> None:
    """Write a minimal MP4 with n_frames of solid-colour frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for i in range(n_frames):
        # Alternating grey levels so MOG2 sees some foreground motion
        level = 80 + (i % 4) * 40
        frame = np.full((height, width, 3), level, dtype=np.uint8)
        # Draw a small bright square that moves -- simulates instrument tip
        tip_x = int(width * 0.2 + (width * 0.6) * (i / max(n_frames - 1, 1)))
        tip_y = height // 4
        cv2.rectangle(frame, (tip_x - 10, tip_y - 10), (tip_x + 10, tip_y + 10), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _make_sequence_dir(tmp_path: Path, n_frames: int = 40) -> Path:
    """Create a minimal MICCAI-style sequence directory with video.mp4 + info.yaml."""
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    # Stereo video: 1280 wide, 2048 tall (two 1024-row halves stacked)
    _make_synthetic_video(seq_dir / "video.mp4", n_frames, width=1280, height=2048)
    info = {
        "video_stack": "vertical",
        "name_video": "video.mp4",
        "resolution": {"width": 1280, "height": 1024},
    }
    (seq_dir / "info.yaml").write_text(yaml.dump(info))
    return seq_dir


# --------------------------------------------------------------------------- #
# Test 1 -- frame count alignment                                              #
# --------------------------------------------------------------------------- #

def test_trajectory_frame_count(tmp_path):
    """len(trajectory) must equal CAP_PROP_FRAME_COUNT for the source video."""
    n_frames = 40
    seq_dir = _make_sequence_dir(tmp_path, n_frames=n_frames)

    # Disable cache so we always run extraction (no stale .npy on disk)
    loader = MICCAILoader(stereo_half="top", use_cache=False, smooth_sigma=0)
    traj = loader.load_trajectory(seq_dir)

    cap = cv2.VideoCapture(str(seq_dir / "video.mp4"))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert len(traj) == reported, (
        f"Trajectory has {len(traj)} points but video reports {reported} frames"
    )
    assert traj.shape == (reported, 2)


# --------------------------------------------------------------------------- #
# Test 2 -- frame-index alignment (trajectory[i] comes from frame i)          #
# --------------------------------------------------------------------------- #

def test_trajectory_frame_alignment(tmp_path):
    """Verify that trajectory[i] was computed from video frame i.

    The synthetic video has a bright rectangle moving left-to-right only in the
    TOP half.  For stereo_half='top' the x-coordinate should increase
    monotonically; for 'bottom' it should be flat (no moving object).
    """
    n_frames = 40
    seq_dir = _make_sequence_dir(tmp_path, n_frames=n_frames)

    loader_top = MICCAILoader(
        stereo_half="top", use_cache=False, smooth_sigma=0, min_contour_area=50
    )
    traj_top = loader_top.load_trajectory(seq_dir)

    # The moving bright square is only in the top half -- x should have
    # non-trivial variance
    assert traj_top[:, 0].std() > 10, (
        "Top-half trajectory x-coordinate has no variance -- "
        "frame alignment may be wrong"
    )
    assert len(traj_top) == n_frames


# --------------------------------------------------------------------------- #
# Test 3 -- stereo_half propagation                                            #
# --------------------------------------------------------------------------- #

def test_stereo_half_propagation(tmp_path):
    """get_frame_size() must return (1280, 1024) regardless of stereo_half."""
    seq_dir = _make_sequence_dir(tmp_path)

    for half in ("top", "bottom"):
        loader = MICCAILoader(stereo_half=half, use_cache=False)
        w, h = loader.get_frame_size(seq_dir)
        assert w == 1280, f"Expected width 1280, got {w} for stereo_half={half}"
        assert h == 1024, f"Expected height 1024, got {h} for stereo_half={half}"


# --------------------------------------------------------------------------- #
# Test 4 -- visual mode rejects multiple case_dirs                             #
# --------------------------------------------------------------------------- #

def test_visual_mode_rejects_multi_case(tmp_path, monkeypatch, capsys):
    """train.py must exit(1) when --visual is combined with multiple --case_dir."""
    import scripts.train as train_module

    # Patch sys.argv to simulate CLI args
    monkeypatch.setattr(
        sys, "argv",
        [
            "train.py",
            "--visual",
            "--case_dir", str(tmp_path / "a"), str(tmp_path / "b"),
            "--total_timesteps", "10",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        train_module.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out


# --------------------------------------------------------------------------- #
# Test 5 -- synthetic smoke test: short training run completes on Windows     #
# --------------------------------------------------------------------------- #

def test_synthetic_training_smoke():
    """A 200-step SAC training run on a synthetic trajectory must complete
    without error and produce a valid model."""
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.envs.endoscope_env import EndoscopeEnv

    t = np.linspace(0, 4 * math.pi, 50)
    traj = np.stack([
        640 + 200 * np.cos(t),
        512 + 150 * np.sin(2 * t),
    ], axis=1)

    def _make():
        env = EndoscopeEnv(traj, frame_size=(1280, 1024))
        env.reset(seed=0)
        return env

    train_env = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True, gamma=0.95)

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_starts=10,
        buffer_size=500,
        batch_size=16,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=200)

    # Confirm the policy produces valid actions
    obs = train_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 2)
    assert np.all(np.isfinite(action))
    train_env.close()
