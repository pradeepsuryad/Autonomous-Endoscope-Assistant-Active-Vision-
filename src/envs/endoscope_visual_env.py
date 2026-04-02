"""
Endoscope Visual Gymnasium Environment
=======================================

An image-based variant of :class:`~src.envs.EndoscopeEnv` where the agent
observes the **actual pixel crop** from the surgical video rather than a
pre-computed (dx, dy) offset.

This is a more realistic formulation: a real endoscope controller receives
raw camera images, not ground-truth instrument offsets.  A CNN policy
(e.g. SB3's ``CnnPolicy`` with ``NatureCNN``) learns to look at the pixels
and decide where to pan the camera.

Environment summary
-------------------
* **Observation space**: ``Box(0, 255, shape=(84, 84, 3), dtype=uint8)`` —
  an 84×84 RGB crop of the surgical video centred on the current crop window.
* **Action space**: ``Box(-1, 1, shape=(2,))`` — ``[delta_x, delta_y]``
  velocity command scaled internally by ``max_velocity``.
* **Reward**: identical to :class:`EndoscopeEnv` — dense negative distance +
  boundary penalty + smoothness penalty.
* **Episode**: one video sequence; terminates at the last frame.

Usage
-----
    >>> from src.envs import EndoscopeVisualEnv
    >>> from src.data import MICCAILoader
    >>> loader = MICCAILoader()
    >>> traj = loader.load_trajectory("data/case_1/1")
    >>> env = EndoscopeVisualEnv(
    ...     video_path="data/case_1/1/video.mp4",
    ...     trajectory=traj,
    ...     stereo_half="top",
    ... )
    >>> obs, info = env.reset()
    >>> print(obs.shape)   # (84, 84, 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EndoscopeVisualEnv(gym.Env):
    """Image-observation endoscope tracking environment.

    At every step the agent receives an ``(84, 84, 3)`` uint8 RGB image —
    the region of the surgical video currently visible through the crop window.
    The instrument tip position is **not** given directly; the agent must learn
    to detect it from pixels and keep it centred.

    Parameters
    ----------
    video_path : str or Path
        Path to the ``video.mp4`` for this sequence.
    trajectory : np.ndarray, shape (N, 2)
        Instrument-tip trajectory in single-view pixel coordinates, as
        returned by :meth:`~src.data.MICCAILoader.load_trajectory`.
        Used to compute rewards (ground-truth tip position per frame).
    stereo_half : str
        ``"top"`` (left camera, default) or ``"bottom"`` (right camera).
    frame_size : tuple of int
        ``(width, height)`` of the single stereo view in pixels.
        Defaults to ``(1280, 512)`` for the MICCAI EndoVis dataset.
    crop_size : tuple of int
        ``(width, height)`` of the moving crop window in pixels.
        Defaults to ``(400, 400)``.
    obs_size : int
        Side length (pixels) of the square observation image fed to the CNN.
        Defaults to ``84`` (standard for ``NatureCNN``).
    max_velocity : float
        Maximum pixel displacement per step.  Defaults to ``50.0``.
    boundary_penalty : float
        Penalty when the crop window hits the frame edge.  Defaults to ``-10.0``.
    velocity_penalty_weight : float
        Weight for the action-norm smoothness penalty.  Defaults to ``0.01``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        video_path: str | Path,
        trajectory: np.ndarray,
        stereo_half: str = "top",
        frame_size: Tuple[int, int] = (1280, 512),
        crop_size: Tuple[int, int] = (400, 400),
        obs_size: int = 84,
        max_velocity: float = 50.0,
        boundary_penalty: float = -10.0,
        velocity_penalty_weight: float = 0.01,
    ) -> None:
        super().__init__()

        self.video_path = str(video_path)
        if not Path(self.video_path).is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(f"trajectory must have shape (N, 2), got {trajectory.shape}")
        if len(trajectory) < 2:
            raise ValueError("trajectory must contain at least 2 frames.")

        self.trajectory = trajectory.astype(np.float64)
        self.n_frames = len(self.trajectory)
        self.stereo_half = stereo_half

        self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])
        self.crop_w, self.crop_h = int(crop_size[0]), int(crop_size[1])
        self.obs_size = int(obs_size)
        self.max_velocity = float(max_velocity)
        self.boundary_penalty = float(boundary_penalty)
        self.velocity_penalty_weight = float(velocity_penalty_weight)

        self._half_crop_w = self.crop_w / 2.0
        self._half_crop_h = self.crop_h / 2.0

        # Valid range for crop centre
        self._cx_min = self._half_crop_w
        self._cx_max = self.frame_w - self._half_crop_w
        self._cy_min = self._half_crop_h
        self._cy_max = self.frame_h - self._half_crop_h

        # Spaces
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size, self.obs_size, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Runtime state (initialised in reset())
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.crop_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.current_step: int = 0

    # ------------------------------------------------------------------ #
    # gymnasium.Env interface                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # (Re-)open the video and seek to frame 0
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {self.video_path}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.current_step = 0

        # Initialise crop centre near the first tool position + jitter
        first_tip = self.trajectory[0]
        jitter_x = self.np_random.uniform(-self._half_crop_w * 0.5, self._half_crop_w * 0.5)
        jitter_y = self.np_random.uniform(-self._half_crop_h * 0.5, self._half_crop_h * 0.5)
        cx = float(np.clip(first_tip[0] + jitter_x, self._cx_min, self._cx_max))
        cy = float(np.clip(first_tip[1] + jitter_y, self._cy_min, self._cy_max))
        self.crop_center = np.array([cx, cy], dtype=np.float64)

        # Read the first video frame
        ret, raw = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read first frame from {self.video_path}")
        self.current_frame = self._decode_frame(raw)

        obs = self._get_obs()
        tool_tip = self.trajectory[0]
        dist = float(np.linalg.norm(tool_tip - self.crop_center))
        return obs, self._build_info(dist, tool_tip)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).flatten()[:2]

        # 1. Move crop window
        velocity = action * self.max_velocity
        new_cx = self.crop_center[0] + velocity[0]
        new_cy = self.crop_center[1] + velocity[1]
        clamped_cx = float(np.clip(new_cx, self._cx_min, self._cx_max))
        clamped_cy = float(np.clip(new_cy, self._cy_min, self._cy_max))
        hit_boundary = (clamped_cx != new_cx) or (clamped_cy != new_cy)
        self.crop_center = np.array([clamped_cx, clamped_cy], dtype=np.float64)

        # 2. Advance to next video frame
        self.current_step += 1
        ret, raw = self.cap.read()
        if ret:
            self.current_frame = self._decode_frame(raw)
        # If video ends before trajectory, keep the last decoded frame

        terminated = self.current_step >= self.n_frames - 1
        frame_idx = min(self.current_step, self.n_frames - 1)
        tool_tip = self.trajectory[frame_idx]

        # 3. Reward
        dx = tool_tip[0] - self.crop_center[0]
        dy = tool_tip[1] - self.crop_center[1]
        dist = float(np.sqrt(dx ** 2 + dy ** 2))
        reward = -dist
        if hit_boundary:
            reward += self.boundary_penalty
        reward -= self.velocity_penalty_weight * float(np.linalg.norm(action))

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), False, self._build_info(dist, tool_tip)

    def render(self) -> Optional[np.ndarray]:
        """Return the current crop as an RGB array (84×84×3)."""
        if self.current_frame is None:
            return None
        return self._get_obs()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _decode_frame(self, raw: np.ndarray) -> np.ndarray:
        """Crop stereo half and convert BGR → RGB."""
        h = raw.shape[0]
        half_h = h // 2
        half = raw[:half_h] if self.stereo_half == "top" else raw[half_h:]
        return cv2.cvtColor(half, cv2.COLOR_BGR2RGB)

    def _get_obs(self) -> np.ndarray:
        """Extract the crop window from the current frame and resize to obs_size."""
        cx = int(round(self.crop_center[0]))
        cy = int(round(self.crop_center[1]))

        x1 = max(0, cx - self.crop_w // 2)
        y1 = max(0, cy - self.crop_h // 2)
        x2 = min(self.frame_w, x1 + self.crop_w)
        y2 = min(self.frame_h, y1 + self.crop_h)

        crop = self.current_frame[y1:y2, x1:x2]

        # Pad to exact crop_size if the window is at a boundary
        if crop.shape[0] != self.crop_h or crop.shape[1] != self.crop_w:
            padded = np.zeros((self.crop_h, self.crop_w, 3), dtype=np.uint8)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded

        # Resize to (obs_size, obs_size) for the CNN
        resized = cv2.resize(
            crop, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA
        )
        return resized  # (obs_size, obs_size, 3) uint8

    def _build_info(
        self, distance: float, tool_tip: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "distance": distance,
            "crop_center": self.crop_center.tolist(),
            "tool_tip": tool_tip.tolist(),
        }


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import math

    if len(sys.argv) < 2:
        print("Usage: python endoscope_visual_env.py <sequence_dir>")
        print("       e.g. python endoscope_visual_env.py C:/Users/dprad/Downloads/train/train/case_1/1")
        sys.exit(1)

    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
    from src.data.miccai_loader import MICCAILoader

    seq_dir = sys.argv[1]
    loader = MICCAILoader(stereo_half="top")
    traj = loader.load_trajectory(seq_dir)
    w, h = loader.get_frame_size(seq_dir)

    env = EndoscopeVisualEnv(
        video_path=str(_Path(seq_dir) / "video.mp4"),
        trajectory=traj,
        stereo_half="top",
        frame_size=(w, h),
    )

    obs, info = env.reset(seed=42)
    print(f"Observation shape : {obs.shape}  dtype={obs.dtype}")
    print(f"Obs pixel range   : [{obs.min()}, {obs.max()}]")
    print(f"Reset info        : {info}")

    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"After 20 steps    : total reward = {total_reward:.2f}")
    print(f"Final distance    : {info['distance']:.1f} px")
    env.close()
    print("=== Visual env smoke test PASSED ===")
