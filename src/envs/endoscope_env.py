"""
Endoscope Active-Vision Gymnasium Environment
=============================================

This module implements ``EndoscopeEnv``, a custom :class:`gymnasium.Env` that
models an *active-vision* endoscope camera: a crop window that must track a
moving surgical instrument tip across a high-resolution video frame.

The RL agent controls the 2-D velocity (pan/tilt) of the crop window and is
rewarded for keeping the instrument tip centred inside that window.

Environment summary
-------------------
* **State space**: normalised ``(dx, dy)`` offset of the instrument tip relative
  to the crop-window centre — bounded ``[-1, 1]`` per axis.
* **Action space**: ``[delta_x, delta_y]`` velocity commands bounded ``[-1, 1]``,
  scaled internally by ``max_velocity`` to obtain pixel-per-step displacement.
* **Reward**: dense negative Euclidean distance + boundary penalty (if the crop
  window is pushed against the frame edge) + smoothness penalty on action norm.
* **Episode**: one trajectory sequence; terminates when the last frame is reached.

Usage
-----
    >>> import numpy as np
    >>> from src.envs import EndoscopeEnv
    >>> trajectory = np.random.uniform(200, 1720, size=(500, 2))
    >>> env = EndoscopeEnv(trajectory)
    >>> obs, info = env.reset()
    >>> obs_next, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EndoscopeEnv(gym.Env):
    """Active-vision endoscope tracking environment.

    The environment simulates a camera crop window that pans over a full
    high-resolution endoscopic video frame.  At each step the agent issues a
    2-D velocity command that moves the crop centre.  The task is to keep the
    surgical instrument tip inside — and as close to the centre of — the crop
    window as possible.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 2)
        Pre-loaded instrument-tip trajectory in pixel coordinates ``(x, y)``,
        one row per frame.  Use :class:`~src.data.MICCAILoader` to obtain this
        array from MICCAI EndoVis annotation files.
    frame_size : tuple of int, optional
        ``(width, height)`` of the full video frame in pixels.
        Defaults to ``(1920, 1080)``.
    crop_size : tuple of int, optional
        ``(width, height)`` of the crop window in pixels.
        Defaults to ``(400, 400)``.
    max_velocity : float, optional
        Maximum pixel displacement per step along each axis.
        The raw action in ``[-1, 1]`` is multiplied by this value.
        Defaults to ``50.0``.
    boundary_penalty : float, optional
        Scalar penalty added to the reward whenever the crop window reaches
        a frame boundary and is clamped.  Should be negative.
        Defaults to ``-10.0``.
    velocity_penalty_weight : float, optional
        Weight for the action-norm smoothness penalty: ``-w * ||action||``.
        Defaults to ``0.01``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        trajectory: np.ndarray,
        frame_size: Tuple[int, int] = (1920, 1080),
        crop_size: Tuple[int, int] = (400, 400),
        max_velocity: float = 50.0,
        boundary_penalty: float = -10.0,
        velocity_penalty_weight: float = 0.01,
    ) -> None:
        super().__init__()

        # Validate trajectory
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                f"trajectory must have shape (N, 2), got {trajectory.shape}"
            )
        if len(trajectory) < 2:
            raise ValueError("trajectory must contain at least 2 frames.")

        self.trajectory = trajectory.astype(np.float64)
        self.n_frames = len(self.trajectory)

        self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])
        self.crop_w, self.crop_h = int(crop_size[0]), int(crop_size[1])
        self.max_velocity = float(max_velocity)
        self.boundary_penalty = float(boundary_penalty)
        self.velocity_penalty_weight = float(velocity_penalty_weight)

        # Half-crop extents (used for clamping and normalisation)
        self._half_crop_w = self.crop_w / 2.0
        self._half_crop_h = self.crop_h / 2.0

        # Valid range for crop centre so the window never exits the frame
        self._cx_min = self._half_crop_w
        self._cx_max = self.frame_w - self._half_crop_w
        self._cy_min = self._half_crop_h
        self._cy_max = self.frame_h - self._half_crop_h

        # ── Spaces ──────────────────────────────────────────────────────── #
        # Observation: normalised (dx, dy) offset of tool tip w.r.t. crop centre
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: [delta_x, delta_y] velocity command in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (initialised in reset())
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
        """Reset the environment to the start of the trajectory.

        The crop window is initialised near (but with a small random offset from)
        the first instrument-tip position, so the agent must learn to recover
        from imperfect initial alignment.

        Parameters
        ----------
        seed : int, optional
            RNG seed forwarded to the gymnasium parent class.
        options : dict, optional
            Unused; reserved for future extensions.

        Returns
        -------
        observation : np.ndarray, shape (2,)
            Initial normalised ``(dx, dy)`` observation.
        info : dict
            Auxiliary information (same schema as :meth:`step`).
        """
        super().reset(seed=seed)

        self.current_step = 0

        # Place the crop centre at the first tool position + small random jitter
        first_tip = self.trajectory[0]
        jitter_x = self.np_random.uniform(-self._half_crop_w * 0.5, self._half_crop_w * 0.5)
        jitter_y = self.np_random.uniform(-self._half_crop_h * 0.5, self._half_crop_h * 0.5)

        cx = np.clip(first_tip[0] + jitter_x, self._cx_min, self._cx_max)
        cy = np.clip(first_tip[1] + jitter_y, self._cy_min, self._cy_max)
        self.crop_center = np.array([cx, cy], dtype=np.float64)

        obs = self._get_obs()
        info = self._get_info(distance=float(np.linalg.norm(obs)))
        return obs.astype(np.float32), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one frame.

        Parameters
        ----------
        action : np.ndarray, shape (2,)
            Velocity command ``[delta_x, delta_y]`` in ``[-1, 1]``.

        Returns
        -------
        observation : np.ndarray, shape (2,)
            Normalised ``(dx, dy)`` offset after the move.
        reward : float
            Dense reward signal (see class docstring).
        terminated : bool
            ``True`` when the last frame of the trajectory has been reached.
        truncated : bool
            Always ``False`` (no time-limit truncation; use a wrapper if needed).
        info : dict
            ``{"distance": float, "crop_center": list, "tool_tip": list}``
        """
        action = np.asarray(action, dtype=np.float64).flatten()[:2]

        # 1. Scale action to pixel velocity
        velocity = action * self.max_velocity

        # 2. Propose new crop centre
        new_cx = self.crop_center[0] + velocity[0]
        new_cy = self.crop_center[1] + velocity[1]

        # 3. Clamp to valid range; detect boundary hit
        clamped_cx = np.clip(new_cx, self._cx_min, self._cx_max)
        clamped_cy = np.clip(new_cy, self._cy_min, self._cy_max)
        hit_boundary = (clamped_cx != new_cx) or (clamped_cy != new_cy)

        self.crop_center = np.array([clamped_cx, clamped_cy], dtype=np.float64)

        # 4. Advance to next frame
        self.current_step += 1
        terminated = self.current_step >= self.n_frames - 1
        # Use the last valid frame index if we've gone past the end
        frame_idx = min(self.current_step, self.n_frames - 1)
        tool_tip = self.trajectory[frame_idx]  # (tx, ty)

        # 5. Compute offset and normalise
        dx = tool_tip[0] - self.crop_center[0]
        dy = tool_tip[1] - self.crop_center[1]
        # Normalise so that ±1 corresponds to the crop half-width/height
        obs = np.array(
            [dx / self._half_crop_w, dy / self._half_crop_h], dtype=np.float64
        )
        obs = np.clip(obs, -1.0, 1.0)

        # 6. Compute reward
        dist = float(np.sqrt(dx ** 2 + dy ** 2))
        reward = -dist  # dense negative distance
        if hit_boundary:
            reward += self.boundary_penalty
        reward -= self.velocity_penalty_weight * float(np.linalg.norm(action))

        info = self._get_info(distance=dist, tool_tip=tool_tip)

        return obs.astype(np.float32), float(reward), bool(terminated), False, info

    def render(self) -> None:
        """Render the current state.

        A full visual render would require OpenCV.  This stub can be extended
        to draw the frame crop and tool-tip overlay using ``cv2.rectangle`` and
        ``cv2.circle``.
        """
        pass

    def close(self) -> None:
        """Clean up any resources (none allocated in the base implementation)."""
        pass

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """Compute the current normalised (dx, dy) observation."""
        frame_idx = min(self.current_step, self.n_frames - 1)
        tool_tip = self.trajectory[frame_idx]
        dx = tool_tip[0] - self.crop_center[0]
        dy = tool_tip[1] - self.crop_center[1]
        obs = np.array(
            [dx / self._half_crop_w, dy / self._half_crop_h], dtype=np.float64
        )
        return np.clip(obs, -1.0, 1.0)

    def _get_info(
        self,
        distance: float = 0.0,
        tool_tip: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Build the info dictionary returned by :meth:`reset` and :meth:`step`."""
        if tool_tip is None:
            frame_idx = min(self.current_step, self.n_frames - 1)
            tool_tip = self.trajectory[frame_idx]
        return {
            "distance": distance,
            "crop_center": self.crop_center.tolist(),
            "tool_tip": tool_tip.tolist(),
        }


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    print("=== EndoscopeEnv smoke test ===")

    # Build a synthetic sinusoidal trajectory (500 frames, 1920×1080 frame)
    N = 500
    t = np.linspace(0, 4 * math.pi, N)
    traj_x = 960.0 + 400.0 * np.cos(t)          # oscillates around centre
    traj_y = 540.0 + 200.0 * np.sin(2.0 * t)    # figure-8 in y
    trajectory = np.stack([traj_x, traj_y], axis=1)

    env = EndoscopeEnv(
        trajectory=trajectory,
        frame_size=(1920, 1080),
        crop_size=(400, 400),
        max_velocity=50.0,
    )

    obs, info = env.reset(seed=42)
    print(f"Reset observation : {obs}  (shape {obs.shape})")
    print(f"Reset info        : {info}")

    total_reward = 0.0
    step_count = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Episode finished  : {step_count} steps, total reward = {total_reward:.2f}")
    print(f"Final distance    : {info['distance']:.2f} px")
    env.close()
    print("=== Smoke test PASSED ===")
