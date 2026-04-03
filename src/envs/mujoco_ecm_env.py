"""
MuJoCo ECM (Endoscope Camera Manipulator) Environment
======================================================

Physics-based active-vision environment using a simplified da Vinci ECM
arm model.  The agent controls 4 joint velocity commands to keep a moving
surgical instrument tip centred in the endoscope camera's image plane.

This environment sits alongside the video-based EndoscopeEnv and shares the
same reward structure, making it straightforward to compare policies trained
in simulation against policies trained on real video data.

Observation (10-D, float32):
    [0:4]  normalised joint positions  in [-1, 1]  (within joint range)
    [4:8]  normalised joint velocities in [-1, 1]  (within actuator ctrlrange)
    [8:10] (dx, dy) of projected instrument tip on image plane,
           normalised by half-image size -- unbounded, VecNormalize handles scale

Action (4-D, float32):
    Normalised joint velocity commands in [-1, 1].
    Scaled internally to physical units: [0.5, 0.3, 0.05, 1.0] for
    [outer_yaw (rad/s), outer_pitch (rad/s), insertion (m/s), outer_roll (rad/s)].

Reward:
    -distance_px / 100        dense tracking (same shape as EndoscopeEnv)
    + boundary_penalty        if tip projects outside the image frame
    - vel_weight * ||action|| smoothness penalty

Episode terminates when the last frame of the trajectory is reached.

Camera intrinsics match the MICCAI EndoVis calibration:
    fx = fy = 1035 px,  cx = 597 px,  cy = 520 px,  1280x1024 image.

Requirements:
    mujoco >= 3.0.0   (pip install mujoco)
    Python 3.10-3.12 recommended (mujoco wheels may not be available for 3.14+)

Usage:
    >>> from src.envs import MuJoCoECMEnv
    >>> import numpy as np
    >>> traj_3d = np.zeros((200, 3)); traj_3d[:, 2] = -0.66  # static tip
    >>> env = MuJoCoECMEnv(traj_3d)
    >>> obs, info = env.reset()
    >>> print(obs.shape)   # (10,)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

# Default path to the ECM MuJoCo XML model
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_DEFAULT_XML = str(_ASSETS_DIR / "ecm_mujoco.xml")

# Camera intrinsics -- MICCAI EndoVis calibration (M1 matrix from calibration.yaml)
_FX = _FY = 1035.0
_CX = 597.0
_CY = 520.0
_IMG_W = 1280
_IMG_H = 1024

# Physical joint velocity limits: [outer_yaw, outer_pitch, insertion, outer_roll]
# Units: rad/s for hinge joints, m/s for slide joint
_MAX_QVEL = np.array([0.5, 0.3, 0.05, 1.0], dtype=np.float64)

# Default joint positions at reset: insertion at 12 cm, others at zero
_DEFAULT_QPOS = np.array([0.0, 0.0, 0.12, 0.0], dtype=np.float64)


class MuJoCoECMEnv(gym.Env):
    """Physics-based da Vinci ECM camera tracking environment.

    Parameters
    ----------
    trajectory_3d : np.ndarray, shape (N, 3)
        Instrument tip positions in MuJoCo world coordinates (metres), one per
        RL step.  Use :func:`make_synthetic_trajectory_3d` for a synthetic path
        or :func:`miccai_trajectory_to_3d` to lift MICCAI 2D pixel data to 3D.
    n_substeps : int
        Number of MuJoCo physics steps per RL step (controls the effective
        action frequency).  Defaults to ``5`` (40 Hz at timestep=0.005 s).
    boundary_penalty : float
        Reward penalty when the projected tip leaves the image frame.
    velocity_penalty_weight : float
        Weight for the action-norm smoothness penalty.
    render_mode : str or None
        ``"rgb_array"`` to enable camera rendering; ``None`` to disable.
    xml_path : str or None
        Path to the ECM MuJoCo XML model.  Defaults to
        ``assets/ecm_mujoco.xml``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        trajectory_3d: np.ndarray,
        n_substeps: int = 5,
        boundary_penalty: float = -5.0,
        velocity_penalty_weight: float = 0.01,
        render_mode: Optional[str] = None,
        xml_path: Optional[str] = None,
    ) -> None:
        if not _MUJOCO_AVAILABLE:
            raise ImportError(
                "mujoco is not installed.  Run: pip install mujoco>=3.0.0\n"
                "Note: mujoco requires Python 3.10-3.12.  "
                "If you are on Python 3.13+ install Python 3.12 in a venv."
            )
        super().__init__()

        if trajectory_3d.ndim != 2 or trajectory_3d.shape[1] != 3:
            raise ValueError(
                f"trajectory_3d must have shape (N, 3), got {trajectory_3d.shape}"
            )
        if len(trajectory_3d) < 2:
            raise ValueError("trajectory_3d must have at least 2 frames.")

        self.trajectory_3d = trajectory_3d.astype(np.float64)
        self.n_frames = len(trajectory_3d)
        self.n_substeps = int(n_substeps)
        self.boundary_penalty = float(boundary_penalty)
        self.velocity_penalty_weight = float(velocity_penalty_weight)
        self.render_mode = render_mode

        # Load MuJoCo model
        xml = xml_path or _DEFAULT_XML
        if not Path(xml).is_file():
            raise FileNotFoundError(f"ECM MuJoCo XML not found: {xml}")
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)

        # Cache joint indices and limits
        self._n_joints = 4  # outer_yaw, outer_pitch, insertion, outer_roll
        self._qpos_min = self.model.jnt_range[:self._n_joints, 0].copy()
        self._qpos_max = self.model.jnt_range[:self._n_joints, 1].copy()
        self._qpos_range = self._qpos_max - self._qpos_min

        # Camera and mocap IDs
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "endoscope_cam"
        )
        self._mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "instrument_tip"
        )
        # mocap_pos index = mocap body index, not body id
        # Find which mocap index corresponds to this body
        self._mocap_idx = int(self.model.body_mocapid[self._mocap_id])

        # Spaces
        # Obs: [4 qpos_norm] + [4 qvel_norm] + [2 tip_offset_norm] = 10-D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        # Action: 4 normalised joint velocity commands
        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step: int = 0
        self._renderer: Optional[Any] = None

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

        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Initialise joints near default pose with small random jitter
        jitter = self.np_random.uniform(-0.03, 0.03, size=self._n_joints)
        qpos_init = np.clip(
            _DEFAULT_QPOS + jitter,
            self._qpos_min,
            self._qpos_max,
        )
        self.data.qpos[:self._n_joints] = qpos_init
        self.data.qvel[:self._n_joints] = 0.0

        # Place instrument tip at first trajectory point
        self.data.mocap_pos[self._mocap_idx] = self.trajectory_3d[0]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64).flatten()[:4], -1.0, 1.0
        )

        # Scale to physical joint velocity limits and apply to actuators
        self.data.ctrl[:self._n_joints] = action * _MAX_QVEL

        # Advance physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Advance trajectory: move the mocap instrument tip
        self.current_step += 1
        terminated = self.current_step >= self.n_frames - 1
        frame_idx = min(self.current_step, self.n_frames - 1)
        self.data.mocap_pos[self._mocap_idx] = self.trajectory_3d[frame_idx]
        mujoco.mj_forward(self.model, self.data)

        # Compute reward
        dx_px, dy_px, in_front = self._project_tip()
        dist = float(np.sqrt(dx_px ** 2 + dy_px ** 2))
        hit_boundary = (
            abs(dx_px) > _IMG_W / 2
            or abs(dy_px) > _IMG_H / 2
            or not in_front
        )

        reward = -dist / 100.0  # normalise so reward ~ O(1), consistent with EndoscopeEnv
        if hit_boundary:
            reward += self.boundary_penalty
        reward -= self.velocity_penalty_weight * float(np.linalg.norm(action))

        obs = self._get_obs()
        info = self._get_info(dist, in_front)
        return obs, float(reward), bool(terminated), False, info

    def render(self) -> Optional[np.ndarray]:
        """Return 320x256 RGB image from the endoscope camera."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=_IMG_H // 4,
                width=_IMG_W // 4,
            )
        self._renderer.update_scene(self.data, camera="endoscope_cam")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # Camera projection                                                    #
    # ------------------------------------------------------------------ #

    def _project_tip(self) -> Tuple[float, float, bool]:
        """Project the mocap instrument tip onto the camera image plane.

        Returns
        -------
        dx_px : float
            Horizontal offset from image centre (positive = right).
        dy_px : float
            Vertical offset from image centre (positive = down).
        in_front : bool
            Whether the tip is in front of the camera (z_cam < 0).
        """
        tip_world = self.data.mocap_pos[self._mocap_idx].copy()   # (3,)
        cam_pos   = self.data.cam_xpos[self._cam_id].copy()       # (3,)
        # cam_xmat rows = camera x, y, z axes expressed in world coordinates
        cam_mat   = self.data.cam_xmat[self._cam_id].reshape(3, 3)

        # Transform tip into camera frame
        # cam_mat @ (P_world - cam_pos) gives [x_cam, y_cam, z_cam]
        # Camera looks in local -Z, so in_front iff z_cam < 0
        tip_cam = cam_mat @ (tip_world - cam_pos)

        in_front = bool(tip_cam[2] < -1e-4)
        if not in_front:
            # Return large off-screen offsets so the reward pushes back
            return float(_IMG_W), float(_IMG_H), False

        depth = -tip_cam[2]   # positive depth (metres)

        # Standard pinhole projection with MICCAI intrinsics
        # Image convention: y=0 at top, increases downward
        #   px = fx * x_cam / depth + cx
        #   py = cy - fy * y_cam / depth  (y_cam up, image y down)
        px = _FX * tip_cam[0] / depth + _CX
        py = _CY - _FY * tip_cam[1] / depth

        dx_px = float(px - _CX)
        dy_px = float(py - _CY)
        return dx_px, dy_px, True

    # ------------------------------------------------------------------ #
    # Observation / info helpers                                           #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        # Normalised joint positions: map [qpos_min, qpos_max] -> [-1, 1]
        qpos = self.data.qpos[:self._n_joints]
        qpos_norm = 2.0 * (qpos - self._qpos_min) / self._qpos_range - 1.0

        # Normalised joint velocities: divide by max velocity limits
        qvel_norm = self.data.qvel[:self._n_joints] / _MAX_QVEL

        # Projected tip offset normalised by half-image size (unbounded)
        dx, dy, _ = self._project_tip()
        tip_norm = np.array([dx / (_IMG_W / 2.0), dy / (_IMG_H / 2.0)])

        obs = np.concatenate([qpos_norm, qvel_norm, tip_norm])
        return obs.astype(np.float32)

    def _get_info(
        self,
        distance_px: float = 0.0,
        in_front: bool = True,
    ) -> Dict[str, Any]:
        if distance_px == 0.0:
            dx, dy, in_front = self._project_tip()
            distance_px = float(np.sqrt(dx ** 2 + dy ** 2))
        return {
            "distance_px": distance_px,
            "in_front": in_front,
            "qpos": self.data.qpos[:self._n_joints].tolist(),
            "qvel": self.data.qvel[:self._n_joints].tolist(),
        }


# --------------------------------------------------------------------------- #
# Trajectory utilities                                                         #
# --------------------------------------------------------------------------- #

def make_synthetic_trajectory_3d(
    xml_path: Optional[str] = None,
    n_frames: int = 500,
    depth: float = 0.15,
    amplitude_lateral: float = 0.02,
    amplitude_vertical: float = 0.015,
) -> np.ndarray:
    """Generate a Lissajous 3D trajectory in front of the ECM camera.

    The trajectory is computed relative to the camera's actual world position
    at the default joint configuration, so it is always visible regardless of
    the exact ECM kinematic chain.

    Parameters
    ----------
    xml_path : str or None
        Path to ECM XML.  Defaults to ``assets/ecm_mujoco.xml``.
    n_frames : int
        Number of trajectory points.
    depth : float
        Distance in front of camera (metres).
    amplitude_lateral, amplitude_vertical : float
        Amplitudes of lateral and vertical sinusoidal motion (metres).

    Returns
    -------
    np.ndarray, shape (n_frames, 3)
        Instrument tip positions in MuJoCo world coordinates.
    """
    if not _MUJOCO_AVAILABLE:
        raise ImportError("mujoco is required to generate 3D trajectories.")

    xml = xml_path or _DEFAULT_XML
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    # Evaluate at default joint positions
    data.qpos[:4] = _DEFAULT_QPOS
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "endoscope_cam")
    cam_pos = data.cam_xpos[cam_id].copy()                     # (3,)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    cam_right   =  cam_mat[0]   # camera x-axis in world
    cam_up      =  cam_mat[1]   # camera y-axis in world
    cam_forward = -cam_mat[2]   # camera looks in -z; forward = -z_cam axis in world

    t = np.linspace(0, 4 * np.pi, n_frames)
    center = cam_pos + depth * cam_forward  # (3,)

    traj = (
        center[None, :]
        + amplitude_lateral  * np.outer(np.cos(t),      cam_right)
        + amplitude_vertical * np.outer(np.sin(2.0 * t), cam_up)
    )
    return traj.astype(np.float64)


def miccai_trajectory_to_3d(
    traj_2d: np.ndarray,
    xml_path: Optional[str] = None,
    assumed_depth: float = 0.15,
) -> np.ndarray:
    """Back-project a MICCAI 2D pixel trajectory to 3D world coordinates.

    Uses the MICCAI EndoVis camera intrinsics and places the resulting 3D path
    in front of the ECM camera's default-pose position.  This bridges the
    video-based and physics-based environments: you can train the ECM agent on
    instrument motion extracted from real surgical video.

    Parameters
    ----------
    traj_2d : np.ndarray, shape (N, 2)
        Pixel-coordinate trajectory as returned by
        :meth:`~src.data.MICCAILoader.load_trajectory`.
    xml_path : str or None
        Path to ECM XML.  Defaults to ``assets/ecm_mujoco.xml``.
    assumed_depth : float
        Assumed depth of the instrument from the camera (metres).
        Typical laparoscopic distance: 0.10 - 0.20 m.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Instrument tip positions in MuJoCo world coordinates.
    """
    if not _MUJOCO_AVAILABLE:
        raise ImportError("mujoco is required.")

    xml = xml_path or _DEFAULT_XML
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    data.qpos[:4] = _DEFAULT_QPOS
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "endoscope_cam")
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    cam_right   =  cam_mat[0]
    cam_up      =  cam_mat[1]
    cam_forward = -cam_mat[2]

    # Normalised image coordinates (principal-point-relative)
    x_norm = (traj_2d[:, 0] - _CX) / _FX   # (N,)  positive = right
    y_norm = (traj_2d[:, 1] - _CY) / _FY   # (N,)  positive = down in image

    # Back-project: P = cam_pos + d*(cam_forward + x*cam_right - y*cam_up)
    # y is negated because image-y (down) is opposite to cam_up (up)
    center = cam_pos + assumed_depth * cam_forward
    traj_3d = (
        center[None, :]
        + assumed_depth * np.outer(x_norm,  cam_right)
        - assumed_depth * np.outer(y_norm,  cam_up)
    )
    return traj_3d.astype(np.float64)


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    if not _MUJOCO_AVAILABLE:
        print("[ERROR] mujoco is not installed.  pip install mujoco>=3.0.0")
        sys.exit(1)

    print("=== MuJoCoECMEnv smoke test ===")

    traj = make_synthetic_trajectory_3d(n_frames=100)
    print(f"Trajectory shape  : {traj.shape}")
    print(f"Tip X range       : [{traj[:,0].min():.4f}, {traj[:,0].max():.4f}] m")
    print(f"Tip Z range       : [{traj[:,2].min():.4f}, {traj[:,2].max():.4f}] m")

    env = MuJoCoECMEnv(traj)
    obs, info = env.reset(seed=42)
    print(f"Reset obs shape   : {obs.shape}  dtype={obs.dtype}")
    print(f"Reset info        : {info}")

    total_reward = 0.0
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"After 30 steps    : total reward = {total_reward:.3f}")
    print(f"Final dist        : {info['distance_px']:.1f} px")
    env.close()
    print("=== Smoke test PASSED ===")
