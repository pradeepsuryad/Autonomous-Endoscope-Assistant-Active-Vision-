"""
Custom Gymnasium environments for the active-vision endoscope RL project.

Environments
------------
EndoscopeEnv        -- video-based, state obs (dx,dy), MlpPolicy
EndoscopeVisualEnv  -- video-based, image obs 84x84 RGB, CnnPolicy
MuJoCoECMEnv        -- physics-based, 4-DOF ECM arm, MlpPolicy
                       (requires mujoco>=3.0.0 and Python 3.10-3.12)
"""
from .endoscope_env import EndoscopeEnv
from .endoscope_visual_env import EndoscopeVisualEnv

# MuJoCo env is optional -- graceful import failure when mujoco not installed
try:
    from .mujoco_ecm_env import MuJoCoECMEnv
    __all__ = ["EndoscopeEnv", "EndoscopeVisualEnv", "MuJoCoECMEnv"]
except ImportError:
    __all__ = ["EndoscopeEnv", "EndoscopeVisualEnv"]
