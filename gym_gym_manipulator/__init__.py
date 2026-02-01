"""
Minimal stub package so LeRobot can import `gym_gym_manipulator`
and create gym env `gym_gym_manipulator/None`.

This is only needed because lerobot-train always instantiates an eval env.
"""
from .env import register_envs
register_envs()
