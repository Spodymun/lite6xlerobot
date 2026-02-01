from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DummyManipulatorEnv(gym.Env):
    """
    Minimal gymnasium env to satisfy LeRobot's env factory.

    Observation: dict with "env" (shape [env_dim]).
    Action: Box (shape [action_dim]).

    It will never be used for real evaluation if eval_freq is huge,
    but must be instantiable.
    """
    metadata = {"render_modes": []}

    def __init__(self, env_dim: int = 9, action_dim: int = 13, episode_length: int = 1, **kwargs):
        super().__init__()
        self.env_dim = int(env_dim)
        self.action_dim = int(action_dim)
        self.episode_length = int(episode_length)
        self._t = 0

        self.observation_space = spaces.Dict({
            "env": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = {"env": np.zeros((self.env_dim,), dtype=np.float32)}
        info = {}
        return obs, info

    def step(self, action):
        self._t += 1
        obs = {"env": np.zeros((self.env_dim,), dtype=np.float32)}
        reward = 0.0
        terminated = self._t >= self.episode_length
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def register_envs():
    # Important: the id LeRobot tries is `gym_gym_manipulator/None`
    gym.register(
        id="gym_gym_manipulator/None",
        entry_point="gym_gym_manipulator.env:DummyManipulatorEnv",
    )
