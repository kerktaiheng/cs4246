from __future__ import annotations

from typing import Any


class Trainer:
    """Placeholder trainer that can be extended with PPO/SAC/Nash-Q."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def train(self, steps: int = 10) -> dict[str, float]:
        obs = self.env.reset()
        rewards = []
        for _ in range(steps):
            action = 0.0
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        return {"steps": float(len(rewards)), "reward": float(sum(rewards))}
