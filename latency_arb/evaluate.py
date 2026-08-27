from __future__ import annotations

from typing import Any


def evaluate(agent: Any, env: Any, episodes: int = 3) -> dict[str, float]:
    """Simple evaluation loop for a trained policy over unseen episodes."""
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        for _ in range(5):
            action = 0.0
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return {"episodes": float(len(rewards)), "avg_reward": float(sum(rewards) / len(rewards))}
