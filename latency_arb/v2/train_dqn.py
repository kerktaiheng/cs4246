"""@file train_dqn.py
@brief Double DQN on the same fee-conditioned semi-MDP as the version 2 PPO agent.
@details Stable-Baselines3 DQN with one change: the TD target uses the online network to
choose the next action and the target network to value it (van Hasselt et al., Double
DQN), which reduces the max-operator overestimation that would otherwise favour trading
on noisy, mostly unprofitable entries. Same training table, fee sampling, observation
scaling and decision epochs as PPO (latency_arb/v2/train.py). Checkpoints are saved at
fixed step counts; selection happens on validation days only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from latency_arb.v2.replay import FastLeadLagEnv, OBS_NAMES, ScreenConfig
from latency_arb.v2.table import load_table
from latency_arb.v2.train import TRAIN_FEES


class DoubleDQN(DQN):
    """@brief SB3 DQN with the Double-DQN target."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            with th.no_grad():
                next_actions = self.q_net(data.next_observations).argmax(dim=1, keepdim=True)
                next_q = th.gather(self.q_net_target(data.next_observations), dim=1, index=next_actions)
                target = data.rewards + (1 - data.dones) * discounts * next_q
            current = th.gather(self.q_net(data.observations), dim=1, index=data.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


class Diagnostics(BaseCallback):
    """@brief Log entry/exit rates every `every` steps and save fixed-step checkpoints."""

    def __init__(self, out: Path, checkpoint_every: int, every: int = 10_000):
        super().__init__()
        self.out, self.ck, self.next_ck, self.every, self.next_log = out, checkpoint_every, checkpoint_every, every, every
        self.prev = None
        self.log = (out / "training_log.jsonl").open("w")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_ck:
            self.model.save(self.out / f"checkpoint_{self.next_ck}.zip")
            self.next_ck += self.ck
        if self.num_timesteps >= self.next_log:
            self.next_log += self.every
            stats = self.training_env.get_attr("stats")
            total = {k: sum(s[k] for s in stats) for k in stats[0]}
            delta = total if self.prev is None else {k: total[k] - self.prev[k] for k in total}
            self.prev = total
            row = {"timesteps": self.num_timesteps, "epsilon": float(self.model.exploration_rate),
                   "entry_rate": delta["entries"] / max(delta["entry_decisions"], 1),
                   "exit_rate": delta["exits"] / max(delta["exit_decisions"], 1),
                   "trades_per_episode": delta["trades"] / max(delta["episodes"], 1),
                   "reward_per_episode_bps": delta["reward"] / max(delta["episodes"], 1),
                   "episodes": delta["episodes"]}
            if "train/loss" in self.model.logger.name_to_value:
                row["train/loss"] = float(self.model.logger.name_to_value["train/loss"])
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
        return True

    def _on_training_end(self) -> None:
        self.log.close()


def train(table: str, out: str, seed: int, steps: int, latency_index: int = 3,
          checkpoint_every: int = 250_000) -> Path:
    th.set_num_threads(1)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=False)
    T = load_table(table)
    if T["meta"]["split"] != "train":
        raise ValueError("DQN may only be trained on the training split")
    env = FastLeadLagEnv(T, TRAIN_FEES, latency_index, ScreenConfig(), seed * 1000)
    settings = {"algorithm": "double_dqn", "table": table, "seed": seed, "steps": steps,
                "net_arch": [64, 64], "gamma": 0.99, "buffer_size": 200_000, "batch_size": 256,
                "learning_rate": 1e-4, "learning_starts": 10_000, "train_freq": 4, "gradient_steps": 1,
                "target_update_interval": 5_000, "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02,
                "train_fees_bps_per_side": TRAIN_FEES, "latency_ms": T["meta"]["offsets_ms"][latency_index],
                "observation": OBS_NAMES, "train_days": T["meta"]["days"]}
    model = DoubleDQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=200_000, learning_starts=10_000,
                      batch_size=256, gamma=0.99, train_freq=4, gradient_steps=1,
                      target_update_interval=5_000, exploration_fraction=0.2, exploration_initial_eps=1.0,
                      exploration_final_eps=0.02, policy_kwargs={"net_arch": [64, 64]}, seed=seed,
                      device="cpu", verbose=0)
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=1))
    start = time.time()
    model.learn(total_timesteps=steps, callback=Diagnostics(out_dir, checkpoint_every))
    model.save(out_dir / "final.zip")
    settings["elapsed_s"] = round(time.time() - start, 1)
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=1))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--latency-index", type=int, default=3)
    a = p.parse_args()
    train(a.table, a.out, a.seed, a.steps, a.latency_index)


if __name__ == "__main__":
    main()
