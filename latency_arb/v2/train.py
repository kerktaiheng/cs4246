"""@file train.py
@brief Train the version 2 fee-conditioned PPO agent on training days only.
@details Stable-Baselines3 PPO learns from the fast replay of 3-24 September. The
fee per side is sampled per window and observed by the policy, so one network must
learn how selective to be at each cost level. Checkpoints are saved at fixed step
counts; selection between them happens later on validation days only.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from latency_arb.v2.replay import FastLeadLagEnv, OBS_NAMES, ScreenConfig
from latency_arb.v2.table import load_table

TRAIN_FEES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


class Diagnostics(BaseCallback):
    """@brief Log entry/exit rates per rollout and save fixed-step checkpoints."""

    def __init__(self, out: Path, checkpoint_every: int):
        super().__init__()
        self.out, self.every, self.next_save = out, checkpoint_every, checkpoint_every
        self.prev = None
        self.log = (out / "training_log.jsonl").open("w")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.model.save(self.out / f"checkpoint_{self.next_save}.zip")
            self.next_save += self.every
        return True

    def _on_rollout_end(self) -> None:
        stats = self.training_env.get_attr("stats")
        total = {k: sum(s[k] for s in stats) for k in stats[0]}
        delta = total if self.prev is None else {k: total[k] - self.prev[k] for k in total}
        self.prev = total
        row = {"timesteps": self.num_timesteps,
               "entry_rate": delta["entries"] / max(delta["entry_decisions"], 1),
               "exit_rate": delta["exits"] / max(delta["exit_decisions"], 1),
               "trades_per_episode": delta["trades"] / max(delta["episodes"], 1),
               "reward_per_episode_bps": delta["reward"] / max(delta["episodes"], 1),
               "episodes": delta["episodes"], "entry_decisions": delta["entry_decisions"]}
        for key in ("train/entropy_loss", "train/value_loss", "train/approx_kl", "train/explained_variance"):
            if key in self.logger.name_to_value:
                row[key] = float(self.logger.name_to_value[key])
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()

    def _on_training_end(self) -> None:
        self.log.close()


def train(table: str, out: str, seed: int, steps: int, *, ent_coef: float = 0.01,
          n_envs: int = 8, n_steps: int = 512, batch_size: int = 512, n_epochs: int = 10,
          learning_rate: float = 3e-4, gamma: float = 0.99, checkpoint_every: int = 250_000,
          latency_index: int = 1, screen: ScreenConfig = ScreenConfig()) -> Path:
    torch.set_num_threads(1)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=False)
    T = load_table(table)
    if T["meta"]["split"] != "train":
        raise ValueError("PPO may only be trained on the training split")
    envs = DummyVecEnv([lambda i=i: FastLeadLagEnv(T, TRAIN_FEES, latency_index, screen, seed * 1000 + i)
                        for i in range(n_envs)])
    model = PPO("MlpPolicy", envs, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                learning_rate=learning_rate, gamma=gamma, gae_lambda=0.95, ent_coef=ent_coef,
                clip_range=0.2, policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                seed=seed, device="cpu", verbose=0)
    settings = {"table": table, "seed": seed, "steps": steps, "ent_coef": ent_coef, "n_envs": n_envs,
                "n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs,
                "learning_rate": learning_rate, "gamma": gamma, "gae_lambda": 0.95, "clip_range": 0.2,
                "net_arch": [64, 64], "train_fees_bps_per_side": TRAIN_FEES,
                "latency_ms": T["meta"]["offsets_ms"][latency_index],
                "screen": screen.__dict__, "observation": OBS_NAMES, "train_days": T["meta"]["days"],
                "versions": {"torch": torch.__version__, "python": platform.python_version()}}
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
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--checkpoint-every", type=int, default=250_000)
    p.add_argument("--latency-index", type=int, default=1)
    a = p.parse_args()
    train(a.table, a.out, a.seed, a.steps, ent_coef=a.ent_coef,
          checkpoint_every=a.checkpoint_every, latency_index=a.latency_index)


if __name__ == "__main__":
    main()
