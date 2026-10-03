"""@file train.py
@brief Reproducible CPU PPO training using only chronological training episodes.

@details This module delegates PPO updates to Stable-Baselines3. It normalizes
observations using training data only, preserves dollar rewards apart from the
explicit environment reward_scale, and exports frozen inference statistics.
It never loads validation/test episodes or selects a checkpoint on their returns.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from latency_arb.data.schema import read_manifest, load_manifest_entry
from latency_arb.env.latency_sim import Action, EnvConfig, LatencySimEnv, OBSERVATION_NAMES


@dataclass(frozen=True)
class TrainingConfig:
    """@brief Explicit PPO settings recorded alongside every trained model.

    @details The one-environment CPU configuration avoids process scheduling
    variability. Stable-Baselines3 completes entire n_steps rollouts, so the
    actual transition count can exceed total_timesteps by at most n_steps - 1.
    Reproducibility is expected on the same installed software and hardware.
    """

    total_timesteps: int = 30_000
    seed: int = 7
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    clip_obs: float = 10.0
    verbose: int = 0
    episode_cache_size: int = 2

    def validate(self) -> None:
        """@brief Reject ambiguous, invalid, or non-finite PPO settings early."""
        for name in ("total_timesteps", "n_steps", "batch_size", "n_epochs", "hidden_size", "episode_cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        if self.n_steps < 2 or self.batch_size < 2:
            raise ValueError("PPO needs at least two rollout steps and batch samples.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps exactly.")
        for name in ("learning_rate", "clip_range", "max_grad_norm", "clip_obs"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        for name in ("ent_coef", "vf_coef"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")


class ArchiveReplayEnv(gym.Env):
    """@brief Replay training archives with a bounded least-recently-used cache.

    @details Construction reads only manifest metadata. Each reset selects one
    training episode using a seeded local generator, verifies its file checksum,
    and loads it only when absent from the cache. Completed archives are sampled
    uniformly: because the whole archive is played, expected transition visits
    already scale with archive duration. Weighting archive selection by duration
    again would overrepresent long periods quadratically.

    @details At most cache_size archive environments are retained after reset;
    loading a cache miss can temporarily hold one additional episode. No
    validation or test archive is opened by this environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, manifest_path: str | Path, config: EnvConfig | None = None,
        cache_size: int = 2,
    ) -> None:
        """@brief Validate training metadata and establish spaces without bulk I/O.

        @param manifest_path Chronologically split archive manifest.
        @param config Cost, risk, and decision-cadence settings for each archive.
        @param cache_size Maximum number of parsed archive environments retained.
        @throws ValueError If metadata or cache configuration is invalid.
        """
        super().__init__()
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer.")
        self.manifest_path = Path(manifest_path).resolve()
        self.document, self.entries = read_manifest(self.manifest_path, split="train")
        self.config = config or EnvConfig()
        self.cache_size = cache_size
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(len(OBSERVATION_NAMES),), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(Action))
        self._cache: OrderedDict[int, LatencySimEnv] = OrderedDict()
        self._active: LatencySimEnv | None = None
        self._active_index: int | None = None
        self.verified_files: dict[str, str] = {}
        self.episode_selections = np.zeros(len(self.entries), dtype=np.int64)
        self.cache_miss_loads = 0

    @property
    def cached_episode_count(self) -> int:
        """@brief Report retained archive count for diagnostics and memory tests."""
        return len(self._cache)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """@brief Select and lazily open one training archive with deterministic RNG.

        @param seed Optional seed; omitted resets continue the existing RNG stream.
        @param options Optional episode_index selects a training entry for tests.
        @return Raw observation and replay accounting with training-archive identity.
        """
        super().reset(seed=seed)
        selected = (options or {}).get("episode_index")
        if selected is None:
            selected = int(self.np_random.integers(len(self.entries)))
        if (
            isinstance(selected, bool)
            or not isinstance(selected, (int, np.integer))
            or not 0 <= selected < len(self.entries)
        ):
            raise ValueError("episode_index is outside the training archive collection.")
        selected = int(selected)
        self._active = None
        if selected not in self._cache:
            ## @details The shared loader checks hash, day, split, bounds, and schema
            # before the new replay environment can expose its first observation.
            entry = self.entries[selected]
            episode = load_manifest_entry(self.manifest_path, entry)
            self._cache[selected] = LatencySimEnv(episode, config=self.config)
            self.verified_files[entry["path"]] = entry["content_sha256"]
            self.cache_miss_loads += 1
        self._cache.move_to_end(selected)
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.close()
        self._active = self._cache[selected]
        self._active_index = selected
        self.episode_selections[selected] += 1
        observation, info = self._active.reset(seed=seed, options={"episode_index": 0})
        info["training_episode_index"] = selected
        return observation, info

    def step(self, action):
        """@brief Delegate one decision to the active causal replay environment."""
        if self._active is None:
            raise RuntimeError("Call reset before stepping an archive environment.")
        observation, reward, terminated, truncated, info = self._active.step(action)
        info["training_episode_index"] = self._active_index
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """@brief Release cached arrays and environments after a training run."""
        for environment in self._cache.values():
            environment.close()
        self._cache.clear()
        self._active = None


class _TrainingHistory(BaseCallback):
    """@brief Persist compact training diagnostics after each collected rollout."""

    def __init__(self, path: Path) -> None:
        """@brief Retain a fresh JSON-lines destination owned by this run."""
        super().__init__()
        self.path = path
        self.completed_episodes = 0
        self.closed_trades = 0
        self.zero_trade_episodes = 0

    def _on_step(self) -> bool:
        """@brief Count completed episodes and actual fills from the first rollout.

        @details Monitor writes every completed episode to CSV independently of
        rollout boundaries; these cumulative counts make collapse to zero trading
        visible even after the rolling statistics window drops old episodes.
        """
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is not None:
                trades = int(episode["trade_count"])
                self.completed_episodes += 1
                self.closed_trades += trades
                self.zero_trade_episodes += int(trades == 0)
        return True

    def _on_rollout_end(self) -> None:
        """@brief Record recent completed-episode reward without extra rollouts.

        @details Monitor rewards include EnvConfig.reward_scale and therefore are
        diagnostics rather than reported trading P&L. Held-out economic metrics
        are produced by the evaluation pipeline using the unscaled environment.
        """
        recent = list(self.model.ep_info_buffer or ())
        record: dict[str, Any] = {
            "timesteps": self.num_timesteps,
            "completed_episodes_total": self.completed_episodes,
            "closed_trades_total": self.closed_trades,
            "zero_trade_episode_fraction": (
                self.zero_trade_episodes / self.completed_episodes
                if self.completed_episodes else None
            ),
            "recent_mean_trade_count": (
                float(np.mean([episode["trade_count"] for episode in recent]))
                if recent else None
            ),
            "recent_mean_pnl_usd": (
                float(np.mean([episode["pnl"] for episode in recent]))
                if recent else None
            ),
            "recent_mean_fees_paid_usd": (
                float(np.mean([episode["fees_paid"] for episode in recent]))
                if recent else None
            ),
            "recent_completed_episodes": len(recent),
            "recent_mean_scaled_episode_reward": (
                float(np.mean([episode["r"] for episode in recent])) if recent else None
            ),
            "recent_mean_episode_length": (
                float(np.mean([episode["l"] for episode in recent])) if recent else None
            ),
        }
        ## @details Count decisions from the training rollout itself so abstention
        # is measurable without sampling additional episodes or changing seeds.
        decisions = self.model.rollout_buffer.actions.astype(np.int64).reshape(-1)
        counts = np.bincount(decisions, minlength=4)
        record["rollout_action_counts"] = {
            name: int(counts[index])
            for index, name in enumerate(("hold", "long", "short", "exit"))
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    """@brief Hash an artifact in bounded memory for reproducible provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_policy(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    env_config: EnvConfig | None = None,
) -> dict[str, Any]:
    """@brief Train a final PPO checkpoint and save everything inference needs.

    @param manifest_path Checked manifest containing chronological split labels.
    @param output_dir New or empty directory for model, normalization, and metadata.
    @param config PPO hyperparameters; defaults favor a short CPU experiment.
    @param env_config Identical execution-cost/risk assumptions used for evaluation.
    @return JSON-serializable run metadata, including requested and actual steps.
    @throws ValueError If configuration or training data are invalid.
    @throws FileExistsError If output_dir is nonempty, preserving earlier runs.
    @details Only split='train' is passed to the data loader. Validation/test
    filenames can be recorded in manifest metadata but their data are never read.
    No checkpoint search or validation-driven hyperparameter tuning happens here.
    """
    settings = config or TrainingConfig()
    settings.validate()
    execution = replace(env_config or EnvConfig(), log_history=False)
    manifest_file = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {destination}")

    ## @details Read chronology and source metadata without loading every archive.
    # Individual training files are verified only when selected at an episode reset;
    # this keeps memory proportional to the cache, not the full month of recordings.
    archive_env = ArchiveReplayEnv(
        manifest_file, config=execution, cache_size=settings.episode_cache_size
    )
    train_entries = archive_env.entries
    manifest_sha256 = _sha256_file(manifest_file)
    destination.mkdir(parents=True, exist_ok=True)

    ## @details Pin stochastic sources and CPU kernels before model construction.
    # PPO additionally seeds Python, NumPy, Torch, action spaces, and vector reset.
    # Restoring Torch process options in finally avoids surprising library callers.
    prior_threads = torch.get_num_threads()
    prior_determinism = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    vector_env: VecNormalize | None = None
    start = time.perf_counter()
    try:
        raw_env = DummyVecEnv([
            lambda: Monitor(
                archive_env,
                filename=str(destination / "training_episodes.monitor.csv"),
                info_keywords=("trade_count", "pnl", "fees_paid", "training_episode_index"),
            )
        ])
        vector_env = VecNormalize(
            raw_env,
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=settings.clip_obs,
            gamma=settings.gamma,
        )
        model = PPO(
            "MlpPolicy",
            vector_env,
            learning_rate=settings.learning_rate,
            n_steps=settings.n_steps,
            batch_size=settings.batch_size,
            n_epochs=settings.n_epochs,
            gamma=settings.gamma,
            gae_lambda=settings.gae_lambda,
            clip_range=settings.clip_range,
            ent_coef=settings.ent_coef,
            vf_coef=settings.vf_coef,
            max_grad_norm=settings.max_grad_norm,
            policy_kwargs={
                "net_arch": {
                    "pi": [settings.hidden_size, settings.hidden_size],
                    "vf": [settings.hidden_size, settings.hidden_size],
                }
            },
            seed=settings.seed,
            device="cpu",
            verbose=settings.verbose,
        )
        model.learn(
            total_timesteps=settings.total_timesteps,
            callback=_TrainingHistory(destination / "training_metrics.jsonl"),
            progress_bar=False,
        )

        ## @details Freeze statistics immediately after the last training rollout.
        # Export SB3's native state for trusted resumptions and a numeric-only copy
        # for inference. Neither file is fitted or updated using held-out data.
        vector_env.training = False
        vector_env.norm_reward = False
        model.save(destination / "policy.zip")
        vector_env.save(str(destination / "vecnormalize.pkl"))
        np.savez(
            destination / "observation_normalization.npz",
            mean=vector_env.obs_rms.mean,
            variance=vector_env.obs_rms.var,
            count=np.asarray(vector_env.obs_rms.count),
            epsilon=np.asarray(vector_env.epsilon),
            clip_obs=np.asarray(vector_env.clip_obs),
        )

        ## @details Capture cost assumptions, feature order, source checksums,
        # versions, and the final checkpoint rule to make comparisons auditable.
        synthetic_flags = [
            bool(entry.get("synthetic", False)) for entry in train_entries
        ]
        metadata: dict[str, Any] = {
            "artifact_version": 1,
            "algorithm": "stable_baselines3.PPO",
            "checkpoint_rule": "final training iterate; no held-out selection",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha256,
            "training_entries": train_entries,
            "training_episode_count": len(train_entries),
            "training_row_count": sum(int(entry["rows"]) for entry in train_entries),
            "archive_replay": {
                "cache_size": settings.episode_cache_size,
                "sampling": "uniform archive selection; full chronological replay",
                "verified_files": dict(archive_env.verified_files),
                "cache_miss_loads": archive_env.cache_miss_loads,
                "episode_selections": archive_env.episode_selections.tolist(),
            },
            "training_data_kind": (
                "synthetic" if all(synthetic_flags)
                else "mixed" if any(synthetic_flags) else "recorded"
            ),
            "training_split": "train",
            "held_out_data_loaded": False,
            "observation_names": list(OBSERVATION_NAMES),
            "training_config": asdict(settings),
            "environment_config": asdict(execution),
            "requested_timesteps": settings.total_timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_seconds": time.perf_counter() - start,
            "normalization": {
                "training_only": True,
                "frozen_for_inference": True,
                "normalize_reward": False,
                "observation_count": float(vector_env.obs_rms.count),
            },
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "torch", "gymnasium", "stable-baselines3")
            },
            "python_version": platform.python_version(),
            "device": "cpu",
            "torch_threads": 1,
            "artifact_sha256": {
                filename: _sha256_file(destination / filename)
                for filename in (
                    "policy.zip", "vecnormalize.pkl", "observation_normalization.npz"
                )
            },
            "limitations": [
                "Replay does not model the market's reaction to agent orders.",
                "Synthetic results test the pipeline, not real trading profitability.",
                "Exact reproducibility is scoped to the same software and hardware.",
            ],
        }
        (destination / "training.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return metadata
    finally:
        ## @details Release wrappers on success and failure, then restore Torch
        # process settings. Partial artifacts remain visibly incomplete on error.
        if vector_env is not None:
            vector_env.close()
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(prior_determinism, warn_only=prior_warn_only)


def main(argv: list[str] | None = None) -> int:
    """@brief Run the offline trainer from an explicit dataset manifest.

    @param argv Optional argument list for programmatic command-line testing.
    @return Zero after successfully writing and describing the final checkpoint.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--episode-cache-size", type=int, default=2)
    parser.add_argument("--fee-bps", type=float, default=3.5)
    parser.add_argument("--latency-ms", type=float, default=150.0)
    parser.add_argument("--decision-interval-ms", type=float, default=0.0)
    parser.add_argument("--max-holding-ms", type=float, default=30_000.0)
    parser.add_argument("--slippage-bps", type=float, default=0.1)
    parser.add_argument("--position-size-btc", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    ## @details Keep reward scaling explicit and recorded: it changes PPO's
    # learning signal while economic P&L in the environment remains in dollars.
    metadata = train_policy(
        args.manifest,
        args.output,
        TrainingConfig(
            total_timesteps=args.steps,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            ent_coef=args.entropy_coef,
            gamma=args.gamma,
            episode_cache_size=args.episode_cache_size,
            verbose=int(args.verbose),
        ),
        EnvConfig(
            fee_bps=args.fee_bps,
            latency_ms=args.latency_ms,
            decision_interval_ms=args.decision_interval_ms,
            max_holding_ms=args.max_holding_ms,
            slippage_bps=args.slippage_bps,
            position_size_btc=args.position_size_btc,
            reward_scale=args.reward_scale,
        ),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "actual_timesteps": metadata["actual_timesteps"],
        "training_data_kind": metadata["training_data_kind"],
        "held_out_data_loaded": metadata["held_out_data_loaded"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
