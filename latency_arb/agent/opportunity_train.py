"""@file opportunity_train.py
@brief Train-only counterfactual warm-start and continuous opportunity-level PPO.

@details Future replay rows supply supervised targets only. The actor receives
causal features at an eligible decision, then chooses skip or trade. PPO rewards
come from the continuous simulator's actual nonoverlapping portfolio transitions.
No positive-trade bonus, class balancing, validation fitting, or test access occurs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from latency_arb.agent.policy import FrozenNormalizer
from latency_arb.data.schema import load_manifest_entry, read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import (
    FEATURE_NAMES, OpportunityConfig, OpportunityReplayEnv, candidate_indices,
    counterfactual_trade, features_at,
)


def _hash(path: Path) -> str:
    """@brief Hash a local provenance artifact without buffering its full contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path: Path, value: dict) -> None:
    """@brief Atomically publish a completed JSON artifact within an owned directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _new_directory(path: str | Path) -> Path:
    """@brief Preserve every existing nonempty output rather than overwriting a run."""
    directory = Path(path).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _economic_config(config: EnvConfig) -> dict:
    """@brief Identify execution economics independently of logs and reward units."""
    result = asdict(config)
    result.pop("reward_scale")
    result.pop("log_history")
    return result


def _sources() -> dict[str, str]:
    """@brief Fingerprint the simulator, features, normalization, and training logic."""
    root = Path(__file__).resolve().parents[2]
    names = (
        "latency_arb/agent/opportunity_train.py", "latency_arb/agent/policy.py",
        "latency_arb/opportunity.py", "latency_arb/env/latency_sim.py",
        "latency_arb/data/schema.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in names}


def _normalizer(path: Path) -> FrozenNormalizer:
    """@brief Load numeric-only moments; normalization can never learn during replay."""
    with np.load(path, allow_pickle=False) as data:
        return FrozenNormalizer(
            mean=data["mean"], variance=data["variance"], count=float(data["count"]),
            epsilon=float(data["epsilon"]), clip_obs=float(data["clip_obs"]),
        )


def prepare_training_events(
    manifest: str | Path, output: str | Path, execution: EnvConfig | None = None,
    opportunity: OpportunityConfig | None = None,
) -> dict[str, Any]:
    """@brief Cache every causal training candidate and its fixed-rule trade outcome.

    @param manifest Chronological manifest; only train archives are ever opened.
    @param output New directory for feature/target arrays and audited provenance.
    @param execution Frozen fees, delay, risk, and one-second decision cadence.
    @param opportunity Causal entry gate and predeclared automatic exit rule.
    @return Metadata including directory, event_count, and eligible_episode_count.
    @details Eligibility depends exclusively on contemporaneous features and known
    episode bounds. Future rejection, losses, and zero payoffs remain in the table.
    Counterfactual outcomes can overlap; they are labels, never portfolio returns.
    """
    source = Path(manifest).resolve()
    chosen = execution or EnvConfig(fee_bps=4.5, decision_interval_ms=1000.0)
    gate = opportunity or OpportunityConfig()
    _, entries = read_manifest(source, split="train")
    destination = _new_directory(output)
    observations, targets, episode_numbers, event_rows = [], [], [], []
    timestamps, trades, fees, rejections, endings = [], [], [], [], []
    eligible = []
    for episode_number, entry in enumerate(entries):
        episode = load_manifest_entry(source, entry)
        indices = list(candidate_indices(episode, chosen, gate))
        if indices:
            eligible.append(episode_number)
        for index in indices:
            ## @details Features are captured before computing any future target.
            # Never condition inclusion, normalization, or labels' weights on fill
            # success beyond using the realized economic payoff as the target.
            observations.append(np.asarray(features_at(episode, index, chosen), dtype=np.float32))
            outcome = counterfactual_trade(episode, index, chosen, gate)
            targets.append(float(outcome["net_pnl"]))
            episode_numbers.append(episode_number)
            event_rows.append(int(index))
            timestamps.append(int(episode.timestamp_ns[index]))
            trades.append(int(outcome["trade_count"]))
            fees.append(float(outcome["fees_paid"]))
            rejections.append(int(outcome.get("rejection_count", 0)))
            endings.append(int(outcome.get("end_timestamp_ns", 0)))
        if (episode_number + 1) % 250 == 0:
            print(json.dumps({"stage": "training_event_preparation",
                              "archives_checked": episode_number + 1,
                              "events": len(targets)}), flush=True)
    if not observations:
        raise ValueError("No causal opportunities exist in the training split.")
    features = np.asarray(observations, dtype=np.float32)
    payoffs = np.asarray(targets, dtype=np.float64)
    if features.shape != (len(payoffs), len(FEATURE_NAMES)):
        raise ValueError("Training candidate features do not match FEATURE_NAMES.")
    if not np.isfinite(features).all() or not np.isfinite(payoffs).all():
        raise ValueError("Features and counterfactual net payoffs must be finite.")

    ## @details Fit once on natural-frequency training candidates. Labels and
    # validation observations cannot affect these frozen feature statistics.
    moments = features.astype(np.float64)
    np.savez_compressed(
        destination / "events.npz", features=features, net_payoffs=payoffs,
        episode_indices=np.asarray(episode_numbers, dtype=np.int64),
        candidate_indices=np.asarray(event_rows, dtype=np.int64),
        timestamp_ns=np.asarray(timestamps, dtype=np.int64),
        trade_counts=np.asarray(trades, dtype=np.int64),
        fees_paid=np.asarray(fees, dtype=np.float64),
        rejection_counts=np.asarray(rejections, dtype=np.int64),
        end_timestamp_ns=np.asarray(endings, dtype=np.int64),
    )
    np.savez(
        destination / "normalization.npz", mean=moments.mean(axis=0),
        variance=moments.var(axis=0), count=np.asarray(float(len(features))),
        epsilon=np.asarray(1e-8), clip_obs=np.asarray(10.0),
    )
    metadata = {
        "event_version": 1, "directory": str(destination),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(source), "manifest_sha256": _hash(source), "split": "train",
        "feature_names": list(FEATURE_NAMES), "event_count": len(payoffs),
        "eligible_episode_count": len(eligible), "eligible_episode_indices": eligible,
        "training_entries": entries, "execution_config": asdict(chosen),
        "economic_config": _economic_config(chosen), "opportunity_config": asdict(gate),
        "source_sha256": _sources(), "future_rows_used_only_for_targets": True,
        "candidate_rule": "All causal candidates; no future-outcome filtering.",
        "normalization_rule": "All natural-frequency train candidate features; frozen.",
        "positive_target_count": int(np.count_nonzero(payoffs > 0)),
        "negative_target_count": int(np.count_nonzero(payoffs < 0)),
        "zero_target_count": int(np.count_nonzero(payoffs == 0)),
        "rejected_target_count": int(np.count_nonzero(rejections)),
        "counterfactual_warning": "Overlapping labels are not a realizable portfolio.",
        "artifact_sha256": {
            name: _hash(destination / name) for name in ("events.npz", "normalization.npz")
        },
    }
    _write(destination / "events.json", metadata)
    return metadata


class _ArchiveOpportunities(gym.Env):
    """@brief Bounded-cache continuous replay of causally eligible training archives."""

    metadata = {"render_modes": []}

    def __init__(self, manifest: Path, eligible: list[int], execution: EnvConfig,
                 opportunity: OpportunityConfig, cache_size: int = 2) -> None:
        """@brief Retain training metadata and spaces without eagerly loading books."""
        super().__init__()
        _, self.entries = read_manifest(manifest, split="train")
        if not eligible or any(type(index) is not int or not 0 <= index < len(self.entries)
                               for index in eligible):
            raise ValueError("Eligible indices must identify training archives.")
        self.manifest, self.eligible = manifest, eligible
        self.execution, self.opportunity = execution, opportunity
        self.cache_size = cache_size
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(len(FEATURE_NAMES),), dtype=np.float32
        )
        self._cache: OrderedDict[int, OpportunityReplayEnv] = OrderedDict()
        self._active: OpportunityReplayEnv | None = None
        self._active_index = -1
        self.total_underlying_steps = 0

    def reset(self, *, seed=None, options=None):
        """@brief Sample eligible training windows uniformly, retaining chronology within."""
        super().reset(seed=seed)
        index = int(self.np_random.choice(self.eligible))
        self._active = None
        if index not in self._cache:
            episode = load_manifest_entry(self.manifest, self.entries[index])
            if not len(candidate_indices(episode, self.execution, self.opportunity)):
                raise ValueError("Cached eligibility disagrees with the causal candidate rule.")
            self._cache[index] = OpportunityReplayEnv(
                episode, self.execution, self.opportunity
            )
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.close()
        self._active, self._active_index = self._cache[index], index
        observation, info = self._active.reset(seed=seed)
        self.total_underlying_steps += int(info.get("underlying_step_count", 0))
        info["training_episode_index"] = index
        return observation, info

    def step(self, action):
        """@brief Advance the actual portfolio; occupied trade intervals skip candidates."""
        if self._active is None:
            raise RuntimeError("Reset the opportunity environment before stepping.")
        observation, reward, terminated, truncated, info = self._active.step(action)
        self.total_underlying_steps += int(info.get("macro_underlying_steps", 0))
        info["training_episode_index"] = self._active_index
        return observation, reward, terminated, truncated, info

    def close(self):
        """@brief Release retained replay books after fitting or an exception."""
        for env in self._cache.values():
            env.close()
        self._cache.clear()
        self._active = None


class _NormalizeFeatures(gym.ObservationWrapper):
    """@brief Apply immutable training moments to every continuous replay observation."""

    def __init__(self, env: gym.Env, normalizer: FrozenNormalizer) -> None:
        """@brief Preserve the binary action space and fixed-size observation contract."""
        super().__init__(env)
        self.normalizer = normalizer

    def observation(self, observation):
        """@brief Transform raw features without fitting on policy-selected transitions."""
        return self.normalizer.normalize(observation)


class _History(BaseCallback):
    """@brief Log real macro transitions, underlying work, and completed trade outcomes."""

    def __init__(self, path: Path, archive: _ArchiveOpportunities) -> None:
        """@brief Initialize cumulative diagnostics for actual on-policy experience."""
        super().__init__()
        self.path, self.archive = path, archive
        self.episodes = self.trades = self.rejections = 0
        self.pnl = self.fees = 0.0

    def _on_step(self):
        """@brief Count completed actual portfolios independently of warm-start labels."""
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.trades += int(episode["trade_count"])
                self.rejections += int(episode["rejection_count"])
                self.pnl += float(episode["pnl"])
                self.fees += float(episode["fees_paid"])
        return True

    def _on_rollout_end(self):
        """@brief Persist cumulative economic metrics and binary policy decisions."""
        actions = self.model.rollout_buffer.actions.astype(np.int64).reshape(-1)
        row = {
            "macro_steps": self.num_timesteps,
            "underlying_steps": self.archive.total_underlying_steps,
            "completed_episodes": self.episodes, "completed_trades": self.trades,
            "completed_episode_pnl": self.pnl, "fees_paid": self.fees,
            "rejection_count": self.rejections,
            "rollout_trade_decisions": int(np.count_nonzero(actions == 1)),
            "rollout_skip_decisions": int(np.count_nonzero(actions == 0)),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


class OpportunityPolicy:
    """@brief Frozen binary policy accepting raw causal opportunity features."""

    def __init__(self, model: PPO, normalizer: FrozenNormalizer, metadata: dict) -> None:
        """@brief Bind one labeled checkpoint to its exact immutable normalization."""
        if model.observation_space.shape != normalizer.mean.shape:
            raise ValueError("Policy and opportunity normalization shapes disagree.")
        self.model, self.normalizer, self.metadata = model, normalizer, metadata
        self.feature_names = tuple(metadata["feature_names"])

    def predict(self, observation, deterministic: bool = True):
        """@brief Return the standard SB3 action/state tuple after frozen normalization."""
        return self.model.predict(self.normalizer.normalize(observation),
                                  deterministic=deterministic)

    def trade_probability(self, observation):
        """@brief Expose the actor's probability of trade for diagnostics or fixed selection.

        @details Calling this method neither samples an action nor updates moments.
        Any probability cutoff used for model selection must be validation-only.
        """
        values = self.normalizer.normalize(observation)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(torch.as_tensor(values))
            probabilities = distribution.distribution.probs[:, 1].cpu().numpy()
        return float(probabilities[0]) if single else probabilities


def load_opportunity_policy(model_dir: str | Path) -> OpportunityPolicy:
    """@brief Verify and load a trusted warm-start-only or PPO-finetuned artifact.

    @warning SB3 model archives deserialize Python/PyTorch objects. Load trusted
    local artifacts only; content checksums prevent accidental file mismatches.
    """
    directory = Path(model_dir)
    metadata = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != 1 or metadata.get("policy_family") != "opportunity":
        raise ValueError("Unsupported opportunity policy artifact.")
    if tuple(metadata["feature_names"]) != tuple(FEATURE_NAMES):
        raise ValueError("Opportunity feature names differ from the saved policy.")
    for name in ("policy.zip", "normalization.npz"):
        if _hash(directory / name) != metadata["artifact_sha256"].get(name):
            raise ValueError(f"Opportunity artifact checksum mismatch: {name}")
    model = PPO.load(directory / "policy.zip", device="cpu")
    if model.action_space.n != 2:
        raise ValueError("An opportunity policy must have exactly two actions.")
    model.policy.set_training_mode(False)
    return OpportunityPolicy(model, _normalizer(directory / "normalization.npz"), metadata)


def train_opportunity_policy(
    manifest: str | Path, event_data_dir: str | Path, output_dir: str | Path,
    seed: int = 7, total_timesteps: int = 65_536, warm_epochs: int = 50,
    execution: EnvConfig | None = None, opportunity: OpportunityConfig | None = None,
    *, n_steps: int = 1024, batch_size: int = 128, n_epochs: int = 5,
    hidden_size: int = 32, learning_rate: float = 1e-4,
    entropy_coef: float = 0.005, warm_batch_size: int = 256,
) -> dict[str, Any]:
    """@brief Warm-start economically, then run genuine continuous macro-step PPO.

    @details Behavior cloning uses natural-frequency training events, labels
    payoff>0, and weights |payoff| normalized by their global mean. This targets
    expected-dollar regret rather than unweighted win probability. PPO then uses
    actual simulator rewards, gamma=1, and a fixed positive reward scale of 100.
    No counterfactual payoff is substituted for continuous portfolio accounting.
    """
    from latency_arb.agent.train import TrainingConfig

    ## @details Reuse validated PPO numerical constraints while fixing the economic
    # objective and transition discount for irregular-duration macro actions.
    settings = TrainingConfig(
        total_timesteps=total_timesteps, seed=seed, n_steps=n_steps,
        batch_size=batch_size, n_epochs=n_epochs, hidden_size=hidden_size,
        learning_rate=learning_rate, ent_coef=entropy_coef, gamma=1.0,
    )
    settings.validate()
    if type(warm_epochs) is not int or warm_epochs < 1 or type(warm_batch_size) is not int or warm_batch_size < 1:
        raise ValueError("Warm-start epochs and batch size must be positive integers.")
    source, event_dir = Path(manifest).resolve(), Path(event_data_dir).resolve()
    event_meta = json.loads((event_dir / "events.json").read_text(encoding="utf-8"))
    chosen = execution or EnvConfig(**event_meta["execution_config"])
    gate = opportunity or OpportunityConfig(**event_meta["opportunity_config"])
    if event_meta.get("event_version") != 1 or event_meta.get("split") != "train":
        raise ValueError("Only version-one training event data are accepted.")
    if event_meta["manifest_sha256"] != _hash(source):
        raise ValueError("Training event manifest checksum differs.")
    if event_meta["source_sha256"] != _sources():
        raise ValueError("Training event implementation differs; regenerate targets.")
    if event_meta["economic_config"] != _economic_config(chosen) or event_meta["opportunity_config"] != asdict(gate):
        raise ValueError("Training event economics or opportunity rules differ.")
    if event_meta["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Training event feature order differs.")
    for name in ("events.npz", "normalization.npz"):
        if _hash(event_dir / name) != event_meta["artifact_sha256"][name]:
            raise ValueError(f"Training event checksum mismatch: {name}")
    with np.load(event_dir / "events.npz", allow_pickle=False) as data:
        features, payoffs = data["features"].copy(), data["net_payoffs"].copy()
        observed_eligible = np.unique(data["episode_indices"]).tolist()
    if observed_eligible != event_meta["eligible_episode_indices"]:
        raise ValueError("Training archive eligibility disagrees with saved events.")
    normalizer = _normalizer(event_dir / "normalization.npz")
    normalized = normalizer.normalize(features)
    if payoffs.shape != (len(features),) or not np.isfinite(payoffs).all():
        raise ValueError("Counterfactual targets must be finite and align with features.")
    magnitude = np.abs(payoffs)
    if magnitude.mean() <= 0:
        raise ValueError("All training targets are zero; there is no economic learning signal.")
    destination = _new_directory(output_dir)
    execution_for_learning = replace(chosen, reward_scale=100.0, log_history=False)
    prior_threads = torch.get_num_threads()
    prior_determinism = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    archive = _ArchiveOpportunities(
        source, observed_eligible, execution_for_learning, gate
    )
    vector = None
    try:
        vector = DummyVecEnv([lambda: Monitor(
            _NormalizeFeatures(archive, normalizer),
            filename=str(destination / "training_episodes.monitor.csv"),
            info_keywords=("pnl", "trade_count", "fees_paid", "rejection_count",
                           "underlying_step_count", "training_episode_index"),
        )])
        model = PPO(
            "MlpPolicy", vector, seed=seed, device="cpu", learning_rate=learning_rate,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=1.0,
            ent_coef=entropy_coef, policy_kwargs={"net_arch": {
                "pi": [hidden_size, hidden_size], "vf": [hidden_size, hidden_size]
            }},
        )
        x = torch.as_tensor(normalized, dtype=torch.float32)
        y = torch.as_tensor(payoffs > 0, dtype=torch.long)
        weights = torch.as_tensor(magnitude / magnitude.mean(), dtype=torch.float32)
        random = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)

        ## @details Only the actor receives a cloning gradient. The critic is not
        # taught hindsight-optimal returns; its value estimates are learned by PPO
        # from actual policy trajectories. PPO's optimizer starts with fresh state.
        model.policy.set_training_mode(True)
        for epoch in range(warm_epochs):
            permutation = torch.randperm(len(x), generator=random)
            loss_sum = 0.0
            for start in range(0, len(x), warm_batch_size):
                indices = permutation[start:start + warm_batch_size]
                distribution = model.policy.get_distribution(x[indices])
                loss = -(distribution.log_prob(y[indices]) * weights[indices]).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            with (destination / "warm_start_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": epoch + 1, "weighted_actor_loss": loss_sum / len(x),
                    "event_count": len(x), "positive_targets": int((payoffs > 0).sum()),
                    "label_scope": "train-only counterfactuals, not portfolio returns",
                }, allow_nan=False) + "\n")

        common = {
            "artifact_version": 1, "policy_family": "opportunity",
            "feature_names": list(FEATURE_NAMES), "manifest_sha256": _hash(source),
            "training_split": "train", "held_out_data_loaded": False,
            "events_sha256": event_meta["artifact_sha256"]["events.npz"],
            "events_metadata_sha256": _hash(event_dir / "events.json"),
            "source_sha256": _sources(), "seed": seed, "training_config": asdict(settings),
            "execution_config": asdict(execution_for_learning),
            "opportunity_config": asdict(gate), "warm_epochs": warm_epochs,
            "warm_objective": "natural-frequency |net payoff|-weighted binary cross entropy",
            "normalization": "fitted once on training candidates; frozen in every phase",
            "event_count": len(payoffs), "eligible_episode_count": len(observed_eligible),
            "versions": {name: importlib.metadata.version(name) for name in (
                "numpy", "torch", "gymnasium", "stable-baselines3")},
        }

        def save_checkpoint(name: str, kind: str) -> dict:
            """@brief Save weights and immutable moments with an honest phase label."""
            directory = destination / name
            directory.mkdir()
            model.save(directory / "policy.zip")
            shutil.copyfile(event_dir / "normalization.npz", directory / "normalization.npz")
            metadata = {
                **common, "checkpoint_kind": kind, "actual_ppo_macro_steps": int(model.num_timesteps),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_sha256": {
                    filename: _hash(directory / filename)
                    for filename in ("policy.zip", "normalization.npz")
                },
            }
            _write(directory / "training.json", metadata)
            return metadata

        warm = save_checkpoint("warmstart", "supervised_warm_start_only")
        history = _History(destination / "training_metrics.jsonl", archive)
        model.learn(total_timesteps=total_timesteps, callback=history, progress_bar=False)
        final = save_checkpoint("ppo", "supervised_warm_start_plus_on_policy_ppo")
        result = {
            "directory": str(destination),
            "warmstart_dir": str(destination / "warmstart"), "ppo_dir": str(destination / "ppo"),
            "warmstart_model_sha256": warm["artifact_sha256"]["policy.zip"],
            "ppo_model_sha256": final["artifact_sha256"]["policy.zip"],
            "actual_ppo_macro_steps": int(model.num_timesteps),
            "underlying_steps": archive.total_underlying_steps,
            "completed_training_episodes": history.episodes,
            "completed_training_trades": history.trades,
            "completed_training_pnl": history.pnl, "training_fees_paid": history.fees,
            "event_count": len(payoffs), "seed": seed,
            "held_out_data_loaded": False,
        }
        _write(destination / "run.json", result)
        return result
    finally:
        if vector is not None:
            vector.close()
        else:
            archive.close()
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(prior_determinism, warn_only=prior_warn_only)


def main(argv: list[str] | None = None) -> int:
    """@brief Prepare training labels or fit one predeclared opportunity PPO seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=65_536)
    parser.add_argument("--warm-epochs", type=int, default=50)
    parser.add_argument("--fee-bps", type=float, default=4.5)
    parser.add_argument("--min-gap-bps", type=float, default=6.0)
    args = parser.parse_args(argv)
    execution = EnvConfig(fee_bps=args.fee_bps, decision_interval_ms=1000.0)
    gate = OpportunityConfig(min_gap_bps=args.min_gap_bps)
    if args.prepare:
        result = prepare_training_events(args.manifest, args.output, execution, gate)
    else:
        if args.events is None:
            parser.error("--events is required unless --prepare is supplied")
        result = train_opportunity_policy(
            args.manifest, args.events, args.output, seed=args.seed,
            total_timesteps=args.steps, warm_epochs=args.warm_epochs,
            execution=execution, opportunity=gate,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

