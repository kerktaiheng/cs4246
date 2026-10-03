"""@file test_training.py
@brief End-to-end checks for seeded PPO training and frozen held-out inference.
@details These tests perform small real PPO updates, verify saved native SB3
normalization against inference, and make held-out files unavailable to expose
accidental data leakage rather than relying solely on mocks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from latency_arb.agent.policy import FrozenNormalizer, load_policy
import latency_arb.agent.train as training_module
from latency_arb.agent.train import ArchiveReplayEnv, TrainingConfig, train_policy
from latency_arb.data.generate_episodes import create_demo
from latency_arb.data.schema import load_manifest
from latency_arb.env.latency_sim import EnvConfig, LatencySimEnv


@pytest.fixture(scope="module")
def trained_runs(tmp_path_factory):
    """@brief Train two tiny same-seed PPO models with inaccessible held-out data.

    @details Removing validation and test files proves the training path does not
    need them. Comparing full policy tensors checks seeded optimization and
    episode selection, not merely whether deterministic argmax picks one action.
    """
    root = tmp_path_factory.mktemp("ppo_training")
    manifest = create_demo(root / "data", days=6, steps_per_day=64, seed=13)
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in contents["episodes"]:
        if entry["split"] != "train":
            (manifest.parent / entry["path"]).unlink()
    settings = TrainingConfig(
        total_timesteps=256, seed=19, n_steps=32, batch_size=16,
        n_epochs=1, hidden_size=16,
    )
    run_a, run_b = root / "run_a", root / "run_b"
    metadata = train_policy(
        manifest, run_a, settings, EnvConfig(reward_scale=100.0)
    )
    train_policy(manifest, run_b, settings, EnvConfig(reward_scale=100.0))
    return manifest, run_a, run_b, metadata


def test_seeded_training_reproduces_weights_and_statistics(trained_runs):
    """@brief Identical CPU seeds reproduce optimizer results and fitted moments."""
    _, run_a, run_b, metadata = trained_runs
    first, second = load_policy(run_a), load_policy(run_b)
    assert metadata["actual_timesteps"] == 256
    assert metadata["training_split"] == "train"
    assert metadata["held_out_data_loaded"] is False
    assert metadata["training_data_kind"] == "synthetic"
    assert all(entry["split"] == "train" for entry in metadata["training_entries"])
    for name, parameter in first.model.policy.state_dict().items():
        assert torch.equal(parameter, second.model.policy.state_dict()[name]), name
    np.testing.assert_array_equal(first.normalizer.mean, second.normalizer.mean)
    np.testing.assert_array_equal(first.normalizer.variance, second.normalizer.variance)


def test_loaded_policy_matches_native_normalization_and_actions(trained_runs):
    """@brief Exported numeric normalization reproduces SB3's saved transformation."""
    manifest, run_a, _, _ = trained_runs
    policy = load_policy(run_a)
    episode = load_manifest(manifest, split="train")[0]
    env = LatencySimEnv(episode)
    native = VecNormalize.load(
        str(run_a / "vecnormalize.pkl"),
        DummyVecEnv([lambda: LatencySimEnv(episode)]),
    )
    try:
        observation, _ = env.reset(seed=99)
        assert native.training is False
        assert native.norm_reward is False
        normalized = native.normalize_obs(observation)
        np.testing.assert_array_equal(
            policy.normalizer.normalize(observation), normalized
        )
        expected, _ = policy.model.predict(normalized, deterministic=True)
        actual, _ = policy.predict(observation)
        np.testing.assert_array_equal(actual, expected)
    finally:
        native.close()
        env.close()


def test_inference_never_changes_training_statistics(trained_runs):
    """@brief Repeated prediction on shifted observations leaves moments untouched."""
    manifest, run_a, _, _ = trained_runs
    policy = load_policy(run_a)
    episode = load_manifest(manifest, split="train")[0]
    env = LatencySimEnv(episode)
    observation, _ = env.reset(seed=8)
    mean = policy.normalizer.mean.copy()
    variance = policy.normalizer.variance.copy()
    count = policy.normalizer.count
    try:
        for multiplier in (1.0, 10.0, 100.0):
            first, _ = policy.predict(observation * multiplier)
            repeated, _ = policy.predict(observation * multiplier)
            np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(policy.normalizer.mean, mean)
        np.testing.assert_array_equal(policy.normalizer.variance, variance)
        assert policy.normalizer.count == count
        assert not policy.normalizer.mean.flags.writeable
        assert not policy.normalizer.variance.flags.writeable
    finally:
        env.close()


def test_training_preserves_existing_run(trained_runs):
    """@brief A new experiment cannot overwrite an existing checkpoint directory."""
    manifest, run_a, _, _ = trained_runs
    with pytest.raises(FileExistsError, match="not empty"):
        train_policy(manifest, run_a, TrainingConfig(total_timesteps=64))


def test_policy_rejects_mixed_or_corrupted_artifacts(trained_runs, tmp_path):
    """@brief A replaced normalization archive is rejected before model loading."""
    _, run_a, _, _ = trained_runs
    (tmp_path / "training.json").write_bytes((run_a / "training.json").read_bytes())
    (tmp_path / "policy.zip").write_bytes((run_a / "policy.zip").read_bytes())
    (tmp_path / "observation_normalization.npz").write_bytes(b"incorrect statistics")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_policy(tmp_path)


@pytest.mark.parametrize(
    "values",
    [
        {"total_timesteps": 0},
        {"n_steps": 1},
        {"n_steps": 30, "batch_size": 16},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"gamma": 1.1},
        {"ent_coef": -0.1},
        {"episode_cache_size": 0},
    ],
)
def test_training_rejects_invalid_hyperparameters(values):
    """@brief Invalid configurations fail before data loading or training."""
    with pytest.raises(ValueError):
        TrainingConfig(**values).validate()


def test_frozen_normalizer_validates_shapes_values_and_clips():
    """@brief Inference rejects malformed inputs and clips finite extremes."""
    normalizer = FrozenNormalizer(
        mean=np.array([1.0, 2.0]), variance=np.array([4.0, 9.0]),
        count=8.0, epsilon=1e-8, clip_obs=10.0,
    )
    np.testing.assert_allclose(normalizer.normalize(np.array([1.0, 2.0])), [0, 0])
    np.testing.assert_allclose(
        normalizer.normalize(np.array([[1e9, -1e9]])), [[10, -10]]
    )
    with pytest.raises(ValueError, match="features"):
        normalizer.normalize(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        normalizer.normalize(np.array([float("nan"), 2.0]))
    with pytest.raises(ValueError, match="non-negative"):
        FrozenNormalizer(
            mean=np.array([0.0]), variance=np.array([-1.0]),
            count=1.0, epsilon=1e-8, clip_obs=10,
        )


def test_lazy_replay_opens_only_selected_archives_and_bounds_cache(trained_runs, monkeypatch):
    """@brief Construction is metadata-only; reset verifies cache misses on demand.

    @details Held-out files are already absent in the fixture. The selected loader
    is observed directly to prove that an unselected training archive is not read
    eagerly and that evicted archives are reverified rather than trusted forever.
    """
    manifest, _, _, _ = trained_runs
    opened = []
    original = training_module.load_manifest_entry

    def observed_load(path, entry):
        """@brief Record the archive boundary before using the real checked loader."""
        opened.append(entry["path"])
        return original(path, entry)

    monkeypatch.setattr(training_module, "load_manifest_entry", observed_load)
    env = ArchiveReplayEnv(manifest, cache_size=1)
    assert opened == []
    assert env.cached_episode_count == 0
    try:
        env.reset(seed=3, options={"episode_index": 0})
        first = env.entries[0]["path"]
        assert opened == [first]
        env.reset(options={"episode_index": 0})
        assert opened == [first]
        env.reset(options={"episode_index": 1})
        env.reset(options={"episode_index": 0})
        assert opened == [first, env.entries[1]["path"], first]
        assert env.cached_episode_count == 1
        assert env.cache_miss_loads == 3
        assert len(env.verified_files) == 2
        assert all(entry["split"] == "train" for entry in env.entries)
    finally:
        env.close()
    assert env.cached_episode_count == 0


def test_lazy_replay_selection_is_seeded_and_cache_independent(trained_runs):
    """@brief Cache size and hit patterns cannot change the episode RNG stream."""
    manifest, _, _, _ = trained_runs
    small, large = ArchiveReplayEnv(manifest, cache_size=1), ArchiveReplayEnv(manifest, cache_size=3)
    try:
        sequence_a, sequence_b = [], []
        for iteration in range(8):
            kwargs = {"seed": 23} if iteration == 0 else {}
            observation_a, info_a = small.reset(**kwargs)
            observation_b, info_b = large.reset(**kwargs)
            sequence_a.append(info_a["training_episode_index"])
            sequence_b.append(info_b["training_episode_index"])
            np.testing.assert_array_equal(observation_a, observation_b)
        assert sequence_a == sequence_b
        assert len(set(sequence_a)) > 1
        assert small.cached_episode_count <= 1
        assert large.cached_episode_count <= 3
    finally:
        small.close()
        large.close()


def test_lazy_replay_rejects_tampered_selected_archive(tmp_path):
    """@brief File integrity is checked when an archive is first selected."""
    manifest = create_demo(tmp_path / "data", days=3, steps_per_day=32, seed=5)
    env = ArchiveReplayEnv(manifest, cache_size=1)
    selected = manifest.parent / env.entries[0]["path"]
    selected.write_bytes(b"damaged archive")
    try:
        with pytest.raises(ValueError, match="hash"):
            env.reset(seed=1, options={"episode_index": 0})
    finally:
        env.close()


def test_trade_diagnostics_cover_first_completed_episode(trained_runs):
    """@brief Completed-trade counts and unscaled economics are logged from outset."""
    _, run_a, _, metadata = trained_runs
    with (run_a / "training_episodes.monitor.csv").open(encoding="utf-8") as handle:
        episodes = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    assert episodes
    assert all(int(row["trade_count"]) >= 0 for row in episodes)
    assert all(np.isfinite(float(row["pnl"])) for row in episodes)
    assert all(float(row["fees_paid"]) >= 0 for row in episodes)
    records = [
        json.loads(line)
        for line in (run_a / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["completed_episodes_total"] == len(episodes)
    assert records[-1]["closed_trades_total"] == sum(int(row["trade_count"]) for row in episodes)
    assert records[-1]["recent_mean_trade_count"] is not None
    assert 0 <= records[-1]["zero_trade_episode_fraction"] <= 1
    assert metadata["archive_replay"]["cache_size"] == 2
    assert metadata["archive_replay"]["verified_files"]


def test_training_cli_records_real_cadence_controls(trained_runs, tmp_path, capsys):
    """@brief The public CLI applies sparse-opportunity and 1 Hz replay controls."""
    manifest, _, _, _ = trained_runs
    output = tmp_path / "cadenced"
    result = training_module.main([
        "--manifest", str(manifest), "--output", str(output),
        "--steps", "64", "--n-steps", "32", "--batch-size", "16",
        "--epochs", "1", "--gamma", "0.999", "--decision-interval-ms", "1000",
        "--max-holding-ms", "30000", "--episode-cache-size", "1",
    ])
    assert result == 0
    metadata = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert metadata["training_config"]["gamma"] == 0.999
    assert metadata["environment_config"]["decision_interval_ms"] == 1000
    assert metadata["environment_config"]["max_holding_ms"] == 30000
    assert metadata["archive_replay"]["cache_size"] == 1
    assert metadata["held_out_data_loaded"] is False
    assert json.loads(capsys.readouterr().out)["actual_timesteps"] == 64
