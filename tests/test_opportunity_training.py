"""@file test_opportunity_training.py
@brief Real warm-start/PPO tests with unavailable held-out archives.

@details Synthetic books use an exact one-second decision grid with +150/+300 ms
execution rows. Positive and adverse price paths test economic targets, not a
claim about real market profitability. No recorded test data are opened.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import latency_arb.agent.opportunity_train as training
from latency_arb.data.schema import ReplayEpisode, read_manifest, load_manifest_entry
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import FEATURE_NAMES, OpportunityConfig, candidate_indices, features_at


def _manifest(root: Path) -> Path:
    """@brief Build small, valid archives with causal opportunities and exact clocks."""
    root.mkdir(parents=True)
    entries = []
    days = [
        ("2026-09-03", "train", 1.0), ("2026-09-04", "train", -1.0),
        ("2026-09-05", "train", 0.0), ("2026-09-25", "validation", 1.0),
        ("2026-10-03", "test", 1.0),
    ]
    elapsed = np.asarray([second + offset for second in range(90)
                          for offset in (0.0, 0.15, 0.3)], dtype=np.float64)
    for number, (day, split, trend) in enumerate(days):
        start = int(np.datetime64(day, "ns").astype(np.int64))
        timestamps = start + np.rint(elapsed * 1e9).astype(np.int64)
        mid = 100_000.0 + trend * elapsed * 5.0
        leader = mid + 100.0
        levels = np.arange(5, dtype=np.float64)[None, :]
        bid, ask = mid[:, None] - 1.0 - levels, mid[:, None] + 1.0 + levels
        n = len(elapsed)
        episode = ReplayEpisode(
            timestamp_ns=timestamps, binance_bid=leader - 1.0,
            binance_ask=leader + 1.0, hl_bid_prices=bid,
            hl_bid_sizes=np.full((n, 5), 0.1, dtype=np.float64),
            hl_ask_prices=ask, hl_ask_sizes=np.full((n, 5), 0.1, dtype=np.float64),
            binance_imbalance=np.zeros(n), hl_imbalance=np.zeros(n),
            volatility_bps=np.ones(n), hl_quote_age_ms=np.zeros(n),
            hl_received_age_ms=np.zeros(n), funding_rate=np.zeros(n),
            metadata={"day": day, "split": split, "synthetic": True,
                      "episode_id": f"opportunity_fixture_{number}"},
        )
        filename = f"episode_{number}.npz"
        episode.save(root / filename)
        entries.append({
            "path": filename, "split": split, "day": day, "synthetic": True,
            "rows": n, "start_timestamp_ns": int(timestamps[0]),
            "end_timestamp_ns": int(timestamps[-1]),
            "content_sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest(),
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "episodes": entries}), encoding="utf-8")
    return manifest


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """@brief Fit two actual same-seed models after removing every held-out archive."""
    root = tmp_path_factory.mktemp("opportunity_training")
    manifest = _manifest(root / "data")
    document = json.loads(manifest.read_text())
    for entry in document["episodes"]:
        if entry["split"] != "train":
            (manifest.parent / entry["path"]).unlink()
    execution = EnvConfig(fee_bps=4.5, decision_interval_ms=1000.0)
    opportunity = OpportunityConfig(min_gap_bps=6.0)
    events = training.prepare_training_events(manifest, root / "events", execution, opportunity)
    settings = dict(
        seed=19, total_timesteps=32, warm_epochs=2, execution=execution,
        opportunity=opportunity, n_steps=16, batch_size=8, n_epochs=1,
        hidden_size=8, warm_batch_size=32,
    )
    first = training.train_opportunity_policy(manifest, root / "events", root / "first", **settings)
    second = training.train_opportunity_policy(manifest, root / "events", root / "second", **settings)
    return manifest, root, execution, opportunity, events, first, second


def test_event_preparation_preserves_positive_negative_and_zero_fill_targets(trained):
    """@brief Every causal candidate is retained independently of its future profit."""
    manifest, root, execution, opportunity, metadata, _, _ = trained
    _, entries = read_manifest(manifest, "train")
    expected = sum(len(candidate_indices(load_manifest_entry(manifest, entry), execution, opportunity))
                   for entry in entries)
    assert metadata["event_count"] == expected
    assert metadata["positive_target_count"] > 0
    assert metadata["negative_target_count"] > 0
    assert metadata["future_rows_used_only_for_targets"] is True
    assert metadata["eligible_episode_count"] == len(entries)
    with np.load(root / "events" / "events.npz", allow_pickle=False) as data:
        assert data["features"].shape == (expected, len(FEATURE_NAMES))
        first = load_manifest_entry(manifest, entries[int(data["episode_indices"][0])])
        expected_features = features_at(first, int(data["candidate_indices"][0]), execution)
        np.testing.assert_array_equal(data["features"][0], expected_features)
    normalizer = training._normalizer(root / "events" / "normalization.npz")
    assert normalizer.count == expected


def test_real_ppo_checkpoint_is_distinct_and_reproducible(trained):
    """@brief Actual PPO updates change the warm actor and reproduce under one seed."""
    _, _, _, _, _, first, second = trained
    warm = training.load_opportunity_policy(first["warmstart_dir"])
    final = training.load_opportunity_policy(first["ppo_dir"])
    repeated = training.load_opportunity_policy(second["ppo_dir"])
    assert first["actual_ppo_macro_steps"] == 32
    assert first["underlying_steps"] >= 32
    assert first["completed_training_trades"] > 0
    assert first["held_out_data_loaded"] is False
    assert warm.metadata["checkpoint_kind"] == "supervised_warm_start_only"
    assert warm.metadata["actual_ppo_macro_steps"] == 0
    assert final.metadata["checkpoint_kind"] == "supervised_warm_start_plus_on_policy_ppo"
    assert final.metadata["training_config"]["gamma"] == 1.0
    assert any(not torch.equal(value, final.model.policy.state_dict()[name])
               for name, value in warm.model.policy.state_dict().items())
    for name, value in final.model.policy.state_dict().items():
        assert torch.equal(value, repeated.model.policy.state_dict()[name]), name


def test_saved_probability_matches_model_and_normalizer_never_updates(trained):
    """@brief Raw-feature inference reproduces native probabilities with fixed moments."""
    _, root, _, _, _, first, _ = trained
    policy = training.load_opportunity_policy(first["ppo_dir"])
    with np.load(root / "events" / "events.npz", allow_pickle=False) as data:
        raw = data["features"][:4].copy()
    before = policy.normalizer.mean.copy()
    with torch.no_grad():
        expected = policy.model.policy.get_distribution(
            torch.as_tensor(policy.normalizer.normalize(raw))
        ).distribution.probs[:, 1].numpy()
    np.testing.assert_allclose(policy.trade_probability(raw), expected)
    assert policy.trade_probability(raw[0]) == pytest.approx(float(expected[0]))
    actions, _ = policy.predict(raw)
    np.testing.assert_array_equal(actions, expected > 0.5)
    policy.trade_probability(raw * 100)
    np.testing.assert_array_equal(policy.normalizer.mean, before)
    assert not policy.normalizer.mean.flags.writeable


def test_targets_cannot_leak_into_feature_selection(trained, tmp_path, monkeypatch):
    """@brief Changing only future outcome labels leaves candidates/features identical."""
    manifest, root, execution, opportunity, original, _, _ = trained
    call_count = 0

    def future_rejection(*args, **kwargs):
        """@brief Simulate unknown-at-decision rejection for every existing candidate."""
        nonlocal call_count
        call_count += 1
        return {"net_pnl": 0.0, "trade_count": 0, "fees_paid": 0.0,
                "rejection_count": 1, "end_timestamp_ns": 0}

    monkeypatch.setattr(training, "counterfactual_trade", future_rejection)
    changed = training.prepare_training_events(manifest, tmp_path / "changed", execution, opportunity)
    assert call_count == original["event_count"] == changed["event_count"]
    assert changed["zero_target_count"] == changed["event_count"]
    assert changed["rejected_target_count"] == changed["event_count"]
    with np.load(root / "events" / "events.npz", allow_pickle=False) as initial, np.load(
        tmp_path / "changed" / "events.npz", allow_pickle=False
    ) as altered:
        for name in ("features", "episode_indices", "candidate_indices", "timestamp_ns"):
            np.testing.assert_array_equal(initial[name], altered[name])


def test_existing_outputs_and_different_costs_are_rejected(trained):
    """@brief New fitting cannot overwrite checkpoints or silently relabel economic costs."""
    manifest, root, execution, opportunity, _, _, _ = trained
    with pytest.raises(FileExistsError, match="not empty"):
        training.prepare_training_events(manifest, root / "events", execution, opportunity)
    with pytest.raises(FileExistsError, match="not empty"):
        training.train_opportunity_policy(
            manifest, root / "events", root / "first", total_timesteps=32, warm_epochs=1
        )
    with pytest.raises(ValueError, match="economics"):
        training.train_opportunity_policy(
            manifest, root / "events", root / "wrong_cost",
            execution=EnvConfig(fee_bps=3.5, decision_interval_ms=1000),
        )


def test_policy_rejects_changed_numeric_statistics(trained, tmp_path):
    """@brief Saved policy hashes reject normalization from an unrelated run."""
    _, _, _, _, _, first, _ = trained
    source = Path(first["ppo_dir"])
    for name in ("policy.zip", "training.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    (tmp_path / "normalization.npz").write_bytes(b"changed moments")
    with pytest.raises(ValueError, match="checksum"):
        training.load_opportunity_policy(tmp_path)

