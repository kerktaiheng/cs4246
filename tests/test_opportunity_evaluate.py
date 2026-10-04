"""@file test_opportunity_evaluate.py
@brief Synthetic-only checks of complete opportunity-policy evaluation.
@details Fixtures have known spreads, fees, funding, and receive clocks. No
recorded dataset or held-out market outcome is accessed by these tests.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_arb.data.schema import ReplayEpisode
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import FEATURE_NAMES, OpportunityConfig, OpportunityReplayEnv
from latency_arb.opportunity_evaluate import (
    GapThresholdPolicy, SkipPolicy, evaluate_opportunity_policies,
)


def episode(day="2026-09-25", *, gap=12.0, seconds=8) -> ReplayEpisode:
    """@brief Build a liquid constant-price book on exact one-second/+150/+300 grids."""
    start = int(np.datetime64(day, "D").astype(np.int64)) * 86_400_000_000_000
    times = start + (np.arange(seconds, dtype=np.int64)[:, None] * 1_000_000_000
                     + np.array([0, 150_000_000, 300_000_000])).reshape(-1)
    count = len(times)
    zeros = np.zeros(count, dtype=np.float64)
    bn_mid = 100 * (1 + gap / 10000)
    return ReplayEpisode(
        timestamp_ns=times, binance_bid=np.full(count, bn_mid - 0.01),
        binance_ask=np.full(count, bn_mid + 0.01),
        hl_bid_prices=np.full((count, 1), 99.99), hl_bid_sizes=np.full((count, 1), 100.0),
        hl_ask_prices=np.full((count, 1), 100.01), hl_ask_sizes=np.full((count, 1), 100.0),
        binance_imbalance=zeros.copy(), hl_imbalance=zeros.copy(),
        volatility_bps=zeros.copy(), hl_quote_age_ms=zeros.copy(),
        hl_received_age_ms=zeros.copy(), funding_rate=zeros.copy(),
        metadata={"day": day, "episode_id": f"fixture-{day}", "synthetic": True},
    )


def execution(**changes) -> EnvConfig:
    """@brief Choose transparent one-BTC economics and a one-second holding deadline."""
    values = dict(position_size_btc=1.0, fee_bps=1.0, slippage_bps=0.0,
                  decision_interval_ms=1000.0, max_holding_ms=1000.0)
    return EnvConfig(**{**values, **changes})


def opportunity() -> OpportunityConfig:
    """@brief Use only present gaps and a known terminal reserve in test candidates."""
    return OpportunityConfig(min_gap_bps=6.0, exit_gap_bps=0.5, min_remaining_ms=1000)


def manifest(tmp_path: Path, episodes: list[ReplayEpisode],
             splits: list[str] | None = None) -> Path:
    """@brief Save hashed temporary fixture archives without touching recorded data."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, item in enumerate(episodes):
        split = splits[index] if splits is not None else "validation"
        item.metadata["split"] = split
        filename = f"episode-{index}.npz"
        target = item.save(tmp_path / filename)
        entries.append({
            "path": filename, "day": item.metadata["day"], "split": split,
            "rows": len(item), "start_timestamp_ns": int(item.timestamp_ns[0]),
            "end_timestamp_ns": int(item.timestamp_ns[-1]),
            "content_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": entries}))
    return path


def direct(item, policy, config, screen):
    """@brief Independently drive the public environment for economic comparison."""
    env = OpportunityReplayEnv(item, config, screen)
    observation, info = env.reset(seed=0)
    decisions = 0
    while True:
        if info["has_opportunity"]:
            action = int(policy.predict(observation, deterministic=True)[0])
            decisions += 1
        else:
            action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    result = {"info": info, "trades": list(env.base.trades),
              "fills": list(env.base.fills), "history": list(env.base.equity_history),
              "decisions": decisions}
    env.close()
    return result


def test_exact_economics_match_direct_environment_and_ledger(tmp_path):
    """@brief The aggregate must reproduce a continuous public-wrapper replay exactly."""
    item = episode()
    path = manifest(tmp_path / "data", [item])
    config = execution()
    expected = direct(item, GapThresholdPolicy(6), config, opportunity())
    report = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, config, opportunity(),
        tmp_path / "audit",
    )
    actual = report["policies"]["gap"]["aggregate"]
    assert actual["net_pnl"] == pytest.approx(expected["info"]["pnl"])
    assert actual["trade_count"] == len(expected["trades"]) > 0
    assert actual["macro_decision_count"] == expected["decisions"]
    assert actual["fees_paid"] == pytest.approx(sum(row["fee"] for row in expected["fills"]))
    roundtrip_loss = 0.02 + 200 * config.fee_bps / 10000
    assert actual["net_pnl"] == pytest.approx(-roundtrip_loss * actual["trade_count"])
    assert actual["net_return"] == pytest.approx(actual["net_pnl"] / config.initial_cash)
    saved = [json.loads(line) for line in (tmp_path / "audit/gap/trades.jsonl").read_text().splitlines()]
    assert sum(row["net_pnl"] for row in saved) == pytest.approx(actual["net_pnl"])
    assert (tmp_path / "audit/comparison.json").is_file()


def test_full_marked_drawdown_stitches_losses_without_reset_recovery(tmp_path):
    """@brief Two independent losing clips must remain one cumulative drawdown."""
    items = [episode("2026-09-25"), episode("2026-09-26")]
    path = manifest(tmp_path, items)
    expected_curve = [execution().initial_cash]
    cumulative = 0.0
    for item in items:
        result = direct(item, GapThresholdPolicy(6), execution(), opportunity())
        expected_curve.extend(execution().initial_cash + cumulative + row["equity"] -
                              execution().initial_cash for row in result["history"][1:])
        cumulative += result["info"]["pnl"]
    values = np.asarray(expected_curve)
    expected_drawdown = np.max(np.maximum.accumulate(values) - values)
    report = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, execution(), opportunity())
    summary = report["policies"]["gap"]["aggregate"]
    assert summary["net_pnl"] == pytest.approx(cumulative)
    assert summary["max_drawdown_usd"] == pytest.approx(expected_drawdown)
    assert summary["max_drawdown_usd"] > report["policies"]["gap"]["episodes"][0]["max_drawdown_usd"]
    assert len(summary["daily_pnl"]) == 2


class FrozenSpy:
    """@brief Fail on fitting and record deterministic raw-feature inference."""

    def __init__(self, action=0):
        """@brief Initialize externally visible reset and prediction counters."""
        self.action = action
        self.resets = 0
        self.calls = []

    def reset(self):
        """@brief Demonstrate policy state reset once for each complete clip."""
        self.resets += 1

    def predict(self, values, deterministic=True):
        """@brief Record raw observations and require deterministic evaluation."""
        assert deterministic is True
        assert np.asarray(values).shape == (len(FEATURE_NAMES),)
        self.calls.append(np.asarray(values).copy())
        return np.array(self.action), None

    def learn(self, *args, **kwargs):
        """@brief Reject accidental optimization inside an evaluator."""
        raise AssertionError("evaluation cannot learn")

    def fit(self, *args, **kwargs):
        """@brief Reject fitting feature transforms on held-out data."""
        raise AssertionError("evaluation cannot fit")


def test_no_opportunity_clips_are_retained_without_dummy_prediction(tmp_path):
    """@brief All-empty dates remain in coverage, capital, and exposure denominators."""
    path = manifest(tmp_path, [episode(gap=1.0), episode("2026-09-26", gap=1.0)])
    spy = FrozenSpy()
    report = evaluate_opportunity_policies(path, "validation", {"spy": spy},
                                          execution(), opportunity())
    summary = report["policies"]["spy"]["aggregate"]
    assert spy.resets == 2 and spy.calls == []
    assert summary["episode_count"] == 2
    assert summary["no_opportunity_episode_count"] == 2
    assert summary["trade_count"] == summary["macro_decision_count"] == 0
    assert summary["net_pnl"] == summary["fees_paid"] == summary["funding_paid"] == 0
    assert summary["win_rate"] is None and summary["skip_rate"] is None
    assert summary["exposure_rate"] == 0
    assert summary["daily_pnl"] == {"2026-09-25": 0.0, "2026-09-26": 0.0}


def test_skip_and_threshold_are_frozen_raw_feature_policies(tmp_path):
    """@brief Baselines receive the shared raw feature order and never fit or learn."""
    path = manifest(tmp_path, [episode(), episode("2026-09-26")])
    spy = FrozenSpy(0)
    report = evaluate_opportunity_policies(
        path, "validation", {"spy": spy, "skip": SkipPolicy()},
        execution(), opportunity())
    assert spy.resets == 2 and spy.calls
    index = FEATURE_NAMES.index("abs_gap_bps")
    assert spy.calls[0][index] == pytest.approx(12)
    assert report["policies"]["spy"]["aggregate"]["macro_skip_count"] == len(spy.calls)
    assert report["policies"]["skip"]["aggregate"]["net_pnl"] == 0
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    values[index] = 10
    assert GapThresholdPolicy(10).predict(values)[0] == 1
    assert GapThresholdPolicy(10.1).predict(values)[0] == 0


def test_selected_split_only_opens_selected_archives(tmp_path, monkeypatch):
    """@brief Missing train and test files cannot affect validation-only evaluation."""
    import latency_arb.opportunity_evaluate as evaluator

    path = manifest(tmp_path, [
        episode("2026-09-24"), episode("2026-09-25"), episode("2026-09-29")
    ], ["train", "validation", "test"])
    (tmp_path / "episode-0.npz").unlink()
    (tmp_path / "episode-2.npz").unlink()
    actual_loader = evaluator.load_manifest_entry
    opened = []

    def checked_loader(path, entry):
        """@brief Assert the evaluator never attempts an unselected archive read."""
        opened.append(entry["split"])
        assert entry["split"] == "validation"
        return actual_loader(path, entry)

    monkeypatch.setattr(evaluator, "load_manifest_entry", checked_loader)
    report = evaluator.evaluate_opportunity_policies(
        path, "validation", {"skip": SkipPolicy()}, execution(), opportunity())
    assert opened == ["validation"]
    assert report["selected_episode_count"] == 1


def test_nonoverlapping_portfolio_and_daily_trade_counts(tmp_path):
    """@brief Trades share one account and cannot overlap independent trial lifetimes."""
    path = manifest(tmp_path / "data", [episode()])
    report = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, execution(),
        opportunity(), tmp_path / "audit")
    trades = [json.loads(line) for line in (tmp_path / "audit/gap/trades.jsonl").read_text().splitlines()]
    assert len(trades) > 1
    assert all(later["entry_timestamp_ns"] >= earlier["exit_timestamp_ns"]
               for earlier, later in zip(trades, trades[1:]))
    summary = report["policies"]["gap"]["aggregate"]
    assert summary["daily_trades"] == {"2026-09-25": len(trades)}


def test_overlapping_archive_ranges_fail_before_any_replay(tmp_path):
    """@brief A corrupted manifest cannot double-count overlapping replay periods."""
    path = manifest(tmp_path, [episode(), episode("2026-09-26")])
    document = json.loads(path.read_text())
    document["episodes"][1]["start_timestamp_ns"] = document["episodes"][0]["start_timestamp_ns"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="overlap"):
        evaluate_opportunity_policies(path, "validation", {"skip": SkipPolicy()},
                                      execution(), opportunity())


def test_funding_and_reward_scaling_preserve_reported_economics(tmp_path):
    """@brief Funding debits the held position once while reward scaling changes no dollars."""
    item = episode()
    item.funding_rate[3] = 0.001
    path = manifest(tmp_path, [item])
    first = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, execution(), opportunity())
    second = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)},
        execution(reward_scale=100), opportunity())
    left, right = [result["policies"]["gap"]["aggregate"] for result in (first, second)]
    assert left["funding_paid"] == pytest.approx(0.1)
    assert left["net_pnl"] == right["net_pnl"]
    assert right["scaled_reward_sum"] == pytest.approx(left["net_pnl"] * 100)


def test_future_entry_rejection_remains_in_evaluation(tmp_path):
    """@brief Eligibility now cannot remove an order that lacks depth at its arrival."""
    item = episode()
    item.hl_ask_sizes[1, 0] = 0.001
    path = manifest(tmp_path, [item])
    report = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, execution(), opportunity())
    summary = report["policies"]["gap"]["aggregate"]
    assert summary["rejection_count"] >= 1
    assert summary["entry_request_count"] > summary["trade_count"]


def test_audit_curves_are_compact_and_existing_files_are_protected(tmp_path):
    """@brief Detail files are opt-in and never silently overwrite a previous evaluation."""
    path = manifest(tmp_path / "data", [episode()])
    output = tmp_path / "audit"
    evaluate_opportunity_policies(path, "validation", {"gap": GapThresholdPolicy(6)},
                                  execution(log_history=False), opportunity(), output)
    points = (output / "gap/equity_curve.jsonl").read_text().splitlines()
    assert 2 <= len(points) <= 6
    with pytest.raises(FileExistsError):
        evaluate_opportunity_policies(path, "validation", {"gap": GapThresholdPolicy(6)},
                                      execution(), opportunity(), output)


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_baseline_thresholds_are_rejected(threshold):
    """@brief Invalid rules cannot silently turn a fixed control into an arbitrary policy."""
    with pytest.raises(ValueError):
        GapThresholdPolicy(threshold)


def test_drawdown_includes_marks_between_macro_decisions(tmp_path):
    """@brief A temporary intratrade equity peak must survive macro-step aggregation."""
    item = episode()
    item.hl_bid_prices[2, 0] = 119.99
    item.hl_ask_prices[2, 0] = 120.01
    item.binance_bid[2] = 120 * 1.0012 - 0.01
    item.binance_ask[2] = 120 * 1.0012 + 0.01
    path = manifest(tmp_path, [item])
    report = evaluate_opportunity_policies(
        path, "validation", {"gap": GapThresholdPolicy(6)}, execution(), opportunity())
    summary = report["policies"]["gap"]["aggregate"]
    #! @details The position exists at +300 ms, where equity rises by about $20.
    #! It returns before the next whole-second decision, so endpoint-only curves
    #! would miss almost the entire subsequent peak-to-trough loss.
    assert summary["max_drawdown_usd"] > 19
    assert abs(summary["net_pnl"]) < 1
