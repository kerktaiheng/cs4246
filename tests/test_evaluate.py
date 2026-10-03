"""@file test_evaluate.py
@brief Independent economic and selection checks for the evaluation harness.
@details Fixtures use simple constant books so fees and rewards have hand-computable
answers; they do not depend on the simulator's own price or accounting helpers.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from latency_arb.baseline import FlatPolicy, ThresholdPolicy
from latency_arb.data.schema import ReplayEpisode
from latency_arb.env.latency_sim import EnvConfig, OBSERVATION_NAMES
from latency_arb.evaluate import (
    aggregate_results, compare_policies, cost_sweep, fee_break_even,
    maximum_drawdown, rollout, select_threshold,
)


def episode() -> ReplayEpisode:
    """@brief Build a flat, liquid, one-dollar-spread book over two seconds.
    @details Binance has a persistent positive gap, so the threshold policy opens
    one long and holds until the scheduled end-of-episode close.
    """
    count = 21
    return ReplayEpisode(
        timestamp_ns=np.arange(count, dtype=np.int64) * 100_000_000,
        binance_bid=np.full(count, 100.2, dtype=np.float64),
        binance_ask=np.full(count, 100.3, dtype=np.float64),
        hl_bid_prices=np.full((count, 1), 99.99, dtype=np.float64),
        hl_bid_sizes=np.full((count, 1), 100.0, dtype=np.float64),
        hl_ask_prices=np.full((count, 1), 100.01, dtype=np.float64),
        hl_ask_sizes=np.full((count, 1), 100.0, dtype=np.float64),
        binance_imbalance=np.zeros(count), hl_imbalance=np.zeros(count),
        volatility_bps=np.zeros(count), hl_quote_age_ms=np.zeros(count),
        hl_received_age_ms=np.zeros(count), funding_rate=np.zeros(count),
        metadata={"day": "1970-01-01", "episode_id": "flat-book", "synthetic": True},
    )


def test_flat_policy_has_exactly_zero_cost_and_return():
    """@brief Abstention must produce no spurious action reward, fee, or trade."""
    result = rollout(FlatPolicy(), episode())
    assert result["metrics"]["net_pnl"] == 0.0
    assert result["metrics"]["fees_paid"] == 0.0
    assert result["metrics"]["trade_count"] == 0
    assert result["metrics"]["win_rate"] is None
    assert result["metrics"]["abstention_rate"] == 1.0
    assert result["metrics"]["decision_count"] == len(episode()) - 1


def test_roundtrip_is_full_episode_and_reconciles_fee_arithmetic():
    """@brief A flat-price long must lose one spread and exactly two taker fees."""
    config = EnvConfig(position_size_btc=1.0, fee_bps=3.5, slippage_bps=0.0)
    result = rollout(ThresholdPolicy(8.0), episode(), config)
    expected = -0.02 - (100.01 + 99.99) * 3.5 / 10_000
    assert result["metrics"]["net_pnl"] == pytest.approx(expected)
    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["decision_count"] > 5
    assert result["metrics"]["scaled_reward_sum"] == pytest.approx(expected * config.reward_scale)
    assert result["trades"][0]["net_pnl"] == pytest.approx(expected)


def test_reward_scale_cannot_change_reported_economic_return():
    """@brief PPO reward tuning must leave reported dollars and fill paths unchanged."""
    config = EnvConfig(position_size_btc=1.0, fee_bps=1.0, slippage_bps=0.0)
    first = rollout(ThresholdPolicy(8.0), episode(), config)
    second = rollout(ThresholdPolicy(8.0), episode(), replace(config, reward_scale=100))
    assert first["metrics"]["net_pnl"] == second["metrics"]["net_pnl"]
    assert second["metrics"]["scaled_reward_sum"] == pytest.approx(
        first["metrics"]["scaled_reward_sum"] * 100)


def test_drawdown_stitches_episode_resets_without_fake_recovery():
    """@brief Consecutive independent losses must remain cumulative in the report."""
    config = EnvConfig(position_size_btc=1.0, fee_bps=0, slippage_bps=0)
    result = rollout(ThresholdPolicy(), episode(), config)
    summary = aggregate_results([result, result])
    assert summary["net_pnl"] == pytest.approx(-0.04)
    assert summary["max_drawdown_usd"] == pytest.approx(0.04)
    assert maximum_drawdown([100, 110, 104, 108, 90]) == 20
    assert maximum_drawdown([]) == 0


def test_baseline_obeys_signed_inventory_and_pending_orders():
    """@brief A fixed rule should exit on convergence and never reverse or resubmit."""
    values = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
    gap = OBSERVATION_NAMES.index("gap_bps")
    inventory = OBSERVATION_NAMES.index("inventory")
    pending = OBSERVATION_NAMES.index("pending_action")
    values[pending] = -1
    policy = ThresholdPolicy(8)
    values[gap] = 10
    assert policy.predict(values)[0] == 1
    values[gap] = -10
    assert policy.predict(values)[0] == 2
    values[inventory] = 0.1
    assert policy.predict(values)[0] == 3
    values[pending] = 3
    assert policy.predict(values)[0] == 0


def test_cost_sweep_reexecutes_policies_and_brackets_are_honest():
    """@brief Larger fees worsen a fixed round trip; no-trade policies have no break-even."""
    config = EnvConfig(position_size_btc=1.0, slippage_bps=0)
    rows = cost_sweep([episode()], {"flat": FlatPolicy(), "threshold": ThresholdPolicy()},
                      config, [0.5, 3.5], [150], [1])
    trades = [row for row in rows if row["policy"] == "threshold"]
    assert trades[0]["net_pnl"] > trades[1]["net_pnl"]
    statuses = {row["policy"]: row["status"] for row in fee_break_even(rows)}
    assert statuses == {"flat": "no_trades", "threshold": "nonpositive_at_all_sampled_fees"}
    artificial = [
        {"policy": "p", "latency_ms": 150, "spread_multiplier": 1, "fee_bps": fee,
         "net_pnl": pnl, "trade_count": 2}
        for fee, pnl in [(0.5, 2), (1.5, -1)]
    ]
    bracket = fee_break_even(artificial)[0]
    assert bracket["highest_profitable_sampled_fee_bps"] == 0.5
    assert bracket["next_nonpositive_sampled_fee_bps"] == 1.5


def test_threshold_selection_never_requests_test_split(monkeypatch):
    """@brief Assert selection can only request validation, using an independent loader spy."""
    requested = []

    def fake_loader(path, split):
        """@brief Record the requested split and return a known constant-price fixture."""
        requested.append(split)
        return [episode()]

    monkeypatch.setattr("latency_arb.evaluate.load_manifest", fake_loader)
    selected = select_threshold("unused-manifest.json", [8, 100])
    assert requested == ["validation"]
    assert selected["entry_threshold_bps"] == 100
    assert selected["selection_split"] == "validation"


def test_comparison_writes_auditable_artifacts(tmp_path):
    """@brief Verify report output contains decisions, costs, fills, and trade records."""
    report = compare_policies([episode()], {"flat": FlatPolicy()}, output_dir=tmp_path)
    assert report["policies"]["flat"]["aggregate"]["synthetic"] is True
    assert (tmp_path / "comparison.json").is_file()
    assert (tmp_path / "flat" / "decisions.jsonl").read_text()
    assert (tmp_path / "flat" / "trades.jsonl").read_text() == ""


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_invalid_thresholds_rejected(threshold):
    """@brief Reject boundaries that could silently invalidate the fixed-rule control."""
    with pytest.raises(ValueError):
        ThresholdPolicy(threshold)



def timed_market(times_ms, mids, gaps_bps=None) -> ReplayEpisode:
    """@brief Build an independent irregular replay fixture with hand-computable costs.

    @param times_ms Exact integer millisecond offsets for quote and execution events.
    @param mids Hyperliquid midprices; each side is exactly one dollar from mid.
    @param gaps_bps Optional Binance-to-Hyperliquid midpoint gaps at each event.
    @details This fixture uses only public ReplayEpisode arrays, avoiding the
    simulator's price, timing, or accounting helpers as a source of expected results.
    """
    mids = np.asarray(mids, dtype=np.float64)
    gaps = np.zeros(len(mids)) if gaps_bps is None else np.asarray(gaps_bps, dtype=np.float64)
    binance_mids = mids * (1 + gaps / 10_000)
    zeros = np.zeros(len(mids), dtype=np.float64)
    return ReplayEpisode(
        timestamp_ns=np.asarray(times_ms, dtype=np.int64) * 1_000_000,
        binance_bid=binance_mids - .25, binance_ask=binance_mids + .25,
        hl_bid_prices=(mids - 1)[:, None], hl_ask_prices=(mids + 1)[:, None],
        hl_bid_sizes=np.full((len(mids), 1), 10., dtype=np.float64),
        hl_ask_sizes=np.full((len(mids), 1), 10., dtype=np.float64),
        binance_imbalance=zeros.copy(), hl_imbalance=zeros.copy(),
        volatility_bps=zeros.copy(), hl_quote_age_ms=zeros.copy(),
        hl_received_age_ms=zeros.copy(), funding_rate=zeros.copy(),
        metadata={"day": "1970-01-01", "episode_id": "timed-book", "synthetic": True},
    )


class ScriptedPolicy:
    """@brief Issue a short prescribed action sequence, then HOLD indefinitely.

    @details Tests control only policy requests. The real environment still decides
    which orders execute, when risk terminates the run, and which costs are charged.
    """

    def __init__(self, actions):
        """@brief Copy the requested actions into a private forward-only iterator."""
        self.actions = iter(actions)

    def predict(self, observation, deterministic=True):
        """@brief Return the next prescribed action without inspecting future data."""
        del observation, deterministic
        return next(self.actions, 0), None


def test_risk_stop_reports_actual_end_separately_from_planned_horizon():
    """@brief A 300 ms breach exits at 450 ms, rather than claiming two seconds of coverage."""
    market = timed_market([0, 150, 300, 450, 1000, 1150, 2000],
                          [100, 100, 95, 90, 120, 120, 120])
    config = EnvConfig(position_size_btc=1, fee_bps=0, slippage_bps=0,
                       initial_cash=1000, max_drawdown_usd=5, decision_interval_ms=1000)
    result = rollout(ScriptedPolicy([1]), market, config)
    metrics = result["metrics"]
    # @details Buy at 101 at 150 ms; the delayed stop sells at 89 at 450 ms.
    # The rebound at one second is outside this policy's actual evaluated exposure.
    assert metrics["net_pnl"] == pytest.approx(-12)
    assert metrics["trade_count"] == 1
    assert metrics["decision_count"] == 1
    assert metrics["termination_reason"] == "max_drawdown"
    assert metrics["end_timestamp_ns"] == 450_000_000
    assert metrics["scheduled_end_timestamp_ns"] == 2_000_000_000
    assert result["trades"][0]["exit_timestamp_ns"] == metrics["end_timestamp_ns"]


def test_reported_drawdown_includes_peaks_between_policy_decisions():
    """@brief Full replay marks retain a 30 USD intrasecond drawdown hidden by endpoints."""
    market = timed_market([0, 150, 200, 300, 1000, 1150, 2000],
                          [100, 100, 120, 90, 100, 100, 100])
    config = EnvConfig(position_size_btc=1, fee_bps=0, slippage_bps=0,
                       initial_cash=1000, max_drawdown_usd=100, decision_interval_ms=1000)
    result = rollout(ScriptedPolicy([1]), market, config)
    # @details Entry costs 101. The 120 mid marks equity at 1019; the subsequent
    # 90 mid marks 989. Policy endpoints are only 1000,999,998, hiding that excursion.
    assert result["metrics"]["max_drawdown_usd"] == pytest.approx(30)
    assert max(result["equity"]) == pytest.approx(1019)
    assert min(result["equity"]) == pytest.approx(989)
    endpoint_equity = [1000] + [row["equity_after"] for row in result["decisions"]]
    assert maximum_drawdown(endpoint_equity) == pytest.approx(2)
    assert aggregate_results([result])["max_drawdown_usd"] == pytest.approx(30)


def test_reference_opportunities_do_not_shrink_when_a_policy_stops_early():
    """@brief Risk and flat policies share the full-grid reference denominator despite different coverage."""
    market = timed_market([0, 150, 300, 450, 1000, 1150, 2000],
                          [100, 100, 95, 90, 120, 120, 120],
                          [8, 0, 0, 0, -8, 0, 8])
    config = EnvConfig(position_size_btc=1, fee_bps=0, slippage_bps=0,
                       initial_cash=1000, max_drawdown_usd=5, decision_interval_ms=1000)
    stopped = rollout(ScriptedPolicy([1]), market, config)
    flat = rollout(FlatPolicy(), market, config)
    # @details Scheduled decisions at 0 and 1000 ms both exceed the absolute seven
    # bps reference. The terminal 2000 ms row is not another policy opportunity.
    assert stopped["metrics"]["reference_opportunity_count"] == 2
    assert flat["metrics"]["reference_opportunity_count"] == 2
    assert stopped["metrics"]["reference_gap_bps"] == 7
    assert stopped["metrics"]["flat_decision_count"] == 1
    assert flat["metrics"]["flat_decision_count"] == 2
    assert stopped["metrics"]["trade_selectivity"] == pytest.approx(.5)
    assert stopped["metrics"]["flat_entry_rate"] == pytest.approx(1)
    assert flat["metrics"]["trade_selectivity"] == 0
    combined = aggregate_results([stopped, stopped])
    assert combined["reference_opportunity_count"] == 4
    assert combined["trade_selectivity"] == pytest.approx(.5)
    assert combined["flat_entry_rate"] == pytest.approx(1)


def test_reference_count_excludes_execution_rows_and_terminal_snapshot():
    """@brief Large gaps between policy decisions must not inflate the opportunity denominator."""
    market = timed_market([0, 150, 300, 1000, 1150, 1300, 2000],
                          [100] * 7, [0, 20, -20, 8, 20, -20, 20])
    result = rollout(FlatPolicy(), market, EnvConfig(decision_interval_ms=1000))
    assert result["metrics"]["decision_count"] == 2
    assert result["metrics"]["reference_opportunity_count"] == 1
    assert result["metrics"]["trade_selectivity"] == 0


def test_zero_reference_opportunities_remain_undefined_even_when_policy_trades():
    """@brief A strategy trading below seven bps cannot turn an empty denominator into zero selectivity."""
    market = timed_market([0, 150, 1000, 1150, 2000], [100] * 5)
    config = EnvConfig(position_size_btc=1, fee_bps=0, slippage_bps=0,
                       decision_interval_ms=1000)
    result = rollout(ScriptedPolicy([1]), market, config)
    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["reference_opportunity_count"] == 0
    assert result["metrics"]["trade_selectivity"] is None
    assert result["metrics"]["flat_entry_rate"] == pytest.approx(1)
    assert aggregate_results([result])["trade_selectivity"] is None


def test_reference_selectivity_can_exceed_one_for_below_threshold_trades():
    """@brief Two completed trades against one reference opportunity must report ratio two."""
    market = timed_market([0, 150, 1000, 1150, 2000, 2150, 3000, 3150, 4000],
                          [100] * 9, [8, 0, 0, 0, 0, 0, 0, 0, 0])
    config = EnvConfig(position_size_btc=1, fee_bps=0, slippage_bps=0,
                       decision_interval_ms=1000)
    result = rollout(ScriptedPolicy([1, 3, 1, 3]), market, config)
    # @details Requests at 0/1000 and 2000/3000 ms form two independent round trips.
    # Only the first decision satisfies the shared reference-gap criterion.
    assert result["metrics"]["trade_count"] == 2
    assert result["metrics"]["reference_opportunity_count"] == 1
    assert result["metrics"]["trade_selectivity"] == pytest.approx(2)
    assert result["metrics"]["flat_entry_rate"] == pytest.approx(1)
    assert aggregate_results([result])["trade_selectivity"] == pytest.approx(2)
