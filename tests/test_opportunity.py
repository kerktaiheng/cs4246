"""@file test_opportunity.py
@brief Economic and causal contract tests for binary opportunity replay.
@details These fixtures are synthetic and use exact integer nanoseconds. Tests
compare complete execution ledgers or independent hand arithmetic, rather than
accepting improved activity as evidence of a trading edge.
"""
from dataclasses import fields, replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from latency_arb.data.schema import ReplayEpisode
from latency_arb.env.latency_sim import Action, EnvConfig, LatencySimEnv, OBSERVATION_NAMES
from latency_arb.opportunity import (
    FEATURE_NAMES, OpportunityConfig, OpportunityReplayEnv,
    candidate_indices, counterfactual_trade, features_at,
)

EPOCH = 1_700_000_000_000_000_000


def episode(seconds=40, gap=10.0):
    """@brief Make one-second decisions plus exact 150/300ms execution snapshots."""
    times = (np.arange(seconds, dtype=np.int64)[:, None] * 1_000_000_000
             + np.asarray([0, 150_000_000, 300_000_000], dtype=np.int64)).reshape(-1)
    n = len(times)
    hl = np.full(n, 10_000.0)
    bn = hl * (1 + gap / 10_000)
    zeros = np.zeros(n)
    return ReplayEpisode(
        timestamp_ns=EPOCH + times, binance_bid=bn - .25, binance_ask=bn + .25,
        hl_bid_prices=(hl - .5)[:, None], hl_ask_prices=(hl + .5)[:, None],
        hl_bid_sizes=np.ones((n, 1)), hl_ask_sizes=np.ones((n, 1)),
        binance_imbalance=zeros.copy(), hl_imbalance=zeros.copy(),
        volatility_bps=zeros.copy(), hl_quote_age_ms=zeros.copy(),
        hl_received_age_ms=zeros.copy(), funding_rate=zeros.copy(),
        metadata={"synthetic": True, "decision_indices": list(range(0, n, 3))},
    )


def execution(**changes):
    """@brief Share the requested actual-cost assumptions between all comparisons."""
    return replace(EnvConfig(fee_bps=4.5, latency_ms=150, slippage_bps=.1,
                             position_size_btc=.001, decision_interval_ms=1000,
                             max_holding_ms=30_000, log_history=True), **changes)


def set_prices(item, rows, mids, gaps):
    """@brief Edit synthetic current books while preserving uncrossed validity."""
    mids, gaps = np.asarray(mids), np.asarray(gaps)
    item.hl_bid_prices[rows, 0], item.hl_ask_prices[rows, 0] = mids - .5, mids + .5
    bn = mids * (1 + gaps / 10_000)
    item.binance_bid[rows], item.binance_ask[rows] = bn - .25, bn + .25


def copy_episode(item):
    """@brief Deep-copy source arrays for controlled future-only perturbations."""
    return ReplayEpisode(**{f.name: (dict(item.metadata) if f.name == "metadata"
                                     else getattr(item, f.name).copy())
                             for f in fields(ReplayEpisode)})


def direct_action(observation):
    """@brief Independent fixed-exit controller using only present observations."""
    value = dict(zip(OBSERVATION_NAMES, observation, strict=True))
    if value["pending_action"] >= 0:
        return Action.HOLD
    if value["inventory"] > 0 and value["gap_bps"] <= .5:
        return Action.EXIT
    if value["inventory"] < 0 and value["gap_bps"] >= -.5:
        return Action.EXIT
    return Action.HOLD


def complete(env, action=1):
    """@brief Collect every macro reward until the wrapper's actual termination."""
    observation, info = env.reset(seed=7)
    reward = 0.0
    while True:
        observation, value, done, truncated, info = env.step(action)
        reward += value
        if done or truncated:
            return reward, info


def test_candidate_screen_uses_current_clocks_depth_and_known_horizon():
    """@brief Later fill failure does not erase an otherwise eligible current row."""
    item = episode()
    item.hl_quote_age_ms[3] = 1001
    item.hl_received_age_ms[6] = 1001
    item.hl_ask_sizes[9, 0] = .0009
    item.hl_ask_sizes[1, 0] = .00001  # Future entry depth must not filter row zero.
    actual = candidate_indices(item, execution(), OpportunityConfig())
    np.testing.assert_array_equal(actual, [0, 12, 15, 18, 21, 24])
    assert all(int(item.timestamp_ns[index]) % 1_000_000_000 == 0 for index in actual)


def test_future_mutation_cannot_change_current_features_or_candidate_membership():
    """@brief Price, age, and depth changes strictly after t preserve its inputs."""
    before, config = episode(), execution()
    after = copy_episode(before)
    cutoff = 15  # t=5 seconds, with all history available.
    rows = np.arange(cutoff + 1, len(after))
    set_prices(after, rows, np.linspace(9000, 11000, len(rows)), np.full(len(rows), -20.0))
    after.hl_quote_age_ms[rows] = 5000
    after.hl_ask_sizes[rows] = .00001
    np.testing.assert_array_equal(features_at(before, cutoff, config), features_at(after, cutoff, config))
    screen = OpportunityConfig(min_remaining_ms=0)
    np.testing.assert_array_equal(candidate_indices(before, config, screen)[0:6],
                                  candidate_indices(after, config, screen)[0:6])


def test_features_use_original_prefix_exact_lags_and_cost_units():
    """@brief t=5s features use t=4/t=0, with both sides' fees and slippage."""
    item = episode()
    set_prices(item, np.asarray([0, 12, 15]), np.asarray([9900., 9950., 10000.]),
               np.asarray([2., 4., 10.]))
    vector = dict(zip(FEATURE_NAMES, features_at(item, 15, execution()), strict=True))
    assert vector["abs_gap_bps"] == pytest.approx(10)
    assert vector["gap_excess_roundtrip_cost_bps"] == pytest.approx(-.2)
    assert vector["directional_gap_1s_bps"] == pytest.approx(4)
    assert vector["directional_gap_5s_bps"] == pytest.approx(2)
    assert vector["directional_hl_return_1s_bps"] == pytest.approx((10000 / 9950 - 1) * 10000)
    assert vector["directional_hl_return_5s_bps"] == pytest.approx((10000 / 9900 - 1) * 10000)
    assert vector["history_1s_available"] == vector["history_5s_available"] == 1
    initial = dict(zip(FEATURE_NAMES, features_at(item, 0, execution()), strict=True))
    assert initial["history_1s_available"] == initial["history_5s_available"] == 0
    assert initial["directional_hl_return_5s_bps"] == 0


def test_short_features_orient_imbalance_and_lag_gap_to_entry_direction():
    """@brief A negative present gap makes bearish imbalance a positive feature."""
    item = episode(gap=-10)
    item.binance_imbalance[:] = -.4
    item.hl_imbalance[:] = .2
    values = dict(zip(FEATURE_NAMES, features_at(item, 15, execution()), strict=True))
    assert values["abs_gap_bps"] == pytest.approx(10)
    assert values["directional_binance_imbalance"] == pytest.approx(.4)
    assert values["directional_hl_imbalance"] == pytest.approx(-.2)
    assert values["directional_gap_5s_bps"] == pytest.approx(10)


def test_macro_trade_uses_150ms_books_and_reconciles_both_fees():
    """@brief One macro contains entry and convergence exit before the next candidate."""
    item = episode(seconds=6)
    set_prices(item, np.asarray([1, 2]), np.asarray([10001., 10001.]), np.asarray([10., 10.]))
    set_prices(item, np.asarray([3, 4, 5]), np.asarray([10020., 10021., 10021.]), np.zeros(3))
    env = OpportunityReplayEnv(item, execution(reward_scale=100), OpportunityConfig(min_remaining_ms=0))
    observation, initial = env.reset()
    observation, reward, done, _, info = env.step(1)
    buy, sell = 10001.5 * 1.00001, 10020.5 * .99999
    expected_fees = .001 * (buy + sell) * .00045
    expected = .001 * (sell - buy) - expected_fees
    assert not done and info["has_opportunity"]
    assert info["timestamp_ns"] == EPOCH + 2_000_000_000
    assert info["macro_underlying_steps"] == 2
    assert env.base.fills[0]["timestamp_ns"] == EPOCH + 150_000_000
    assert env.base.fills[1]["timestamp_ns"] == EPOCH + 1_150_000_000
    assert info["fees_paid"] == pytest.approx(expected_fees)
    assert info["pnl"] == pytest.approx(expected)
    assert reward == pytest.approx(expected * 100)
    assert env.base.trades[0]["net_pnl"] == pytest.approx(expected)
    np.testing.assert_array_equal(observation, features_at(item, 6, execution()))


def test_macro_rewards_equal_full_ledger_without_overlapping_positions():
    """@brief Repeated entries remain sequential and include every exit cost."""
    item = episode(seconds=12)
    # @details Every odd second converges; even seconds may expose a new entry.
    for second in range(1, 12, 2):
        rows = np.arange(second * 3, second * 3 + 3)
        set_prices(item, rows, np.full(3, 10000.), np.zeros(3))
    env = OpportunityReplayEnv(item, execution(reward_scale=23), OpportunityConfig(min_remaining_ms=0))
    reward, info = complete(env)
    assert info["trade_count"] == 6
    assert reward == pytest.approx(info["pnl"] * 23)
    assert sum(trade["net_pnl"] for trade in env.base.trades) == pytest.approx(info["pnl"])
    assert info["fees_paid"] > 0 and env.base.inventory == 0
    assert all(first["exit_timestamp_ns"] < second["entry_timestamp_ns"]
               for first, second in zip(env.base.trades, env.base.trades[1:]))


@pytest.mark.parametrize("gap", [10., -10.])
def test_counterfactual_suffix_matches_flat_prefix_replay(gap):
    """@brief Moving only a flat start preserves fills, funding, costs, and horizon."""
    item = episode(gap=gap)
    # @details These prefix returns remain visible to features even though the
    # counterfactual engine starts its economic account at the candidate at t=5.
    item.funding_rate[16] = .001  # Entry row settlement is not owed.
    item.funding_rate[18] = .002  # Carried position settlement is owed.
    rows = np.arange(21, 24)
    set_prices(item, rows, np.full(3, 10001.), np.zeros(3))
    costs, screen, index = execution(), OpportunityConfig(), 15
    label = counterfactual_trade(item, index, costs, screen)
    direct = LatencySimEnv(item, costs)
    observation, info = direct.reset()
    while info["index"] < index:
        observation, _, _, _, info = direct.step(Action.HOLD)
    action = Action.LONG if gap > 0 else Action.SHORT
    while True:
        observation, _, done, _, info = direct.step(action)
        if done or (info["inventory"] == 0 and info["pending_action"] is None):
            break
        action = direct_action(observation)
    for key in ("trade_count", "fees_paid", "rejection_count", "funding_paid"):
        assert label[key] == pytest.approx(info[key])
    assert label["net_pnl"] == pytest.approx(info["pnl"])
    assert label["end_timestamp_ns"] == info["timestamp_ns"]
    assert label["trade_count"] == 1


@pytest.mark.parametrize("failure", ["age", "receive_age", "depth"])
def test_future_fill_rejection_remains_a_zero_trade_label(failure):
    """@brief Entry opportunities cannot be filtered by eventual fill feasibility."""
    item = episode()
    if failure == "age":
        item.hl_quote_age_ms[1] = 1001
    elif failure == "receive_age":
        item.hl_received_age_ms[1] = 1001
    else:
        item.hl_ask_sizes[1, 0] = .00001
    costs, screen = execution(), OpportunityConfig()
    assert candidate_indices(item, costs, screen)[0] == 0
    label = counterfactual_trade(item, 0, costs, screen)
    assert label["net_pnl"] == label["fees_paid"] == label["trade_count"] == 0
    assert label["rejection_count"] == 1
    assert label["end_timestamp_ns"] == EPOCH + 1_000_000_000
    env = OpportunityReplayEnv(item, costs, screen)
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert reward == 0 and not done and info["has_opportunity"]
    assert info["rejection_count"] == 1 and info["trade_count"] == 0


def test_zero_opportunities_finish_safely_and_do_not_invent_trade_rewards():
    """@brief Reset may exhaust a quiet session; the first step terminates once."""
    env = OpportunityReplayEnv(episode(gap=1), execution(), OpportunityConfig())
    observation, info = env.reset(seed=4)
    assert not info["has_opportunity"] and np.isfinite(observation).all()
    assert info["pnl"] == 0
    observation, reward, done, truncated, info = env.step(1)
    assert done and not truncated and reward == 0
    assert info["trade_count"] == 0 and not env.base.fills
    with pytest.raises(RuntimeError):
        env.step(0)


def test_skip_advances_to_next_candidate_without_fees_or_entry():
    """@brief Abstention is a real zero-exposure action, never a reward bonus."""
    env = OpportunityReplayEnv(episode(), execution(), OpportunityConfig())
    env.reset()
    _, reward, done, _, info = env.step(0)
    assert not done and reward == 0 and info["trade_count"] == 0
    assert info["timestamp_ns"] == EPOCH + 1_000_000_000
    assert info["underlying_step_count"] == 1 and not env.base.fills


def test_maximum_holding_exit_uses_fill_deadline_without_new_horizon():
    """@brief A t=0.15 entry exits at t=30.15, not at a wrapper-created endpoint."""
    item = episode()
    env = OpportunityReplayEnv(item, execution(), OpportunityConfig())
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert done and info["trade_count"] == 1
    assert env.base.trades[0]["exit_timestamp_ns"] == EPOCH + 30_150_000_000
    assert env.base.trades[0]["exit_reason"] == "max_holding"
    assert info["timestamp_ns"] == int(item.timestamp_ns[-1])
    assert reward == pytest.approx(info["pnl"])


def test_risk_termination_and_reward_are_owned_by_base_environment():
    """@brief A drawdown trigger finishes its delayed exit before macro termination."""
    item = episode()
    rows = np.arange(2, len(item))
    set_prices(item, rows, np.full(len(rows), 9900.), np.full(len(rows), 10.))
    env = OpportunityReplayEnv(item, execution(max_drawdown_usd=.05), OpportunityConfig())
    env.reset()
    _, reward, done, _, info = env.step(1)
    assert done and info["termination_reason"] == "max_drawdown"
    assert info["trade_count"] == 1 and info["inventory"] == 0
    assert reward == pytest.approx(info["pnl"])
    assert info["timestamp_ns"] < int(item.timestamp_ns[-1])


def test_convergence_at_execution_row_is_not_an_extra_policy_decision():
    """@brief A brief 300ms convergence cannot submit an exit between whole seconds."""
    item = episode()
    set_prices(item, np.asarray([2]), np.asarray([10000.]), np.zeros(1))
    label = counterfactual_trade(item, 0, execution(), OpportunityConfig())
    assert label["end_timestamp_ns"] == EPOCH + 31_000_000_000
    assert label["trade_count"] == 1


def test_invalid_settings_and_indices_fail_explicitly():
    """@brief Unsupported clocks and bogus rows cannot silently alter training data."""
    item = episode()
    with pytest.raises(ValueError):
        OpportunityConfig(min_gap_bps=.5)
    with pytest.raises(ValueError):
        OpportunityConfig(min_remaining_ms=-1)
    with pytest.raises(ValueError):
        OpportunityReplayEnv(item, execution(decision_interval_ms=0))
    with pytest.raises(ValueError):
        features_at(item, -1, execution())
    with pytest.raises(ValueError):
        counterfactual_trade(item, 1, execution(), OpportunityConfig())
    env = OpportunityReplayEnv(item, execution())
    with pytest.raises(ValueError):
        env.reset(options={"start_index": 3})


def test_gymnasium_contract_for_normal_and_empty_episodes():
    """@brief Seeded replay and safe empty termination meet Gymnasium's API."""
    check_env(OpportunityReplayEnv(episode(), execution(), OpportunityConfig()), skip_render_check=True)
    check_env(OpportunityReplayEnv(episode(gap=0), execution(), OpportunityConfig()), skip_render_check=True)
