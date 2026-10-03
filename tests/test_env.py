"""@brief Hand-calculated contract tests for delayed fills and portfolio rewards.

@details Prices and quantities deliberately use small round numbers. Assertions
verify economic outcomes independently of environment implementation formulas.
"""
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from latency_arb.data.schema import ReplayEpisode
from latency_arb.env.latency_sim import Action, EnvConfig, LatencySimEnv, OBSERVATION_NAMES


EPOCH_NS = 1_700_000_000_000_000_000


def book_episode(mids=None, n=12, interval_ms=100, half_spread=1.0, sizes=10.0,
                 funding=None) -> ReplayEpisode:
    """@brief Construct a valid one-level episode with exact integer receive times."""
    mids = np.full(n, 100.0, dtype=np.float64) if mids is None else np.asarray(mids, dtype=np.float64)
    n = len(mids)
    zero = np.zeros(n, dtype=np.float64)
    return ReplayEpisode(
        timestamp_ns=EPOCH_NS + np.arange(n, dtype=np.int64) * int(interval_ms * 1e6),
        binance_bid=mids - 0.5, binance_ask=mids + 0.5,
        hl_bid_prices=(mids - half_spread)[:, None],
        hl_bid_sizes=np.full((n, 1), sizes, dtype=np.float64),
        hl_ask_prices=(mids + half_spread)[:, None],
        hl_ask_sizes=np.full((n, 1), sizes, dtype=np.float64),
        binance_imbalance=zero.copy(), hl_imbalance=zero.copy(), volatility_bps=zero.copy(),
        hl_quote_age_ms=zero.copy(), hl_received_age_ms=zero.copy(),
        funding_rate=zero.copy() if funding is None else np.asarray(funding, dtype=np.float64),
        metadata={"synthetic": True},
    )


def config(**overrides) -> EnvConfig:
    """@brief Use one BTC and zero extra slippage so expected costs are transparent."""
    return replace(EnvConfig(position_size_btc=1.0, slippage_bps=0,
                             initial_cash=1_000, max_drawdown_usd=500), **overrides)


def finish(env, action=Action.HOLD):
    """@brief Replay to completion while accumulating the actual returned rewards."""
    rewards = 0.0
    while not env._done:
        _, reward, _, _, _ = env.step(action)
        rewards += reward
    return rewards


@pytest.mark.parametrize("side", [Action.LONG, Action.SHORT])
def test_round_trip_charges_spread_and_both_side_fees(side):
    """@brief A stationary 99/101 book loses 2 USD spread and 0.07 USD fees."""
    env = LatencySimEnv(book_episode(), config())
    env.reset(seed=7)
    _, first_reward, _, _, _ = env.step(side)
    _, second_reward, _, _, _ = env.step(Action.HOLD)
    _, third_reward, _, _, _ = env.step(Action.EXIT)
    total = first_reward + second_reward + third_reward + finish(env)
    assert env.trade_count == 1
    assert env.trades[0]["gross_pnl"] == pytest.approx(-2)
    assert env.fees_paid == pytest.approx(0.07)
    assert env.pnl == pytest.approx(-2.07)
    assert total == pytest.approx(-2.07)
    assert env.trades[0]["net_pnl"] == pytest.approx(env.pnl)
    assert env.inventory == 0


def test_150ms_delay_fills_future_book_and_preserves_nanoseconds():
    """@brief Decisions at 0/200 ms execute at 200/400 ms on their actual books."""
    episode = book_episode(mids=[100, 100, 110, 115, 120, 120, 120, 120])
    env = LatencySimEnv(episode, config())
    env.reset()
    observation, reward, _, _, _ = env.step(Action.LONG)
    assert reward == 0
    assert env.inventory == 0
    assert observation[OBSERVATION_NAMES.index("pending_action")] == Action.LONG
    assert observation[OBSERVATION_NAMES.index("pending_remaining_ms")] == 50
    env.step(Action.HOLD)
    assert env.fills[0]["price"] == 111
    assert env.fills[0]["timestamp_ns"] == EPOCH_NS + 200_000_000
    assert env.fills[0]["decision_timestamp_ns"] == EPOCH_NS
    env.step(Action.EXIT)
    env.step(Action.HOLD)
    assert env.fills[1]["price"] == 119
    assert env.trades[0]["net_pnl"] == pytest.approx(8 - 0.0805)


def test_unchanged_past_produces_identical_observation_and_pending_state():
    """@brief Altering later prices cannot change earlier decisions or observations."""
    left = book_episode()
    right = book_episode(mids=[100, 100, 100, 200, 300, 400, 500, 600, 600, 600, 600, 600])
    a, b = LatencySimEnv(left, config()), LatencySimEnv(right, config())
    np.testing.assert_array_equal(a.reset(seed=1)[0], b.reset(seed=1)[0])
    first_a, first_b = a.step(Action.LONG), b.step(Action.LONG)
    np.testing.assert_array_equal(first_a[0], first_b[0])
    assert first_a[1:] == first_b[1:]
    assert a._pending == b._pending


@pytest.mark.parametrize("side,expected", [(Action.LONG, 1.0), (Action.SHORT, -1.0)])
def test_funding_sign_and_settlement_precedes_same_row_entry(side, expected):
    """@brief Entry-row funding is not owed; carried longs pay and shorts receive."""
    rates = np.zeros(12)
    rates[2], rates[3], rates[5] = 0.02, 0.003, 0.007
    env = LatencySimEnv(book_episode(funding=rates), config(fee_bps=0))
    env.reset()
    env.step(side)
    env.step(Action.HOLD)  # Entry executes after row-two settlement.
    assert env.funding_paid == 0
    env.step(Action.HOLD)  # Carry through 0.3 USD event.
    env.step(Action.EXIT)
    env.step(Action.HOLD)  # Exit row-five event is owed before the close.
    assert env.funding_paid == pytest.approx(expected)
    assert env.trades[0]["funding_paid"] == pytest.approx(expected)
    assert env.pnl == pytest.approx(-2 - expected)


def test_depth_vwap_uses_multiple_levels_without_partial_entry():
    """@brief Buying one BTC consumes 0.25 at 101 and 0.75 at 102, totaling 101.75."""
    episode = book_episode()
    episode.hl_ask_prices = np.tile([101., 102.], (12, 1))
    episode.hl_bid_prices = np.tile([99., 98.], (12, 1))
    episode.hl_ask_sizes = np.tile([0.25, 2.], (12, 1))
    episode.hl_bid_sizes = np.tile([0.25, 2.], (12, 1))
    env = LatencySimEnv(episode, config(fee_bps=0))
    env.reset()
    env.step(Action.LONG)
    env.step(Action.HOLD)
    assert env.fills[0]["price"] == pytest.approx(101.75)
    env.step(Action.EXIT)
    env.step(Action.HOLD)
    assert env.fills[1]["price"] == pytest.approx(98.25)
    assert env.pnl == pytest.approx(-3.5)
    shallow = LatencySimEnv(book_episode(sizes=0.5), config())
    shallow.reset()
    shallow.step(Action.LONG)
    shallow.step(Action.HOLD)
    assert shallow.inventory == 0
    assert shallow.fees_paid == 0
    assert shallow.rejection_count == 1


def test_forced_exit_penalizes_unseen_residual_depth():
    """@brief Terminal exit prices unobserved residual at worse than displayed depth."""
    episode = book_episode()
    episode.hl_bid_sizes[-1, 0] = 0.25
    env = LatencySimEnv(episode, config(fee_bps=0, forced_liquidity_penalty_bps=100))
    env.reset()
    env.step(Action.LONG)
    finish(env)
    # @details 0.25 BTC at 99 and 0.75 at 98.01 gives a 98.2575 USD forced sale.
    assert env.trades[0]["exit_price"] == pytest.approx(98.2575)
    assert env.trades[0]["extrapolated_exit_quantity_btc"] == pytest.approx(0.75)
    assert env.pnl == pytest.approx(98.2575 - 101)


def test_voluntary_exit_insufficient_depth_does_not_erase_inventory():
    """@brief Failed voluntary closure preserves exposure until a costed mandatory exit."""
    episode = book_episode()
    episode.hl_bid_sizes[4, 0] = 0.25
    env = LatencySimEnv(episode, config(fee_bps=0))
    env.reset()
    env.step(Action.LONG)
    env.step(Action.HOLD)
    env.step(Action.EXIT)
    env.step(Action.HOLD)
    assert env.inventory == 1
    assert env.trade_count == 0
    assert env.rejection_count == 1
    finish(env)
    assert env.trade_count == 1
    assert env.pnl == pytest.approx(-2)


def test_spread_and_slippage_stress_are_paid_on_both_sides():
    """@brief Doubling spread gives 98/102; 10 bps slippage produces 97.902/102.102."""
    env = LatencySimEnv(book_episode(), config(fee_bps=0, spread_multiplier=2, slippage_bps=10))
    env.reset()
    env.step(Action.LONG)
    finish(env)
    assert env.trades[0]["entry_price"] == pytest.approx(102.102)
    assert env.trades[0]["exit_price"] == pytest.approx(97.902)
    assert env.pnl == pytest.approx(-4.2)


@pytest.mark.parametrize("age_field", ["hl_quote_age_ms", "hl_received_age_ms"])
def test_freshness_is_checked_at_decision_and_delayed_fill(age_field):
    """@brief Neither stale source quotes nor stale receipts can open a position."""
    episode = book_episode()
    getattr(episode, age_field)[0] = 1001
    env = LatencySimEnv(episode, config())
    env.reset()
    _, _, _, _, info = env.step(Action.LONG)
    assert info["action_result"] == "stale_entry_rejected"
    getattr(episode, age_field)[0] = 0
    getattr(episode, age_field)[2] = 1001
    env.reset()
    env.step(Action.LONG)
    env.step(Action.HOLD)
    assert env.inventory == 0
    assert env.rejection_count == 1


def test_policy_cancellation_and_no_pyramiding_or_instant_reversal():
    """@brief EXIT can cancel a pending entry; opposite entries cannot reverse exposure."""
    env = LatencySimEnv(book_episode(), config())
    env.reset()
    env.step(Action.LONG)
    env.step(Action.EXIT)
    assert not env.fills
    assert env.cancellation_count == 1
    env.step(Action.LONG)
    env.step(Action.LONG)
    assert env.inventory == 1
    _, _, _, _, info = env.step(Action.SHORT)
    assert info["action_result"] == "position_limit"
    assert env.inventory == 1
    assert len(env.fills) == 1


def test_terminal_exit_is_costed_and_late_entries_are_rejected_or_cancelled():
    """@brief The horizon closes carried inventory and never turns a pending entry into profit."""
    env = LatencySimEnv(book_episode(), config(reward_scale=7))
    env.reset()
    _, reward, _, _, _ = env.step(Action.LONG)
    rewards = reward + finish(env)
    assert env.trades[0]["exit_reason"] == "episode_end"
    assert env.fills[-1]["decision_timestamp_ns"] == env.episode.timestamp_ns[-1] - 150_000_000
    assert rewards / 7 == pytest.approx(env.pnl)
    late = LatencySimEnv(book_episode(n=3), config())
    late.reset()
    _, reward, _, _, info = late.step(Action.LONG)
    assert info["action_result"] == "insufficient_exit_time"
    assert reward == 0
    finish(late)
    assert late.pnl == 0
    # @details Irregular spacing can jump over an entry deadline to the final row.
    episode = book_episode(n=3)
    episode.timestamp_ns = EPOCH_NS + np.asarray([0, 100_000_000, 900_000_000], dtype=np.int64)
    pending = LatencySimEnv(episode, config())
    pending.reset()
    pending.step(Action.LONG)
    finish(pending)
    assert pending.cancellation_count == 1
    assert pending.pnl == 0


def test_drawdown_schedules_a_delayed_exit_and_reports_realized_overshoot():
    """@brief A 6 USD breach at 300 ms exits at 500 ms, paying the later 89 bid."""
    episode = book_episode(mids=[100, 100, 100, 95, 93, 90, 90, 90, 90, 90])
    env = LatencySimEnv(episode, config(fee_bps=0, max_drawdown_usd=5))
    env.reset()
    env.step(Action.LONG)
    env.step(Action.HOLD)
    _, _, terminated, truncated, info = env.step(Action.HOLD)
    assert terminated and not truncated
    assert info["termination_reason"] == "max_drawdown"
    assert env.fills[-1]["decision_timestamp_ns"] == EPOCH_NS + 300_000_000
    assert env.fills[-1]["timestamp_ns"] == EPOCH_NS + 500_000_000
    assert env.trades[0]["exit_price"] == 89
    assert env.pnl == pytest.approx(-12)
    assert env.inventory == 0
    with pytest.raises(RuntimeError):
        env.step(Action.HOLD)


def test_max_holding_exit_is_scheduled_from_entry_and_pays_all_costs():
    """@brief Entry at 200 ms with a 250 ms limit exits first available row at 500 ms."""
    env = LatencySimEnv(book_episode(), config(max_holding_ms=250))
    env.reset()
    env.step(Action.LONG)
    finish(env)
    assert env.trades[0]["exit_reason"] == "max_holding"
    assert env.trades[0]["holding_time_ms"] == 300
    assert env.fills[-1]["due_timestamp_ns"] == EPOCH_NS + 450_000_000
    assert env.pnl == pytest.approx(-2.07)


def test_zero_latency_is_an_explicit_stress_setting_and_checks_entry_cost_risk():
    """@brief Zero delay fills current prices and immediately respects a cost-triggered stop."""
    env = LatencySimEnv(book_episode(), config(latency_ms=0, fee_bps=0, max_drawdown_usd=0.5))
    env.reset()
    _, reward, done, _, _ = env.step(Action.LONG)
    assert done
    assert env.fills[0]["timestamp_ns"] == EPOCH_NS
    assert env.fills[1]["timestamp_ns"] == EPOCH_NS
    assert reward == pytest.approx(-2)


def test_hold_has_no_action_bonus_and_disabled_history_retains_metrics():
    """@brief The abstention baseline earns zero; audit-disabled training still accounts."""
    env = LatencySimEnv(book_episode(), config(log_history=False))
    env.reset()
    assert finish(env) == 0
    assert env.pnl == 0
    env.reset()
    env.step(Action.LONG)
    finish(env)
    assert env.pnl == pytest.approx(-2.07)
    assert env.trade_count == 1
    assert not env.trades and not env.fills and not env.equity_history


def test_seeding_and_gymnasium_api_contract():
    """@brief Gym's independent checker verifies spaces, seeded reset, and step signatures."""
    env = LatencySimEnv([book_episode(), book_episode(mids=np.full(12, 200.))], config())
    check_env(env, skip_render_check=True)
    first = env.reset(seed=82)
    second = env.reset(seed=82)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert env.observation_space.contains(first[0])
    with pytest.raises(ValueError):
        env.step(9)
    with pytest.raises(ValueError):
        env.reset(options={"episode_index": 2})


@pytest.mark.parametrize("kwargs", [dict(fee_bps=-1), dict(latency_ms=-1),
    dict(position_size_btc=0), dict(spread_multiplier=0.5), dict(reward_scale=0),
    dict(max_drawdown_usd=np.inf), dict(max_holding_ms=10)])
def test_invalid_execution_parameters_fail_early(kwargs):
    """@brief Invalid cost/risk settings cannot silently produce optimistic rewards."""
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_replay_gaps_must_be_split_before_trading():
    """@brief Refuse to trade through missing five-second-plus market intervals."""
    episode = book_episode(interval_ms=6000)
    with pytest.raises(ValueError, match="gap"):
        LatencySimEnv(episode, config())


def test_terminal_liquidation_uses_first_eligible_row_when_clocks_tie():
    """@brief Equal nanoseconds cannot defer a terminal order to a more favorable later book."""
    episode = book_episode(mids=[100, 100, 100, 100, 100, 200])
    episode.timestamp_ns[-1] = episode.timestamp_ns[-2]
    env = LatencySimEnv(episode, config(fee_bps=0, latency_ms=50))
    env.reset()
    env.step(Action.LONG)
    finish(env)
    assert env.fills[-1]["index"] == 4
    assert env.fills[-1]["price"] == 99
    assert env.pnl == pytest.approx(-2)


def timed_episode(times_ms, mids=None, funding=None):
    """@brief Keep irregular event timestamps exact while reusing the economic fixture."""
    episode = book_episode(n=len(times_ms), mids=mids, funding=funding)
    episode.timestamp_ns = EPOCH_NS + np.asarray(times_ms, dtype=np.int64) * 1_000_000
    return episode


def test_one_second_decisions_keep_exact_150ms_fills_and_both_side_costs():
    """@brief The policy acts at 0/1000 ms while executions occur at 150/1150 ms."""
    times = [0, 100, 150, 300, 1000, 1100, 1150, 1300, 2000, 2150, 3000]
    episode = timed_episode(times, mids=[100, 100, 110, 110, 120, 120, 130, 130, 125, 125, 125])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000))
    env.reset()
    observation, first_reward, done, _, first_info = env.step(Action.LONG)
    assert not done
    assert first_info["timestamp_ns"] == EPOCH_NS + 1_000_000_000
    assert env.fills[0]["timestamp_ns"] == EPOCH_NS + 150_000_000
    assert env.fills[0]["price"] == 111
    assert first_reward == pytest.approx(9 - 0.03885)
    assert observation[OBSERVATION_NAMES.index("holding_time_ms")] == 850
    _, second_reward, done, _, second_info = env.step(Action.EXIT)
    assert not done
    assert second_info["timestamp_ns"] == EPOCH_NS + 2_000_000_000
    assert env.fills[1]["timestamp_ns"] == EPOCH_NS + 1_150_000_000
    assert env.fills[1]["price"] == 129
    assert first_reward + second_reward == pytest.approx(18 - 0.084)
    assert len([row for row in env.action_history if row["event"] == "decision"]) == 2
    assert list(dict.fromkeys(row["timestamp_ns"] for row in env.equity_history)) == [EPOCH_NS + t * 1_000_000 for t in times[:9]]
    assert finish(env) == 0
    assert env.pnl == pytest.approx(17.916)


def test_decision_grid_does_not_drift_after_sparse_arrivals_or_fills():
    """@brief An arrival at 1150 ms crosses the 1000 ms tick, preserving next tick 2000 ms."""
    episode = timed_episode([0, 150, 1150, 2000, 2150, 3000])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, fee_bps=0))
    env.reset()
    _, _, _, _, first = env.step(Action.HOLD)
    assert first["timestamp_ns"] == EPOCH_NS + 1_150_000_000
    assert first["decision_target_timestamp_ns"] == EPOCH_NS + 1_000_000_000
    assert first["next_decision_timestamp_ns"] == EPOCH_NS + 2_000_000_000
    _, _, _, _, second = env.step(Action.LONG)
    assert second["timestamp_ns"] == EPOCH_NS + 2_000_000_000
    assert env.fills[0]["timestamp_ns"] == EPOCH_NS + 2_000_000_000
    env.step(Action.EXIT)
    assert env.fills[1]["timestamp_ns"] == EPOCH_NS + 2_150_000_000
    assert env.pnl == pytest.approx(-2)


def test_intermediate_funding_is_not_skipped_by_one_second_policy_cadence():
    """@brief A carried position pays the 300 ms settlement before the 1000 ms return."""
    times = [0, 150, 300, 1000, 1150, 2000]
    env = LatencySimEnv(timed_episode(times, funding=[0, 0, .01, 0, 0, 0]),
                        config(decision_interval_ms=1000, fee_bps=0))
    env.reset()
    _, reward, _, _, info = env.step(Action.LONG)
    assert reward == pytest.approx(-2)
    assert info["funding_paid"] == pytest.approx(1)
    env.step(Action.EXIT)
    assert env.trades[0]["net_pnl"] == pytest.approx(-3)
    assert env.pnl == pytest.approx(-3)


def test_intermediate_drawdown_closes_before_next_policy_decision():
    """@brief A 300 ms breach exits at 450 ms even when the policy cadence is one second."""
    times = [0, 150, 300, 450, 1000, 1150, 2000]
    episode = timed_episode(times, mids=[100, 100, 95, 90, 120, 120, 120])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, fee_bps=0, max_drawdown_usd=5))
    env.reset()
    _, reward, done, _, info = env.step(Action.LONG)
    assert done
    assert info["termination_reason"] == "max_drawdown"
    assert info["timestamp_ns"] == EPOCH_NS + 450_000_000
    assert env.fills[-1]["decision_timestamp_ns"] == EPOCH_NS + 300_000_000
    assert reward == pytest.approx(-12)
    assert len([row for row in env.action_history if row["event"] == "decision"]) == 1
    assert env.equity_history[-1]["timestamp_ns"] < EPOCH_NS + 1_000_000_000


def test_intermediate_holding_limit_and_terminal_boundary_stop_safely():
    """@brief Automatic exits use event time even when two policy decisions cannot fit."""
    episode = timed_episode([0, 150, 400, 1000, 1150, 2000])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, fee_bps=0, max_holding_ms=250))
    env.reset()
    _, reward, done, _, info = env.step(Action.LONG)
    assert not done and info["inventory"] == 0
    assert env.trades[0]["exit_timestamp_ns"] == EPOCH_NS + 400_000_000
    assert env.trades[0]["exit_reason"] == "max_holding"
    assert reward == pytest.approx(-2)
    short = LatencySimEnv(timed_episode([0, 150, 300, 400, 500]),
                          config(decision_interval_ms=1000, fee_bps=0))
    short.reset()
    _, reward, done, _, info = short.step(Action.LONG)
    assert done and info["inventory"] == 0
    assert info["timestamp_ns"] == EPOCH_NS + 500_000_000
    assert reward == pytest.approx(-2)


def test_cadenced_step_does_not_look_beyond_next_decision_timestamp():
    """@brief Changing post-tick prices leaves earlier observations, fills, and rewards identical."""
    times = [0, 150, 300, 1000, 1150, 2000]
    left = timed_episode(times)
    right = timed_episode(times, mids=[100, 100, 100, 100, 200, 300])
    a = LatencySimEnv(left, config(decision_interval_ms=1000))
    b = LatencySimEnv(right, config(decision_interval_ms=1000))
    np.testing.assert_array_equal(a.reset(seed=12)[0], b.reset(seed=12)[0])
    result_a, result_b = a.step(Action.LONG), b.step(Action.LONG)
    np.testing.assert_array_equal(result_a[0], result_b[0])
    assert result_a[1:] == result_b[1:]
    assert a.fills == b.fills
    assert a.equity_history == b.equity_history


def test_zero_latency_risk_is_checked_before_a_later_rebound_under_slow_cadence():
    """@brief Entry spread loss at zero delay stops immediately instead of awaiting a rebound."""
    episode = timed_episode([0, 150, 1000, 2000], mids=[100, 150, 150, 150])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, latency_ms=0,
                                       fee_bps=0, max_drawdown_usd=.5))
    env.reset()
    _, reward, done, _, info = env.step(Action.LONG)
    assert done
    assert reward == pytest.approx(-2)
    assert info["timestamp_ns"] == EPOCH_NS
    assert info["termination_reason"] == "max_drawdown"


def test_intermediate_peak_is_retained_even_if_decision_endpoints_hide_drawdown():
    """@brief Audit rows retain the 200 ms peak and 300 ms loss missed by endpoint metrics."""
    times = [0, 150, 200, 300, 1000, 1150, 2000]
    episode = timed_episode(times, mids=[100, 100, 120, 90, 100, 100, 100])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, fee_bps=0))
    env.reset()
    _, reward, _, _, info = env.step(Action.LONG)
    assert reward == pytest.approx(-1)
    assert info["max_drawdown"] == pytest.approx(30)
    assert max(row["equity"] for row in env.equity_history) == pytest.approx(1019)
    assert min(row["equity"] for row in env.equity_history) == pytest.approx(989)


@pytest.mark.parametrize("interval", [-1, np.inf, np.nan, 0.0000001])
def test_invalid_policy_cadence_is_rejected(interval):
    """@brief Cadence must be finite, nonnegative, and representable on the integer clock."""
    with pytest.raises(ValueError):
        config(decision_interval_ms=interval)



def test_equity_audit_preserves_peak_immediately_before_exit_costs():
    """@brief An exit at a newly raised mid retains its pre-fill peak and cost drawdown."""
    episode = timed_episode([0, 150, 1000, 1150, 2000], mids=[100, 100, 100, 120, 120])
    env = LatencySimEnv(episode, config(decision_interval_ms=1000, fee_bps=0))
    env.reset()
    env.step(Action.LONG)
    env.step(Action.EXIT)
    equities = np.asarray([row["equity"] for row in env.equity_history])
    observed_drawdown = np.max(np.maximum.accumulate(equities) - equities)
    assert max(equities) == pytest.approx(1019)
    assert equities[-1] == pytest.approx(1018)
    assert observed_drawdown == pytest.approx(env.max_drawdown)
