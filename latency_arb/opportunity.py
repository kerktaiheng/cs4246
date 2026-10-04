"""@file opportunity.py
@brief Causal binary entry decisions around the unchanged economic replay engine.
@details A macro transition ends at the next eligible flat entry decision. Its
reward sums every underlying marked-equity change, including the complete trade
and any delayed exit. Unequal macro durations are intended for undiscounted
finite-session PPO (gamma=1). Counterfactual labels are independent trade trials;
they must never be summed as an executable, nonoverlapping portfolio result.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from latency_arb.data.schema import ReplayEpisode
from latency_arb.env.latency_sim import Action, EnvConfig, LatencySimEnv, OBSERVATION_NAMES

SECOND_NS = 1_000_000_000
_GAP = OBSERVATION_NAMES.index("gap_bps")


@dataclass(frozen=True)
class OpportunityConfig:
    """@brief Present-information entry screen and fixed convergence exit rule.
    @param min_gap_bps Minimum absolute current gap for considering an entry.
    @param exit_gap_bps Close a long below this gap or a short above its negative.
    @param min_remaining_ms Require this much time before the original fixed end.
    @details The screen never consults later fill success, returns, or liquidity.
    """

    min_gap_bps: float = 6.0
    exit_gap_bps: float = 0.5
    min_remaining_ms: float = 31_000.0

    def __post_init__(self) -> None:
        """@brief Reject ambiguous thresholds before constructing any candidates."""
        values = (self.min_gap_bps, self.exit_gap_bps, self.min_remaining_ms)
        if not np.isfinite(values).all():
            raise ValueError("Opportunity settings must be finite")
        if not 0 <= self.exit_gap_bps < self.min_gap_bps:
            raise ValueError("Require 0 <= exit_gap_bps < min_gap_bps")
        if self.min_remaining_ms < 0:
            raise ValueError("min_remaining_ms must be nonnegative")


# @details Directional features multiply by the present entry direction, making
# economically mirrored long/short states comparable without exposing raw price.
# Lag availability masks distinguish a genuine zero return from missing history.
FEATURE_NAMES = (
    "abs_gap_bps", "gap_excess_roundtrip_cost_bps", "binance_spread_bps",
    "hl_spread_bps", "directional_binance_imbalance", "directional_hl_imbalance",
    "volatility_bps", "hl_quote_age_ms", "hl_received_age_ms",
    "directional_gap_1s_bps", "directional_gap_5s_bps",
    "directional_binance_return_1s_bps", "directional_binance_return_5s_bps",
    "directional_hl_return_1s_bps", "directional_hl_return_5s_bps",
    "history_1s_available", "history_5s_available", "time_remaining_ms",
)


def _check_index(episode: ReplayEpisode, index: int) -> int:
    """@brief Preserve exact row identity and reject Python's negative indexing."""
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer replay row")
    if not 0 <= index < len(episode):
        raise ValueError("index is outside the replay episode")
    return int(index)


def _check_cadence(episode: ReplayEpisode, execution: EnvConfig) -> None:
    """@brief Require the simulator policy clock to align with UTC whole seconds.
    @details The execution timeline remains untouched: only policy decisions are
    constrained. Exact replay timestamps, rather than array stride, define cadence.
    """
    if execution.decision_interval_ms != 1000.0:
        raise ValueError("Opportunity replay requires decision_interval_ms=1000")
    if int(episode.timestamp_ns[0]) % SECOND_NS:
        raise ValueError("Opportunity episodes must start on an integer second")


def candidate_indices(episode: ReplayEpisode, execution: EnvConfig,
                      opportunity: OpportunityConfig) -> np.ndarray:
    """@brief Find current-information entry opportunities on the integer grid.
    @return Sorted int64 original-episode indices, at most one per timestamp.
    @details Check the present gap, both HL freshness clocks, displayed entry-side
    quantity, and known terminal reserve only. Later stale/depth rejection remains
    possible and must remain in labels and evaluation. A minimum of two latency
    intervals additionally preserves the base simulator's strict entry reserve.
    """
    _check_cadence(episode, execution)
    timestamps = episode.timestamp_ns
    gap = episode.gap_bps
    remaining = timestamps[-1] - timestamps
    size = np.where(gap > 0, episode.hl_ask_sizes.sum(axis=1),
                    episode.hl_bid_sizes.sum(axis=1))
    mask = ((timestamps % SECOND_NS == 0)
            & (np.abs(gap) >= opportunity.min_gap_bps)
            & (episode.hl_quote_age_ms <= execution.max_quote_age_ms)
            & (episode.hl_received_age_ms <= execution.max_received_age_ms)
            & (size >= execution.position_size_btc)
            & (remaining >= int(round(opportunity.min_remaining_ms * 1e6)))
            & (remaining > 2 * int(round(execution.latency_ms * 1e6))))
    # @details If source messages share a timestamp, the policy only sees the
    # first row reached by the base clock. A later duplicate cannot become a new
    # decision merely because its book would pass the screen.
    first = np.r_[True, timestamps[1:] != timestamps[:-1]]
    return np.flatnonzero(mask & first).astype(np.int64)


def features_at(episode: ReplayEpisode, index: int, execution: EnvConfig) -> np.ndarray:
    """@brief Build the same causal feature vector for labels, training and inference.
    @param index Original episode index, including when an execution uses a suffix.
    @details Lag lookups are backward-as-of the exact t-1s/t-5s clock, never nearest
    or forward filled. Prefix history remains available at random training starts.
    The round-trip cost is a current top-book proxy: spread stress plus two fees
    and two slippage charges. It is not a prediction of the delayed fill or exit.
    Existing volatility is retained in its original sampled-row units. No future
    high/low, absolute price, realized label, or episode identifier is a feature.
    """
    index = _check_index(episode, index)
    now = int(episode.timestamp_ns[index])
    hl = float((episode.hl_bid_prices[index, 0] + episode.hl_ask_prices[index, 0]) / 2)
    bn = float((episode.binance_bid[index] + episode.binance_ask[index]) / 2)
    gap = (bn / hl - 1) * 10_000
    direction = 1.0 if gap >= 0 else -1.0
    bn_spread = float((episode.binance_ask[index] - episode.binance_bid[index]) / bn * 10_000)
    hl_spread = float((episode.hl_ask_prices[index, 0] - episode.hl_bid_prices[index, 0]) / hl * 10_000)
    cost = hl_spread * execution.spread_multiplier + 2 * (execution.fee_bps + execution.slippage_bps)
    # @details Missing lag history is encoded as zeros plus explicit masks. The
    # search array is limited to the observed prefix, including duplicate times.
    old_gaps, bn_returns, hl_returns, available = [], [], [], []
    for seconds in (1, 5):
        lag = int(np.searchsorted(episode.timestamp_ns[:index + 1],
                                  now - seconds * SECOND_NS, side="right")) - 1
        available.append(float(lag >= 0))
        if lag < 0:
            old_gaps.append(0.0)
            bn_returns.append(0.0)
            hl_returns.append(0.0)
            continue
        old_bn = float((episode.binance_bid[lag] + episode.binance_ask[lag]) / 2)
        old_hl = float((episode.hl_bid_prices[lag, 0] + episode.hl_ask_prices[lag, 0]) / 2)
        old_gaps.append(direction * (old_bn / old_hl - 1) * 10_000)
        bn_returns.append(direction * (bn / old_bn - 1) * 10_000)
        hl_returns.append(direction * (hl / old_hl - 1) * 10_000)
    values = [abs(gap), abs(gap) - cost, bn_spread, hl_spread,
              direction * episode.binance_imbalance[index],
              direction * episode.hl_imbalance[index], episode.volatility_bps[index],
              episode.hl_quote_age_ms[index], episode.hl_received_age_ms[index],
              *old_gaps, *bn_returns, *hl_returns, *available,
              (int(episode.timestamp_ns[-1]) - now) / 1e6]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Opportunity features must be a finite float32 vector")
    return result


def _fixed_action(observation: np.ndarray, info: dict,
                  opportunity: OpportunityConfig) -> Action:
    """@brief Apply convergence exits only on whole-second, nonpending decisions.
    @details HOLD never cancels delayed orders. Maximum holding, terminal closure,
    and drawdown remain exclusively enforced by the unchanged base environment.
    """
    if info["pending_action"] is not None or int(info["timestamp_ns"]) % SECOND_NS:
        return Action.HOLD
    inventory, gap = float(info["inventory"]), float(observation[_GAP])
    if ((inventory > 0 and gap <= opportunity.exit_gap_bps)
            or (inventory < 0 and gap >= -opportunity.exit_gap_bps)):
        return Action.EXIT
    return Action.HOLD


def _suffix(episode: ReplayEpisode, index: int) -> ReplayEpisode:
    """@brief Retain a causal start and the original terminal horizon using views.
    @details All original arrays, including funding events and execution snapshots,
    survive from index onward. Metadata indices are remapped for audit only. The
    original episode remains the source of lagged features, and is never modified.
    """
    arrays = {field.name: getattr(episode, field.name)[index:]
              for field in fields(ReplayEpisode) if field.name != "metadata"}
    metadata = dict(episode.metadata)
    metadata.update({"opportunity_original_start_timestamp_ns": int(episode.timestamp_ns[0]),
                     "opportunity_start_index": index,
                     "opportunity_original_end_timestamp_ns": int(episode.timestamp_ns[-1])})
    if "decision_indices" in metadata:
        metadata["decision_indices"] = [int(row) - index for row in metadata["decision_indices"] if row >= index]
    return ReplayEpisode(**arrays, metadata=metadata)


def counterfactual_trade(episode: ReplayEpisode, index: int, execution: EnvConfig,
                         opportunity: OpportunityConfig) -> dict:
    """@brief Replay one candidate's trade outcome without modifying source data.
    @return Economic label with net_pnl, trade_count, fees_paid, rejection_count,
        end_timestamp_ns and entry metadata; all values are unscaled USD/accounting.
    @details The trial ends once the entry has been processed and the account is
    flat with no pending order, or the unchanged simulator terminates. It reuses
    the original fixed terminal timestamp, not an invented label-length horizon.
    Labels may overlap in time and are unsuitable for portfolio P&L aggregation.
    """
    index = _check_index(episode, index)
    if not np.any(candidate_indices(episode, execution, opportunity) == index):
        raise ValueError("counterfactual index is not a causal entry candidate")
    base = LatencySimEnv(_suffix(episode, index), execution)
    observation, info = base.reset(seed=0)
    action = Action.LONG if float(observation[_GAP]) > 0 else Action.SHORT
    total_reward, steps = 0.0, 0
    while True:
        observation, reward, terminated, truncated, info = base.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or (info["inventory"] == 0 and info["pending_action"] is None):
            break
        action = _fixed_action(observation, info, opportunity)
    result = {"net_pnl": float(info["pnl"]), "trade_count": int(info["trade_count"]),
              "fees_paid": float(info["fees_paid"]), "funding_paid": float(info["funding_paid"]),
              "rejection_count": int(info["rejection_count"]),
              "end_timestamp_ns": int(info["timestamp_ns"]),
              "start_timestamp_ns": int(episode.timestamp_ns[index]),
              "index": index, "underlying_step_count": steps,
              "scaled_reward": float(total_reward), "termination_reason": info["termination_reason"]}
    base.close()
    return result


class OpportunityReplayEnv(gym.Env):
    """@brief Continuous, nonoverlapping binary entry decisions over one episode.
    @details An action is accepted only at the currently exposed candidate. The
    base engine alone owns cash, fills, risk, and position state. Automatic exits
    and candidate seeking use its public step interface, preserving all intermediate
    rows. Features use the original episode so lag history survives skipped time.
    Account features are intentionally omitted from the compact feature vector;
    the base engine still enforces its configured drawdown constraint.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode: ReplayEpisode, execution: EnvConfig,
                 opportunity: OpportunityConfig | None = None):
        """@brief Validate one immutable replay and establish the binary Gym spaces."""
        super().__init__()
        self.episode, self.execution = episode, execution
        self.opportunity = opportunity or OpportunityConfig()
        self.base = LatencySimEnv(episode, execution)
        self.candidates = candidate_indices(episode, execution, self.opportunity)
        self._candidate_set = set(self.candidates.tolist())
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self._done = True
        self._base_done = True
        self.underlying_step_count = 0

    def _has_opportunity(self) -> bool:
        """@brief Suppress every candidate while an order or position is active."""
        return (not self._base_done and self._info["index"] in self._candidate_set
                and self._info["inventory"] == 0 and self._info["pending_action"] is None)

    def _advance(self, action: Action) -> float:
        """@brief Advance only through the public simulator and count actual decisions."""
        self._observation, reward, terminated, truncated, self._info = self.base.step(action)
        self._base_done = terminated or truncated
        self.underlying_step_count += 1
        return float(reward)

    def _seek(self) -> float:
        """@brief Process fixed exits and flat waiting until the next causal candidate."""
        reward = 0.0
        while not self._base_done and not self._has_opportunity():
            reward += self._advance(_fixed_action(self._observation, self._info, self.opportunity))
        return reward

    def _result_info(self, steps: int) -> dict:
        """@brief Preserve base economics and add transparent macro-clock diagnostics."""
        return {**self._info, "has_opportunity": self._has_opportunity(),
                "underlying_step_count": self.underlying_step_count,
                "macro_underlying_steps": steps}

    def _features(self) -> np.ndarray:
        """@brief Always index shared features against the original episode prefix."""
        return features_at(self.episode, int(self._info["index"]), self.execution)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """@brief Start flat and seek the first candidate without concealing exposure.
        @details Before the first entry, HOLD has exactly zero economic reward.
        Empty episodes finish this seek and expose has_opportunity=False; one safe
        no-op step then returns terminated=True, as reset has no termination flag.
        """
        super().reset(seed=seed)
        if options:
            raise ValueError("OpportunityReplayEnv does not support reset options")
        self._observation, self._info = self.base.reset(seed=seed)
        self._done, self._base_done = False, False
        self.underlying_step_count = 0
        reward = self._seek()
        if not np.isclose(reward, 0.0, rtol=0, atol=1e-9):
            raise AssertionError("Flat initial candidate seeking changed account equity")
        return self._features(), self._result_info(self.underlying_step_count)

    def step(self, action):
        """@brief Execute one skip/trade choice and return its complete economic reward.
        @details A rejected future entry may return to another candidate immediately;
        an accepted position suppresses all overlapping opportunities until flat.
        Empty episodes accept either action as the same zero-reward terminal no-op.
        """
        if self._done:
            raise RuntimeError("Call reset before stepping opportunity replay")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid opportunity action: {action!r}")
        start_steps = self.underlying_step_count
        if self._base_done:
            self._done = True
            info = self._result_info(0)
            info.update({"opportunity_action": int(action), "action_result": "no_opportunity"})
            return self._features(), 0.0, True, False, info
        if not self._has_opportunity():
            raise AssertionError("Binary action requested outside a flat candidate")
        # @details Direction comes from the causal raw observation, never from an
        # inference-normalized vector or any future execution price.
        base_action = Action.HOLD
        if int(action) == 1:
            base_action = Action.LONG if float(self._observation[_GAP]) > 0 else Action.SHORT
        decision_timestamp = int(self._info["timestamp_ns"])
        reward = self._advance(base_action)
        reward += self._seek()
        self._done = self._base_done
        info = self._result_info(self.underlying_step_count - start_steps)
        info.update({"opportunity_action": int(action),
                     "opportunity_decision_timestamp_ns": decision_timestamp,
                     "opportunity_elapsed_ms": (int(info["timestamp_ns"]) - decision_timestamp) / 1e6})
        return self._features(), reward, self._done, False, info

    def close(self) -> None:
        """@brief Delegate cleanup without altering replay arrays or stored accounting."""
        self.base.close()
