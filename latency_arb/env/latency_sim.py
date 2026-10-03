"""@brief Delayed, cost-aware replay of one BTC perpetual position.

@details Every action uses only the current row. A submitted market order executes
at the first recorded row at or after its due time, never at its decision price.
Funding events settle positions carried into a row before that row's fills. Reward
is the change in marked portfolio equity, so unscaled rewards sum to terminal net
PnL. Episode-end and maximum-holding exits are scheduled in advance; entries need
sufficient replay time for the terminal exit. No exchange client or live trading
capability exists in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
from gymnasium import spaces

try:
    from common.gym_base import GymBaseEnv
except ModuleNotFoundError:  # Package can also be imported from its parent folder.
    from models.common.gym_base import GymBaseEnv

from ..data.schema import ReplayEpisode


class Action(IntEnum):
    """@brief The four policy actions; HOLD preserves any existing inventory."""

    HOLD = 0
    LONG = 1
    SHORT = 2
    EXIT = 3


# @brief Raw observation units are explicit in names; normalize on training data only.
OBSERVATION_NAMES = (
    "gap_bps", "binance_spread_bps", "hl_spread_bps", "binance_imbalance",
    "hl_imbalance", "volatility_bps", "hl_quote_age_ms", "hl_received_age_ms",
    "inventory", "holding_time_ms", "pending_action", "pending_remaining_ms",
    "drawdown_usd", "equity_pnl_usd", "time_remaining_ms", "hl_mid_price",
    "unrealized_pnl_usd",
)


@dataclass(frozen=True)
class EnvConfig:
    """@brief Explicit execution assumptions and policy-independent risk limits.

    @param fee_bps Taker fee per executed side, not a round-trip estimate.
    @param latency_ms Delay between an order decision and first eligible replay row.
    @param decision_interval_ms Policy cadence independent of execution latency.
        Zero yields after every replay row; positive values yield on a fixed grid
        anchored to the first episode timestamp while processing all intervening rows.
    @param slippage_bps Additional adverse movement applied after depth VWAP.
    @param spread_multiplier Stress factor on each displayed price's distance to mid.
    @param position_size_btc Fixed absolute inventory cap; no pyramiding or reversal.
    @param max_holding_ms Pre-scheduled holding deadline, measured from entry fill.
    @param max_drawdown_usd Equity drawdown that triggers a delayed mandatory exit.
    @param max_quote_age_ms Maximum exchange quote age accepted for new entries.
    @param max_received_age_ms Maximum time since receiving a quote for new entries.
    @param max_replay_gap_ms Refuse episodes with larger holes; split them beforehand.
    @param forced_liquidity_penalty_bps Adverse residual-price penalty when a forced
        exit exceeds displayed depth. This conservative extrapolation is an explicit
        assumption, not evidence that hidden liquidity actually exists.
    @param initial_cash Initial USD collateral used to calculate account equity.
    @param reward_scale Positive multiplier on equity changes; no action bonus.
    @param log_history Store detailed audits; disable for memory-efficient training.
    """

    fee_bps: float = 3.5
    latency_ms: float = 150.0
    decision_interval_ms: float = 0.0
    slippage_bps: float = 0.1
    spread_multiplier: float = 1.0
    position_size_btc: float = 0.001
    max_holding_ms: float = 30_000.0
    max_drawdown_usd: float = 100.0
    max_quote_age_ms: float = 1_000.0
    max_received_age_ms: float = 1_000.0
    max_replay_gap_ms: float = 5_000.0
    forced_liquidity_penalty_bps: float = 25.0
    initial_cash: float = 10_000.0
    reward_scale: float = 1.0
    log_history: bool = True

    def __post_init__(self) -> None:
        """@brief Reject nonfinite or economically nonsensical model parameters."""
        for name, value in asdict(self).items():
            if name != "log_history" and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("latency_ms", "decision_interval_ms", "max_quote_age_ms", "max_received_age_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("fee_bps", "slippage_bps", "forced_liquidity_penalty_bps"):
            if not 0 <= getattr(self, name) < 10_000:
                raise ValueError(f"{name} must lie in [0, 10000)")
        for name in ("position_size_btc", "max_holding_ms", "max_drawdown_usd",
                     "max_replay_gap_ms", "initial_cash", "reward_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if 0 < self.decision_interval_ms < 0.000001:
            raise ValueError("decision_interval_ms must be zero or at least one nanosecond")
        if self.spread_multiplier < 1:
            raise ValueError("spread_multiplier must be at least one")
        if self.max_holding_ms < self.latency_ms:
            raise ValueError("max_holding_ms must be at least latency_ms")


class LatencySimEnv(GymBaseEnv):
    """@brief Gymnasium replay with an auditable single-position accounting ledger.

    @details A step processes every replay row up to the next policy decision. With
    decision_interval_ms=0 it advances one row; otherwise a fixed episode-start grid
    controls decisions independently of fills. On drawdown breach, that step
    advances without additional agent actions until its delayed risk exit completes
    or the already-scheduled terminal exit occurs. HOLD never cancels a pending
    order. EXIT cancels a pending unfilled entry; all other actions while pending
    are ignored. An opposite-direction entry while invested is ignored rather than
    allowing an instantaneous reversal. Voluntary exits require displayed depth;
    only mandatory exits extrapolate missing depth at a penalized price.
    """

    def __init__(self, episode: ReplayEpisode | Sequence[ReplayEpisode],
                 config: EnvConfig | None = None, render_mode: str | None = None):
        """@brief Validate immutable replay inputs and establish real Gym spaces.

        @param episode One contiguous episode or a collection sampled at reset.
        @param config Cost and risk assumptions shared by every compared policy.
        """
        super().__init__(render_mode=render_mode)
        if render_mode is not None:
            raise ValueError("LatencySimEnv does not provide a render mode")
        self.config = config or EnvConfig()
        self.episodes = (episode,) if isinstance(episode, ReplayEpisode) else tuple(episode)
        if not self.episodes:
            raise ValueError("At least one replay episode is required")
        for item in self.episodes:
            item.validate()
            if len(item) < 2:
                raise ValueError("An episode needs at least two snapshots")
            if np.any(np.diff(item.timestamp_ns) > self.config.max_replay_gap_ms * 1e6):
                raise ValueError("Replay gap exceeds max_replay_gap_ms; split the episode")
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(len(OBSERVATION_NAMES),), dtype=np.float32)
        self._done = True
        self.episode = self.episodes[0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """@brief Reset all accounting and select an episode using a local seeded RNG.

        @param options Optional episode_index selects a particular held-out episode.
        @return Pair of raw float32 observation and current accounting information.
        """
        super().reset(seed=seed)
        options = options or {}
        episode_index = options.get("episode_index")
        if episode_index is None:
            episode_index = int(self.np_random.integers(len(self.episodes)))
        if not isinstance(episode_index, (int, np.integer)) or not 0 <= episode_index < len(self.episodes):
            raise ValueError("episode_index is outside the supplied episode collection")
        self.episode_index = int(episode_index)
        self.episode = self.episodes[self.episode_index]
        self._index = 0
        # @details The policy clock is anchored at reset rather than shifted by a
        # delayed fill. An execution at +150 ms never changes the next +1000 ms tick.
        self._decision_origin_ns = self._now
        self._next_decision_ns = self._decision_origin_ns + self._decision_interval_ns
        self._done = False
        self._risk_closing = False
        self._pending: dict | None = None
        self._open_trade: dict | None = None
        self.cash = self.config.initial_cash
        self.inventory = 0.0
        self.equity = self.config.initial_cash
        self.pnl = 0.0
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        self.trade_count = 0
        self.rejection_count = 0
        self.cancellation_count = 0
        self.termination_reason: str | None = None
        self.trades: list[dict] = []
        self.fills: list[dict] = []
        self.action_history: list[dict] = []
        self.equity_history: list[dict] = []
        self._record_equity()
        return self._observation(), self._info()

    @property
    def _now(self) -> int:
        """@brief Preserve nanosecond timestamp precision as a Python integer."""
        return int(self.episode.timestamp_ns[self._index])

    @property
    def _last_time(self) -> int:
        """@brief Known replay horizon used solely for episode boundary management."""
        return int(self.episode.timestamp_ns[-1])

    @property
    def _latency_ns(self) -> int:
        """@brief Convert configured milliseconds into the integer replay clock."""
        return int(round(self.config.latency_ms * 1_000_000))

    @property
    def _decision_interval_ns(self) -> int:
        """@brief Represent the independent policy cadence on the exact replay clock."""
        return int(round(self.config.decision_interval_ms * 1_000_000))

    def _mid(self) -> float:
        """@brief Mark inventory using the current Hyperliquid top-of-book midpoint."""
        return float((self.episode.hl_bid_prices[self._index, 0]
                      + self.episode.hl_ask_prices[self._index, 0]) / 2)

    def _mark(self) -> None:
        """@brief Recompute equity and drawdown after a price, fill, or funding event."""
        self.equity = float(self.cash + self.inventory * self._mid())
        self.pnl = self.equity - self.config.initial_cash
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)

    def _fresh(self) -> bool:
        """@brief Apply both exchange-clock and local-receipt staleness guards."""
        return (self.episode.hl_quote_age_ms[self._index] <= self.config.max_quote_age_ms
                and self.episode.hl_received_age_ms[self._index] <= self.config.max_received_age_ms)

    def _queue(self, action: Action, reason: str = "policy", forced: bool = False) -> None:
        """@brief Store the submission clock without consulting any future price."""
        self._pending = {"action": int(action), "decision_timestamp_ns": self._now,
                         "decision_index": self._index, "due_timestamp_ns": self._now + self._latency_ns,
                         "reason": reason, "forced": forced}

    def _cancel(self, reason: str) -> None:
        """@brief Record an unfilled order cancellation separately from execution."""
        if self._pending is not None:
            self.cancellation_count += 1
            if self.config.log_history:
                self.action_history.append({"timestamp_ns": self._now, "index": self._index,
                                            "event": "cancel", "reason": reason,
                                            "action": self._pending["action"]})
            self._pending = None

    def _submit_action(self, action: Action) -> str:
        """@brief Enforce inventory, pending-order, freshness, and boundary constraints."""
        if self._pending is not None:
            if action == Action.EXIT and self.inventory == 0:
                self._cancel("policy_cancel")
                return "cancelled_entry"
            return "pending_order"
        if action == Action.HOLD:
            return "hold"
        if action == Action.EXIT:
            if self.inventory == 0:
                return "already_flat"
            self._queue(action)
            return "submitted_exit"
        if self.inventory != 0:
            return "position_limit"
        if not self._fresh():
            self.rejection_count += 1
            return "stale_entry_rejected"
        # @details Two latency intervals reserve entry execution and a terminal exit.
        # No book at that future time is inspected when making this admissibility check.
        if self._now + self._latency_ns >= self._last_time - self._latency_ns:
            self.rejection_count += 1
            return "insufficient_exit_time"
        self._queue(action)
        return "submitted_entry"

    def _execution_price(self, direction: int, quantity: float, forced: bool) -> tuple[float, float] | None:
        """@brief Consume actual current depth, optionally penalizing forced residuals.

        @param direction Positive for a buy, negative for a sell.
        @return VWAP and extrapolated BTC quantity, or None for insufficient depth.
        """
        if direction > 0:
            prices = self.episode.hl_ask_prices[self._index]
            sizes = self.episode.hl_ask_sizes[self._index]
        else:
            prices = self.episode.hl_bid_prices[self._index]
            sizes = self.episode.hl_bid_sizes[self._index]
        mid = self._mid()
        remaining, notional = quantity, 0.0
        worst_price = float(prices[0])
        # @details Spread stress scales each level away from the midpoint, preserving
        # ordering; the base spread and every consumed depth level are already paid.
        for raw_price, size in zip(prices, sizes):
            price = mid + (float(raw_price) - mid) * self.config.spread_multiplier
            if price <= 0:
                raise ValueError("Spread stress produced a nonpositive execution price")
            worst_price = price
            take = min(remaining, float(size))
            notional += take * price
            remaining -= take
            if remaining <= quantity * 1e-12:
                remaining = 0.0
                break
        extrapolated = remaining
        if remaining > 0:
            if not forced:
                return None
            # @details This explicit penalty prevents forced liquidation from becoming
            # a cost-free escape when the dataset omits liquidity beyond its last level.
            residual_price = worst_price * (1 + direction * self.config.forced_liquidity_penalty_bps / 10_000)
            notional += remaining * residual_price
        price = notional / quantity
        price *= 1 + direction * self.config.slippage_bps / 10_000
        return float(price), float(extrapolated)

    def _execute(self, order: dict) -> bool:
        """@brief Fill an eligible order and update cash, fees, inventory, and audit.

        @details A rejected entry never creates partial exposure. A normal exit also
        requires sufficient depth. Mandatory exits conservatively price any residual.
        """
        entering = order["action"] != int(Action.EXIT)
        if entering:
            if not self._fresh() or self._now >= self._last_time - self._latency_ns:
                self.rejection_count += 1
                if self.config.log_history:
                    self.action_history.append({"event": "reject", "timestamp_ns": self._now,
                                                "index": self._index, "action": order["action"],
                                                "reason": "stale_or_insufficient_exit_time"})
                return False
            direction = 1 if order["action"] == int(Action.LONG) else -1
            quantity = self.config.position_size_btc
        else:
            if self.inventory == 0:
                return False
            direction = -1 if self.inventory > 0 else 1
            quantity = abs(self.inventory)
        execution = self._execution_price(direction, quantity, order["forced"])
        if execution is None:
            self.rejection_count += 1
            if self.config.log_history:
                self.action_history.append({"event": "reject", "timestamp_ns": self._now,
                                            "index": self._index, "action": order["action"],
                                            "reason": "insufficient_depth"})
            return False
        price, extrapolated = execution
        # @details Preserve the marked pre-fill equity as well as the post-fill
        # balance. An exit can occur at a new market peak and immediately pay costs;
        # recording only its final balance would conceal this within-row drawdown.
        self._record_equity()
        fee = quantity * price * self.config.fee_bps / 10_000
        self.cash -= direction * quantity * price + fee
        self.fees_paid += fee
        if entering:
            self.inventory = direction * quantity
            self._open_trade = {"side": "long" if direction > 0 else "short", "direction": direction,
                                "quantity_btc": quantity, "entry_price": price, "entry_fee": fee,
                                "entry_timestamp_ns": self._now, "entry_index": self._index,
                                "entry_decision_timestamp_ns": order["decision_timestamp_ns"],
                                "funding_paid": 0.0}
        else:
            trade = self._open_trade
            assert trade is not None
            gross = trade["direction"] * quantity * (price - trade["entry_price"])
            completed = {**trade, "exit_timestamp_ns": self._now, "exit_index": self._index,
                         "exit_decision_timestamp_ns": order["decision_timestamp_ns"],
                         "exit_price": price, "exit_fee": fee, "fees": trade["entry_fee"] + fee,
                         "gross_pnl": gross, "net_pnl": gross - trade["entry_fee"] - fee - trade["funding_paid"],
                         "holding_time_ms": (self._now - trade["entry_timestamp_ns"]) / 1e6,
                         "exit_reason": order["reason"], "extrapolated_exit_quantity_btc": extrapolated}
            self.trade_count += 1
            if self.config.log_history:
                self.trades.append(completed)
            self.inventory = 0.0
            self._open_trade = None
        if self.config.log_history:
            self.fills.append({**order, "timestamp_ns": self._now, "index": self._index,
                               "side": "buy" if direction > 0 else "sell", "quantity_btc": quantity,
                               "price": price, "fee": fee, "extrapolated_quantity_btc": extrapolated})
        self._mark()
        return True

    def _scheduled_exit(self, reason: str, due_timestamp_ns: int) -> None:
        """@brief Execute a previously known holding or episode-end exit schedule.

        @details Its order was scheduled latency_ms before the known deadline. The
        first available row at or after that deadline supplies all execution prices.
        """
        self._cancel(reason)
        order = {"action": int(Action.EXIT), "decision_timestamp_ns": due_timestamp_ns - self._latency_ns,
                 "decision_index": None, "due_timestamp_ns": due_timestamp_ns, "forced": True,
                 "reason": reason}
        self._execute(order)

    def _advance_row(self) -> None:
        """@brief Process settlement, scheduled exits, ordinary fills, and risk in order."""
        self._index += 1
        # @details Positive funding is paid by longs and received by shorts. A new
        # fill on this exact row does not retrospectively own the settlement exposure.
        rate = float(self.episode.funding_rate[self._index])
        payment = self.inventory * self._mid() * rate
        self.cash -= payment
        self.funding_paid += payment
        if self._open_trade is not None:
            self._open_trade["funding_paid"] += payment
        self._mark()
        # @details Execute on the first row reaching the known terminal timestamp,
        # even if several source messages share that nanosecond. Waiting for the last
        # duplicate would choose a later book than the order's first eligible arrival.
        if self._now >= self._last_time:
            self._cancel("episode_end")
            if self.inventory:
                self._scheduled_exit("episode_end", self._last_time)
        if self._index == len(self.episode) - 1:
            self._done = True
            self.termination_reason = "max_drawdown" if self._risk_closing else "episode_end"
        else:
            if self._open_trade is not None:
                deadline = (self._open_trade["entry_timestamp_ns"]
                            + int(round(self.config.max_holding_ms * 1e6)))
                if self._now >= deadline:
                    self._scheduled_exit("max_holding", deadline)
            if self._pending is not None and self._now >= self._pending["due_timestamp_ns"]:
                order, self._pending = self._pending, None
                self._execute(order)
            self._mark()
            self._check_risk()
        self._mark()
        self._record_equity()

    def _check_risk(self) -> None:
        """@brief Trigger delayed liquidation immediately when marked loss breaches.

        @details Execution costs can exceed the trigger during the subsequent delay,
        so the drawdown setting is not a guaranteed maximum realized loss. A zero-
        latency experiment executes its risk order on the triggering row itself.
        """
        if not self._risk_closing and self.peak_equity - self.equity >= self.config.max_drawdown_usd:
            self._risk_closing = True
            self.termination_reason = "max_drawdown"
            self._cancel("max_drawdown")
            if self.inventory:
                self._queue(Action.EXIT, reason="max_drawdown", forced=True)
        if self._risk_closing and self._pending is not None and self._pending["due_timestamp_ns"] <= self._now:
            order, self._pending = self._pending, None
            self._execute(order)
        if self._risk_closing and self.inventory == 0:
            self._done = True
            self.termination_reason = "max_drawdown"

    def step(self, action):
        """@brief Submit a policy action and advance the causal replay event clock.

        @details Positive decision cadence advances through every intervening quote,
        settlement, delayed fill and risk event until the first row reaching the
        next scheduled grid tick. The final reward includes their combined equity
        change. No intermediate market row is exposed as an extra policy decision.
        @return Observation, scaled equity change, terminated, truncated=False, info.
        @throws RuntimeError If the caller steps an episode before or after its run.
        """
        if self._done:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        action = Action(int(action))
        previous_equity = self.equity
        decision_target_ns = self._next_decision_ns
        result = self._submit_action(action)
        if self.config.log_history:
            self.action_history.append({"event": "decision", "timestamp_ns": self._now,
                                        "index": self._index, "action": int(action), "result": result,
                                        "inventory": self.inventory, "equity": self.equity})
        # @details A zero-latency stress experiment may fill the decision row. Every
        # positive delay, however small, must reach a later eligible replay timestamp.
        if self._pending is not None and self._pending["due_timestamp_ns"] <= self._now:
            order, self._pending = self._pending, None
            self._execute(order)
            self._check_risk()
            self._record_equity()
        # @details Advance at least one source row in event-cadence mode. At positive
        # cadence, every internal event is still processed in source order, so a
        # 150 ms fill is not incorrectly postponed until a one-second policy tick.
        if not self._done:
            self._advance_row()
        while not self._done and (self._risk_closing or
                                  (self._decision_interval_ns > 0 and self._now < decision_target_ns)):
            self._advance_row()
        # @details Sparse timestamps can overshoot a grid tick. Skip missed ticks
        # arithmetically without issuing fictitious extra actions, and retain the
        # original grid phase rather than drifting to the arrival/fill timestamp.
        if self._decision_interval_ns > 0:
            completed_ticks = (self._now - self._decision_origin_ns) // self._decision_interval_ns
            self._next_decision_ns = (self._decision_origin_ns
                                      + (completed_ticks + 1) * self._decision_interval_ns)
        reward = (self.equity - previous_equity) * self.config.reward_scale
        info = self._info()
        info["action_result"] = result
        info["decision_target_timestamp_ns"] = decision_target_ns if self._decision_interval_ns > 0 else None
        return self._observation(), float(reward), self._done, False, info

    def _observation(self) -> np.ndarray:
        """@brief Construct causal market features and all policy-relevant account state."""
        row = self._index
        mid = self._mid()
        binance_mid = float((self.episode.binance_bid[row] + self.episode.binance_ask[row]) / 2)
        holding_ms, unrealized = 0.0, 0.0
        if self._open_trade is not None:
            holding_ms = (self._now - self._open_trade["entry_timestamp_ns"]) / 1e6
            unrealized = self.inventory * (mid - self._open_trade["entry_price"])
        pending_action, remaining_ms = -1.0, 0.0
        if self._pending is not None:
            pending_action = float(self._pending["action"])
            remaining_ms = max(0, self._pending["due_timestamp_ns"] - self._now) / 1e6
        values = ((binance_mid - mid) / mid * 10_000,
                  (self.episode.binance_ask[row] - self.episode.binance_bid[row]) / binance_mid * 10_000,
                  (self.episode.hl_ask_prices[row, 0] - self.episode.hl_bid_prices[row, 0]) / mid * 10_000,
                  self.episode.binance_imbalance[row], self.episode.hl_imbalance[row],
                  self.episode.volatility_bps[row], self.episode.hl_quote_age_ms[row],
                  self.episode.hl_received_age_ms[row], self.inventory, holding_ms,
                  pending_action, remaining_ms, self.peak_equity - self.equity, self.pnl,
                  (self._last_time - self._now) / 1e6, mid, unrealized)
        observation = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation contains nonfinite values")
        return observation

    def _record_equity(self) -> None:
        """@brief Record cash-flow-consistent marked equity for drawdown analysis."""
        if self.config.log_history:
            record = {"timestamp_ns": self._now, "index": self._index,
                      "equity": self.equity, "pnl": self.pnl,
                      "inventory": self.inventory, "drawdown": self.peak_equity - self.equity}
            # @details Zero-delay actions may revisit the same exact state before
            # filling. Suppress only identical records; distinct pre/post-fill
            # balances at the same timestamp remain separate audit events.
            if not self.equity_history or record != self.equity_history[-1]:
                self.equity_history.append(record)

    def _info(self) -> dict:
        """@brief Expose accounting summaries even when detailed histories are disabled."""
        return {"timestamp_ns": self._now, "index": self._index, "episode_index": self.episode_index,
                "equity": self.equity, "pnl": self.pnl, "cash": self.cash, "inventory": self.inventory,
                "fees_paid": self.fees_paid, "funding_paid": self.funding_paid,
                "trade_count": self.trade_count, "drawdown": self.peak_equity - self.equity,
                "max_drawdown": self.max_drawdown, "rejection_count": self.rejection_count,
                "cancellation_count": self.cancellation_count, "termination_reason": self.termination_reason,
                "decision_interval_ms": self.config.decision_interval_ms,
                "next_decision_timestamp_ns": self._next_decision_ns if self._decision_interval_ns > 0 else None,
                "pending_action": None if self._pending is None else self._pending["action"]}
