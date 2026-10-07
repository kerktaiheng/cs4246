"""@file replay.py
@brief Fast single-position replay over the v2 decision table (training and selection).
@details The runner mirrors LatencySimEnv's economics for one fixed-size Hyperliquid
position: an order decided at whole second t fills on the row t + latency with depth
VWAP, slippage and a taker fee on the fill notional; entries must be fresh at both
the decision and the fill row and leave time for a delayed terminal exit; positions
are force-closed 30 s after the entry fill or at the window's last row. Equity is
marked at the HL mid on decision rows, so per-step rewards sum to net P&L.
Only the unchanged simulator (simeval.py) produces reported validation/test results;
tests/test_v2.py checks that this runner reproduces it trade for trade.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

EPISODE_SECONDS = 300
MAX_HOLD_S = 30

OBS_NAMES = (
    "in_position", "dir_dgap", "dir_gap", "dir_basis", "dir_bret1", "dir_bret5",
    "dir_hret1", "dir_hret5", "vol30", "hl_spread", "quote_age", "received_age",
    "dir_hl_imbalance", "dir_binance_imbalance", "fee", "holding", "unrealized",
    "entry_dgap", "time_remaining", "segment_age",
)
# @details Fixed, hand-chosen scales (not fitted to any split) keep inputs near unit size.
_SCALE = np.array([1, 5, 5, 5, 5, 5, 5, 5, 2, 0.5, 1000, 1000, 1, 1, 2, 30, 5, 5, 300, 600], float)


@dataclass(frozen=True)
class ScreenConfig:
    """@brief Present-information screen deciding when the agent is asked to enter.
    @param min_dgap_bps Minimum |gap - basis| for a flat decision point.
    @param warmup_s Seconds of uninterrupted history required for the basis.
    """

    min_dgap_bps: float = 2.0
    warmup_s: int = 30


def candidate_mask(T: dict, screen: ScreenConfig) -> np.ndarray:
    """@brief Flat decision points: large de-meaned gap, warm basis, fresh decision quote."""
    return ((np.abs(T["dgap"]) >= screen.min_dgap_bps) & (T["seg_age"] >= screen.warmup_s)
            & T["fresh"][:, 0])


def flat_features(T: dict, g: np.ndarray, direction: np.ndarray, fee: float) -> np.ndarray:
    """@brief Vectorised observations for flat decision points (no position state)."""
    d = direction.astype(float)
    k = T["k"][g].astype(float)
    cols = [
        np.zeros(len(g)), d * T["dgap"][g], d * T["gap"][g], d * T["basis"][g],
        d * np.nan_to_num(T["bret1"][g]), d * np.nan_to_num(T["bret5"][g]),
        d * np.nan_to_num(T["hret1"][g]), d * np.nan_to_num(T["hret5"][g]),
        T["vol30"][g], T["hl_spread_bps"][g], T["qa_ms"][g], T["ra_ms"][g],
        d * (2 * T["h_imb"][g] - 1), d * (2 * T["b_imb"][g] - 1), np.full(len(g), fee),
        np.zeros(len(g)), np.zeros(len(g)), np.zeros(len(g)),
        (EPISODE_SECONDS - 1 - k), np.minimum(T["seg_age"][g], 600).astype(float),
    ]
    obs = np.stack(cols, axis=1) / _SCALE
    return np.clip(obs, -10, 10).astype(np.float32)


def position_features(T: dict, g: int, d: int, fee: float, holding_s: float,
                      entry_price: float, entry_dgap: float) -> np.ndarray:
    """@brief Observation while holding; unrealised P&L uses the decision-row exit side."""
    base = flat_features(T, np.array([g]), np.array([d]), fee)[0].astype(float) * _SCALE
    exit_side = T["h_bid"][g] if d > 0 else T["h_ask"][g]
    base[0] = 1.0
    base[15] = holding_s
    base[16] = d * (exit_side - entry_price) / entry_price * 1e4
    base[17] = d * entry_dgap
    return np.clip(base / _SCALE, -10, 10).astype(np.float32)


class EpisodeRunner:
    """@brief Step one five-minute window through decision points with exact accounting.
    @param T Decision table; e episode index; fee per side in bps; j execution-offset index.
    @param candidates Boolean mask over table rows where a flat policy is consulted.
    @param direction Entry direction (+1 long HL, -1 short HL) per table row.
    """

    def __init__(self, T: dict, e: int, fee: float, j: int, candidates: np.ndarray,
                 direction: np.ndarray, size: float = 0.001, initial_cash: float = 10_000.0):
        self.T, self.e, self.fee, self.j = T, e, fee, j
        self.g0 = e * EPISODE_SECONDS
        self.cand, self.dirn, self.q = candidates, direction, size
        self.latency_ns = int(T["meta"]["offsets_ms"][j]) * 1_000_000
        self.last_t = int(T["meta"]["episodes"][e]["last_timestamp_ns"])
        self.cash, self.inv = initial_cash, 0.0
        self.k = 0
        self.pos: dict | None = None
        self.trades: list[dict] = []
        self.rejections = 0
        self.entry_requests = 0
        self.done = False
        self.ref_price = float(T["h_mid"][self.g0])

    def equity(self, g: int) -> float:
        """@brief Marked equity at the decision row of table index g."""
        return self.cash + self.inv * float(self.T["h_mid"][g])

    def _fill(self, g: int, side: int, forced: bool) -> float | None:
        """@brief Fill price at row (g, j) for a buy (+1) or sell (-1), as the simulator."""
        j = self.j
        if forced:
            return float(self.T["buy_forced"][g, j] if side > 0 else self.T["sell_forced"][g, j])
        price = self.T["buy"][g, j] if side > 0 else self.T["sell"][g, j]
        return float(price) if np.isfinite(price) else None

    def _terminal_price(self, side: int) -> float:
        """@brief Forced exit on the window's last row (last offset of second 299)."""
        g = self.g0 + EPISODE_SECONDS - 1
        last = len(self.T["meta"]["offsets_ms"]) - 1
        return float(self.T["buy_forced"][g, last] if side > 0 else self.T["sell_forced"][g, last])

    def _close(self, price: float, g: int | None, reason: str) -> None:
        d = self.pos["dir"]
        fee = self.q * price * self.fee / 1e4
        self.cash -= -d * self.q * price + fee
        gross = d * self.q * (price - self.pos["entry_price"])
        self.trades.append({"entry_g": self.pos["g"], "exit_g": g, "dir": d,
                            "entry_price": self.pos["entry_price"], "exit_price": price,
                            "net_pnl": gross - self.pos["fee"] - fee, "fees": self.pos["fee"] + fee,
                            "reason": reason, "hold_s": (g - self.pos["g"]) if g is not None else None,
                            "entry_same_snapshot": self.pos.get("same_snapshot")})
        self.inv, self.pos = 0.0, None

    def _forced_due(self) -> bool:
        return self.pos is not None and self.k - self.pos["k"] >= MAX_HOLD_S

    def next_decision(self) -> tuple[int, str] | None:
        """@brief Advance through automatic seconds to the next policy decision.
        @return (table index, 'entry' or 'exit'), or None when the window is finished.
        """
        T = self.T
        while self.k < EPISODE_SECONDS:
            g = self.g0 + self.k
            if self.pos is not None:
                if self._forced_due():
                    if int(T["fill_t"][g, self.j]) >= self.last_t:
                        break
                    self._close(self._fill(g, -self.pos["dir"], forced=True), g, "max_holding")
                    self.k += 1
                    continue
                return g, "exit"
            if self.cand[g] and int(T["t_ns"][g]) + self.latency_ns < self.last_t - self.latency_ns:
                return g, "entry"
            self.k += 1
        if self.pos is not None:
            self._close(self._terminal_price(-self.pos["dir"]), None, "episode_end")
        self.done = True
        return None

    def apply(self, g: int, kind: str, act: bool, entry_dgap: float = 0.0) -> None:
        """@brief Apply a decision at table row g and move to the next second."""
        T, j = self.T, self.j
        if kind == "entry" and act:
            self.entry_requests += 1
            d = int(self.dirn[g])
            ok = bool(T["fresh"][g, j]) and int(T["fill_t"][g, j]) < self.last_t - self.latency_ns
            price = self._fill(g, d, forced=False) if ok else None
            if price is None:
                self.rejections += 1
            else:
                fee = self.q * price * self.fee / 1e4
                self.cash -= d * self.q * price + fee
                self.inv = d * self.q
                self.pos = {"dir": d, "entry_price": price, "fee": fee, "k": self.k, "g": g,
                            "entry_dgap": float(T["dgap"][g]),
                            "same_snapshot": bool(T["same_snapshot"][g, j]) if "same_snapshot" in T else None}
        elif kind == "exit" and act:
            if int(T["fill_t"][g, j]) >= self.last_t:
                self._close(self._terminal_price(-self.pos["dir"]), None, "episode_end")
            else:
                price = self._fill(g, -self.pos["dir"], forced=False)
                if price is None:
                    self.rejections += 1
                else:
                    self._close(price, g, "policy")
        self.k += 1


class FastLeadLagEnv(gym.Env):
    """@brief Semi-MDP training environment: decide only at screened entries and while holding.
    @details Action 1 means enter (in the de-meaned gap direction) when flat, or exit
    when holding; action 0 means skip or keep holding. Reward is the change in marked
    equity between decision points in basis points of the window's reference notional.
    The fee per side is drawn per episode from `fees` and is part of the observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, T: dict, fees=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5), j: int = 1,
                 screen: ScreenConfig = ScreenConfig(), seed: int = 0):
        super().__init__()
        self.T, self.fees, self.j = T, tuple(fees), j
        self.cand = candidate_mask(T, screen)
        self.dirn = np.where(T["dgap"] >= 0, 1, -1).astype(np.int8)
        per_ep = self.cand.reshape(-1, EPISODE_SECONDS).any(axis=1)
        self.episodes = np.flatnonzero(per_ep)
        self.observation_space = spaces.Box(-10, 10, shape=(len(OBS_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)
        self.runner: EpisodeRunner | None = None
        # @details Exploration diagnostics read by the training callback.
        self.stats = {"entry_decisions": 0, "entries": 0, "exit_decisions": 0, "exits": 0,
                      "episodes": 0, "reward": 0.0, "trades": 0}

    def _obs(self) -> np.ndarray:
        g, kind = self.pending
        r = self.runner
        if kind == "entry":
            return flat_features(self.T, np.array([g]), np.array([self.dirn[g]]), self.fee)[0]
        return position_features(self.T, g, r.pos["dir"], self.fee, r.k - r.pos["k"],
                                 r.pos["entry_price"], r.pos["entry_dgap"])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while True:
            e = int(self.rng.choice(self.episodes))
            self.fee = float(self.rng.choice(self.fees))
            self.runner = EpisodeRunner(self.T, e, self.fee, self.j, self.cand, self.dirn)
            self.pending = self.runner.next_decision()
            if self.pending is not None:
                break
        self.scale = 1.0 / (self.runner.q * self.runner.ref_price * 1e-4)
        self.last_equity = self.runner.equity(self.pending[0])
        return self._obs(), {}

    def step(self, action):
        g, kind = self.pending
        key = "entry" if kind == "entry" else "exit"
        self.stats[f"{key}_decisions"] += 1
        self.stats["entries" if key == "entry" else "exits"] += int(bool(action))
        self.runner.apply(g, kind, bool(action))
        self.pending = self.runner.next_decision()
        if self.pending is None:
            eq = self.runner.cash
        else:
            eq = self.runner.equity(self.pending[0])
        reward = (eq - self.last_equity) * self.scale
        self.last_equity = eq
        done = self.pending is None
        self.stats["reward"] += reward
        if done:
            self.stats["episodes"] += 1
            self.stats["trades"] += len(self.runner.trades)
        obs = np.zeros(len(OBS_NAMES), np.float32) if done else self._obs()
        return obs, float(reward), done, False, {}


def backtest(T: dict, fee: float, j: int, candidates: np.ndarray, direction: np.ndarray,
             entry_fn, exit_fn) -> dict:
    """@brief Deterministic pass over every window of a split.
    @param entry_fn (g) -> bool for a flat candidate row g.
    @param exit_fn (g, k, pos) -> bool while holding; pos has k, dir, entry_price, entry_dgap.
    """
    n_ep = len(T["meta"]["episodes"])
    trades, rejections, requests = [], 0, 0
    for e in range(n_ep):
        r = EpisodeRunner(T, e, fee, j, candidates, direction)
        while True:
            nd = r.next_decision()
            if nd is None:
                break
            g, kind = nd
            act = entry_fn(g) if kind == "entry" else exit_fn(g, r.k, r.pos)
            r.apply(g, kind, bool(act))
        for t in r.trades:
            t["episode"] = e
        trades.extend(r.trades)
        rejections += r.rejections
        requests += r.entry_requests
    return summarize(T, trades, rejections, requests)


def summarize(T: dict, trades: list[dict], rejections: int, requests: int) -> dict:
    """@brief Aggregate fast-replay trades into the same headline metrics as the report."""
    days = T["meta"]["days"]
    eps = T["meta"]["episodes"]
    daily = {d: 0.0 for d in days}
    daily_n = {d: 0 for d in days}
    for t in trades:
        day = eps[t["episode"]]["day"]
        daily[day] += t["net_pnl"]
        daily_n[day] += 1
    pnl = np.array([t["net_pnl"] for t in trades]) if trades else np.zeros(0)
    curve = np.cumsum(pnl) if len(pnl) else np.zeros(1)
    peak = np.maximum.accumulate(np.r_[0.0, curve])
    n_windows = len(eps)
    window_pnl = np.zeros(n_windows)
    for t in trades:
        window_pnl[t["episode"]] += t["net_pnl"]
    same = [t for t in trades if t.get("entry_same_snapshot")]
    return {"net_pnl": float(pnl.sum()), "trade_count": len(trades),
            "window_pnl": window_pnl.tolist(),
            "same_snapshot_entries": {"trades": len(same), "net_pnl": float(sum(t["net_pnl"] for t in same))},
            "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
            "mean_trade_usd": float(pnl.mean()) if len(pnl) else None,
            "fees_paid": float(sum(t["fees"] for t in trades)),
            "max_drawdown_usd": float((peak - np.r_[0.0, curve]).max()),
            "daily_pnl": daily, "daily_trades": daily_n,
            "positive_days": int(sum(v > 0 for v in daily.values())),
            "rejections": rejections, "entry_requests": requests, "trades": trades}
