"""@file table.py
@brief Causal one-second decision table for the version 2 lead-lag agent.
@details Every retained five-minute window contributes 300 decision seconds. Context
features (the slow cross-venue basis, lagged returns, realised volatility) are computed
over the whole UTC day in time order and use only earlier decision rows. A day is cut
into segments wherever retained windows are not adjacent, and every rolling quantity
restarts at a segment boundary, so no value crosses an excluded outage.

Execution prices are precomputed for each decision second at every recorded execution
offset with the same depth-VWAP, spread-multiplier and slippage arithmetic as
LatencySimEnv._execution_price. The table is a speed cache for training and selection;
reported validation and test results come from the unchanged simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from latency_arb.data.schema import load_manifest_entry, read_manifest

SECOND_NS = 1_000_000_000
EPISODE_SECONDS = 300


@dataclass(frozen=True)
class TableConfig:
    """@brief Feature and execution constants fixed before any policy is fitted.
    @param basis_span_s EMA span (seconds) of the slow cross-venue basis.
    @param vol_window_s Window of one-second Binance returns for realised volatility.
    @param position_size_btc Order size used to price depth VWAP.
    @param slippage_bps Extra adverse price movement per fill (as the simulator).
    @param spread_multiplier Spread stress applied around the HL midpoint.
    @param fill_convention "received": fill at the latest HL book received by t+L (the
        simulator's convention). "hl_clock": fill at HL's own book as of t+L, i.e. the last
        snapshot whose exchange timestamp is <= t+L, even if it arrives later (stress test).
    """

    basis_span_s: int = 60
    vol_window_s: int = 30
    position_size_btc: float = 0.001
    slippage_bps: float = 0.1
    spread_multiplier: float = 1.0
    fill_convention: str = "received"


def _vwap(prices: np.ndarray, sizes: np.ndarray, mid: np.ndarray, quantity: float,
          direction: int, cfg: TableConfig, forced_penalty_bps: float = 25.0):
    """@brief Vectorised copy of LatencySimEnv._execution_price, level by level.
    @return (voluntary price or NaN where displayed depth is insufficient, depth flag,
        forced price that prices any residual at the worst level plus the penalty).
    @details Subtracting level by level (not cumsum differences) reproduces the
    simulator's floating-point behaviour exactly.
    """
    rows = len(mid)
    remaining = np.full(rows, quantity)
    notional = np.zeros(rows)
    worst = np.zeros(rows)
    for level in range(prices.shape[1]):
        active = remaining > 0
        price = mid + (prices[:, level] - mid) * cfg.spread_multiplier
        worst = np.where(active, price, worst)
        take = np.where(active, np.minimum(remaining, sizes[:, level]), 0.0)
        notional += take * price
        remaining = np.where(active, remaining - take, remaining)
        remaining = np.where(active & (remaining <= quantity * 1e-12), 0.0, remaining)
    ok = remaining == 0
    voluntary = notional / quantity * (1 + direction * cfg.slippage_bps / 10_000)
    forced_notional = notional + remaining * worst * (1 + direction * forced_penalty_bps / 10_000)
    forced = forced_notional / quantity * (1 + direction * cfg.slippage_bps / 10_000)
    return np.where(ok, voluntary, np.nan), ok, forced


def _ema_previous(values: np.ndarray, segment: np.ndarray, span: int) -> np.ndarray:
    """@brief EMA that excludes the current value; restarts at each segment start."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    level = values[0]
    for i in range(len(values)):
        if i == 0 or segment[i] != segment[i - 1]:
            level = values[i]
        out[i] = level
        level = alpha * values[i] + (1 - alpha) * level
    return out


def _lag(values: np.ndarray, segment_age: np.ndarray, lag: int) -> np.ndarray:
    """@brief Value exactly `lag` decision seconds earlier within the same segment, else NaN."""
    out = np.full_like(values, np.nan)
    out[lag:] = values[:-lag]
    out[segment_age < lag] = np.nan
    return out


def _rolling_std_previous(returns: np.ndarray, segment_age: np.ndarray, window: int) -> np.ndarray:
    """@brief Std of the last `window` one-second returns ending at the current second."""
    r = np.nan_to_num(returns)
    c1 = np.concatenate([[0.0], np.cumsum(r)])
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    n = np.minimum(segment_age, window).astype(float)
    idx = np.arange(len(r)) + 1
    lo = (idx - n).astype(int)
    s1 = c1[idx] - c1[lo]
    s2 = c2[idx] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(n > 1, (s2 - s1 * s1 / np.maximum(n, 1)) / np.maximum(n - 1, 1), 0.0)
    return np.sqrt(np.maximum(var, 0.0))


def build_table(manifest_path: str | Path, split: str, cfg: TableConfig | None = None,
                episode_ids: set[str] | None = None) -> dict:
    """@brief Load one split and return aligned per-decision-second arrays.
    @details Arrays have length 300 * number of retained windows, ordered by time.
    `offsets_ms` lists the execution offsets; column j of the fill arrays is the row
    `offsets_ms[j]` after the decision timestamp (j=0 is the decision row itself).
    `episode_ids` optionally restricts the table to named windows (used by tests).
    """
    cfg = cfg or TableConfig()
    manifest, entries = read_manifest(manifest_path, split=split)
    offsets = [0] + list(manifest["execution_offsets_ms"])
    rows_per_second = len(offsets)
    if episode_ids is not None:
        entries = [e for e in entries if e["episode_id"] in episode_ids]
    entries = sorted(entries, key=lambda e: e["start_timestamp_ns"])
    n = EPISODE_SECONDS * len(entries)
    nof = rows_per_second
    out = {name: np.empty(n) for name in (
        "b_mid", "h_mid", "h_bid", "h_ask", "hl_spread_bps", "qa_ms", "ra_ms", "b_imb", "h_imb")}
    out["t_ns"] = np.empty(n, np.int64)
    out["episode"] = np.repeat(np.arange(len(entries), dtype=np.int32), EPISODE_SECONDS)
    out["k"] = np.tile(np.arange(EPISODE_SECONDS, dtype=np.int16), len(entries))
    buy = np.empty((n, nof)); sell = np.empty((n, nof)); mid_fill = np.empty((n, nof))
    buy_forced = np.empty((n, nof)); sell_forced = np.empty((n, nof))
    buy_ok = np.empty((n, nof), bool); sell_ok = np.empty((n, nof), bool); fresh = np.empty((n, nof), bool)
    fill_t = np.empty((n, nof), np.int64)
    same_snapshot = np.zeros((n, nof), bool)
    episode_meta = []
    for e_index, entry in enumerate(entries):
        ep = load_manifest_entry(manifest_path, entry)
        ts = ep.timestamp_ns
        if len(ts) != EPISODE_SECONDS * rows_per_second:
            raise ValueError(f"unexpected row count in {entry['episode_id']}")
        rows = np.arange(EPISODE_SECONDS) * rows_per_second
        sl = slice(e_index * EPISODE_SECONDS, (e_index + 1) * EPISODE_SECONDS)
        if not np.all(ts[rows] == ts[0] + np.arange(EPISODE_SECONDS, dtype=np.int64) * SECOND_NS):
            raise ValueError(f"decision rows are not whole seconds in {entry['episode_id']}")
        hb, ha = ep.hl_bid_prices[:, 0], ep.hl_ask_prices[:, 0]
        hmid = (hb + ha) / 2
        out["t_ns"][sl] = ts[rows]
        out["b_mid"][sl] = ((ep.binance_bid + ep.binance_ask) / 2)[rows]
        out["h_mid"][sl] = hmid[rows]
        out["h_bid"][sl] = hb[rows]
        out["h_ask"][sl] = ha[rows]
        out["hl_spread_bps"][sl] = ((ha - hb) / hmid * 1e4)[rows]
        out["qa_ms"][sl] = ep.hl_quote_age_ms[rows]
        out["ra_ms"][sl] = ep.hl_received_age_ms[rows]
        out["b_imb"][sl] = ep.binance_imbalance[rows]
        out["h_imb"][sl] = ep.hl_imbalance[rows]
        bp, bok, bf = _vwap(ep.hl_ask_prices, ep.hl_ask_sizes, hmid, cfg.position_size_btc, 1, cfg)
        sp, sok, sf = _vwap(ep.hl_bid_prices, ep.hl_bid_sizes, hmid, cfg.position_size_btc, -1, cfg)
        fr = (ep.hl_quote_age_ms <= 1000.0) & (ep.hl_received_age_ms <= 1000.0)
        # @details Exchange timestamp of the latest HL snapshot held at each row.
        ets = np.maximum.accumulate(ts - np.round(ep.hl_quote_age_ms * 1e6).astype(np.int64))
        for j in range(nof):
            r = rows + j
            if cfg.fill_convention == "hl_clock" and j > 0:
                target = ts[rows] + int(offsets[j]) * 1_000_000
                r = np.maximum(np.searchsorted(ets, target, side="right") - 1, rows)
            elif cfg.fill_convention not in ("received", "hl_clock"):
                raise ValueError("unknown fill convention")
            same_snapshot[sl, j] = ets[r] == ets[rows]
            buy[sl, j] = bp[r]; sell[sl, j] = sp[r]; buy_ok[sl, j] = bok[r]; sell_ok[sl, j] = sok[r]
            buy_forced[sl, j] = bf[r]; sell_forced[sl, j] = sf[r]
            nominal = rows + j
            fresh[sl, j] = fr[nominal]; mid_fill[sl, j] = hmid[r]; fill_t[sl, j] = ts[nominal]
        episode_meta.append({"episode_id": entry["episode_id"], "day": entry["day"],
                             "path": entry["path"], "last_timestamp_ns": int(ts[-1])})
    t = out["t_ns"]
    segment = np.concatenate([[0], np.cumsum(np.diff(t) != SECOND_NS)])
    seg_start = np.r_[0, np.flatnonzero(np.diff(segment)) + 1]
    seg_age = np.arange(n) - np.repeat(seg_start, np.diff(np.r_[seg_start, n]))
    gap = (out["b_mid"] - out["h_mid"]) / out["h_mid"] * 1e4
    basis = _ema_previous(gap, segment, cfg.basis_span_s)
    bret1 = (out["b_mid"] / _lag(out["b_mid"], seg_age, 1) - 1) * 1e4
    bret5 = (out["b_mid"] / _lag(out["b_mid"], seg_age, 5) - 1) * 1e4
    hret1 = (out["h_mid"] / _lag(out["h_mid"], seg_age, 1) - 1) * 1e4
    hret5 = (out["h_mid"] / _lag(out["h_mid"], seg_age, 5) - 1) * 1e4
    out.update({
        "gap": gap, "basis": basis, "dgap": gap - basis, "segment": segment, "seg_age": seg_age,
        "bret1": bret1, "bret5": bret5, "hret1": hret1, "hret5": hret5,
        "vol30": _rolling_std_previous(bret1, seg_age, cfg.vol_window_s),
        "buy": buy, "sell": sell, "buy_ok": buy_ok, "sell_ok": sell_ok, "fresh": fresh,
        "buy_forced": buy_forced, "sell_forced": sell_forced, "same_snapshot": same_snapshot,
        "mid_fill": mid_fill, "fill_t": fill_t,
    })
    out["meta"] = {"manifest": str(manifest_path), "split": split, "offsets_ms": offsets, "table_version": 2,
                   "episodes": episode_meta, "config": asdict(cfg),
                   "days": sorted({m["day"] for m in episode_meta})}
    return out


def save_table(table: dict, path: str | Path) -> str:
    """@brief Cache a table as npz plus JSON metadata; return the npz SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in table.items() if k != "meta"}
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(table["meta"], indent=1))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table(path: str | Path) -> dict:
    """@brief Load a cached table into memory."""
    path = Path(path)
    with np.load(path) as z:
        table = {k: z[k] for k in z.files}
    table["meta"] = json.loads(path.with_suffix(".json").read_text())
    return table
