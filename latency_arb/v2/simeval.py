"""@file simeval.py
@brief Evaluate version 2 policies in the unchanged LatencySimEnv with version 1 metrics.
@details Each window is replayed by latency_arb.evaluate.rollout (log_history on, trade
ledger reconciled with terminal equity) and combined by aggregate_results, so reported
numbers use the same accounting as every earlier attempt. Work is split over processes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os

import numpy as np

from latency_arb.data.schema import load_manifest_entry, read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.evaluate import aggregate_results, rollout
from latency_arb.v2.policies import SimAdapter, make_decider
from latency_arb.v2.table import load_table

_CACHE: dict = {}


def _table(path: str) -> dict:
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = load_table(path)
    return _CACHE[path]


def _chunk(args) -> list[dict]:
    table_path, spec, config_dict, episodes = args
    T = _table(table_path)
    config = EnvConfig(**config_dict)
    if config.decision_interval_ms != 1000.0:
        raise ValueError("v2 policies decide once per whole second; use decision_interval_ms=1000")
    decider = make_decider(spec, T, config.fee_bps)
    manifest = T["meta"]["manifest"]
    _, entries = read_manifest(manifest, split=T["meta"]["split"])
    by_id = {e["episode_id"]: e for e in entries}
    results = []
    for e in episodes:
        meta = T["meta"]["episodes"][e]
        episode = load_manifest_entry(manifest, by_id[meta["episode_id"]])
        if not decider.candidates[e * 300:(e + 1) * 300].any():
            # @details No decision point: the policy holds flat for the whole window.
            # Replaying is still done so that window metrics are identical in form.
            pass
        out = rollout(SimAdapter(decider, T, e, config.latency_ms), episode, config)
        out.pop("fills"); out.pop("decisions"); out.pop("environment_actions")
        results.append(out)
    return results


def evaluate_sim(table_path: str, spec: dict, config: EnvConfig, workers: int | None = None,
                 keep_trades: bool = True) -> dict:
    """@brief Replay every window of the table's split; return aggregate and trades."""
    if config.decision_interval_ms != 1000.0:
        raise ValueError("v2 policies decide once per whole second; use decision_interval_ms=1000")
    T = _table(table_path)
    n = len(T["meta"]["episodes"])
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    chunks = [list(c) for c in np.array_split(np.arange(n), workers * 4) if len(c)]
    cfg = asdict(config)
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_chunk, [(table_path, spec, cfg, c) for c in chunks]))
    results = [r for part in parts for r in part]
    aggregate = aggregate_results(results)
    trades = [dict(t, episode_id=r["metrics"]["episode_id"], day=r["metrics"]["day"])
              for r in results for t in r["trades"]] if keep_trades else []
    daily_trades: dict[str, int] = {}
    for r in results:
        m = r["metrics"]
        daily_trades[m["day"]] = daily_trades.get(m["day"], 0) + m["trade_count"]
    aggregate["daily_trades"] = daily_trades
    aggregate["positive_days"] = int(sum(v > 0 for v in aggregate["daily_pnl"].values()))
    return {"aggregate": aggregate, "trades": trades, "policy": spec, "config": cfg}
