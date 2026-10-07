"""@file experiment.py
@brief Version 2 protocol: validation selection, frozen selection file, then test.
@details Stage `select` scores every PPO checkpoint and every baseline configuration on
validation days with the fast replay (which reproduces the simulator trade for trade)
and writes selection.json with its hash. Stage `evaluate` refuses to run without that
file, replays the frozen choices in the unchanged simulator on validation and test
days, runs the predeclared stress grid, and writes day-level bootstrap intervals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from latency_arb.env.latency_sim import EnvConfig
from latency_arb.v2.policies import make_decider
from latency_arb.v2.replay import backtest
from latency_arb.v2.table import load_table

FEES = (0.5, 1.0, 2.0, 3.5)
RAW_X = (1, 2, 4, 7, 10, 20)
DGAP_X = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
DGAP_H = (1, 2, 3, 5, 10, 30)
CONV_E = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
CONV_H = (3, 10, 30)
TABLES = {"validation": "data/v2-tables/validation.npz", "test_A": "data/v2-tables/test.npz",
          "test_B": "data/v2-tables/fresh_oct03.npz"}
LATENCY_INDEX = {150: 1, 300: 2, 450: 3, 600: 4}
_T: dict = {}


def _table(path: str) -> dict:
    """@brief Keep at most one table per worker process (memory is limited to ~7 GB)."""
    if path not in _T:
        _T.clear()
        _T[path] = load_table(path)
    return _T[path]


def _fast(args) -> dict:
    """@brief One fast-replay run; returns compact metrics (no trade list)."""
    path, spec, fee, latency = args
    import torch
    torch.set_num_threads(1)
    T = _table(path)
    d = make_decider(spec, T, fee)
    out = backtest(T, fee, LATENCY_INDEX[latency], d.candidates, d.direction, d.entry, d.exit)
    out.pop("trades")
    return {"spec": spec, "fee": fee, "latency_ms": latency, **out}


def _pool_map(fn, jobs, workers):
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, jobs, chunksize=1))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(run_dir: Path, workers: int, latency: int = 150, objective: str = "sum_usd") -> Path:
    """@brief Score candidates on validation only, at the group's latency, and freeze them."""
    target = run_dir / "selection.json"
    if target.exists():
        raise FileExistsError("selection.json already exists; selection is frozen")
    val = TABLES["validation"]
    checkpoints = sorted(str(p) for p in (run_dir / "models").glob("*/checkpoint_*.zip"))
    checkpoints += sorted(str(p) for p in (run_dir / "models").glob("*/final.zip"))
    jobs = [(val, {"kind": "ppo", "path": c}, f, latency) for c in checkpoints for f in FEES]
    jobs += [(val, {"kind": "raw_rule", "X": x}, f, latency) for x in RAW_X for f in FEES]
    jobs += [(val, {"kind": "dgap_rule", "X": x, "H": h}, f, latency)
             for x, h in itertools.product(DGAP_X, DGAP_H) for f in FEES]
    jobs.append((val, {"kind": "flat"}, 1.0, latency))
    start = time.time()
    rows = _pool_map(_fast, jobs, workers)
    by_ckpt: dict[str, dict] = {}
    for r in rows:
        if r["spec"]["kind"] == "ppo":
            item = by_ckpt.setdefault(r["spec"]["path"], {"path": r["spec"]["path"], "per_fee": {}})
            item["per_fee"][str(r["fee"])] = {k: r[k] for k in ("net_pnl", "trade_count", "win_rate",
                                                                 "positive_days", "daily_pnl")}
    for item in by_ckpt.values():
        item["score_usd"] = sum(v["net_pnl"] for v in item["per_fee"].values())
        item["trades"] = sum(v["trade_count"] for v in item["per_fee"].values())
        item["run"] = Path(item["path"]).parent.name
    # @details Amendment 4 (450 ms group): scale-free per-fee rank sum, lowest wins.
    items = list(by_ckpt.values())
    for f in FEES:
        order = sorted(items, key=lambda c: -c["per_fee"][str(f)]["net_pnl"])
        for rank, c in enumerate(order, 1):
            c["rank_sum"] = c.get("rank_sum", 0) + rank
    by_sum = sorted(items, key=lambda c: (-c["score_usd"], c["trades"]))
    by_rank = sorted(items, key=lambda c: (c["rank_sum"], c["trades"]))
    if objective not in ("sum_usd", "rank_sum"):
        raise ValueError("objective must be sum_usd or rank_sum")
    ranked = by_rank if objective == "rank_sum" else by_sum
    best_per_run: dict[str, dict] = {}
    for c in ranked:
        best_per_run.setdefault(c["run"], c)

    def best_rule(kind):
        out = {}
        for f in FEES:
            cands = [r for r in rows if r["spec"]["kind"] == kind and r["fee"] == f]
            best = sorted(cands, key=lambda r: (-r["net_pnl"], r["trade_count"]))[0]
            out[str(f)] = {"spec": best["spec"], "net_pnl": best["net_pnl"],
                           "trade_count": best["trade_count"]}
        return out

    selection = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": _sha(run_dir / "protocol.json"),
        "latency_ms": latency,
        "validation_table_sha256": _sha(Path(val)),
        "selected_ppo": {"path": ranked[0]["path"], "sha256": _sha(Path(ranked[0]["path"])),
                         "run": ranked[0]["run"], "score_usd": ranked[0]["score_usd"],
                         "per_fee": ranked[0]["per_fee"]},
        "best_checkpoint_per_run": {k: {"path": v["path"], "score_usd": v["score_usd"],
                                        "per_fee": v["per_fee"]} for k, v in best_per_run.items()},
        "raw_rule_per_fee": best_rule("raw_rule"),
        "dgap_rule_per_fee": best_rule("dgap_rule"),
        "objective": objective,
        "selected_by_sum_usd": by_sum[0]["path"], "selected_by_rank_sum": by_rank[0]["path"],
        "all_ppo_checkpoints": [{k: c[k] for k in ("path", "run", "score_usd", "trades", "rank_sum", "per_fee")}
                                for c in ranked],
        "all_rules": [{"spec": r["spec"], "fee": r["fee"], "net_pnl": r["net_pnl"],
                       "trade_count": r["trade_count"]} for r in rows if r["spec"]["kind"] != "ppo"],
        "elapsed_s": round(time.time() - start, 1),
    }
    target.write_text(json.dumps(selection, indent=1))
    (run_dir / "selection.sha256").write_text(_sha(target) + "\n")
    return target


def select_extra(run_dir: Path, workers: int, latency: int) -> Path:
    """@brief Amendment 2: tune the convergence-exit rule on validation and freeze it."""
    target = run_dir / "selection_extra.json"
    if target.exists():
        raise FileExistsError("selection_extra.json already exists; it is frozen")
    val = TABLES["validation"]
    jobs = [(val, {"kind": "dgap_conv_rule", "X": x, "E": e, "H": h}, f, latency)
            for x, e, h in itertools.product(DGAP_X, CONV_E, CONV_H) for f in FEES]
    rows = _pool_map(_fast, jobs, workers)
    best = {}
    for f in FEES:
        cands = sorted([r for r in rows if r["fee"] == f], key=lambda r: (-r["net_pnl"], r["trade_count"]))
        best[str(f)] = {"spec": cands[0]["spec"], "net_pnl": cands[0]["net_pnl"],
                        "trade_count": cands[0]["trade_count"]}
    out = {"amendment": 2, "latency_ms": latency, "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dgap_conv_rule_per_fee": best,
           "all": [{"spec": r["spec"], "fee": r["fee"], "net_pnl": r["net_pnl"], "trade_count": r["trade_count"]}
                   for r in rows]}
    target.write_text(json.dumps(out, indent=1))
    (run_dir / "selection_extra.sha256").write_text(_sha(target) + "\n")
    return target


def _bootstrap_days(daily: dict, reps: int = 10_000, seed: int = 0) -> dict:
    """@brief Percentile CI of total P&L by resampling days with replacement."""
    values = np.array(list(daily.values()), float)
    rng = np.random.default_rng(seed)
    sims = rng.choice(values, size=(reps, len(values)), replace=True).sum(axis=1)
    return {"total": float(values.sum()), "ci90": [float(np.percentile(sims, 5)), float(np.percentile(sims, 95))]}


def _bootstrap_trades(trades: list[dict], reps: int = 10_000, seed: int = 0) -> dict | None:
    if not trades:
        return None
    pnl = np.array([t["net_pnl"] for t in trades], float)
    rng = np.random.default_rng(seed)
    means = rng.choice(pnl, size=(reps, len(pnl)), replace=True).mean(axis=1)
    return {"mean_usd": float(pnl.mean()), "ci90_mean": [float(np.percentile(means, 5)),
                                                         float(np.percentile(means, 95))],
            "share_resamples_mean_positive": float((means > 0).mean())}


def evaluate(run_dir: Path, workers: int, stages: tuple[str, ...], latency: int = 150) -> None:
    """@brief Replay frozen choices in the unchanged simulator; never re-select."""
    from latency_arb.v2.simeval import evaluate_sim
    sel_path = run_dir / "selection.json"
    if (run_dir / "selection.sha256").read_text().strip() != _sha(sel_path):
        raise RuntimeError("selection.json does not match its frozen hash")
    sel = json.loads(sel_path.read_text())
    if sel.get("latency_ms", 150) != latency:
        raise RuntimeError("evaluation latency differs from the frozen selection latency")
    out_dir = run_dir / "sim"
    out_dir.mkdir(exist_ok=True)

    def policies_for(fee: float, include_all_runs: bool):
        f = str(fee)
        pols = {"ppo_selected": {"kind": "ppo", "path": sel["selected_ppo"]["path"]},
                "raw_rule_tuned": sel["raw_rule_per_fee"][f]["spec"],
                "dgap_rule_tuned": sel["dgap_rule_per_fee"][f]["spec"],
                "flat": {"kind": "flat"}}
        extra = run_dir / "selection_extra.json"
        if extra.exists():
            if (run_dir / "selection_extra.sha256").read_text().strip() != _sha(extra):
                raise RuntimeError("selection_extra.json does not match its frozen hash")
            pols["dgap_conv_rule_tuned"] = json.loads(extra.read_text())["dgap_conv_rule_per_fee"][f]["spec"]
        if include_all_runs:
            for run, v in sel["best_checkpoint_per_run"].items():
                pols[f"ppo_{run}"] = {"kind": "ppo", "path": v["path"]}
        return pols

    def run(split, name, spec, config, tag):
        target = out_dir / f"{split}__{tag}__{name}.json"
        if target.exists():
            return json.loads(target.read_text())
        res = evaluate_sim(TABLES[split], spec, config, workers=workers)
        res["bootstrap_days"] = _bootstrap_days(res["aggregate"]["daily_pnl"])
        res["bootstrap_trades"] = _bootstrap_trades(res["trades"])
        res["split"], res["name"], res["tag"] = split, name, tag
        target.write_text(json.dumps(res, default=float))
        agg = res["aggregate"]
        print(json.dumps({"split": split, "tag": tag, "policy": name, "net_pnl": round(agg["net_pnl"], 4),
                          "trades": agg["trade_count"], "win": agg["win_rate"],
                          "pos_days": res["aggregate"]["positive_days"]}), flush=True)
        return res

    if "validation" in stages:
        for fee in FEES:
            for name, spec in policies_for(fee, False).items():
                run("validation", name, spec, EnvConfig(fee_bps=fee, latency_ms=latency, decision_interval_ms=1000),
                    f"fee{fee}_lat{latency}")
    for split in ("test_A", "test_B"):
        if split in stages:
            for fee in FEES:
                for name, spec in policies_for(fee, False).items():
                    run(split, name, spec, EnvConfig(fee_bps=fee, latency_ms=latency, decision_interval_ms=1000),
                        f"fee{fee}_lat{latency}")
    if "fast" in stages:
        fast_secondary(run_dir, sel, policies_for, latency, workers)
    if "stress" in stages:
        for fee in FEES:
            pols = policies_for(fee, False)
            pols.pop("flat")
            for lat in [x for x in (300, 450, 600) if x > latency]:
                for name, spec in pols.items():
                    run("test_A", name, spec, EnvConfig(fee_bps=fee, latency_ms=lat, decision_interval_ms=1000),
                        f"fee{fee}_lat{lat}")
            for name, spec in pols.items():
                run("test_A", name, spec, EnvConfig(fee_bps=fee, latency_ms=latency, spread_multiplier=1.5,
                                                    decision_interval_ms=1000), f"fee{fee}_lat{latency}_spread1.5")
        pols = policies_for(3.5, False)
        pols.pop("flat")
        for name, spec in pols.items():
            run("test_A", name, spec, EnvConfig(fee_bps=4.5, latency_ms=latency, decision_interval_ms=1000),
                f"fee4.5_lat{latency}")


def _fast_full(args) -> dict:
    """@brief Fast replay with trades kept, for bootstrap intervals."""
    path, spec, fee, latency = args
    import torch
    torch.set_num_threads(1)
    T = _table(path)
    d = make_decider(spec, T, fee)
    out = backtest(T, fee, LATENCY_INDEX[latency], d.candidates, d.direction, d.entry, d.exit)
    trades = out.pop("trades")
    return {"spec": spec, "fee": fee, "latency_ms": latency, "table": path, **out,
            "bootstrap_days": _bootstrap_days(out["daily_pnl"]), "bootstrap_trades": _bootstrap_trades(trades)}


def fast_secondary(run_dir: Path, sel: dict, policies_for, latency: int, workers: int) -> None:
    """@brief Secondary results with the fast replay, which reproduces the simulator trade for
    trade (tests/test_v2.py): validation, every PPO run on both test sets, and the stress grid.
    Spread stress uses a table rebuilt with spread_multiplier=1.5.
    """
    out_dir = run_dir / "fast"
    out_dir.mkdir(exist_ok=True)
    jobs, names = [], []
    for split in ("validation", "test_A", "test_B"):
        for fee in FEES:
            for name, spec in policies_for(fee, True).items():
                jobs.append((TABLES[split], spec, fee, latency))
                names.append((split, f"fee{fee}_lat{latency}", name))
    stress_lat = [x for x in (150, 300, 450, 600) if x != latency]
    for fee in FEES:
        pols = policies_for(fee, False)
        pols.pop("flat")
        for lat in stress_lat:
            for name, spec in pols.items():
                jobs.append((TABLES["test_A"], spec, fee, lat))
                names.append(("test_A", f"fee{fee}_lat{lat}", name))
        for name, spec in pols.items():
            jobs.append(("data/v2-tables/test_spread1.5.npz", spec, fee, latency))
            names.append(("test_A", f"fee{fee}_lat{latency}_spread1.5", name))
    pols = policies_for(3.5, False)
    pols.pop("flat")
    for name, spec in pols.items():
        jobs.append((TABLES["test_A"], spec, 4.5, latency))
        names.append(("test_A", f"fee4.5_lat{latency}", name))
    # @details Amendment 4: fills at HL's own book as of t+L (exchange clock), frozen policies.
    for split, table in (("test_A", "data/v2-tables/test_hlclock.npz"), ("test_B", "data/v2-tables/fresh_oct03_hlclock.npz")):
        for fee in FEES:
            pols = policies_for(fee, False)
            pols.pop("flat")
            for name, spec in pols.items():
                jobs.append((table, spec, fee, latency))
                names.append((split, f"fee{fee}_lat{latency}_hlclock", name))
    todo = [(j, n) for j, n in zip(jobs, names) if not (out_dir / f"{n[0]}__{n[1]}__{n[2]}.json").exists()]
    # @details Group jobs by table so each worker mostly reuses its one cached table, and
    # save every result as soon as it completes so an interruption loses nothing.
    todo.sort(key=lambda item: item[0][0])
    with ProcessPoolExecutor(min(workers, 8)) as pool:
        futures = {pool.submit(_fast_full, j): n for j, n in todo}
        for future in as_completed(futures):
            split, tag, name = futures[future]
            res = future.result()
            (out_dir / f"{split}__{tag}__{name}.json").write_text(json.dumps(res, default=float))
            print(json.dumps({"fast": 1, "split": split, "tag": tag, "policy": name,
                              "net_pnl": round(res["net_pnl"], 4), "trades": res["trade_count"]}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "select_extra", "evaluate"])
    p.add_argument("--run-dir", default="runs/v2-leadlag-ppo")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--stages", default="validation,test_A,test_B,stress")
    p.add_argument("--latency", type=int, default=150, choices=sorted(LATENCY_INDEX))
    p.add_argument("--objective", default="sum_usd", choices=["sum_usd", "rank_sum"])
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    if a.stage == "select":
        print(select(run_dir, a.workers, a.latency, a.objective))
    elif a.stage == "select_extra":
        print(select_extra(run_dir, a.workers, a.latency))
    else:
        evaluate(run_dir, a.workers, tuple(a.stages.split(",")), a.latency)


if __name__ == "__main__":
    main()
