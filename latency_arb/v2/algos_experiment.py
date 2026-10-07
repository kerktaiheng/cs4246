"""@file algos_experiment.py
@brief Amendment 5: select and evaluate Double DQN, fitted-Q iteration and the bandit ablation.
@details Same rules as the 450 ms PPO group: per-fee rank sum on validation (fast replay,
which reproduces the simulator), a hash-frozen selection file, then the unchanged simulator
on test_A and test_B at 450 ms, HL-clock fill stress with the fast replay, and paired
per-window comparisons with hour-block bootstrap intervals against the selected PPO and the
strongest de-meaned rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from latency_arb.env.latency_sim import EnvConfig
from latency_arb.v2.experiment import FEES, TABLES, _fast_full
from latency_arb.v2.stats import paired, window_pnl

RUN = Path("runs/v2-algos-lat450")
PPO = Path("runs/v2-leadlag-ppo-lat450")
LATENCY = 450


def candidates() -> dict[str, list[dict]]:
    dqn = sorted(str(p) for p in (RUN / "dqn").glob("seed*/checkpoint_*.zip")) + \
        sorted(str(p) for p in (RUN / "dqn").glob("seed*/final.zip"))
    return {"dqn": [{"kind": "dqn", "path": p} for p in dqn],
            "fqi": [{"kind": "fqi", "dir": str(p)} for p in sorted((RUN / "batch").glob("seed*/fqi"))],
            "bandit": [{"kind": "bandit", "dir": str(p)} for p in sorted((RUN / "batch").glob("seed*/bandit"))]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(workers: int) -> None:
    target = RUN / "selection.json"
    if target.exists():
        raise FileExistsError("selection is frozen")
    jobs = [(algo, spec, fee) for algo, specs in candidates().items() for spec in specs for fee in FEES]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_fast_full, (TABLES["validation"], spec, fee, LATENCY)): (algo, spec, fee)
                   for algo, spec, fee in jobs}
        for fut in as_completed(futures):
            algo, spec, fee = futures[fut]
            r = fut.result()
            rows.append({"algo": algo, "spec": spec, "fee": fee, "net_pnl": r["net_pnl"],
                         "trade_count": r["trade_count"]})
            print(json.dumps(rows[-1]), flush=True)
    out = {"latency_ms": LATENCY, "objective": "per-fee rank sum on validation", "selected": {}, "all": rows,
           "protocol_amendment_5_sha256": _sha(RUN / "protocol_amendment_5.json")}
    for algo in candidates():
        specs = {json.dumps(r["spec"], sort_keys=True) for r in rows if r["algo"] == algo}
        score = {s: {"rank_sum": 0, "trades": 0, "per_fee": {}} for s in specs}
        for fee in FEES:
            ordered = sorted([r for r in rows if r["algo"] == algo and r["fee"] == fee], key=lambda r: -r["net_pnl"])
            for rank, r in enumerate(ordered, 1):
                key = json.dumps(r["spec"], sort_keys=True)
                score[key]["rank_sum"] += rank
                score[key]["trades"] += r["trade_count"]
                score[key]["per_fee"][str(fee)] = {"net_pnl": r["net_pnl"], "trade_count": r["trade_count"]}
        best = min(score.items(), key=lambda kv: (kv[1]["rank_sum"], kv[1]["trades"]))
        out["selected"][algo] = {"spec": json.loads(best[0]), **best[1]}
    target.write_text(json.dumps(out, indent=1))
    (RUN / "selection.sha256").write_text(_sha(target) + "\n")


def evaluate(workers: int) -> None:
    from latency_arb.v2.simeval import evaluate_sim
    sel_path = RUN / "selection.json"
    if (RUN / "selection.sha256").read_text().strip() != _sha(sel_path):
        raise RuntimeError("selection.json does not match its hash")
    sel = json.loads(sel_path.read_text())
    (RUN / "sim").mkdir(exist_ok=True)
    (RUN / "fast").mkdir(exist_ok=True)
    for split in ("test_A", "test_B"):
        for fee in FEES:
            for algo, v in sel["selected"].items():
                target = RUN / "sim" / f"{split}__fee{fee}_lat{LATENCY}__{algo}.json"
                if target.exists():
                    continue
                res = evaluate_sim(TABLES[split], v["spec"], EnvConfig(fee_bps=fee, latency_ms=LATENCY,
                                                                       decision_interval_ms=1000), workers=workers)
                res["split"], res["algo"] = split, algo
                target.write_text(json.dumps(res, default=float))
                a = res["aggregate"]
                print(json.dumps({"split": split, "fee": fee, "algo": algo, "net_pnl": round(a["net_pnl"], 4),
                                  "trades": a["trade_count"]}), flush=True)
    hl = {"test_A": "data/v2-tables/test_hlclock.npz", "test_B": "data/v2-tables/fresh_oct03_hlclock.npz"}
    jobs = [(split, fee, algo, v["spec"]) for split in hl for fee in FEES for algo, v in sel["selected"].items()
            if not (RUN / "fast" / f"{split}__fee{fee}_lat{LATENCY}_hlclock__{algo}.json").exists()]
    with ProcessPoolExecutor(min(workers, 6)) as pool:
        futures = {pool.submit(_fast_full, (hl[s], spec, f, LATENCY)): (s, f, a) for s, f, a, spec in jobs}
        for fut in as_completed(futures):
            s, f, a = futures[fut]
            (RUN / "fast" / f"{s}__fee{f}_lat{LATENCY}_hlclock__{a}.json").write_text(json.dumps(fut.result(), default=float))


def compare() -> dict:
    sel = json.loads((RUN / "selection.json").read_text())
    ppo_sel = json.loads((PPO / "selection.json").read_text())
    extra = json.loads((PPO / "selection_extra.json").read_text())
    out = {}
    for split in ("test_A", "test_B"):
        eps = json.loads(Path(TABLES[split]).with_suffix(".json").read_text())["episodes"]
        for fee in FEES:
            f = str(fee)
            tag = f"fee{fee}_lat{LATENCY}"
            fixed, conv = ppo_sel["dgap_rule_per_fee"][f], extra["dgap_conv_rule_per_fee"][f]
            strongest = "dgap_conv_rule_tuned" if conv["net_pnl"] >= fixed["net_pnl"] else "dgap_rule_tuned"
            ppo, n_ppo = window_pnl(PPO / "sim" / f"{split}__{tag}__ppo_selected.json", eps)
            rule, n_rule = window_pnl(PPO / "sim" / f"{split}__{tag}__{strongest}.json", eps)
            row = {"ppo": {"net": float(ppo.sum()), "trades": n_ppo}, "strongest_rule": strongest,
                   "rule": {"net": float(rule.sum()), "trades": n_rule}}
            for algo in sel["selected"]:
                w, n = window_pnl(RUN / "sim" / f"{split}__{tag}__{algo}.json", eps)
                row[algo] = {"net": float(w.sum()), "trades": n}
                for name, base, nb in (("vs_ppo", ppo, n_ppo), ("vs_rule", rule, n_rule)):
                    st = paired(w, base, eps)
                    lo, hi = st["ci90_hour_block"]
                    st["verdict"] = ("inconclusive" if min(n, nb) < 100 or lo <= 0 <= hi
                                     else ("better" if lo > 0 else "worse"))
                    row[algo][name] = {k: st[k] for k in ("sum_diff_usd", "ci90_hour_block", "days_positive",
                                                          "hours_positive", "verdict")}
            out.setdefault(split, {})[f] = row
    (RUN / "stats.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "evaluate", "compare"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.stage == "select":
        select(a.workers)
    elif a.stage == "evaluate":
        evaluate(a.workers)
    else:
        for split, fees in compare().items():
            for fee, row in fees.items():
                print(split, fee, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
                                              for k, v in row.items()})[:700])


if __name__ == "__main__":
    main()
