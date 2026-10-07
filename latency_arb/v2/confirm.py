"""@file confirm.py
@brief Confirmatory evaluation of the frozen version 2 choices on newly prepared days.
@details Nothing is trained or selected here. The PPO checkpoints and rule settings come
from the hash-checked selection files of both latency groups; the new days are prepared
with runs/v2_prepare_data.py beforehand. Results go to runs/v2-confirm/<label>/.

Example (after exporting and preparing 4 October onward):
    .venv/bin/python -m latency_arb.v2.confirm \
        --manifest data/v2-fresh-2026-10-04/manifest.json --label oct04
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from latency_arb.env.latency_sim import EnvConfig
from latency_arb.v2.simeval import evaluate_sim
from latency_arb.v2.stats import paired
from latency_arb.v2.table import build_table, save_table

FEES = ("0.5", "1.0", "2.0", "3.5")
GROUPS = {150: Path("runs/v2-leadlag-ppo"), 450: Path("runs/v2-leadlag-ppo-lat450")}


def _frozen(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise RuntimeError(f"{path} does not match its frozen hash")
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    out = Path("runs/v2-confirm") / a.label
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "table.npz"
    if not table_path.exists():
        save_table(build_table(a.manifest, "test"), table_path)
    episodes = json.loads(table_path.with_suffix(".json").read_text())["episodes"]
    summary = {}
    for latency, run_dir in GROUPS.items():
        sel = _frozen(run_dir / "selection.json")
        extra = _frozen(run_dir / "selection_extra.json")
        for fee in FEES:
            algos = Path("runs/v2-algos-lat450/selection.json")
            extra_algos = ({f"{k}_selected": v["spec"] for k, v in _frozen(algos)["selected"].items()}
                           if latency == 450 and algos.exists() else {})
            policies = {**extra_algos, "ppo_selected": {"kind": "ppo", "path": sel["selected_ppo"]["path"]},
                        "dgap_conv_rule_tuned": extra["dgap_conv_rule_per_fee"][fee]["spec"],
                        "dgap_rule_tuned": sel["dgap_rule_per_fee"][fee]["spec"],
                        "raw_rule_tuned": sel["raw_rule_per_fee"][fee]["spec"]}
            config = EnvConfig(fee_bps=float(fee), latency_ms=latency, decision_interval_ms=1000)
            windows = {}
            for name, spec in policies.items():
                target = out / f"lat{latency}__fee{fee}__{name}.json"
                if target.exists():
                    res = json.loads(target.read_text())
                else:
                    res = evaluate_sim(str(table_path), spec, config, workers=a.workers)
                    target.write_text(json.dumps(res, default=float))
                index = {e["episode_id"]: i for i, e in enumerate(episodes)}
                import numpy as np
                w = np.zeros(len(episodes))
                for t in res["trades"]:
                    w[index[t["episode_id"]]] += t["net_pnl"]
                windows[name] = w
                summary.setdefault(f"lat{latency}", {}).setdefault(fee, {})[name] = {
                    "net_pnl": res["aggregate"]["net_pnl"], "trades": res["aggregate"]["trade_count"]}
            strongest = ("dgap_conv_rule_tuned" if extra["dgap_conv_rule_per_fee"][fee]["net_pnl"]
                         >= sel["dgap_rule_per_fee"][fee]["net_pnl"] else "dgap_rule_tuned")
            summary[f"lat{latency}"][fee]["paired_vs_strongest_rule"] = paired(
                windows["ppo_selected"], windows[strongest], episodes)
            print(latency, fee, json.dumps(summary[f"lat{latency}"][fee], default=float)[:400], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
