"""Collect version 2 simulator results into tables and figures for the report.

Reads runs/v2-leadlag-ppo*/sim/*.json, selection.json and training logs. Writes
docs/report_v2/generated/results.json and PNG figures in docs/report_v2/figures/.
Every number in the version 2 report should be traceable to results.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
GROUPS = {"lat150": ROOT / "runs/v2-leadlag-ppo", "lat450": ROOT / "runs/v2-leadlag-ppo-lat450"}
FEES = ["0.5", "1.0", "2.0", "3.5"]
LABEL = {"ppo_selected": "PPO (selected)", "raw_rule_tuned": "Raw-gap rule (v1 control)",
         "dgap_rule_tuned": "De-meaned rule, fixed hold", "dgap_conv_rule_tuned": "De-meaned rule, convergence exit",
         "flat": "Always flat"}
COLOR = {"ppo_selected": "#186c7a", "raw_rule_tuned": "#d18b31", "dgap_rule_tuned": "#7a4f9a",
         "dgap_conv_rule_tuned": "#c0504d", "flat": "#718096"}
NAMES = ["ppo_selected", "dgap_conv_rule_tuned", "dgap_rule_tuned", "raw_rule_tuned", "flat"]


def load_group(run_dir: Path) -> dict:
    out = {}
    for f in sorted((run_dir / "sim").glob("*.json")):
        split, tag, name = f.stem.split("__")
        r = json.loads(f.read_text())
        a = r["aggregate"]
        out.setdefault(split, {}).setdefault(tag, {})[name] = {
            "policy": r["policy"], "net_pnl": a["net_pnl"], "trade_count": a["trade_count"],
            "win_rate": a["win_rate"], "max_drawdown_usd": a["max_drawdown_usd"],
            "fees_paid": a["fees_paid"], "daily_pnl": a["daily_pnl"], "daily_trades": a.get("daily_trades"),
            "positive_days": a.get("positive_days"), "rejections": a["rejection_count"],
            "mean_trade_usd": (a["net_pnl"] / a["trade_count"]) if a["trade_count"] else None,
            "bootstrap_days": r.get("bootstrap_days"), "bootstrap_trades": r.get("bootstrap_trades"),
            "reference_opportunity_count": a.get("reference_opportunity_count"),
            "trade_selectivity": a.get("trade_selectivity"),
            "mean_hold_ms": float(np.mean([t["holding_time_ms"] for t in r["trades"]])) if r["trades"] else None,
            "exit_reasons": {k: sum(t["exit_reason"] == k for t in r["trades"])
                             for k in {t["exit_reason"] for t in r["trades"]}},
        }
    return out


def load_fast(run_dir: Path) -> dict:
    out = {}
    for f in sorted((run_dir / "fast").glob("*.json")):
        split, tag, name = f.stem.split("__")
        r = json.loads(f.read_text())
        out.setdefault(split, {}).setdefault(tag, {})[name] = {
            "policy": r["spec"], "net_pnl": r["net_pnl"], "trade_count": r["trade_count"],
            "win_rate": r["win_rate"], "max_drawdown_usd": r["max_drawdown_usd"], "fees_paid": r["fees_paid"],
            "daily_pnl": r["daily_pnl"], "daily_trades": r["daily_trades"], "positive_days": r["positive_days"],
            "rejections": r["rejections"], "mean_trade_usd": r["mean_trade_usd"],
            "bootstrap_days": r["bootstrap_days"], "bootstrap_trades": r["bootstrap_trades"]}
    return out


def training_curves(run_dir: Path) -> dict:
    curves = {}
    for log in sorted((run_dir / "models").glob("*/training_log.jsonl")):
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        curves[log.parent.name] = {k: [r.get(k) for r in rows] for k in
                                   ("timesteps", "entry_rate", "exit_rate", "trades_per_episode",
                                    "reward_per_episode_bps")}
    return curves


def bar_by_fee(res: dict, split: str, latency: int, path: Path, title: str) -> None:
    names = [n for n in NAMES if n != "flat" and n in res[split][f"fee1.0_lat{latency}"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [res[split][f"fee{f}_lat{latency}"][n]["net_pnl"] for f in FEES]
        ax.bar(np.arange(len(FEES)) + (i - (len(names) - 1) / 2) * width, vals, width, label=LABEL[n], color=COLOR[n])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(FEES)), [f"{f} bps" for f in FEES])
    ax.set_xlabel("taker fee per side")
    ax.set_ylabel("net P&L (USD, 0.001 BTC per trade)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cumulative(run_dir: Path, split: str, tag: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for n in ("ppo_selected", "dgap_conv_rule_tuned", "dgap_rule_tuned", "raw_rule_tuned"):
        f = run_dir / "sim" / f"{split}__{tag}__{n}.json"
        if not f.exists():
            continue
        trades = json.loads(f.read_text())["trades"]
        trades.sort(key=lambda t: t["exit_timestamp_ns"])
        if trades:
            x = [(t["exit_timestamp_ns"] - trades[0]["entry_timestamp_ns"]) / 86400e9 for t in trades]
            ax.plot(x, np.cumsum([t["net_pnl"] for t in trades]), label=LABEL[n], color=COLOR[n])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("days since first trade")
    ax.set_ylabel("cumulative net P&L (USD)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def learning_plot(curves: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for name, c in curves.items():
        x = np.array(c["timesteps"]) / 1e6
        smooth = lambda v: np.convolve(np.array(v, float), np.ones(10) / 10, mode="same")
        ax[0].plot(x, smooth(c["entry_rate"]), label=name, lw=1)
        ax[1].plot(x, smooth(c["reward_per_episode_bps"]), label=name, lw=1)
    ax[0].set_ylabel("share of screened entries taken")
    ax[1].set_ylabel("training reward per window (bps)")
    for a in ax:
        a.set_xlabel("training steps (millions)")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    (OUT / "generated").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    results = {}
    for g, run_dir in GROUPS.items():
        if not (run_dir / "selection.json").exists():
            continue
        latency = 150 if g == "lat150" else 450
        sel = json.loads((run_dir / "selection.json").read_text())
        extra = json.loads((run_dir / "selection_extra.json").read_text()) if (run_dir / "selection_extra.json").exists() else {}
        res = load_group(run_dir)
        fast = load_fast(run_dir)
        curves = training_curves(run_dir)
        results[g] = {"latency_ms": latency, "selection": {
            "selected_ppo": sel["selected_ppo"], "best_checkpoint_per_run": sel["best_checkpoint_per_run"],
            "raw_rule_per_fee": sel["raw_rule_per_fee"], "dgap_rule_per_fee": sel["dgap_rule_per_fee"],
            "dgap_conv_rule_per_fee": extra.get("dgap_conv_rule_per_fee"),
            "n_ppo_checkpoints": len(sel["all_ppo_checkpoints"])}, "sim": res, "fast": fast,
            "training_final": {k: {m: v[m][-1] for m in v} for k, v in curves.items()}}
        learning_plot(curves, OUT / "figures" / f"learning_{g}.png", f"PPO training, fill latency {latency} ms")
        for split, title in (("test_A", "Test 29 Sep-2 Oct"), ("validation", "Validation 25-28 Sep"),
                             ("test_B", "Test 3 Oct")):
            if split in res and f"fee1.0_lat{latency}" in res[split]:
                bar_by_fee(res, split, latency, OUT / "figures" / f"bars_{split}_{g}.png",
                           f"{title}: net P&L by fee, fill latency {latency} ms")
        if "test_A" in res:
            for fee in ("1.0", "2.0"):
                cumulative(run_dir, "test_A", f"fee{fee}_lat{latency}", OUT / "figures" / f"cum_test_A_fee{fee}_{g}.png",
                           f"Test 29 Sep-2 Oct, fee {fee} bps per side, latency {latency} ms")
    (OUT / "generated" / "results.json").write_text(json.dumps(results, indent=1, default=float))
    for g, r in results.items():
        print(g, "SIM")
        for split, tags in r["sim"].items():
            for tag in sorted(tags):
                row = "  ".join(f"{n}={v['net_pnl']:+.3f}/{v['trade_count']}" for n, v in sorted(tags[tag].items()))
                print(f"  {split:10s} {tag:24s} {row}")


if __name__ == "__main__":
    main()
