"""@file stats.py
@brief Paired comparisons registered in protocol amendment 4.
@details All policies are replayed on identical five-minute windows, so the comparison is
paired: d_w = PPO P&L minus rule P&L in window w. The summed difference gets an hour-block
bootstrap interval (windows in the same UTC hour are resampled together), and sign tests
run over days and hours. A fee's comparison is inconclusive if either policy has fewer
than 100 trades or the 90% interval contains zero. The pooled statistic adds every fee's
per-window differences after dividing by that fee's validation per-window SD of the rule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

FEES = ("0.5", "1.0", "2.0", "3.5")
TABLE_META = {"validation": "data/v2-tables/validation.json", "test_A": "data/v2-tables/test.json",
              "test_B": "data/v2-tables/fresh_oct03.json"}


def window_pnl(path: Path, episodes: list[dict]) -> tuple[np.ndarray, int]:
    """@brief Per-window net P&L and trade count from a sim or fast result file."""
    r = json.loads(path.read_text())
    if "window_pnl" in r:
        return np.asarray(r["window_pnl"], float), int(r["trade_count"])
    index = {e["episode_id"]: i for i, e in enumerate(episodes)}
    out = np.zeros(len(episodes))
    for t in r["trades"]:
        out[index[t["episode_id"]]] += t["net_pnl"]
    return out, int(r["aggregate"]["trade_count"])


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(ppo: np.ndarray, rule: np.ndarray, episodes: list[dict], reps: int = 10_000, seed: int = 0) -> dict:
    d = ppo - rule
    hours = [e["episode_id"][:10] + f"T{(int(e['episode_id'][11:]) * 5) // 60:02d}" for e in episodes]
    days = [e["day"] for e in episodes]
    by_hour, by_day = defaultdict(float), defaultdict(float)
    for x, h, day in zip(d, hours, days):
        by_hour[h] += x
        by_day[day] += x
    blocks = np.array(list(by_hour.values()))
    rng = np.random.default_rng(seed)
    sims = rng.choice(blocks, size=(reps, len(blocks)), replace=True).sum(axis=1)
    pos_h, nonzero_h = int((blocks > 0).sum()), int((blocks != 0).sum())
    day_vals = np.array(list(by_day.values()))
    pos_d, nonzero_d = int((day_vals > 0).sum()), int((day_vals != 0).sum())
    return {"sum_diff_usd": float(d.sum()), "ci90_hour_block": [float(np.percentile(sims, 5)), float(np.percentile(sims, 95))],
            "share_resamples_positive": float((sims > 0).mean()),
            "per_day_diff": {k: float(v) for k, v in sorted(by_day.items())},
            "days_positive": f"{pos_d}/{nonzero_d}", "sign_test_days_p": binom_two_sided(pos_d, nonzero_d),
            "hours_positive": f"{pos_h}/{nonzero_h}", "sign_test_hours_p": binom_two_sided(pos_h, nonzero_h),
            "n_hours": len(blocks)}


def analyse(run_dir: Path, latency: int) -> dict:
    sel = json.loads((run_dir / "selection.json").read_text())
    extra = json.loads((run_dir / "selection_extra.json").read_text())
    out = {"latency_ms": latency, "comparisons": {}}
    val_eps = json.loads(Path(TABLE_META["validation"]).read_text())["episodes"]
    for split in ("test_A", "test_B"):
        eps = json.loads(Path(TABLE_META[split]).read_text())["episodes"]
        pooled = np.zeros(len(eps))
        for fee in FEES:
            tag = f"fee{fee}_lat{latency}"
            # @details Strongest de-meaned rule: the better of the two on validation at this fee.
            fixed, conv = sel["dgap_rule_per_fee"][fee], extra["dgap_conv_rule_per_fee"][fee]
            strongest = "dgap_conv_rule_tuned" if conv["net_pnl"] >= fixed["net_pnl"] else "dgap_rule_tuned"
            src = run_dir / "sim"
            p, n_p = window_pnl(src / f"{split}__{tag}__ppo_selected.json", eps)
            entry = {"strongest_rule": strongest}
            for rule in (strongest, "raw_rule_tuned"):
                r, n_r = window_pnl(src / f"{split}__{tag}__{rule}.json", eps)
                stats = paired(p, r, eps)
                stats["ppo_trades"], stats["rule_trades"] = n_p, n_r
                stats["ppo_net"], stats["rule_net"] = float(p.sum()), float(r.sum())
                lo, hi = stats["ci90_hour_block"]
                stats["verdict"] = ("inconclusive" if min(n_p, n_r) < 100 or lo <= 0 <= hi
                                    else ("PPO better" if lo > 0 else "rule better"))
                entry[rule] = stats
                if rule == strongest:
                    vfile = run_dir / "fast" / f"validation__{tag}__{strongest}.json"
                    vr, _ = window_pnl(vfile, val_eps)
                    sd = float(np.std(vr)) or 1.0
                    entry["validation_rule_window_sd"] = sd
                    pooled += (p - r) / sd
            out["comparisons"].setdefault(split, {})[fee] = entry
        pooled_stats = paired(pooled, np.zeros(len(eps)), eps)
        out["comparisons"][split]["pooled_standardised"] = {k: pooled_stats[k] for k in (
            "sum_diff_usd", "ci90_hour_block", "share_resamples_positive", "days_positive",
            "sign_test_days_p", "hours_positive", "sign_test_hours_p")}
        out["comparisons"][split]["pooled_standardised"]["note"] = "units: validation per-window SDs, summed over windows and fees"
    (run_dir / "stats.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--latency", type=int, required=True)
    a = p.parse_args()
    out = analyse(Path(a.run_dir), a.latency)
    for split, fees in out["comparisons"].items():
        for fee, e in fees.items():
            if fee == "pooled_standardised":
                print(split, "POOLED", json.dumps(e)[:300])
                continue
            s = e[e["strongest_rule"]]
            print(f"{split} fee {fee}: PPO {s['ppo_net']:+.3f} ({s['ppo_trades']}) vs {e['strongest_rule']} {s['rule_net']:+.3f} ({s['rule_trades']}) "
                  f"diff {s['sum_diff_usd']:+.3f} CI90 [{s['ci90_hour_block'][0]:+.3f},{s['ci90_hour_block'][1]:+.3f}] "
                  f"days+ {s['days_positive']} hours+ {s['hours_positive']} p_h={s['sign_test_hours_p']:.3g} -> {s['verdict']}; "
                  f"vs raw: {e['raw_rule_tuned']['sum_diff_usd']:+.3f} -> {e['raw_rule_tuned']['verdict']}")


if __name__ == "__main__":
    main()
