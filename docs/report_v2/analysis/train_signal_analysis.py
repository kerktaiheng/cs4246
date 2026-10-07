"""Training-day signal diagnosis for the version 2 report (training split only).

For every decision second with |signal| >= X, enter one 0.001 BTC Hyperliquid position
in the signal direction at the recorded book `latency` after the decision (depth VWAP
plus 0.1 bps slippage, as in the simulator) and exit 3 s later the same way. The gross
edge already includes the spread and slippage on both fills; net = gross - 2 * fee.
Signals: the raw gap (version 1) and the gap minus its causal 60 s EMA basis (version 2).
Writes train_signal_analysis.json and two figures next to this script.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latency_arb.v2.table import load_table

HERE = Path(__file__).parent
T = load_table("data/v2-tables/train.npz")
days = np.array([T["meta"]["episodes"][e]["day"] for e in T["episode"]])
k = T["k"].astype(int)
THRESHOLDS = [1, 2, 3, 4, 5, 6, 8, 10, 15]
FEES = [0.5, 1.0, 2.0, 3.5]
HOLD = 3


def outcome(signal, entry_shift_s, j, hold=HOLD):
    """Gross bps for entering at (g + entry_shift_s, j) and exiting `hold` s later at offset j."""
    n = len(signal)
    g = np.arange(n)
    ge, gx = g + entry_shift_s, g + entry_shift_s + hold
    ok = (k + entry_shift_s + hold <= 299) & (gx < n)
    ge, gx = np.minimum(ge, n - 1), np.minimum(gx, n - 1)
    long_g = (T["sell"][gx, j] - T["buy"][ge, j]) / T["buy"][ge, j] * 1e4
    short_g = (T["sell"][ge, j] - T["buy"][gx, j]) / T["sell"][ge, j] * 1e4
    gross = np.where(signal > 0, long_g, short_g)
    ok &= np.isfinite(gross) & T["fresh"][ge, j] & (T["seg_age"] >= 30)
    return gross, ok


def table_for(signal, entry_shift_s=0, j=1):
    gross, ok = outcome(signal, entry_shift_s, j)
    rows = []
    for X in THRESHOLDS:
        m = ok & (np.abs(signal) >= X)
        gr = gross[m]
        per_day = {d: float(gr[days[m] == d].sum()) for d in sorted(set(days))}
        row = {"threshold_bps": X, "n": int(m.sum()), "per_day": round(m.sum() / len(set(days)), 1),
               "gross_bps": float(gr.mean()) if len(gr) else None}
        for f in FEES:
            row[f"net_bps_fee{f}"] = float(gr.mean() - 2 * f) if len(gr) else None
            day_net = {d: float((gr[days[m] == d] - 2 * f).sum()) for d in per_day}
            row[f"positive_days_fee{f}"] = int(sum(v > 0 for v in day_net.values()))
        rows.append(row)
    return rows


raw, dgap = T["gap"], T["dgap"]
result = {"split": "train", "days": sorted(set(days)), "hold_s": HOLD,
          "raw_gap": table_for(raw), "dgap": table_for(dgap), "decay": []}
# Fill delay after the decision: 150/300/450/600 ms recorded offsets, then +1 s and +2 s.
for label, shift, j in [("150ms", 0, 1), ("300ms", 0, 2), ("450ms", 0, 3), ("600ms", 0, 4),
                        ("1150ms", 1, 1), ("2150ms", 2, 1)]:
    gross, ok = outcome(dgap, shift, j)
    for X in (3, 5, 8):
        m = ok & (np.abs(dgap) >= X)
        result["decay"].append({"fill_delay": label, "threshold_bps": X, "n": int(m.sum()),
                                "gross_bps": float(gross[m].mean())})
daily = {}
for d in sorted(set(days)):
    m = days == d
    daily[d] = {"median_gap_bps": float(np.median(raw[m])), "median_abs_dgap_bps": float(np.median(np.abs(dgap[m]))),
                "share_abs_gap_ge7": float((np.abs(raw[m]) >= 7).mean()),
                "share_abs_dgap_ge7": float((np.abs(dgap[m]) >= 7).mean())}
result["daily"] = daily
(HERE / "train_signal_analysis.json").write_text(json.dumps(result, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for name, rows, style in [("raw gap (v1)", result["raw_gap"], "o--"), ("gap minus 60 s basis (v2)", result["dgap"], "o-")]:
    ax[0].plot([r["threshold_bps"] for r in rows], [r["gross_bps"] for r in rows], style, label=name)
for f in (1.0, 2.0, 3.5):
    ax[0].axhline(2 * f, color="grey", lw=0.8, ls=":")
    ax[0].text(15.2, 2 * f, f"fee {f}/side", fontsize=8, va="center")
ax[0].set_xlabel("signal threshold (bps)")
ax[0].set_ylabel("mean gross edge per trade (bps)")
ax[0].set_title("Training days: edge by signal (fill +150 ms, hold 3 s)", fontsize=10)
ax[0].legend(fontsize=8)
order = ["150ms", "300ms", "450ms", "600ms", "1150ms", "2150ms"]
xs = [150, 300, 450, 600, 1150, 2150]
for X in (3, 5, 8):
    ys = [next(r["gross_bps"] for r in result["decay"] if r["fill_delay"] == o and r["threshold_bps"] == X) for o in order]
    ax[1].plot(xs, ys, "o-", label=f"|gap - basis| >= {X} bps")
ax[1].set_xlabel("fill delay after decision second (ms)")
ax[1].set_ylabel("mean gross edge per trade (bps)")
ax[1].set_title("Training days: edge decay with fill delay", fontsize=10)
ax[1].axhline(0, color="black", lw=0.6)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "train_signal_edge.png", dpi=200)
print(json.dumps({"dgap": result["dgap"][:6], "decay": result["decay"]}, indent=0)[:3000])
