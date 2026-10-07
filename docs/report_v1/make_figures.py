"""Figures and derived statistics for the CS4246 final report, version 1.

Reads only existing project artifacts (prepared episodes and saved run outputs)
and writes PDF figures to ``figures/`` and the numbers behind them to
``generated/derived_stats.json``. It never trains, evaluates, or modifies runs.

Run from anywhere with the project environment:

    /home/kerk/models/.venv/bin/python docs/report_v1/make_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
GEN = HERE / "generated"
RUN1 = ROOT / "runs" / "recorded-ppo-2026-10-04"
RUN2 = ROOT / "runs" / "opportunity-ppo-2026-10-04"
DATA = ROOT / "data" / "recorded-30d"
FRESH = ROOT / "data" / "recorded-fresh-2026-10-03"

# Reference categorical palette (validated: adjacent CVD dE >= 9.1 on white).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLIT_FILL = {"train": "#ffffff", "validation": "#f0efec", "test": "#e4e3df", "fresh": "#d6d5d0"}
# Text width of the report (A4 with 1 in margins), so figures are included without scaling
# and their text keeps its nominal size (9 pt minimum).
W = 6.27

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. Daily cross-venue gap statistics from prepared decision rows
# --------------------------------------------------------------------------
def daily_gap_stats() -> dict:
    cache = GEN / "daily_gap_stats.json"
    if cache.exists():
        return json.loads(cache.read_text())
    per_day: dict[str, list[np.ndarray]] = defaultdict(list)
    demeaned: dict[str, list[np.ndarray]] = defaultdict(list)
    split_of: dict[str, str] = {}
    for manifest_path, label in ((DATA / "manifest.json", None), (FRESH / "manifest.json", "fresh")):
        manifest = json.loads(manifest_path.read_text())
        for ep in manifest["episodes"]:
            with np.load(manifest_path.parent / ep["path"], allow_pickle=False) as a:
                idx = np.arange(0, len(a["timestamp_ns"]), 3)
                bm = (a["binance_bid"][idx] + a["binance_ask"][idx]) / 2
                hm = (a["hl_bid_prices"][idx, 0] + a["hl_ask_prices"][idx, 0]) / 2
                g = (bm - hm) / hm * 1e4
            per_day[ep["day"]].append(g)
            demeaned[ep["day"]].append(g - g.mean())
            split_of[ep["day"]] = label or ep["split"]
    out = {}
    for day in sorted(per_day):
        g = np.concatenate(per_day[day])
        d = np.concatenate(demeaned[day])
        out[day] = {
            "split": split_of[day],
            "decisions": int(g.size),
            "signed_median_bps": float(np.median(g)),
            "abs_median_bps": float(np.median(np.abs(g))),
            "count_abs_ge_7": int((np.abs(g) >= 7).sum()),
            "frac_abs_ge_7": float((np.abs(g) >= 7).mean()),
            "frac_window_demeaned_abs_ge_7": float((np.abs(d) >= 7).mean()),
        }
    cache.write_text(json.dumps(out, indent=1))
    return out


def fig_daily_gap(stats: dict) -> None:
    days = list(stats)
    x = np.arange(len(days))
    med = np.array([stats[d]["signed_median_bps"] for d in days])
    frac = np.array([100 * stats[d]["frac_abs_ge_7"] for d in days])
    splits = [stats[d]["split"] for d in days]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True)
    for ax in (ax1, ax2):
        start = 0
        for i in range(1, len(days) + 1):
            if i == len(days) or splits[i] != splits[start]:
                ax.axvspan(start - 0.5, i - 0.5, color=SPLIT_FILL[splits[start]], zorder=0, lw=0)
                start = i
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.bar(x, med, width=0.7, color=BLUE)
    ax1.set_ylabel("median signed gap\n(bps)")
    ax1.set_title("(a) Daily median signed gap (Binance mid minus Hyperliquid mid)", loc="left")
    ax1.annotate("18-23 Sep:\npersistent basis", xy=(18, -9.6), xytext=(8.5, -8.8),
                 fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    ax2.bar(x, np.maximum(frac, 1e-3), width=0.7, color=ORANGE)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 100)
    ax2.set_ylabel("% of decision\nseconds, |gap| >= 7 bps")
    ax2.set_title("(b) Share of one-second decision rows with |gap| >= 7 bps (log scale)", loc="left")
    ticks = [i for i, d in enumerate(days) if d in ("2026-09-03", "2026-09-08", "2026-09-13", "2026-09-18", "2026-09-25", "2026-09-29", "2026-10-03")]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([days[i][5:] for i in ticks])
    ax2.set_xlim(-0.6, len(days) - 0.4)
    ymax = ax2.get_ylim()[1]
    for name, label in (("train", "train"), ("validation", "val"), ("test", "test"), ("fresh", "3 Oct")):
        pos = [i for i, s in enumerate(splits) if s == name]
        ax2.text(np.mean(pos), ymax * 0.35, label, ha="center", va="top", fontsize=9, color=INK2)
    fig.tight_layout(h_pad=0.6)
    _save(fig, "fig_daily_gap")


# --------------------------------------------------------------------------
# 2. Attempt 1 training collapse
# --------------------------------------------------------------------------
SEEDS1 = [("ppo_seed7", "seed 7", BLUE, "-"), ("ppo_seed17", "seed 17", ORANGE, "--"),
          ("ppo_seed27", "seed 27", AQUA, "-."), ("ppo_seed7_entropy005", "seed 7, ent 0.05", YELLOW, ":")]


def attempt1_training() -> dict:
    out = {}
    for key, *_ in SEEDS1:
        rows = [json.loads(l) for l in (RUN1 / "models" / key / "training_metrics.jsonl").read_text().splitlines()]
        steps = np.array([r["timesteps"] for r in rows])
        acts = [r["rollout_action_counts"] for r in rows]
        trade_share = np.array([(a["long"] + a["short"]) / sum(a.values()) for a in acts])
        mean_trades = np.array([r["recent_mean_trade_count"] for r in rows])
        below = steps[np.argmax(trade_share < 0.01)] if (trade_share < 0.01).any() else None
        with open(RUN1 / "models" / key / "training_episodes.monitor.csv") as fh:
            fh.readline()
            mon = list(csv.DictReader(fh))
        tc = np.array([int(r["trade_count"]) for r in mon])
        pnl = np.array([float(r["pnl"]) for r in mon])
        fees = np.array([float(r["fees_paid"]) for r in mon])
        dec = np.array_split(np.arange(len(tc)), 10)
        out[key] = {
            "steps": steps.tolist(),
            "trade_share": trade_share.tolist(),
            "recent_mean_trades": mean_trades.tolist(),
            "first_step_trade_share_below_1pct": int(below) if below is not None else None,
            "final_rollout_actions": acts[-1],
            "episodes": int(len(tc)),
            "trades_total": int(tc.sum()),
            "net_pnl_total": float(pnl.sum()),
            "fees_total": float(fees.sum()),
            "zero_trade_episode_share": float((tc == 0).mean()),
            "decile_mean_trades": [float(tc[d].mean()) for d in dec],
            "decile_zero_trade_share": [float((tc[d] == 0).mean()) for d in dec],
            "decile_mean_pnl": [float(pnl[d].mean()) for d in dec],
        }
    return out


def fig_training_collapse(tr: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 3.0))
    for key, label, color, ls in SEEDS1:
        s = np.array(tr[key]["steps"]) / 1e3
        ax1.plot(s, 100 * np.array(tr[key]["trade_share"]), color=color, ls=ls, label=label)
        ax2.plot(s, tr[key]["recent_mean_trades"], color=color, ls=ls, label=label)
    ax1.set_xlabel("training timesteps (thousands)")
    ax1.set_ylabel("LONG + SHORT share\nper rollout (%)")
    ax1.set_title("(a) Entries among sampled actions", loc="left")
    ax1.set_ylim(0, 70)
    ax2.set_xlabel("training timesteps (thousands)")
    ax2.set_ylabel("mean closed trades, last\n100 training episodes")
    ax2.set_title("(b) Trades per episode (rolling)", loc="left")
    ax2.set_ylim(0, 52)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 1))
    _save(fig, "fig_training_collapse")


# --------------------------------------------------------------------------
# 3. Attempt 1 validation: gross edge per trade against the fee per trade
# --------------------------------------------------------------------------
THRESH_KEYS = {"threshold_0": 1, "threshold_1": 2, "threshold_2": 4, "threshold_3": 7,
               "threshold_4": 10, "threshold_5": 20, "threshold_6": 100}


def attempt1_threshold_economics() -> dict:
    rep = json.loads((RUN1 / "validation" / "baselines.json").read_text())["report"]["policies"]
    out = {}
    for key, bps in THRESH_KEYS.items():
        agg = rep[key]["aggregate"]
        gross = sum(e["pnl_before_fees_and_funding"] for e in rep[key]["episodes"])
        n = agg["trade_count"]
        out[str(bps)] = {
            "trades": n,
            "net_pnl": agg["net_pnl"],
            "fees": agg["fees_paid"],
            "pre_fee_pnl": gross,
            "pre_fee_per_trade": gross / n if n else None,
            "fee_per_trade": agg["fees_paid"] / n if n else None,
            "net_per_trade": agg["net_pnl"] / n if n else None,
        }
    return out


def fig_edge_vs_fee(econ: dict) -> None:
    keys = [k for k in ("1", "2", "4", "7", "10") if econ[k]["trades"]]
    x = np.arange(len(keys))
    gross = np.array([100 * econ[k]["pre_fee_per_trade"] for k in keys])
    fee = np.array([100 * econ[k]["fee_per_trade"] for k in keys])
    net = gross - fee
    w = 0.26
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.bar(x - w - 0.02, gross, w, color=BLUE, label="before fees (after spread, slippage, delay)")
    ax.bar(x, -fee, w, color=ORANGE, label="fees (3.5 bps per side, both sides)")
    ax.bar(x + w + 0.02, net, w, color=INK2, label="net")
    ax.axhline(0, color=INK2, lw=0.8)
    for i, k in enumerate(keys):
        ax.text(x[i], -7.9, f"n = {econ[k]['trades']:,}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} bps" for k in keys])
    ax.set_xlabel("entry threshold on |gap| (validation, 25-28 Sep)")
    ax.set_ylabel("US cents per trade\n(0.001 BTC)")
    ax.set_ylim(-8.4, 9.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_edge_vs_fee")


# --------------------------------------------------------------------------
# 4. Attempt 1 test: cost sensitivity of the threshold control
# --------------------------------------------------------------------------
def attempt1_sensitivity() -> list[dict]:
    with open(RUN1 / "sensitivity.csv") as fh:
        return [
            {k: (float(v) if k not in ("policy",) and v != "" else v) for k, v in r.items()}
            for r in csv.DictReader(fh)
        ]


def fig_sensitivity(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(W, 3.1))
    combos = [((150.0, 1.0), BLUE, "-", "o"), ((150.0, 1.5), ORANGE, "--", "s"),
              ((300.0, 1.0), AQUA, "-.", "^"), ((300.0, 1.5), YELLOW, ":", "D")]
    for (lat, spr), color, ls, mk in combos:
        sel = sorted((r for r in rows if r["policy"] == "threshold" and r["latency_ms"] == lat
                      and r["spread_multiplier"] == spr), key=lambda r: r["fee_bps"])
        ax.plot([r["fee_bps"] for r in sel], [r["net_pnl"] for r in sel], color=color, ls=ls,
                marker=mk, ms=4.5, label=f"threshold, {int(lat)} ms, spread x{spr:g}")
    ppo = [r for r in rows if r["policy"] in ("ppo", "flat")]
    assert all(r["trade_count"] == 0 and r["net_pnl"] == 0 for r in ppo)
    ax.plot([0.5, 3.5], [0, 0], color=INK, lw=2.2, label="PPO and flat (0 trades in every cell)")
    ax.set_xlabel("fee per side (bps)")
    ax.set_ylabel("net P&L on test (USD)")
    ax.set_xticks([0.5, 1.0, 2.0, 3.5])
    ax.set_ylim(-0.15, 2.3)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


# --------------------------------------------------------------------------
# 5. Attempt 2 training labels
# --------------------------------------------------------------------------
DENSE = {"2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21", "2026-09-22", "2026-09-23"}
BUCKETS = [(6, 7), (7, 8), (8, 10), (10, 15), (15, 20), (20, np.inf)]


def attempt2_labels() -> dict:
    a = np.load(RUN2 / "events" / "events.npz", allow_pickle=False)
    pay = a["net_payoffs"]
    gap = a["features"][:, 0].astype(float)
    ts = a["timestamp_ns"]
    days = np.array([np.datetime64(int(t), "ns").astype("datetime64[D]").astype(str) for t in ts])
    dense = np.isin(days, sorted(DENSE))
    traded = a["trade_counts"] > 0
    dur = (a["end_timestamp_ns"] - ts) / 1e9
    per_day = {}
    for d in sorted(set(days)):
        m = days == d
        per_day[d] = {"candidates": int(m.sum()), "positive_share": float((pay[m] > 0).mean()),
                      "mean_payoff": float(pay[m].mean())}
    buckets = []
    for lo, hi in BUCKETS:
        m = (gap >= lo) & (gap < hi)
        buckets.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "n": int(m.sum()),
                        "positive_share": float((pay[m] > 0).mean()),
                        "mean_payoff": float(pay[m].mean())})
    w = np.abs(pay) / np.abs(pay).mean()
    return {
        "events": int(pay.size),
        "positive": int((pay > 0).sum()),
        "negative": int((pay < 0).sum()),
        "zero": int((pay == 0).sum()),
        "mean_payoff": float(pay.mean()),
        "median_payoff": float(np.median(pay)),
        "mean_positive_payoff": float(pay[pay > 0].mean()),
        "mean_negative_payoff": float(pay[pay < 0].mean()),
        "mean_fee_traded": float(a["fees_paid"][traded].mean()),
        "dense_share": float(dense.mean()),
        "dense_positive_share": float((pay[dense] > 0).mean()),
        "dense_mean_payoff": float(pay[dense].mean()),
        "other_count": int((~dense).sum()),
        "other_positive_share": float((pay[~dense] > 0).mean()),
        "other_mean_payoff": float(pay[~dense].mean()),
        "traded_share_duration_ge_30s": float((dur[traded] >= 30).mean()),
        "traded_median_duration_s": float(np.median(dur[traded])),
        # Duration runs from the decision second to the decision row at which the
        # account was flat again, so a 30 s holding-limit exit shows as 31 s.
        "dense_traded_share_duration_ge_30s": float((dur[traded & dense] >= 30).mean()),
        "dense_traded_share_duration_eq_31s": float((dur[traded & dense] == 31).mean()),
        "dense_traded_median_duration_s": float(np.median(dur[traded & dense])),
        "other_traded_share_duration_ge_30s": float((dur[traded & ~dense] >= 30).mean()),
        "other_traded_median_duration_s": float(np.median(dur[traded & ~dense])),
        "dense_positive": int((pay[dense] > 0).sum()),
        "positive_weight_share": float(w[pay > 0].sum() / w.sum()),
        "per_day": per_day,
        "gap_buckets": buckets,
    }


# --------------------------------------------------------------------------
# 6. Venue statistics by split (quote age, receive age, spreads, prices, stale gaps)
# --------------------------------------------------------------------------
STALE_MS = 1000.0  # entry freshness limit on both Hyperliquid ages


def split_venue_stats() -> dict:
    """Per-split statistics at decision rows (rows 0, 3, 6, ...) of every episode.

    quote age = decision time minus the exchange timestamp of the latest Hyperliquid
    snapshot; receive age = decision time minus its local receive time; their
    difference is local receive time minus exchange timestamp.
    """
    acc: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for manifest_path, label in ((DATA / "manifest.json", None), (FRESH / "manifest.json", "fresh")):
        manifest = json.loads(manifest_path.read_text())
        for ep in manifest["episodes"]:
            with np.load(manifest_path.parent / ep["path"], allow_pickle=False) as a:
                idx = np.arange(0, len(a["timestamp_ns"]), 3)
                bb, ba = a["binance_bid"][idx], a["binance_ask"][idx]
                hb, ha = a["hl_bid_prices"][idx, 0], a["hl_ask_prices"][idx, 0]
                bm, hm = (bb + ba) / 2, (hb + ha) / 2
                d = acc[label or ep["split"]]
                d["gap"].append((bm - hm) / hm * 1e4)
                d["quote_age"].append(a["hl_quote_age_ms"][idx])
                d["receive_age"].append(a["hl_received_age_ms"][idx])
                d["hl_mid"].append(hm)
                d["hl_spread"].append((ha - hb) / hm * 1e4)
                d["bn_spread"].append((ba - bb) / bm * 1e4)
    out: dict = {}
    pooled: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "validation", "test", "fresh"):
        c = {k: np.concatenate(v) for k, v in acc[split].items()}
        if split != "fresh":
            for k, v in c.items():
                pooled[k].append(v)
        qa, ra, g = c["quote_age"], c["receive_age"], np.abs(c["gap"])
        transit = qa - ra
        stale = (qa > STALE_MS) | (ra > STALE_MS)
        row = {
            "decisions": int(g.size),
            "hl_quote_age_median_ms": float(np.median(qa)),
            "hl_quote_age_p99_ms": float(np.percentile(qa, 99)),
            "hl_receive_age_median_ms": float(np.median(ra)),
            "hl_receive_minus_exchange_median_ms": float(np.median(transit)),
            "hl_receive_minus_exchange_p99_ms": float(np.percentile(transit, 99)),
            "hl_spread_median_bps": float(np.median(c["hl_spread"])),
            "binance_spread_median_bps": float(np.median(c["bn_spread"])),
            "hl_mid_median_usd": float(np.median(c["hl_mid"])),
            "hl_mid_min_usd": float(c["hl_mid"].min()),
            "hl_mid_max_usd": float(c["hl_mid"].max()),
            "share_stale": float(stale.mean()),
        }
        for t in (7, 10):
            big = g >= t
            row[f"count_abs_ge_{t}"] = int(big.sum())
            row[f"frac_abs_ge_{t}"] = float(big.mean())
            row[f"stale_count_abs_ge_{t}"] = int((stale & big).sum())
            row[f"stale_share_abs_ge_{t}"] = float((stale & big).sum() / big.sum()) if big.any() else None
        out[split] = row
    p = {k: np.concatenate(v) for k, v in pooled.items()}
    manifest = json.loads((DATA / "manifest.json").read_text())
    seconds = 30 * 86400
    out["research_range"] = {
        "hl_mid_median_usd": float(np.median(p["hl_mid"])),
        "hl_mid_mean_usd": float(p["hl_mid"].mean()),
        "hl_spread_median_bps": float(np.median(p["hl_spread"])),
        "binance_spread_median_bps": float(np.median(p["bn_spread"])),
        "binance_snapshots_per_s": manifest["raw_rows"]["binance"] / seconds,
        "hyperliquid_snapshots_per_s": manifest["raw_rows"]["hyperliquid"] / seconds,
    }
    return out


# --------------------------------------------------------------------------
# 7. Attempt 1 threshold rules re-priced at 4.5 bps per side
# --------------------------------------------------------------------------
def attempt1_fee_rescale(econ: dict, fee_new: float = 4.5, fee_old: float = 3.5) -> dict:
    """The threshold rule's decisions do not depend on the fee and the fee is
    proportional to traded value, so its trades are unchanged and fees scale by
    fee_new / fee_old. Applies the declared selection rule (net P&L, then fewer
    trades, then larger threshold) to the re-priced validation results."""
    scale = fee_new / fee_old
    val = {}
    for k, e in econ.items():
        fees = e["fees"] * scale
        val[k] = {"trades": e["trades"], "fees": fees, "net_pnl": e["pre_fee_pnl"] - fees}
    chosen = sorted(val, key=lambda k: (-round(val[k]["net_pnl"], 12), val[k]["trades"], -float(k)))[0]
    test = json.loads((RUN1 / "research_outcome.json").read_text())["held_out_threshold_control"]
    test_fees = test["fees_paid"] * scale
    return {
        "fee_bps": fee_new,
        "validation": val,
        "selected_threshold_bps": float(chosen),
        "selected_validation_trades": val[chosen]["trades"],
        "test_threshold_10": {
            "trades": test["trade_count"],
            "pre_fee_pnl": test["net_pnl"] + test["fees_paid"],
            "fees": test_fees,
            "net_pnl": test["net_pnl"] + test["fees_paid"] - test_fees,
        },
        "round_trip_cost_bps": {"3.5": 2 * 3.5 + 2 * 0.1 + 0.12, "4.5": 2 * 4.5 + 2 * 0.1 + 0.12},
    }


def fig_attempt2_labels(lab: dict) -> None:
    days = sorted(lab["per_day"])
    x = np.arange(len(days))
    n = np.array([lab["per_day"][d]["candidates"] for d in days])
    colors = [ORANGE if d in DENSE else BLUE for d in days]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W - 0.12, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.bar(x, n, width=0.7, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 2e5)
    ticks = [i for i, d in enumerate(days) if d.endswith(("-03", "-08", "-13", "-18", "-23"))]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([days[i][5:] for i in ticks])
    ax1.set_ylabel("candidates (log scale)")
    ax1.set_title("(a) Training candidates per day", loc="left")
    ax1.text(0, 1.3e5, "orange: 18-23 Sep,\n99.57% of candidates", fontsize=9, color=INK, va="top")

    # Horizontal bars so the gap ranges and the positive shares stay legible at 9 pt.
    b = lab["gap_buckets"]
    labels = [f"{x['lo']}-{x['hi']}" if x["hi"] else f"{x['lo']}+" for x in b]
    yb = np.arange(len(b))[::-1]
    mean = np.array([100 * x["mean_payoff"] for x in b])
    ax2.barh(yb, mean, height=0.6, color=INK2)
    ax2.axvline(0, color=INK2, lw=0.8)
    for i, x_ in enumerate(b):
        ax2.text(mean[i] - 0.25, yb[i], f"{100 * x_['positive_share']:.1f}%", ha="right", va="center",
                 fontsize=9, color=INK)
    ax2.set_yticks(yb)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_ylabel("|gap| at candidate (bps)")
    ax2.set_xlabel("mean net payoff (US cents per label)")
    ax2.set_xlim(-10.5, 0.5)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) Payoff by gap (% positive)", loc="left")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_attempt2_labels")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    gap = daily_gap_stats()
    fig_daily_gap(gap)
    tr = attempt1_training()
    fig_training_collapse(tr)
    econ = attempt1_threshold_economics()
    fig_edge_vs_fee(econ)
    sens = attempt1_sensitivity()
    fig_sensitivity(sens)
    lab = attempt2_labels()
    fig_attempt2_labels(lab)
    venue = split_venue_stats()
    rescale = attempt1_fee_rescale(econ)
    summary = {
        "daily_gap": gap,
        "attempt1_training": {k: {kk: vv for kk, vv in v.items() if kk not in ("steps", "trade_share", "recent_mean_trades")}
                              for k, v in tr.items()},
        "attempt1_threshold_economics": econ,
        "attempt1_threshold_at_4p5_bps": rescale,
        "attempt2_labels": lab,
        "split_venue_stats": venue,
    }
    (GEN / "derived_stats.json").write_text(json.dumps(summary, indent=1))
    print("wrote", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
