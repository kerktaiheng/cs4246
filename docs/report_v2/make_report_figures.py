"""Figures, tables and numbers for the CS4246 final report, version 2.

Reads only saved run artifacts and analysis outputs; it never trains, evaluates, or
modifies anything under runs/ or data/:

- runs/v2-leadlag-ppo/ and runs/v2-leadlag-ppo-lat450/: protocol and amendments,
  selection.json, selection_extra.json, sim/*.json, fast/*.json, stats.json and the
  PPO training logs (models/*/training_log.jsonl).
- runs/v2-algos-lat450/: protocol_amendment_5.json, selection.json, sim/*.json,
  fast/*hlclock*.json, stats.json, dqn/seed*/{settings.json,training_log.jsonl} and
  batch/seed*/fqi/meta.json.
- docs/report_v2/analysis/train_signal_analysis.json.

Writes PDF and PNG figures to figures/, LaTeX table bodies to generated/tab_*.tex, and
every number used in the report to generated/report_numbers.json.

    /home/kerk/models/.venv/bin/python docs/report_v2/make_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
GEN = HERE / "generated"
R150 = ROOT / "runs" / "v2-leadlag-ppo"
R450 = ROOT / "runs" / "v2-leadlag-ppo-lat450"
RALG = ROOT / "runs" / "v2-algos-lat450"
ANALYSIS = HERE / "analysis"
FEES = ["0.5", "1.0", "2.0", "3.5"]
SPLITS = ["test_A", "test_B"]

# Categorical palette in fixed slot order (validated light-mode adjacent CVD dE >= 9.1).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# One colour per policy, the same in every figure.
POLICY = {
    "ppo": ("PPO", BLUE),
    "dqn": ("Double DQN", ORANGE),
    "fqi": ("FQI", AQUA),
    "bandit": ("Contextual bandit", YELLOW),
    "conv": ("Best rule (convergence exit)", MAGENTA),
    "raw": ("Raw-gap rule (version 1)", GREEN),
    "fixed": ("De-meaned rule, fixed hold", VIOLET),
}
FILE_NAME = {"ppo": "ppo_selected", "conv": "dgap_conv_rule_tuned", "fixed": "dgap_rule_tuned",
             "raw": "raw_rule_tuned", "flat": "flat", "dqn": "dqn", "fqi": "fqi", "bandit": "bandit"}
W = 6.27  # text width in inches (A4, 1 in margins)

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "legend.frameon": False,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- loading
def sim_summary(path: Path) -> dict:
    """Aggregate plus per-trade statistics from one unchanged-simulator result file."""
    r = load(path)
    a, trades = r["aggregate"], r["trades"]
    out = {"net": a["net_pnl"], "trades": a["trade_count"], "win_rate": a["win_rate"],
           "fees": a["fees_paid"], "max_dd": a["max_drawdown_usd"], "positive_days": a.get("positive_days"),
           "daily_pnl": a["daily_pnl"], "daily_trades": a.get("daily_trades"),
           "rejections": a["rejection_count"], "entry_requests": a.get("entry_request_count"),
           "policy": r["policy"]}
    if trades:
        notional = np.array([t["entry_price"] * t["quantity_btc"] for t in trades])
        net = np.array([t["net_pnl"] for t in trades]) / notional * 1e4
        gross = np.array([t["gross_pnl"] for t in trades]) / notional * 1e4
        hold = np.array([t["holding_time_ms"] for t in trades]) / 1000
        reasons = [t["exit_reason"] for t in trades]
        out.update({"net_bps_per_trade": float(net.mean()), "gross_bps_per_trade": float(gross.mean()),
                    "mean_hold_s": float(hold.mean()), "median_hold_s": float(np.median(hold)),
                    "exit_policy_share": reasons.count("policy") / len(reasons),
                    "exit_max_hold_share": reasons.count("max_holding") / len(reasons),
                    "exit_episode_end_share": reasons.count("episode_end") / len(reasons),
                    "trades_sorted": sorted(((t["exit_timestamp_ns"], t["net_pnl"]) for t in trades))})
    return out


def fast_summary(path: Path) -> dict:
    r = load(path)
    ss = r.get("same_snapshot_entries") or {"trades": 0, "net_pnl": 0.0}
    return {"net": r["net_pnl"], "trades": r["trade_count"], "win_rate": r["win_rate"],
            "same_snapshot_trades": ss["trades"], "same_snapshot_net": ss["net_pnl"],
            "positive_days": r["positive_days"], "daily_pnl": r["daily_pnl"],
            "boot_days_ci90": r["bootstrap_days"]["ci90"], "fees": r["fees_paid"]}


def collect() -> dict:
    N: dict = {}
    # Protocol registration times.
    prot = {"protocol": load(R150 / "protocol.json")["registered_at_utc"]}
    for i in range(1, 6):
        p = (R150 if i < 5 else RALG) / f"protocol_amendment_{i}.json"
        prot[f"amendment_{i}"] = load(p)["registered_at_utc"]
    N["protocol_times_utc"] = prot

    # Selections.
    for tag, run in (("lat150", R150), ("lat450", R450)):
        sel, extra = load(run / "selection.json"), load(run / "selection_extra.json")
        N[f"selection_{tag}"] = {
            "selected_ppo": sel["selected_ppo"]["path"], "frozen_at_utc": sel["frozen_at_utc"],
            "selected_by_sum_usd": sel.get("selected_by_sum_usd"),
            "n_checkpoints": len(sel["all_ppo_checkpoints"]),
            "ppo_val": {f: {k: sel["selected_ppo"]["per_fee"][f][k] for k in ("net_pnl", "trade_count")}
                        for f in FEES},
            "raw": sel["raw_rule_per_fee"], "fixed": sel["dgap_rule_per_fee"],
            "conv": extra["dgap_conv_rule_per_fee"],
            "checkpoints": [{"path": c["path"], "score_usd": c["score_usd"], "total_trades": c["trades"],
                             "per_fee": ({f: c["per_fee"][f]["net_pnl"] for f in FEES} if "per_fee" in c else None),
                             "trades": ({f: c["per_fee"][f]["trade_count"] for f in FEES} if "per_fee" in c else None)}
                            for c in sel["all_ppo_checkpoints"]]}
    algsel = load(RALG / "selection.json")
    N["selection_algos"] = {
        "selected": {a: {"spec": v["spec"], "rank_sum": v["rank_sum"],
                         "per_fee": v["per_fee"]} for a, v in algsel["selected"].items()},
        "all": algsel["all"]}

    # Unchanged-simulator results.
    sim = {"lat450": {}, "lat150": {}}
    for split in SPLITS:
        for f in FEES:
            tag = f"fee{f}_lat450"
            row = {}
            for p in ("ppo", "conv", "fixed", "raw", "flat"):
                row[p] = sim_summary(R450 / "sim" / f"{split}__{tag}__{FILE_NAME[p]}.json")
            for p in ("dqn", "fqi", "bandit"):
                row[p] = sim_summary(RALG / "sim" / f"{split}__{tag}__{p}.json")
            sim["lat450"].setdefault(split, {})[f] = row
    for split in SPLITS:
        for f in FEES:
            tag = f"fee{f}_lat150"
            sim["lat150"].setdefault(split, {})[f] = {
                p: sim_summary(R150 / "sim" / f"{split}__{tag}__{FILE_NAME[p]}.json")
                for p in ("ppo", "conv", "fixed", "raw", "flat")}
    N["sim"] = sim

    # Fast-replay stress results (450 ms group and algorithms) and per-seed PPO tests.
    fast = {}
    for run in (R450, RALG, R150):
        for path in sorted((run / "fast").glob("*.json")):
            split, tag, name = path.stem.split("__")
            fast.setdefault(run.name, {}).setdefault(split, {}).setdefault(tag, {})[name] = fast_summary(path)
    N["fast"] = fast

    # Paired statistics.
    N["paired_lat450"] = load(R450 / "stats.json")["comparisons"]
    N["paired_lat150"] = load(R150 / "stats.json")["comparisons"]
    N["paired_algos"] = load(RALG / "stats.json")

    # Training logs and batch metadata.
    def curves(path: Path) -> dict:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        keys = sorted({k for r in rows for k in r})
        return {k: [r.get(k) for r in rows] for k in keys}

    N["train_ppo450"] = {p.parent.name: curves(p) for p in sorted((R450 / "models").glob("*/training_log.jsonl"))}
    N["train_ppo150"] = {p.parent.name: curves(p) for p in sorted((R150 / "models").glob("*/training_log.jsonl"))}
    N["train_dqn"] = {p.parent.name: curves(p) for p in sorted((RALG / "dqn").glob("*/training_log.jsonl"))}
    N["dqn_settings"] = {p.parent.name: {k: v for k, v in load(p).items() if k not in ("observation", "train_days")}
                         for p in sorted((RALG / "dqn").glob("*/settings.json"))}
    N["ppo_settings"] = {f"{r.name}/{p.parent.name}": {k: v for k, v in load(p).items() if k not in ("train_days",)}
                         for r in (R150, R450) for p in sorted((r / "models").glob("*/settings.json"))}
    N["fqi_meta"] = {p.parent.parent.name: load(p) for p in sorted((RALG / "batch").glob("*/fqi/meta.json"))}
    N["train_signal"] = load(ANALYSIS / "train_signal_analysis.json")
    return N


# --------------------------------------------------------------------------- figures
def fig_smdp() -> None:
    """Schematic of the decision process: two position phases and when the agent is asked."""
    fig = plt.figure(figsize=(W, 3.1))
    ax = fig.add_axes([0, 0.33, 1, 0.67])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.75, 3.1)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.66, body, ha="center", va="top", fontsize=8.3, color=INK2, linespacing=1.35)

    box(0.2, 0.55, 3.5, 2.4, "Flat:  $z=\\varnothing$",
        "asked only at screened seconds:\n$|g_t-\\hat b_t|\\geq 2$ bps, 30 s of history,\nHL quote fresh"
        "\n$\\mathcal{A}(s)=\\{\\mathrm{skip},\\mathrm{enter}\\}$", BLUE)
    box(6.3, 0.55, 3.5, 2.4, "Holding:  $z=(d,h,p_e,\\tilde g_e)$",
        "asked every second\n$\\mathcal{A}(s)=\\{\\mathrm{hold},\\mathrm{exit}\\}$\nforced exit at $h=30$ s"
        "\nor at the window end", ORANGE)
    arrow = dict(arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK2)
    ax.add_patch(FancyArrowPatch((3.75, 2.35), (6.25, 2.35), connectionstyle="arc3,rad=-0.2", **arrow))
    ax.text(5.0, 2.62, "enter\n(fill $L$ ms later)", ha="center", va="bottom", fontsize=8.3, color=INK,
            linespacing=1.2)
    ax.add_patch(FancyArrowPatch((6.25, 1.15), (3.75, 1.15), connectionstyle="arc3,rad=-0.2", **arrow))
    ax.text(5.0, 0.86, "exit or forced exit\n(fill $L$ ms later)", ha="center", va="top", fontsize=8.3,
            color=INK, linespacing=1.2)
    loop = dict(arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2)
    ax.add_patch(FancyArrowPatch((1.45, 0.53), (2.45, 0.53), connectionstyle="arc3,rad=1.1", **loop))
    ax.text(1.95, -0.62, "skip: wait for the next\nscreened second", ha="center", va="bottom", fontsize=8,
            color=INK, linespacing=1.15)
    ax.add_patch(FancyArrowPatch((7.55, 0.53), (8.55, 0.53), connectionstyle="arc3,rad=1.1", **loop))
    ax.text(8.05, -0.62, "hold: next second", ha="center", va="bottom", fontsize=8, color=INK)

    # Timeline of decision epochs (illustration, not data).
    tx = fig.add_axes([0.03, 0.02, 0.94, 0.22])
    secs = np.arange(0, 40)
    entry, exit_ = 11, 19
    flat_epochs = [3, 4, entry, 27, 28, 35]
    hold_epochs = list(range(entry + 1, exit_ + 1))
    tx.set_xlim(-0.5, 39.5)
    tx.set_ylim(-1.0, 1.1)
    tx.axis("off")
    tx.scatter(secs, np.zeros_like(secs), s=6, color="#bdbcb7", zorder=1)
    tx.scatter(flat_epochs, [0] * len(flat_epochs), s=36, color=BLUE, zorder=2, label="flat epoch (screened second)")
    tx.scatter(hold_epochs, [0] * len(hold_epochs), s=36, color=ORANGE, zorder=2, label="holding epoch")
    for x, t in ((entry, "enter"), (exit_, "exit")):
        tx.annotate(t, (x, 0.12), (x, 0.75), ha="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    tx.text(19.5, -0.9, "40 whole seconds of a window (illustration). Grey dots: automatic "
            "\u201cstay as you are\u201d seconds.", ha="center", fontsize=7.8, color=INK2)
    tx.legend(loc="upper right", bbox_to_anchor=(1.0, 1.3), ncol=1, fontsize=7.8, handletextpad=0.2,
              labelspacing=0.3)
    save(fig, "fig_smdp")


def fig_train_signal(N: dict) -> None:
    d = N["train_signal"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    for key, label, color in (("raw_gap", "raw gap (version 1)", GREEN), ("dgap", "gap minus 60 s basis (version 2)", BLUE)):
        xs = [r["threshold_bps"] for r in d[key]]
        ys = [r["gross_bps"] for r in d[key]]
        ax[0].plot(xs, ys, marker="o", ms=4, color=color, label=label)
    for f, ls in ((1.0, ":"), (2.0, "--"), (3.5, "-.")):
        ax[0].axhline(2 * f, color=INK2, lw=0.8, ls=ls)
        ax[0].text(17.4, 2 * f + 0.1, f"fee {f:g}", va="bottom", ha="right", fontsize=7.5, color=INK2)
    ax[0].axhline(0, color=INK, lw=0.6)
    ax[0].set_xlabel("signal threshold $X$ (bps)")
    ax[0].set_ylabel("mean gross edge per trade (bps)")
    ax[0].set_title("(a) edge by signal, fill +150 ms, 3 s hold", loc="left")
    ax[0].legend(loc="upper left", fontsize=7.8)
    ax[0].set_xlim(0, 17.5)
    lat = {"150ms": 150, "300ms": 300, "450ms": 450, "600ms": 600, "1150ms": 1150, "2150ms": 2150}
    for X, color in ((3, BLUE), (5, ORANGE), (8, AQUA)):
        rows = [r for r in d["decay"] if r["threshold_bps"] == X]
        ax[1].plot([lat[r["fill_delay"]] for r in rows], [r["gross_bps"] for r in rows], marker="o", ms=4,
                   color=color, label=f"$|g-\\hat b|\\geq{X}$ bps")
    ax[1].axhline(0, color=INK, lw=0.6)
    ax[1].set_xlabel("fill delay after the decision second (ms)")
    ax[1].set_title("(b) decay of the de-meaned edge", loc="left")
    ax[1].legend(fontsize=7.8)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_train_signal")


def fig_algos_by_fee(N: dict) -> None:
    order = ["ppo", "dqn", "fqi", "bandit", "conv", "raw"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.55))
    for ax, f in zip(axes, FEES):
        row = N["sim"]["lat450"]["test_A"][f]
        vals = [row[p]["net"] for p in order]
        ax.bar(range(len(order)), vals, width=0.78, color=[POLICY[p][1] for p in order],
               edgecolor="white", linewidth=1.0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks([])
        ax.set_title(f"fee {f} bps/side", fontsize=9)
        ax.grid(axis="x", visible=False)
        lo, hi = min(0, min(vals)), max(vals)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo))
    axes[0].set_ylabel("net P&L, USD")
    handles = [plt.Rectangle((0, 0), 1, 1, color=POLICY[p][1]) for p in order]
    fig.legend(handles, [POLICY[p][0] for p in order], loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    save(fig, "fig_algos_by_fee")


def fig_paired(N: dict) -> None:
    """PPO minus each competitor, summed over test windows, with hour-block 90% intervals."""
    comps = ["dqn", "fqi", "bandit", "conv"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.3), sharey=True)
    for ax, f in zip(axes, FEES):
        for i, c in enumerate(comps):
            if c == "conv":
                s = N["paired_lat450"]["test_A"][f]["dgap_conv_rule_tuned"]
                d, (lo, hi) = s["sum_diff_usd"], s["ci90_hour_block"]
                n_other = s["rule_trades"]
            else:
                s = N["paired_algos"]["test_A"][f][c]["vs_ppo"]
                d, (lo, hi) = -s["sum_diff_usd"], (-s["ci90_hour_block"][1], -s["ci90_hour_block"][0])
                n_other = N["paired_algos"]["test_A"][f][c]["trades"]
            n_ppo = N["sim"]["lat450"]["test_A"][f]["ppo"]["trades"]
            conclusive = min(n_ppo, n_other) >= 100 and not (lo <= 0 <= hi)
            y = len(comps) - 1 - i
            ax.plot([lo, hi], [y, y], color=POLICY[c][1], lw=2)
            ax.scatter([d], [y], s=34, zorder=3, color=POLICY[c][1] if conclusive else "white",
                       edgecolor=POLICY[c][1], linewidth=1.5)
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_title(f"fee {f} bps/side", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_yticks(range(len(comps)), [("PPO $-$ " + {"dqn": "Double DQN", "fqi": "FQI", "bandit": "bandit",
                                                           "conv": "best rule"}[c]) for c in reversed(comps)])
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    fig.text(0.5, 0.01, "PPO minus competitor, summed over 1,140 test windows (USD); filled marker = conclusive",
             ha="center", fontsize=8.5)
    save(fig, "fig_paired")


def fig_cumulative(N: dict) -> None:
    order = ["ppo", "dqn", "fqi", "bandit", "conv"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    for ax, f in zip(axes, ("1.0", "2.0")):
        row = N["sim"]["lat450"]["test_A"][f]
        t0 = min(row[p]["trades_sorted"][0][0] for p in order if row[p].get("trades_sorted"))
        for p in order:
            ts = row[p].get("trades_sorted") or []
            if not ts:
                continue
            x = (np.array([t for t, _ in ts]) - t0) / 86400e9
            y = np.cumsum([v for _, v in ts])
            ax.plot(x, y, color=POLICY[p][1], lw=1.3, label=POLICY[p][0])
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlabel("days since the first test trade")
        ax.set_title(f"({'a' if f == '1.0' else 'b'}) fee {f} bps per side", loc="left")
    axes[0].set_ylabel("cumulative net P&L (USD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_cumulative")


def smooth(v, k=10):
    v = np.array([np.nan if x is None else x for x in v], float)
    ok = np.isfinite(v)
    out = np.convolve(np.where(ok, v, 0), np.ones(k), "same") / np.maximum(np.convolve(ok, np.ones(k), "same"), 1)
    return out


def fig_learning(N: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.2), sharex="row")
    seed_color = {"7": BLUE, "17": ORANGE, "27": AQUA}
    for run, c in sorted(N["train_ppo450"].items(), key=lambda kv: int(kv[0].split("_")[0][4:])):
        seed = run.split("_")[0].replace("seed", "")
        x = np.array(c["timesteps"]) / 1e6
        ax[0, 0].plot(x, smooth(c["entry_rate"]), color=seed_color[seed], lw=1, label=f"seed {seed}")
        ax[0, 1].plot(x, smooth(c["reward_per_episode_bps"]), color=seed_color[seed], lw=1)
    for run, c in sorted(N["train_dqn"].items(), key=lambda kv: int(kv[0][4:])):
        seed = run.replace("seed", "")
        x = np.array(c["timesteps"]) / 1e6
        ax[1, 0].plot(x, smooth(c["entry_rate"], 5), color=seed_color[seed], lw=1, label=f"seed {seed}")
        ax[1, 1].plot(x, smooth(c["reward_per_episode_bps"], 5), color=seed_color[seed], lw=1)
    for a in ax[1]:
        a.axvspan(0, 0.2, color=GRID, alpha=0.6, lw=0)
    ax[1, 1].text(0.1, 0.62, "$\\epsilon$ decays\n1.0 to 0.02", ha="center", fontsize=7.5, color=INK2,
                  transform=ax[1, 1].get_xaxis_transform())
    ax[0, 0].set_title("(a) PPO, 450 ms: share of screened entries taken", loc="left", fontsize=8.8)
    ax[0, 1].set_title("(b) PPO, 450 ms: training reward per window", loc="left", fontsize=8.8)
    ax[1, 0].set_title("(c) Double DQN: share of screened entries taken", loc="left", fontsize=8.8)
    ax[1, 1].set_title("(d) Double DQN: training reward per window", loc="left", fontsize=8.8)
    for a in ax[:, 1]:
        a.axhline(0, color=INK, lw=0.6)
        a.set_ylabel("bps of notional")
    for a in ax[:, 0]:
        a.set_ylabel("share")
    for a in ax[1]:
        a.set_xlabel("environment steps (millions)")
    for a in ax[0]:
        a.set_xlabel("environment steps (millions)")
    ax[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_learning")


def fig_learning_ppo150(N: dict) -> None:
    """Six 150 ms runs: colour by seed, line style by entropy coefficient."""
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    seed_color = {"7": BLUE, "17": ORANGE, "27": AQUA}
    style = {"ent001": "-", "ent003": "--"}
    runs = sorted(N["train_ppo150"].items(), key=lambda kv: (int(kv[0].split("_")[0][4:]), kv[0]))
    for run, c in runs:
        seed, ent = run.split("_")
        seed = seed.replace("seed", "")
        x = np.array(c["timesteps"]) / 1e6
        lab = f"seed {seed}, entropy {'0.01' if ent == 'ent001' else '0.03'}"
        ax[0].plot(x, smooth(c["entry_rate"]), color=seed_color[seed], ls=style[ent], lw=0.9, label=lab)
        ax[1].plot(x, smooth(c["reward_per_episode_bps"]), color=seed_color[seed], ls=style[ent], lw=0.9)
    ax[0].set_title("(a) share of screened entries taken", loc="left")
    ax[1].set_title("(b) training reward per window (bps)", loc="left")
    ax[1].axhline(0, color=INK, lw=0.6)
    for a in ax:
        a.set_xlabel("environment steps (millions)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.8, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_learning_ppo150")


def fig_ppo_diagnostics(N: dict) -> None:
    keys = (("train/entropy_loss", "(a) entropy loss"), ("train/approx_kl", "(b) approximate KL"),
            ("train/explained_variance", "(c) explained variance"))
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.2))
    seed_color = {"7": BLUE, "17": ORANGE, "27": AQUA}
    for run, c in sorted(N["train_ppo450"].items(), key=lambda kv: int(kv[0].split("_")[0][4:])):
        seed = run.split("_")[0].replace("seed", "")
        x = np.array(c["timesteps"]) / 1e6
        for a, (k, _) in zip(ax, keys):
            a.plot(x, smooth(c[k]), color=seed_color[seed], lw=0.9, label=f"seed {seed}")
    for a, (_, t) in zip(ax, keys):
        a.set_title(t, loc="left")
        a.set_xlabel("steps (millions)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ppo_diagnostics")


def fig_validation_candidates(N: dict) -> None:
    """Every candidate's validation net P&L at each fee, with the frozen choice ringed."""
    groups = ["ppo", "dqn", "fqi", "bandit"]
    cand = {"ppo": [(c["path"], c["per_fee"]) for c in N["selection_lat450"]["checkpoints"]]}
    for a in ("dqn", "fqi", "bandit"):
        specs = {}
        for r in N["selection_algos"]["all"]:
            if r["algo"] == a:
                key = r["spec"].get("path") or r["spec"].get("dir")
                specs.setdefault(key, {})[str(r["fee"])] = r["net_pnl"]
        cand[a] = sorted(specs.items())
    chosen = {"ppo": N["selection_lat450"]["selected_ppo"]}
    for a in ("dqn", "fqi", "bandit"):
        s = N["selection_algos"]["selected"][a]["spec"]
        chosen[a] = s.get("path") or s.get("dir")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.0))
    axes = axes.ravel()
    for ax, f in zip(axes, FEES):
        for i, a in enumerate(groups):
            ys = [pf[f] for _, pf in cand[a]]
            xs = i + rng.uniform(-0.18, 0.18, len(ys))
            ax.scatter(xs, ys, s=12, color=POLICY[a][1], alpha=0.85, lw=0)
            for (key, pf), x in zip(cand[a], xs):
                if key == chosen[a]:
                    ax.scatter([x], [pf[f]], s=60, facecolor="none", edgecolor=INK, lw=1.1, zorder=3)
        rule = N["selection_lat450"]["conv"][f]["net_pnl"]
        ax.axhline(rule, color=MAGENTA, lw=1.1, ls="--")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(4), ["PPO", "Double DQN", "FQI", "bandit"], fontsize=8)
        ax.set_title(f"fee {f} bps/side", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("validation net P&L (USD)")
    axes[2].set_ylabel("validation net P&L (USD)")
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    save(fig, "fig_validation_candidates")


def fig_stress_latency(N: dict) -> None:
    F = N["fast"]["v2-leadlag-ppo-lat450"]["test_A"]
    order = ["ppo", "conv", "fixed", "raw"]
    lats = [150, 300, 450, 600]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45))
    for ax, f in zip(axes, ("0.5", "1.0", "2.0")):
        for p in order:
            ys = [F[f"fee{f}_lat{L}"][FILE_NAME[p]]["net"] for L in lats]
            ax.plot(lats, ys, marker="o", ms=3.5, color=POLICY[p][1], lw=1.4, label=POLICY[p][0])
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(450, color=INK2, lw=0.7, ls=":")
        ax.set_xticks(lats)
        ax.set_xlabel("fill latency $L$ (ms)")
        ax.set_title(f"fee {f} bps/side", fontsize=9)
    axes[0].set_ylabel("test net P&L (USD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "fig_stress_latency")


def fig_selectivity(N: dict) -> None:
    """Trades and mean net edge per trade against the fee, one frozen policy per algorithm."""
    order = ["ppo", "dqn", "fqi", "bandit", "conv"]
    fees = [float(f) for f in FEES]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    for p in order:
        rows = [N["sim"]["lat450"]["test_A"][f][p] for f in FEES]
        ax[0].plot(fees, [r["trades"] for r in rows], marker="o", ms=3.5, color=POLICY[p][1], label=POLICY[p][0])
        ax[1].plot(fees, [r.get("net_bps_per_trade", np.nan) for r in rows], marker="o", ms=3.5, color=POLICY[p][1])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("test trades (log scale)")
    ax[1].set_ylabel("mean net edge per trade (bps)")
    ax[1].axhline(0, color=INK, lw=0.6)
    for a, t in zip(ax, ("(a) trades", "(b) net edge per trade")):
        a.set_xticks(fees)
        a.set_xlabel("taker fee per side (bps)")
        a.set_title(t, loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig_selectivity")


# --------------------------------------------------------------------------- tables
def money(x: float, nd: int = 2) -> str:
    s = f"{x:,.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    if s in ("$-$0.00", "$-$0.000"):
        s = s.replace("$-$", "")
    return s


def write(name: str, text: str) -> None:
    (GEN / f"tab_{name}.tex").write_text(text)


def tables(N: dict) -> None:
    S = N["sim"]["lat450"]
    label = {"ppo": "PPO", "dqn": "Double DQN", "fqi": "FQI", "bandit": "Contextual bandit",
             "conv": "Best rule (convergence exit)", "fixed": "De-meaned rule, fixed hold",
             "raw": "Raw-gap rule (version 1)", "flat": "Always flat"}

    # Main test table (450 ms, test_A).
    lines = []
    for p in ("ppo", "dqn", "fqi", "bandit", "conv", "fixed", "raw", "flat"):
        cells = [f"{money(S['test_A'][f][p]['net'])} & {S['test_A'][f][p]['trades']:,}" for f in FEES]
        lines.append(f"{label[p]} & " + " & ".join(cells) + r" \\")
        if p in ("bandit",):
            lines.append(r"\midrule")
    write("test450", "\n".join(lines) + "\n")

    # 3 October (test_B) at 450 ms.
    lines = []
    for p in ("ppo", "dqn", "fqi", "bandit", "conv", "fixed", "raw"):
        cells = [f"{money(S['test_B'][f][p]['net'])} & {S['test_B'][f][p]['trades']:,}" for f in FEES]
        lines.append(f"{label[p]} & " + " & ".join(cells) + r" \\")
    write("testB450", "\n".join(lines) + "\n")

    # Validation (fast replay, frozen selection files), 450 ms.
    sel = N["selection_lat450"]
    alg = N["selection_algos"]["selected"]
    lines = []
    rows = [("ppo", {f: (sel["ppo_val"][f]["net_pnl"], sel["ppo_val"][f]["trade_count"]) for f in FEES})]
    for a in ("dqn", "fqi", "bandit"):
        rows.append((a, {f: (alg[a]["per_fee"][f]["net_pnl"], alg[a]["per_fee"][f]["trade_count"]) for f in FEES}))
    for p in ("conv", "fixed", "raw"):
        rows.append((p, {f: (sel[p][f]["net_pnl"], sel[p][f]["trade_count"]) for f in FEES}))
    for p, d in rows:
        lines.append(f"{label[p]} & " + " & ".join(f"{money(d[f][0])} & {d[f][1]:,}" for f in FEES) + r" \\")
        if p == "bandit":
            lines.append(r"\midrule")
    write("val450", "\n".join(lines) + "\n")

    # Paired comparisons, test_A, 450 ms.
    P, A = N["paired_lat450"]["test_A"], N["paired_algos"]["test_A"]
    verdict = {"PPO better": "PPO better", "rule better": "rule better", "inconclusive": "inconclusive",
               "better": "PPO better", "worse": "PPO worse"}
    lines = []
    for f in FEES:
        first = True
        for c in ("conv", "raw", "dqn", "fqi", "bandit"):
            if c in ("conv", "raw"):
                s = P[f]["dgap_conv_rule_tuned" if c == "conv" else "raw_rule_tuned"]
                d, (lo, hi) = s["sum_diff_usd"], s["ci90_hour_block"]
                days, v, n_o = s["days_positive"], verdict[s["verdict"]], s["rule_trades"]
            else:
                s = A[f][c]["vs_ppo"]
                d, (lo, hi) = -s["sum_diff_usd"], (-s["ci90_hour_block"][1], -s["ci90_hour_block"][0])
                k, n = s["days_positive"].split("/")
                days = f"{int(n) - int(k)}/{n}"
                v = {"better": "PPO worse", "worse": "PPO better", "inconclusive": "inconclusive"}[s["verdict"]]
                n_o = A[f][c]["trades"]
            name = {"conv": "best rule", "raw": "raw-gap rule (version 1)", "dqn": "Double DQN", "fqi": "FQI",
                    "bandit": "bandit"}[c]
            fee_cell = f"{f}" if first else ""
            first = False
            lines.append(f"{fee_cell} & {name} & {n_o:,} & {money(d)} & [{money(lo)}, {money(hi)}] & {days} & {v}" + r" \\")
        if f != FEES[-1]:
            lines.append(r"\midrule")
    write("paired450", "\n".join(lines) + "\n")

    # Per-trade economics, test_A 450 ms, fees 1.0 and 2.0.
    lines = []
    for f in ("1.0", "2.0"):
        first = True
        for p in ("ppo", "dqn", "fqi", "bandit", "conv"):
            r = S["test_A"][f][p]
            fee_cell = f if first else ""
            first = False
            short = {"ppo": "PPO", "dqn": "Double DQN", "fqi": "FQI", "bandit": "Bandit", "conv": "Best rule"}
            # The bandit never chooses an exit: every trade closes after its fixed 3 s hold.
            exits = ("\\multicolumn{2}{c}{fixed 3\\,s}" if p == "bandit" else
                     f"{100 * r['exit_policy_share']:.0f} & {100 * r['exit_max_hold_share']:.0f}")
            lines.append(f"{fee_cell} & {short[p]} & {r['trades']:,} & {r['gross_bps_per_trade']:.2f} & "
                         f"{r['net_bps_per_trade']:.2f} & {100 * r['win_rate']:.1f} & {r['mean_hold_s']:.1f} & "
                         f"{exits}" + r" \\")
        if f == "1.0":
            lines.append(r"\midrule")
    write("pertrade450", "\n".join(lines) + "\n")

    # 150 ms group: validation and both test sets.
    S1 = N["sim"]["lat150"]
    sel1 = N["selection_lat150"]
    lines = [r"\multicolumn{9}{@{}l}{\emph{Validation, 25--28 Sep (selection files)}} \\"]
    val1 = {"ppo": {f: (sel1["ppo_val"][f]["net_pnl"], sel1["ppo_val"][f]["trade_count"]) for f in FEES}}
    for p in ("conv", "fixed", "raw"):
        val1[p] = {f: (sel1[p][f]["net_pnl"], sel1[p][f]["trade_count"]) for f in FEES}
    for p in ("ppo", "conv", "fixed", "raw"):
        lines.append(f"\\quad {label[p]} & " + " & ".join(f"{money(val1[p][f][0])} & {val1[p][f][1]:,}" for f in FEES) + r" \\")
    for split, name in (("test_A", "Test, 29 Sep--2 Oct"), ("test_B", "Test, 3 Oct")):
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        for p in ("ppo", "conv", "fixed", "raw"):
            cells = [f"{money(S1[split][f][p]['net'])} & {S1[split][f][p]['trades']:,}" for f in FEES]
            lines.append(f"\\quad {label[p]} & " + " & ".join(cells) + r" \\")
    write("lat150", "\n".join(lines) + "\n")

    # Paired, 150 ms group (test_A) against the strongest rule.
    P1 = N["paired_lat150"]["test_A"]
    lines = []
    for f in FEES:
        s = P1[f]["dgap_conv_rule_tuned"]
        lo, hi = s["ci90_hour_block"]
        lines.append(f"{f} & {money(s['ppo_net'])} ({s['ppo_trades']:,}) & {money(s['rule_net'])} ({s['rule_trades']:,}) & "
                     f"{money(s['sum_diff_usd'])} & [{money(lo)}, {money(hi)}] & {s['verdict']}" + r" \\")
    write("paired150", "\n".join(lines) + "\n")

    # Stress table: PPO and best rule under each stress (450 ms group), plus algorithms under HL clock.
    F = N["fast"]["v2-leadlag-ppo-lat450"]["test_A"]
    H = N["fast"]["v2-algos-lat450"]["test_A"]
    lines = []
    for f in FEES:
        cells = []
        for tag in (f"fee{f}_lat150", f"fee{f}_lat300", f"fee{f}_lat450", f"fee{f}_lat600",
                    f"fee{f}_lat450_spread1.5", f"fee{f}_lat450_hlclock"):
            a, b = F[tag]["ppo_selected"], F[tag]["dgap_conv_rule_tuned"]
            cells.append(f"{money(a['net'])} / {money(b['net'])}")
        lines.append(f"{f} & " + " & ".join(cells) + r" \\")
    write("stress450", "\n".join(lines) + "\n")

    lines = []
    for f in FEES:
        cells = []
        for p in ("ppo", "dqn", "fqi", "bandit", "conv", "raw"):
            if p in ("dqn", "fqi", "bandit"):
                r = H[f"fee{f}_lat450_hlclock"][p]
            else:
                r = F[f"fee{f}_lat450_hlclock"][FILE_NAME[p]]
            cells.append(f"{money(r['net'])} & {r['trades']:,}")
        lines.append(f"{f} & " + " & ".join(cells) + r" \\")
    write("hlclock450", "\n".join(lines) + "\n")

    # Fee 4.5 extrapolation.
    lines = []
    for p in ("ppo", "conv", "fixed", "raw"):
        r = F["fee4.5_lat450"][FILE_NAME[p]]
        lines.append(f"{label[p]} & {money(r['net'])} & {r['trades']:,}" + r" \\")
    write("fee45", "\n".join(lines) + "\n")

    # Same-snapshot split (received-book fills), PPO by latency.
    lines = []
    for L in (150, 300, 450, 600):
        cells = []
        for f in ("0.5", "1.0"):
            r = F[f"fee{f}_lat{L}"]["ppo_selected"]
            cells.append(f"{r['same_snapshot_trades']:,} of {r['trades']:,} ({100 * r['same_snapshot_trades'] / r['trades']:.0f}\\%) & "
                         f"{money(r['same_snapshot_net'])} of {money(r['net'])}")
        lines.append(f"{L} & " + " & ".join(cells) + r" \\")
    write("samesnap", "\n".join(lines) + "\n")

    # Appendix: per-day net P&L, test_A 450 ms.
    days = sorted(S["test_A"]["1.0"]["ppo"]["daily_pnl"])
    lines = []
    for f in FEES:
        first = True
        for p in ("ppo", "dqn", "fqi", "bandit", "conv", "raw"):
            r = S["test_A"][f][p]
            cells = " & ".join(f"{money(r['daily_pnl'].get(d, 0.0))} ({(r['daily_trades'] or {}).get(d, 0):,})" for d in days)
            lines.append(f"{f if first else ''} & {label[p]} & {cells}" + r" \\")
            first = False
        if f != FEES[-1]:
            lines.append(r"\midrule")
    write("perday450", "\n".join(lines) + "\n")

    # Appendix: every PPO run on the test sets (fast replay at the group's latency).
    lines = []
    for grp, lat in (("v2-leadlag-ppo", 150), ("v2-leadlag-ppo-lat450", 450)):
        Fg = N["fast"][grp]["test_A"]
        runs = sorted({n for tag in Fg for n in Fg[tag] if n.startswith("ppo_seed")},
                      key=lambda n: (int(n.split("_")[1][4:]), n))
        for run in runs:
            cells = " & ".join(f"{money(Fg[f'fee{f}_lat{lat}'][run]['net'])} & {Fg[f'fee{f}_lat{lat}'][run]['trades']:,}"
                               for f in FEES)
            nm = run.replace("ppo_seed", "seed ").replace("_ent001", ", 0.01").replace("_ent003", ", 0.03")
            lines.append(f"{lat} & {nm} & {cells}" + r" \\")
        lines.append(r"\midrule")
    write("ppo_runs", "\n".join(lines[:-1]) + "\n")

    # Appendix: every algorithm candidate on validation.
    lines = []
    for a in ("dqn", "fqi", "bandit"):
        specs = {}
        for r in N["selection_algos"]["all"]:
            if r["algo"] == a:
                key = r["spec"].get("path") or r["spec"].get("dir")
                specs.setdefault(key, {})[str(r["fee"])] = (r["net_pnl"], r["trade_count"])
        chosen = N["selection_algos"]["selected"][a]["spec"]
        chosen = chosen.get("path") or chosen.get("dir")
        def order(k):
            # Seed, then checkpoint step in numerical order, with the final model last.
            step = k.split("/")[-1].replace("checkpoint_", "").replace(".zip", "")
            return (int(k.split("seed")[1].split("/")[0]), int(step) if step.isdigit() else 10 ** 12, k)

        for key in sorted(specs, key=order):
            seed = key.split("seed")[1].split("/")[0]
            ck = key.split("/")[-1].replace("checkpoint_", "").replace(".zip", "")
            ck = {"fqi": "--", "bandit": "--"}.get(ck, ck)
            if ck.isdigit():
                ck = f"{int(ck):,}"
            mark = r"$\ast$" if key == chosen else ""
            cells = " & ".join(f"{money(specs[key][f][0])} & {specs[key][f][1]:,}" for f in FEES)
            lines.append(f"{label[a] if a != 'fqi' else 'FQI'} & {seed} & {ck}{mark} & {cells}" + r" \\")
        lines.append(r"\midrule")
    write("algo_candidates", "\n".join(lines[:-1]) + "\n")

    # Appendix: full stress grid for the 450 ms group (fast replay).
    lines = []
    for tag_suffix, name in (("lat150", "fill 150 ms"), ("lat300", "fill 300 ms"), ("lat450", "fill 450 ms (primary)"),
                             ("lat600", "fill 600 ms"), ("lat450_spread1.5", "spread $\\times1.5$"),
                             ("lat450_hlclock", "HL-clock fill")):
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        for p in ("ppo", "conv", "fixed", "raw"):
            cells = " & ".join(f"{money(F[f'fee{f}_{tag_suffix}'][FILE_NAME[p]]['net'])} & "
                               f"{F[f'fee{f}_{tag_suffix}'][FILE_NAME[p]]['trades']:,}" for f in FEES)
            lines.append(f"\\quad {label[p]} & {cells}" + r" \\")
    write("stress_full", "\n".join(lines) + "\n")


def numbers(N: dict) -> dict:
    """Compact, report-facing numbers (no per-trade lists)."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "trades_sorted"}
        if isinstance(o, list) and len(o) > 200:
            return f"<list of {len(o)}>"
        return o
    out = strip({k: v for k, v in N.items() if not k.startswith("train_")})
    out["training_final"] = {g: {run: {k: (c[k][-1] if c.get(k) else None) for k in
                                       ("timesteps", "entry_rate", "exit_rate", "trades_per_episode",
                                        "reward_per_episode_bps", "train/explained_variance", "epsilon")}
                                 for run, c in N[g].items()}
                             for g in ("train_ppo450", "train_ppo150", "train_dqn")}
    # Early-training minimum of the smoothed PPO entry share (shows the dip and recovery).
    out["ppo450_entry_min"] = {run: {"min_smoothed_entry_rate": float(np.nanmin(smooth(c["entry_rate"]))),
                                     "at_step": int(c["timesteps"][int(np.nanargmin(smooth(c["entry_rate"])))])}
                               for run, c in N["train_ppo450"].items()}
    out["dqn_first_positive_reward_step"] = {
        run: next((int(t) for t, r in zip(c["timesteps"], smooth(c["reward_per_episode_bps"], 5)) if r > 0), None)
        for run, c in N["train_dqn"].items()}
    out["ppo450_first_positive_reward_step"] = {
        run: next((int(t) for t, r in zip(c["timesteps"], smooth(c["reward_per_episode_bps"])) if r > 0), None)
        for run, c in N["train_ppo450"].items()}
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    N = collect()
    fig_smdp()
    fig_train_signal(N)
    fig_algos_by_fee(N)
    fig_paired(N)
    fig_cumulative(N)
    fig_learning(N)
    fig_learning_ppo150(N)
    fig_ppo_diagnostics(N)
    fig_validation_candidates(N)
    fig_stress_latency(N)
    fig_selectivity(N)
    tables(N)
    (GEN / "report_numbers.json").write_text(json.dumps(numbers(N), indent=1, default=float))
    print("wrote figures, tables and generated/report_numbers.json")


if __name__ == "__main__":
    main()
