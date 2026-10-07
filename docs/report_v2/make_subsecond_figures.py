"""Report-sized copies of the two sub-second analysis figures (Figures 10 and 13).

analysis/subsecond_edge.py draws its figures 13 inches wide for screen use. This script
redraws them at the report's text width with the fonts of make_report_figures.py and
writes new files under figures/; the analysis outputs in analysis/ are not touched.

- fig_edge_latency: drawn only from analysis/subsecond_edge.json.
- fig_timing: the histograms need per-row arrays that the JSON does not store, so they are
  recomputed, one day at a time and read-only, from data/recorded-cache with the helper
  functions of analysis/subsecond_edge.py. The recomputed percentiles and detection shares
  are checked against the JSON before anything is drawn.

    /home/kerk/models/.venv/bin/python docs/report_v2/make_subsecond_figures.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ANALYSIS = HERE / "analysis"
sys.path.insert(0, str(ANALYSIS))
import subsecond_edge as sse  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 6.27
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_edge_latency(doc: dict) -> None:
    R, L = doc["results"], doc["latencies_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.75))
    for ax, x, tag in zip(axes, ["3", "5", "8"], "abc"):
        def curve(combo, s):
            return [R[combo][s][x][str(lat)]["mean_gross_bps"] for lat in L]
        ax.plot(L, curve("recv|local", "grid_all"), color=BLUE, marker="o", ms=2.5,
                label="1 s grid, every signal second")
        ax.plot(L, curve("recv|local", "grid_db"), color=BLUE, ls="--", label="1 s grid, 3 s debounce")
        ax.plot(L, curve("recv|local", "event_db"), color=ORANGE, marker="o", ms=2.5,
                label="every Binance update, 3 s debounce")
        ax.fill_between(L, curve("recv|engine_next", "event_db"), curve("recv|engine_asof", "event_db"),
                        color=AQUA, alpha=0.15, lw=0, label="down to next-snapshot fills")
        ax.plot(L, curve("recv|engine_asof", "event_db"), color=AQUA, marker="s", ms=2.5,
                label="every Binance update, HL-clock fills")
        for f in (1, 2):
            c = 2 * f + 0.2
            ax.axhline(c, color=INK2, lw=0.8, ls=":", label="break-even, 2f + 0.2" if f == 1 else None)
            ax.text(1560, c, f"f = {f}", va="bottom", ha="right", fontsize=7.5, color=INK2)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlim(-40, 1580)
        ax.set_xticks([0, 500, 1000, 1500])
        ax.set_xlabel("latency $L$ (ms)")
        ax.set_title(f"({tag}) $|g-\\hat b|\\geq{x}$ bps", loc="left")
    axes[0].set_ylabel("mean gross edge per trade (bps)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_edge_latency")


def recompute_timing() -> tuple[dict, dict]:
    """Per-row timing arrays and grid detection delays, one day at a time (read-only)."""
    timing: dict = {"b_age": [], "h_age": [], "h_int": []}
    detect: dict = {"3": [], "5": [], "8": []}
    for day in sse.DAYS:
        d = sse.load_day(day)
        day_start = int(np.datetime64(day, "ns").astype(np.int64))
        grid = day_start + np.arange(86_400, dtype=np.int64) * sse.SEC
        d.pop("h_ex_order", None)
        timing["b_age"].append(((d["b_loc"] - d["b_ex"]) / 1e6).astype(np.float32))
        timing["h_age"].append(((d["h_loc"] - d["h_ex"]) / 1e6).astype(np.float32))
        timing["h_int"].append((np.diff(d["h_loc"]) / 1e6).astype(np.float32))
        V = sse.build_view(d, grid, day_start, d["h_loc"], d["h_mid"])
        for x in detect:
            r = sse.detection_delay(d["b_loc"], V["dgap_ev"], V["dgap_grid"], grid, day_start, float(x))
            detect[x].append(r["delay_ms"])
        del d, V, grid
        gc.collect()
        print("processed", day, flush=True)
    return {k: np.concatenate(v) for k, v in timing.items()}, {k: np.concatenate(v) for k, v in detect.items()}


def check(doc: dict, timing: dict, detect: dict) -> None:
    """Stop if the recomputed arrays do not reproduce the stored analysis output."""
    tp = doc["timing_pooled"]
    pairs = (("b_age", "binance_recv_minus_exch_ms"), ("h_age", "hl_recv_minus_exch_ms"),
             ("h_int", "hl_interval_recv_ms"))
    for mine, theirs in pairs:
        for q in (10, 50, 90):
            a, b = float(np.percentile(timing[mine], q)), tp[theirs][f"p{q}"]
            if abs(a - b) > 1e-3 * max(1.0, abs(b)):
                raise RuntimeError(f"{mine} p{q}: {a} != {b}")
    for x, dl in detect.items():
        a = float(np.mean(~np.isfinite(dl)))
        b = doc["detection_delay"][x]["share_not_seen_within_10s"]
        if abs(a - b) > 1e-6:
            raise RuntimeError(f"detection X={x}: {a} != {b}")
    print("recomputed timing and detection match analysis/subsecond_edge.json", flush=True)


def fig_timing(timing: dict, detect: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45))
    ax = axes[0]
    bins = np.arange(0, 1001, 10)
    ax.hist(np.clip(timing["b_age"], 0, 1000), bins=bins, density=True, color=BLUE, alpha=0.85, label="Binance")
    ax.hist(np.clip(timing["h_age"], 0, 1000), bins=bins, density=True, color=ORANGE, alpha=0.85, label="HL")
    ax.set_xticks([0, 500, 1000])
    ax.set_xlabel("receive time $-$ exchange\ntimestamp (ms)")
    ax.set_ylabel("density")
    ax.set_title("(a) feed delay", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(np.clip(timing["h_int"], 0, 1500), bins=np.arange(0, 1501, 20), density=True, color=ORANGE)
    ax.set_xlabel("time between HL books,\nreceive clock (ms)")
    ax.set_title("(b) HL update interval", loc="left")
    ax.set_xticks([0, 500, 1000, 1500])
    ax = axes[2]
    xs = np.arange(0, 3001, 10)
    for x, c in (("3", BLUE), ("5", ORANGE), ("8", AQUA)):
        dl = detect[x]
        dls = np.sort(np.where(np.isfinite(dl), dl, np.inf))
        ax.plot(xs, np.searchsorted(dls, xs, side="right") / len(dl), color=c, label=f"$X={x}$ bps")
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("onset to first whole-second\ndetection (ms)")
    ax.set_ylabel("share of onsets")
    ax.set_title("(c) grid detection delay", loc="left")
    ax.legend(fontsize=7.5, loc="lower right", handlelength=1.2, labelspacing=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_timing")


def main() -> None:
    doc = json.loads((ANALYSIS / "subsecond_edge.json").read_text())
    fig_edge_latency(doc)
    timing, detect = recompute_timing()
    check(doc, timing, detect)
    fig_timing(timing, detect)
    print("wrote figures/fig_edge_latency and figures/fig_timing")


if __name__ == "__main__":
    main()
