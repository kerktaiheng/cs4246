"""@file subsecond_edge.py
@brief Read-only sub-second measurement of the Binance-leads-Hyperliquid edge.
@details Processes six training days one at a time from data/recorded-cache and writes
subsecond_edge.json plus two figures next to this file. No model is trained and no
existing file is modified.

Definitions (all times are our receive clock, local_ts_ns, unless stated):
  gap(t)   = (binance_mid - hl_mid) / hl_mid * 1e4 bps, latest quotes at or before t.
  basis(t) = EMA (span 60 s, alpha = 2/61) of gap sampled at whole seconds strictly
             before floor(t); restarts after any data hole > 2 s; usable after 60 s.
  dgap(t)  = gap(t) - basis(t).
  Trade    : |dgap| >= X at t -> direction d = sign(dgap); enter HL at the book at or
             before t + L (buy ask / sell bid), exit 3 s after entry at the book at or
             before t + L + 3 s crossing the spread again.
  gross    = d * (exit - entry) / entry * 1e4;  net(f) = gross - 2 f - 0.2.

Fill models (which HL book is used for entry and exit):
  local        : latest book we had RECEIVED by t+L (the project's simulator convention).
  engine_asof  : latest book whose HL exchange timestamp <= t+L (book in HL's engine when
                 the order arrives, assuming our clock equals HL's; still at HL's ~0.5 s
                 publish granularity, so the engine may have moved further).
  engine_next  : first book whose HL exchange timestamp >= t+L (pessimistic bracket).
Signal views (which HL book the signal sees):
  recv   : latest HL book received by t (causal, what we actually had).
  engine : latest HL book with exchange timestamp <= t (emulates an HL feed with zero
           publish/transport delay but the same ~0.5 s snapshot cadence; not causal for us).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path("/home/kerk/models")
CACHE = ROOT / "data" / "recorded-cache"
OUT = Path(__file__).resolve().parent

DAYS = ["2026-09-04", "2026-09-08", "2026-09-12", "2026-09-15", "2026-09-19", "2026-09-23"]
LATENCIES_MS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500]
THRESHOLDS = [3.0, 5.0, 8.0]
FEES = [1.0, 2.0, 3.5]
SLIP_ROUND_TRIP = 0.2
HOLD_NS = 3_000_000_000
DEBOUNCE_NS = 3_000_000_000
HOLE_NS = 2_000_000_000
SEC = 1_000_000_000
SPAN = 60
WARMUP_S = 60
DETECT_HORIZON_S = 10
MATCH_LAT_NS = 150_000_000
SETS = ("grid_all", "grid_db", "event_db")
COMBOS = (("recv", "local"), ("recv", "engine_asof"), ("recv", "engine_next"),
          ("engine", "engine_asof"), ("engine", "engine_next"))


def load_day(day: str) -> dict:
    """@brief Load only the needed columns of one day; drop invalid or crossed books."""
    b = pq.read_table(CACHE / f"{day}.binance.parquet",
                      columns=["local_ts_ns", "exchange_ts_ns", "full_source_valid", "bid", "ask"])
    bl = b["local_ts_ns"].to_numpy(); bx = b["exchange_ts_ns"].to_numpy()
    bv = b["full_source_valid"].to_numpy(); bb = b["bid"].to_numpy(); ba = b["ask"].to_numpy()
    del b
    keep = (bv == 1) & (bb > 0) & (ba > bb)
    h = pq.read_table(CACHE / f"{day}.hyperliquid.parquet",
                      columns=["local_ts_ns", "exchange_ts_ns", "full_source_valid",
                               "bid_prices", "ask_prices"])
    hl = h["local_ts_ns"].to_numpy(); hx = h["exchange_ts_ns"].to_numpy()
    hv = h["full_source_valid"].to_numpy()
    hb = pc.list_element(h["bid_prices"], 0).to_numpy()
    ha = pc.list_element(h["ask_prices"], 0).to_numpy()
    del h
    hkeep = (hv == 1) & (hb > 0) & (ha > hb)
    d = {"b_loc": bl[keep], "b_ex": bx[keep], "b_mid": (bb[keep] + ba[keep]) / 2,
         "h_loc": hl[hkeep], "h_ex": hx[hkeep], "h_bid": hb[hkeep], "h_ask": ha[hkeep],
         "b_rows": int(len(bl)), "b_kept": int(keep.sum()),
         "h_rows": int(len(hl)), "h_kept": int(hkeep.sum())}
    if not (np.all(np.diff(d["b_loc"]) >= 0) and np.all(np.diff(d["h_loc"]) >= 0)):
        raise ValueError(f"{day}: local timestamps not sorted")
    if not np.all(np.diff(d["h_ex"]) >= 0):
        o = np.argsort(d["h_ex"], kind="stable")
        d["h_ex_order"] = o
    d["h_mid"] = (d["h_bid"] + d["h_ask"]) / 2
    return d


def asof(ts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """@brief Index of the last element <= q (-1 if none)."""
    return np.searchsorted(ts, q, side="right") - 1


def hole_break_seconds(ts: np.ndarray, day_start: int, n_sec: int) -> np.ndarray:
    """@brief Grid seconds at or after the end of an inter-update gap > 2 s."""
    out = np.zeros(n_sec, bool)
    gaps = np.diff(ts)
    ends = ts[1:][gaps > HOLE_NS]
    k = np.ceil((ends - day_start) / SEC).astype(np.int64)
    k = k[(k >= 0) & (k < n_sec)]
    out[k] = True
    return out


def ema_previous(gap: np.ndarray, start: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """@brief Basis at second s = EMA of gap at valid seconds < s in the same segment."""
    alpha = 2.0 / (SPAN + 1.0)
    out = np.full(len(gap), np.nan)
    level = np.nan
    for i in range(len(gap)):
        if not valid[i]:
            level = np.nan
            continue
        if start[i]:
            level = gap[i]
        out[i] = level
        level = alpha * gap[i] + (1 - alpha) * level
    return out


def build_view(d: dict, grid: np.ndarray, day_start: int, h_ts: np.ndarray,
               h_mid: np.ndarray) -> dict:
    """@brief Grid gap/basis/dgap and event-time dgap for one HL timestamp view."""
    n = len(grid)
    ib = asof(d["b_loc"], grid); ih = asof(h_ts, grid)
    ok_idx = (ib >= 0) & (ih >= 0)
    ibc = np.maximum(ib, 0); ihc = np.maximum(ih, 0)
    age_b = grid - d["b_loc"][ibc]; age_h = grid - h_ts[ihc]
    valid = ok_idx & (age_b <= HOLE_NS) & (age_h <= HOLE_NS)
    gap = (d["b_mid"][ibc] - h_mid[ihc]) / h_mid[ihc] * 1e4
    brk = hole_break_seconds(d["b_loc"], day_start, n) | hole_break_seconds(h_ts, day_start, n)
    prev_valid = np.r_[False, valid[:-1]]
    start = valid & (~prev_valid | brk)
    seg = np.cumsum(start)
    seg_start_idx = np.flatnonzero(start)
    seg_age = np.full(n, -1, np.int64)
    if len(seg_start_idx):
        first = seg_start_idx[np.maximum(seg - 1, 0)]
        seg_age = np.where(valid & (seg > 0), np.arange(n) - first, -1)
    basis = ema_previous(gap, start, valid)
    warm = valid & (seg_age >= WARMUP_S)
    dgap_grid = np.where(warm, gap - basis, np.nan)
    # Event-time dgap at every Binance update (basis of the current whole second).
    t = d["b_loc"]
    sec = (t - day_start) // SEC
    in_day = (sec >= 0) & (sec < n)
    secc = np.clip(sec, 0, n - 1)
    ihe = asof(h_ts, t)
    ihec = np.maximum(ihe, 0)
    age_he = t - h_ts[ihec]
    ev_ok = in_day & (ihe >= 0) & (age_he <= HOLE_NS) & warm[secc]
    gap_e = (d["b_mid"] - h_mid[ihec]) / h_mid[ihec] * 1e4
    dgap_ev = np.where(ev_ok, gap_e - basis[secc], np.nan)
    return {"dgap_grid": dgap_grid, "dgap_ev": dgap_ev, "warm": warm,
            "n_warm": int(warm.sum()), "n_valid": int(valid.sum())}


def fill_index(mode: str, d: dict, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@brief HL row used for a fill at time q under a fill model, plus a validity mask."""
    if mode == "local":
        ts = d["h_loc"]
        i = asof(ts, q); ic = np.maximum(i, 0)
        ok = (i >= 0) & (q - ts[ic] <= HOLE_NS) & (q <= ts[-1])
        return ic, ok
    ts = d["h_ex_sorted"]
    if mode == "engine_asof":
        i = asof(ts, q); ic = np.maximum(i, 0)
        ok = (i >= 0) & (q - ts[ic] <= HOLE_NS) & (q <= ts[-1])
    elif mode == "engine_next":
        i = np.searchsorted(ts, q, side="left"); ic = np.minimum(i, len(ts) - 1)
        ok = (i < len(ts)) & (ts[ic] - q <= HOLE_NS)
    else:
        raise ValueError(mode)
    return d["h_ex_rowmap"][ic], ok


def trade_gross(d: dict, t: np.ndarray, direction: np.ndarray, lat_ns: int,
                mode: str) -> np.ndarray:
    """@brief Gross bps of a 3 s round trip entered at t+L (NaN where a fill is invalid)."""
    ie, oke = fill_index(mode, d, t + lat_ns)
    ix, okx = fill_index(mode, d, t + lat_ns + HOLD_NS)
    buy = direction > 0
    entry = np.where(buy, d["h_ask"][ie], d["h_bid"][ie])
    exitp = np.where(buy, d["h_bid"][ix], d["h_ask"][ix])
    g = direction * (exitp - entry) / entry * 1e4
    return np.where(oke & okx, g, np.nan)


def debounce(times: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """@brief Greedy non-overlapping selection: after a signal, none for DEBOUNCE_NS."""
    ct = times[cand]
    sel = []
    i = 0
    while i < len(ct):
        sel.append(i)
        i = int(np.searchsorted(ct, ct[i] + DEBOUNCE_NS, side="left"))
    return cand[np.asarray(sel, dtype=np.int64)] if sel else cand[:0]


def detection_delay(t_ev: np.ndarray, dgap_ev: np.ndarray, dgap_grid: np.ndarray,
                    grid: np.ndarray, day_start: int, x: float) -> dict:
    """@brief For each onset (first Binance update with |dgap|>=X after one below X or of
    opposite sign), delay until the first whole second with same-sign |dgap|>=X, and how
    long the event-time condition persisted."""
    sig = np.where(np.abs(dgap_ev) >= x, np.sign(dgap_ev), 0.0)
    sig = np.nan_to_num(sig)
    prev = np.r_[0.0, sig[:-1]]
    onset = np.flatnonzero((sig != 0) & (sig != prev))
    t0 = t_ev[onset]; s0 = sig[onset]
    # Persistence of the event-time condition: next Binance update where sig changes.
    change = np.flatnonzero(sig[1:] != sig[:-1]) + 1
    nxt = np.searchsorted(change, onset, side="right")
    end_idx = np.where(nxt < len(change), change[np.minimum(nxt, len(change) - 1)], len(sig) - 1)
    persist_ms = (t_ev[end_idx] - t0) / 1e6
    first_sec = (t0 - day_start + SEC - 1) // SEC
    ks = first_sec[:, None] + np.arange(DETECT_HORIZON_S)[None, :]
    ksc = np.clip(ks, 0, len(grid) - 1)
    g = dgap_grid[ksc]
    hit = (ks < len(grid)) & (np.abs(np.nan_to_num(g)) >= x) & (np.sign(np.nan_to_num(g)) == s0[:, None])
    any_hit = hit.any(axis=1)
    k = np.argmax(hit, axis=1)
    det_t = grid[ksc[np.arange(len(k)), k]]
    delay_ms = np.where(any_hit, (det_t - t0) / 1e6, np.nan)
    return {"delay_ms": delay_ms.astype(np.float32), "first_slot": (any_hit & (k == 0)),
            "persist_ms": persist_ms.astype(np.float32), "t0": t0, "dir": s0,
            "det_t": det_t, "seen": any_hit}


def pct(a: np.ndarray, qs=(10, 50, 90)) -> dict:
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def process_day(day: str, acc: dict, timing: dict, detect: dict) -> dict:
    d = load_day(day)
    day_start = int(np.datetime64(day, "ns").astype(np.int64))
    n_sec = 86_400
    grid = day_start + np.arange(n_sec, dtype=np.int64) * SEC
    order = d.pop("h_ex_order", None)
    rowmap = order if order is not None else np.arange(len(d["h_ex"]))
    d["h_ex_rowmap"] = rowmap
    d["h_ex_sorted"] = d["h_ex"][rowmap]
    meta = {"day": day, "binance_rows": d["b_rows"], "binance_kept": d["b_kept"],
            "hl_rows": d["h_rows"], "hl_kept": d["h_kept"],
            "binance_tob_mid_change_share": float(np.mean(np.diff(d["b_mid"]) != 0)),
            "hl_mid_change_share": float(np.mean(np.diff(d["h_mid"]) != 0))}

    # --- timing (measurement 3) ---
    b_age = ((d["b_loc"] - d["b_ex"]) / 1e6).astype(np.float32)
    h_age = ((d["h_loc"] - d["h_ex"]) / 1e6).astype(np.float32)
    h_int = (np.diff(d["h_loc"]) / 1e6).astype(np.float32)
    h_int_ex = (np.diff(d["h_ex_sorted"]) / 1e6).astype(np.float32)
    b_int = (np.diff(d["b_loc"]) / 1e6).astype(np.float32)
    ih = asof(d["h_loc"], grid); ib = asof(d["b_loc"], grid)
    okg = (ih >= 0) & (ib >= 0)
    h_eff = ((grid - d["h_ex"][np.maximum(ih, 0)]) / 1e6)[okg].astype(np.float32)
    b_eff = ((grid - d["b_ex"][np.maximum(ib, 0)]) / 1e6)[okg].astype(np.float32)
    for k, v in (("binance_recv_minus_exch_ms", b_age), ("hl_recv_minus_exch_ms", h_age),
                 ("hl_interval_recv_ms", h_int), ("hl_interval_exch_ms", h_int_ex),
                 ("binance_interval_recv_ms", b_int), ("hl_info_age_at_grid_ms", h_eff),
                 ("binance_info_age_at_grid_ms", b_eff)):
        timing.setdefault(k, []).append(v)
    meta["timing"] = {
        "binance_recv_minus_exch_ms": pct(b_age), "hl_recv_minus_exch_ms": pct(h_age),
        "hl_interval_recv_ms": pct(h_int, (10, 50, 90, 99)),
        "hl_interval_exch_ms": pct(h_int_ex, (10, 50, 90, 99)),
        "hl_updates_per_s": float(len(d["h_loc"]) / ((d["h_loc"][-1] - d["h_loc"][0]) / 1e9)),
        "binance_updates_per_s": float(len(d["b_loc"]) / ((d["b_loc"][-1] - d["b_loc"][0]) / 1e9)),
    }
    del b_age, h_age, h_int, h_int_ex, b_int, h_eff, b_eff, ih, ib, okg

    # --- signals and trades (measurements 1, 2, 4) ---
    views = {"recv": build_view(d, grid, day_start, d["h_loc"], d["h_mid"]),
             "engine": build_view(d, grid, day_start, d["h_ex_sorted"], d["h_mid"][rowmap])}
    meta["warm_seconds"] = {v: views[v]["n_warm"] for v in views}
    meta["valid_seconds"] = {v: views[v]["n_valid"] for v in views}
    t_ev = d["b_loc"]
    for view, fill in COMBOS:
        V = views[view]
        for x in THRESHOLDS:
            cg = np.flatnonzero(np.abs(np.nan_to_num(V["dgap_grid"])) >= x)
            ce = np.flatnonzero(np.abs(np.nan_to_num(V["dgap_ev"])) >= x)
            sets = {"grid_all": (grid[cg], np.sign(V["dgap_grid"][cg])),
                    "grid_db": None, "event_db": None}
            sg = debounce(grid, cg); se = debounce(t_ev, ce)
            sets["grid_db"] = (grid[sg], np.sign(V["dgap_grid"][sg]))
            sets["event_db"] = (t_ev[se], np.sign(V["dgap_ev"][se]))
            for sname, (tt, dd) in sets.items():
                for lat in LATENCIES_MS:
                    g = trade_gross(d, tt, dd, lat * 1_000_000, fill)
                    g = g[np.isfinite(g)]
                    key = f"{view}|{fill}|{sname}|{x:g}|{lat}"
                    acc.setdefault(key, []).append(
                        (day, int(len(g)), float(g.sum()), float((g * g).sum()),
                         int(len(tt))))
            if view == "recv" and fill == "local":
                dd = detection_delay(t_ev, V["dgap_ev"], V["dgap_grid"], grid, day_start, x)
                # Matched episodes: trade at the onset vs at the grid's first detection.
                for mf in ("local", "engine_asof", "engine_next"):
                    ge = trade_gross(d, dd["t0"], dd["dir"], MATCH_LAT_NS, mf)
                    gg = trade_gross(d, dd["det_t"], dd["dir"], MATCH_LAT_NS, mf)
                    dd[f"g_onset_{mf}"] = ge.astype(np.float32)
                    dd[f"g_grid_{mf}"] = np.where(dd["seen"], gg, np.nan).astype(np.float32)
                for k_ in ("t0", "dir", "det_t"):
                    dd.pop(k_)
                detect.setdefault(f"{x:g}", []).append(dd)
    del views, d, grid
    gc.collect()
    return meta


def summarise_cell(rows: list) -> dict:
    n = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
    ss = np.array([r[3] for r in rows]); sig = np.array([r[4] for r in rows])
    tot = int(n.sum())
    mean = float(s.sum() / tot) if tot else None
    daily = [float(si / ni) if ni else None for si, ni in zip(s, n)]
    sd = float(np.sqrt(max(ss.sum() / tot - mean ** 2, 0.0))) if tot else None
    out = {"days": [r[0] for r in rows], "n_per_day": [int(v) for v in n],
           "signals_per_day": [int(v) for v in sig],
           "mean_n_per_day": float(n.mean()), "mean_gross_bps": mean,
           "trade_sd_bps": sd, "daily_mean_gross_bps": daily}
    dm = np.array([v for v in daily if v is not None])
    out["share_days_gross_pos"] = float(np.mean(dm > 0)) if len(dm) else None
    for f in FEES:
        c = 2 * f + SLIP_ROUND_TRIP
        out[f"net_f{f:g}_bps"] = None if mean is None else mean - c
        out[f"share_days_net_f{f:g}_pos"] = float(np.mean(dm > c)) if len(dm) else None
    out["daily_mean_min"] = float(dm.min()) if len(dm) else None
    out["daily_mean_max"] = float(dm.max()) if len(dm) else None
    return out


def main() -> None:
    acc: dict = {}
    timing: dict = {}
    detect: dict = {}
    day_meta = []
    for day in DAYS:
        print("processing", day, flush=True)
        day_meta.append(process_day(day, acc, timing, detect))
        gc.collect()

    results: dict = {}
    for key, rows in acc.items():
        view, fill, sname, x, lat = key.split("|")
        results.setdefault(f"{view}|{fill}", {}).setdefault(sname, {}).setdefault(x, {})[lat] = \
            summarise_cell(rows)

    pooled_timing = {}
    for k, arrs in timing.items():
        a = np.concatenate(arrs)
        qs = (10, 50, 90, 99) if "interval" in k else (10, 50, 90)
        pooled_timing[k] = pct(a, qs)
        if k == "hl_interval_recv_ms":
            pooled_timing["hl_interval_recv_share_gt_1000ms"] = float(np.mean(a > 1000))
            pooled_timing["hl_interval_recv_mean_ms"] = float(a.mean())
        if k == "binance_recv_minus_exch_ms":
            pooled_timing["binance_recv_minus_exch_p1_ms"] = float(np.percentile(a, 1))
    detection = {}
    for x, lst in detect.items():
        dl = np.concatenate([r["delay_ms"] for r in lst])
        fs = np.concatenate([r["first_slot"] for r in lst])
        ps = np.concatenate([r["persist_ms"] for r in lst])
        seen = np.isfinite(dl)
        matched = {}
        for mf in ("local", "engine_asof", "engine_next"):
            ge = np.concatenate([r[f"g_onset_{mf}"] for r in lst])
            gg = np.concatenate([r[f"g_grid_{mf}"] for r in lst])
            both = np.isfinite(ge) & np.isfinite(gg)
            unseen = np.isfinite(ge) & ~seen
            matched[mf] = {
                "n_seen_pairs": int(both.sum()),
                "mean_gross_onset_trade_seen": float(ge[both].mean()),
                "mean_gross_grid_detection_trade_seen": float(gg[both].mean()),
                "mean_diff_onset_minus_grid": float((ge[both] - gg[both]).mean()),
                "n_unseen_onsets": int(unseen.sum()),
                "mean_gross_onset_trade_unseen": float(ge[unseen].mean()),
            }
        detection[x] = {
            "onsets_per_day": float(len(dl) / len(DAYS)),
            "share_seen_at_first_whole_second_ie_within_1s": float(fs.mean()),
            "share_seen_within_3s": float(np.mean(seen & (dl <= 3000))),
            "share_not_seen_within_10s": float(np.mean(~seen)),
            "delay_ms_given_seen": pct(dl, (10, 25, 50, 75, 90)),
            "mean_delay_ms_given_seen": float(np.nanmean(dl)),
            "event_condition_persistence_ms": pct(ps, (10, 25, 50, 75, 90)),
            "share_persist_lt_1000ms": float(np.mean(ps < 1000)),
            "matched_L150_gross_bps": matched,
        }

    # Decomposition (measurement 4). Per-trade mean gross and per-day net totals for a
    # ladder of improvements, under each fill model. Different rows trade different
    # signal sets, so per-trade differences are not strictly additive.
    def cell(combo, s_, x_, lat):
        c = results[combo][s_][x_][str(lat)]
        m = c["mean_gross_bps"]; n = c["mean_n_per_day"]
        row = {"mean_gross_bps": m, "trades_per_day": n}
        for f in FEES:
            row[f"net_total_per_day_f{f:g}_bps"] = None if m is None else n * (m - 2 * f - SLIP_ROUND_TRIP)
        return row
    decomp = {}
    for x in [f"{v:g}" for v in THRESHOLDS]:
        out = {}
        for fill in ("local", "engine_asof", "engine_next"):
            rc = f"recv|{fill}"
            ladder = {
                "A_grid_recv_L150": cell(rc, "grid_db", x, 150),
                "B_event_recv_L150": cell(rc, "event_db", x, 150),
                "C_event_recv_L100_bookticker_mean": cell(rc, "event_db", x, 100),
                "C2_event_recv_L50_bookticker_max": cell(rc, "event_db", x, 50),
                "D_event_recv_L0": cell(rc, "event_db", x, 0),
            }
            if fill != "local":
                ec = f"engine|{fill}"
                ladder.update({
                    "E_event_engineview_L150_faster_hl": cell(ec, "event_db", x, 150),
                    "F_event_engineview_L100_both": cell(ec, "event_db", x, 100),
                    "G_event_engineview_L0_ideal": cell(ec, "event_db", x, 0),
                    "Ag_grid_engineview_L150": cell(ec, "grid_db", x, 150),
                })
                ideal = ladder["G_event_engineview_L0_ideal"]["mean_gross_bps"]
            else:
                ideal = ladder["D_event_recv_L0"]["mean_gross_bps"]
            m = {k: v["mean_gross_bps"] for k, v in ladder.items()}
            A = m["A_grid_recv_L150"]; lost = ideal - A
            fr = lambda v: None if (v is None or not lost or lost <= 0) else v / lost
            gains = {"lost_edge_ideal_minus_A": lost,
                     "event_trigger_B_minus_A": m["B_event_recv_L150"] - A,
                     "bookticker_mean_C_minus_B": m["C_event_recv_L100_bookticker_mean"] - m["B_event_recv_L150"],
                     "bookticker_max_C2_minus_B": m["C2_event_recv_L50_bookticker_max"] - m["B_event_recv_L150"]}
            if fill != "local":
                gains["faster_hl_E_minus_B"] = m["E_event_engineview_L150_faster_hl"] - m["B_event_recv_L150"]
                gains["faster_hl_on_grid_Ag_minus_A"] = m["Ag_grid_engineview_L150"] - A
            out[fill] = {"ladder": ladder, "gains_per_trade_bps": gains,
                         "fraction_of_lost_edge": {k: fr(v) for k, v in gains.items()
                                                   if k != "lost_edge_ideal_minus_A"}}
        decomp[x] = out

    doc = {
        "definitions": {
            "gap": "(binance_mid - hl_mid)/hl_mid*1e4, as-of on local_ts_ns",
            "basis": f"EMA span {SPAN}s (alpha=2/{SPAN + 1}) of gap at whole seconds < floor(t); "
                     f"restart after data hole > 2 s; warm-up {WARMUP_S} s",
            "trade": "enter HL at t+L book (ask if dgap>0, bid if <0), exit at book at t+L+3s crossing spread",
            "net": "gross - 2*fee - 0.2 (0.1 bps slippage per fill)",
            "sets": {"grid_all": "every whole second with |dgap|>=X (overlapping trades)",
                     "grid_db": "whole seconds, 3 s debounce (non-overlapping)",
                     "event_db": "every Binance update time, 3 s debounce (non-overlapping)"},
            "fills": {"local": "HL book received by t+L (simulator convention)",
                      "engine_asof": "HL book with exchange ts <= t+L (needs clock sync)",
                      "engine_next": "first HL book with exchange ts >= t+L (pessimistic)"},
            "views": {"recv": "signal uses HL book received by t (causal)",
                      "engine": "signal uses HL book with exchange ts <= t (emulated zero-delay HL feed)"},
            "fresh": "a quote or fill book older than 2 s invalidates the decision/trade",
        },
        "days": DAYS, "latencies_ms": LATENCIES_MS, "thresholds_bps": THRESHOLDS,
        "fees_bps": FEES, "day_meta": day_meta, "timing_pooled": pooled_timing,
        "detection_delay": detection, "decomposition_L150": decomp, "results": results,
    }
    (OUT / "subsecond_edge.json").write_text(json.dumps(doc, indent=1))
    print("wrote json", flush=True)
    del acc
    make_figures(doc, timing, detect)


SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRIDC = "#e4e3df"
S1 = "#2a78d6"; S2 = "#eb6834"; S3 = "#1baf7a"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(doc: dict, timing: dict, detect: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = doc["results"]; L = doc["latencies_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False, facecolor=SURFACE)
    for ax, x in zip(axes, ["3", "5", "8"]):
        _style(ax)
        def curve(combo, s):
            return [R[combo][s][x][str(l)]["mean_gross_bps"] for l in L]
        ax.plot(L, curve("recv|local", "grid_all"), color=S1, lw=2, marker="o", ms=4,
                label="1 s grid, every signal second")
        ax.plot(L, curve("recv|local", "grid_db"), color=S1, lw=2, ls="--",
                label="1 s grid, 3 s debounce")
        ax.plot(L, curve("recv|local", "event_db"), color=S2, lw=2, marker="o", ms=4,
                label="every Binance update, 3 s debounce")
        ax.fill_between(L, curve("recv|engine_next", "event_db"), curve("recv|engine_asof", "event_db"),
                        color=S3, alpha=0.15, lw=0,
                        label="  band down to next HL snapshot (pessimistic fill)")
        ax.plot(L, curve("recv|engine_asof", "event_db"), color=S3, lw=2, marker="s", ms=4,
                label="every Binance update, fill at HL book by HL exchange time")
        for f, c in ((1, 2.2), (2, 4.2)):
            ax.axhline(c, color=INK2, lw=1, ls=":",
                       label="break-even gross at fee f bps/side (2f + 0.2)" if f == 1 else None)
            ax.text(1690, c + 0.08, f"f = {f}", va="bottom", ha="right", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("execution latency L (ms after decision)", color=INK, fontsize=10)
        ax.set_title(f"|dgap| ≥ {x} bps", color=INK, fontsize=11, loc="left")
        ax.set_xlim(-30, 1700)
    axes[0].set_ylabel("mean gross edge per trade (bps)", color=INK, fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.suptitle("HL 3 s round-trip gross edge vs latency, 6 training days (Sep 2026)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(OUT / "edge_vs_latency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURFACE)
    ax = axes[0]; _style(ax)
    bins = np.arange(0, 1001, 10)
    b = np.concatenate(timing["binance_recv_minus_exch_ms"]); hh = np.concatenate(timing["hl_recv_minus_exch_ms"])
    ax.hist(np.clip(b, 0, 1000), bins=bins, density=True, color=S1, alpha=0.85, label="Binance depth20@100ms")
    ax.hist(np.clip(hh, 0, 1000), bins=bins, density=True, color=S2, alpha=0.85, label="Hyperliquid l2Book")
    ax.set_xlabel("receive time − exchange timestamp (ms)", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.set_title("Feed delay as recorded (clock offsets included)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    del b, hh
    ax = axes[1]; _style(ax)
    hi = np.concatenate(timing["hl_interval_recv_ms"])
    ax.hist(np.clip(hi, 0, 2000), bins=np.arange(0, 2001, 20), density=True, color=S2)
    ax.set_xlabel("time between HL book updates, receive clock (ms)", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.set_title("HL update interval", color=INK, fontsize=10, loc="left")
    del hi
    ax = axes[2]; _style(ax)
    for x, c in (("3", S1), ("5", S2), ("8", S3)):
        dl = np.concatenate([r["delay_ms"] for r in detect[x]])
        n = len(dl)
        dls = np.sort(np.where(np.isfinite(dl), dl, np.inf))
        xs = np.arange(0, 3001, 10)
        ys = np.searchsorted(dls, xs, side="right") / n
        ax.plot(xs, ys, color=c, lw=2, label=f"X = {x} bps")
    ax.set_xlabel("onset to first whole-second detection (ms)", color=INK)
    ax.set_ylabel("cumulative share of onsets", color=INK)
    ax.set_ylim(0, 1)
    ax.set_title("1 s grid detection delay after onset", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "timing.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote figures", flush=True)


if __name__ == "__main__":
    main()
