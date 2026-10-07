"""@file fqi.py
@brief Batch fitted-Q iteration that exploits exogenous market dynamics (amendment 5).
@details Our 0.001 BTC orders do not change the replayed market, so from every training
decision second the outcome of every action can be read from the table. This removes the
exploration problem that on-policy and off-policy deep RL face:

- Exit (optimal stopping). While holding, the actions are hold or exit. Q(s, exit) is the
  expected net payoff of exiting now (regressed on the decision-time features; the actual
  fill price is an outcome, never an input). Q(s, hold) is fitted by iterating
  Q(s_h, hold) <- max_a Q(s_{h+1}, a), with the forced exit 30 s after entry (or the
  window end) as the terminal payoff. One MLP with two heads.
- Entry. Q(s, enter) is the expected net payoff of a trade that is then managed by the
  learned exit, regressed on the flat features (fee included); Q(s, skip) = 0. The agent
  enters when Q(s, enter) > 0. This is greedy in the entry decision: it ignores the option
  of waiting for a better entry, and the opportunity cost of being busy while holding.
- Contextual bandit (ablation). Same entry regression but every trade exits after a fixed
  3 s hold (the training-day analysis horizon), so no sequential decision is learned.

Payoffs are in basis points of entry notional and use the same fill arithmetic as the
simulator (via the v2 table). Training uses the training split only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from latency_arb.v2.replay import MAX_HOLD_S, ScreenConfig, _SCALE, candidate_mask, flat_features
from latency_arb.v2.table import load_table
from latency_arb.v2.train import TRAIN_FEES

EPISODE_SECONDS = 300


def mlp(outputs: int) -> nn.Module:
    return nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))


def position_batch(T: dict, g: np.ndarray, d: np.ndarray, fee: np.ndarray, hold: np.ndarray,
                   entry_price: np.ndarray, entry_dgap: np.ndarray) -> np.ndarray:
    """@brief Vectorised replay.position_features for many rows with per-row fees."""
    obs = np.empty((len(g), 20), np.float32)
    for f in np.unique(fee):
        m = fee == f
        obs[m] = flat_features(T, g[m], d[m], float(f))
    raw = obs.astype(np.float64) * _SCALE
    exit_side = np.where(d > 0, T["h_bid"][g], T["h_ask"][g])
    raw[:, 0] = 1.0
    raw[:, 15] = hold
    raw[:, 16] = d * (exit_side - entry_price) / entry_price * 1e4
    raw[:, 17] = d * entry_dgap
    return np.clip(raw / _SCALE, -10, 10).astype(np.float32)


def build_paths(T: dict, j: int, fees: np.ndarray, rng: np.random.Generator, per_candidate: int = 2) -> dict:
    """@brief Every admissible training entry, its holding path and exact payoffs (bps).
    @details Mirrors replay.EpisodeRunner: entry fill at (g, j) if fresh and before the
    terminal reserve; decisions while holding at seconds h = 1..29 after the entry decision;
    a voluntary exit fills at (g + h, j) if displayed depth suffices; the forced exit is at
    h = 30 (row g + 30, offset j) or at the window's last row if that comes first.
    """
    screen = ScreenConfig()
    cand = candidate_mask(T, screen)
    latency_ns = int(T["meta"]["offsets_ms"][j]) * 1_000_000
    last_t = np.array([e["last_timestamp_ns"] for e in T["meta"]["episodes"]], np.int64)[T["episode"]]
    k = T["k"].astype(int)
    d_all = np.where(T["dgap"] >= 0, 1, -1)
    entry_px_all = np.where(d_all > 0, T["buy"][:, j], T["sell"][:, j])
    ok = (cand & (T["t_ns"] + latency_ns < last_t - latency_ns) & T["fresh"][:, j]
          & (T["fill_t"][:, j] < last_t - latency_ns) & np.isfinite(entry_px_all))
    g0 = np.repeat(np.flatnonzero(ok), per_candidate)
    fee = rng.choice(fees, size=len(g0)).astype(np.float64)
    d = d_all[g0]
    pe = entry_px_all[g0]
    ke = k[g0]
    H = MAX_HOLD_S - 1
    hs = np.arange(1, H + 1)
    G = g0[:, None] + hs[None, :]
    valid = (ke[:, None] + hs[None, :]) <= EPISODE_SECONDS - 1
    G = np.where(valid, G, g0[:, None])
    # Voluntary exit payoff at each decision second (NaN where depth is insufficient).
    px = np.where(d[:, None] > 0, T["sell"][G, j], T["buy"][G, j])
    terminal_row = (T["episode"][g0] * EPISODE_SECONDS + EPISODE_SECONDS - 1)
    last_j = len(T["meta"]["offsets_ms"]) - 1
    at_end = T["fill_t"][G, j] >= last_t[g0][:, None]
    px_terminal = np.where(d > 0, T["sell_forced"][terminal_row, last_j], T["buy_forced"][terminal_row, last_j])
    px = np.where(at_end, px_terminal[:, None], px)

    def payoff(exit_px):
        return (d[:, None] * (exit_px - pe[:, None]) / pe[:, None] * 1e4
                - fee[:, None] - fee[:, None] * exit_px / pe[:, None])

    X = payoff(px)
    # Forced exit: 30 s after entry if inside the window, else the window's last row.
    g30 = g0 + MAX_HOLD_S
    inside = (ke + MAX_HOLD_S <= EPISODE_SECONDS - 1)
    g30 = np.where(inside, g30, terminal_row)
    forced_px = np.where(d > 0, T["sell_forced"][g30, j], T["buy_forced"][g30, j])
    late = inside & (T["fill_t"][np.where(inside, g0 + MAX_HOLD_S, g0), j] >= last_t[g0])
    forced_px = np.where(~inside | late, px_terminal, forced_px)
    F = payoff(forced_px[:, None])[:, 0]
    n_dec = valid.sum(axis=1)
    rows = np.flatnonzero(valid.reshape(-1))
    feats = position_batch(T, G.reshape(-1)[rows], np.repeat(d, H)[rows], np.repeat(fee, H)[rows],
                           np.tile(hs, len(g0))[rows], np.repeat(pe, H)[rows],
                           np.repeat(T["dgap"][g0], H)[rows])
    S = np.zeros((len(g0), H, 20), np.float32)
    S.reshape(-1, 20)[rows] = feats
    return {"g": g0, "d": d, "fee": fee, "pe": pe, "S": S, "X": X.astype(np.float32),
            "F": F.astype(np.float32), "n_dec": n_dec, "valid": valid,
            "flat": np.concatenate([flat_features(T, g0[fee == f], d[fee == f], float(f)) for f in np.unique(fee)]),
            "flat_order": np.concatenate([np.flatnonzero(fee == f) for f in np.unique(fee)])}


def _fit(net, x, y, mask=None, epochs=1, lr=1e-3, batch=4096, seed=0, opt=None):
    torch.manual_seed(seed)
    opt = opt or torch.optim.Adam(net.parameters(), lr=lr)
    n = len(x)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    mt = None if mask is None else torch.as_tensor(mask)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net(xt[idx])
            err = nn.functional.smooth_l1_loss(pred, yt[idx], reduction="none", beta=2.0)
            if mt is not None:
                err = err * mt[idx]
            loss = err.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return opt


def train_fqi(P: dict, seed: int, iterations: int = 15) -> tuple[nn.Module, dict]:
    """@brief Two-head exit network: column 0 = Q(s, hold), column 1 = Q(s, exit)."""
    torch.manual_seed(seed)
    net = mlp(2)
    S, X, F, valid, n_dec = P["S"], P["X"], P["F"], P["valid"], P["n_dec"]
    N, H, _ = S.shape
    flatS = S.reshape(-1, 20)
    rows = np.flatnonzero(valid.reshape(-1))
    x = flatS[rows]
    exit_ok = np.isfinite(X).reshape(-1)[rows]
    Xc = np.nan_to_num(X, nan=0.0).reshape(-1)[rows]
    opt, history = None, []
    for it in range(iterations):
        with torch.no_grad():
            q = net(torch.as_tensor(flatS)).numpy().reshape(N, H, 2)
        # Value of the next decision second: max over actions; exit only where fillable.
        v = np.where(np.isfinite(X), np.maximum(q[..., 0], q[..., 1]), q[..., 0])
        nxt = np.empty((N, H), np.float32)
        nxt[:, :-1] = v[:, 1:]
        last = n_dec - 1
        nxt[np.arange(N), last] = F  # holding at the last decision second leads to the forced exit
        hold_target = nxt.reshape(-1)[rows]
        y = np.stack([hold_target, Xc], axis=1).astype(np.float32)
        m = np.stack([np.ones_like(hold_target), exit_ok.astype(np.float32)], axis=1)
        opt = _fit(net, x, y, m, epochs=1, seed=seed * 100 + it, opt=opt)
        exit_rate = float((q[..., 1] >= q[..., 0])[valid & np.isfinite(X)].mean())
        history.append({"iteration": it, "share_exit_preferred": exit_rate,
                        "mean_v_first": float(v[:, 0].mean())})
    return net, {"history": history}


def realised(P: dict, exit_policy) -> np.ndarray:
    """@brief Net payoff (bps) of each path when exits follow `exit_policy` (bool array N x H)."""
    X, F, valid = P["X"], P["F"], P["valid"]
    choose = exit_policy & valid & np.isfinite(X)
    first = np.where(choose.any(axis=1), choose.argmax(axis=1), -1)
    out = F.copy()
    has = first >= 0
    out[has] = X[np.flatnonzero(has), first[has]]
    return out


def train_entry(P: dict, payoff: np.ndarray, seed: int, epochs: int = 8) -> nn.Module:
    net = mlp(1)
    order = P["flat_order"]
    y = payoff[order].reshape(-1, 1).astype(np.float32)
    _fit(net, P["flat"], y, epochs=epochs, seed=seed)
    return net


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default="data/v2-tables/train.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--latency-index", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(2)
    start = time.time()
    T = load_table(a.table)
    if T["meta"]["split"] != "train":
        raise ValueError("training split only")
    rng = np.random.default_rng(a.seed)
    P = build_paths(T, a.latency_index, np.array(TRAIN_FEES), rng)
    del T
    out = Path(a.out)
    for algo in ("fqi", "bandit"):
        (out / algo).mkdir(parents=True, exist_ok=False)
    exit_net, info = train_fqi(P, a.seed)
    with torch.no_grad():
        q = exit_net(torch.as_tensor(P["S"].reshape(-1, 20))).numpy().reshape(P["S"].shape[0], -1, 2)
    fqi_payoff = realised(P, q[..., 1] >= q[..., 0])
    hold3 = np.zeros_like(P["valid"])
    hold3[:, 2:] = True  # exit at the first fillable second with h >= 3
    bandit_payoff = realised(P, hold3)
    fqi_entry = train_entry(P, fqi_payoff, a.seed)
    bandit_entry = train_entry(P, bandit_payoff, a.seed)
    meta = {"seed": a.seed, "latency_ms": None, "paths": int(len(P["g"])), "fees": list(TRAIN_FEES),
            "fqi_iterations": info["history"], "elapsed_s": None,
            "mean_payoff_bps": {"fqi_exit_all_entries": float(fqi_payoff.mean()),
                                "fixed_3s_all_entries": float(bandit_payoff.mean())}}
    torch.save(exit_net.state_dict(), out / "fqi" / "exit_q.pt")
    torch.save(fqi_entry.state_dict(), out / "fqi" / "entry_q.pt")
    torch.save(bandit_entry.state_dict(), out / "bandit" / "entry_q.pt")
    meta["elapsed_s"] = round(time.time() - start, 1)
    for algo in ("fqi", "bandit"):
        (out / algo / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("paths", "mean_payoff_bps", "elapsed_s")}))


if __name__ == "__main__":
    main()
