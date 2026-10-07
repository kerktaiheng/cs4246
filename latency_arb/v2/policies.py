"""@file policies.py
@brief Version 2 policies as (screen, direction, entry rule, exit rule) and a simulator adapter.
@details Every policy only reads table rows at or before its decision second. The same
decision functions drive the fast replay (selection) and the unchanged LatencySimEnv
(reported results) through `SimAdapter`, which reconstructs the decision second and the
open position from the simulator's own observation vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from latency_arb.env.latency_sim import OBSERVATION_NAMES, Action
from latency_arb.v2.replay import (EPISODE_SECONDS, ScreenConfig, candidate_mask, flat_features,
                                   position_features)

_OBS = {name: i for i, name in enumerate(OBSERVATION_NAMES)}


@lru_cache(maxsize=8)
def _load_ppo(path: str):
    """@brief Load a PPO checkpoint and return a deterministic action function.
    @details Calls the policy's own torch modules (mlp_extractor.policy_net, action_net)
    and takes argmax of the logits, which is what SB3's deterministic predict computes for
    a Discrete action space, without SB3's per-call preprocessing overhead.
    """
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    policy = PPO.load(path, device="cpu").policy.eval()

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
            logits = policy.action_net(policy.mlp_extractor.policy_net(policy.extract_features(x, policy.pi_features_extractor)))
        return logits.argmax(dim=1).numpy()
    return act


@lru_cache(maxsize=8)
def _load_dqn(path: str):
    """@brief Greedy action from a saved (Double) DQN: argmax of the online Q-network."""
    import torch
    from stable_baselines3 import DQN
    torch.set_num_threads(1)
    q_net = DQN.load(path, device="cpu", custom_objects={"learning_rate": 1e-4}).q_net.eval()

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return q_net(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)).argmax(dim=1).numpy()
    return act


@lru_cache(maxsize=8)
def _load_net(path: str, outputs: int):
    import torch
    from latency_arb.v2.fqi import mlp
    torch.set_num_threads(1)
    net = mlp(outputs)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def q(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)).numpy()
    return q


@dataclass
class Decider:
    """@brief Policy decision functions bound to one table and one fee level."""

    candidates: np.ndarray
    direction: np.ndarray
    entry: object
    exit: object


def make_decider(spec: dict, T: dict, fee: float) -> Decider:
    """@brief Build decision functions for a policy spec.
    @param spec {"kind": "flat"} | {"kind": "raw_rule", "X": bps} |
        {"kind": "dgap_rule", "X": bps, "H": seconds} | {"kind": "ppo", "path": zip}.
    """
    kind = spec["kind"]
    fresh = T["fresh"][:, 0]
    if kind == "flat":
        none = np.zeros(len(T["k"]), bool)
        return Decider(none, np.ones(len(none), np.int8), lambda g: False, lambda g, k, p: False)
    if kind == "raw_rule":
        # @details Version 1's fixed-threshold control: enter in the raw gap direction,
        # exit once the directional gap has converged to 0.5 bps (or by forced exits).
        X = float(spec["X"])
        cand = (np.abs(T["gap"]) >= X) & fresh
        dirn = np.where(T["gap"] >= 0, 1, -1).astype(np.int8)
        gap = T["gap"]
        return Decider(cand, dirn, lambda g: True, lambda g, k, p: p["dir"] * gap[g] <= 0.5)
    screen = ScreenConfig(**spec.get("screen", {}))
    cand = candidate_mask(T, screen)
    dirn = np.where(T["dgap"] >= 0, 1, -1).astype(np.int8)
    if kind == "dgap_rule":
        # @details Same de-meaned signal and screen as the agent, fixed threshold and hold.
        X, H = float(spec["X"]), int(spec["H"])
        dgap = T["dgap"]
        return Decider(cand, dirn, lambda g: abs(dgap[g]) >= X, lambda g, k, p: k - p["k"] >= H)
    if kind == "dgap_conv_rule":
        # @details Amendment 2: de-meaned entry plus a convergence exit (directional
        # de-meaned gap back to E bps) or a holding cap of H seconds.
        X, E, H = float(spec["X"]), float(spec["E"]), int(spec["H"])
        dgap = T["dgap"]
        return Decider(cand, dirn, lambda g: abs(dgap[g]) >= X,
                       lambda g, k, p: p["dir"] * dgap[g] <= E or k - p["k"] >= H)
    if kind in ("fqi", "bandit"):
        # @details Amendment 5: enter when the fitted entry value is positive; exit by the
        # fitted stopping rule (fqi) or after a fixed 3 s hold (bandit ablation).
        entry_q = _load_net(str(Path(spec["dir"]) / "entry_q.pt"), 1)
        idx = np.flatnonzero(cand)
        enter = np.zeros(len(cand), bool)
        if len(idx):
            enter[idx] = entry_q(flat_features(T, idx, dirn[idx], fee))[:, 0] > 0
        if kind == "bandit":
            return Decider(cand, dirn, lambda g: bool(enter[g]), lambda g, k, p: k - p["k"] >= 3)
        exit_q = _load_net(str(Path(spec["dir"]) / "exit_q.pt"), 2)

        def stop(g, k, p):
            q = exit_q(position_features(T, g, p["dir"], fee, k - p["k"], p["entry_price"], p["entry_dgap"]))[0]
            return bool(q[1] >= q[0])
        return Decider(cand, dirn, lambda g: bool(enter[g]), stop)
    if kind in ("ppo", "dqn"):
        act = _load_ppo(spec["path"]) if kind == "ppo" else _load_dqn(spec["path"])
        idx = np.flatnonzero(cand)
        enter = np.zeros(len(cand), bool)
        if len(idx):
            obs = flat_features(T, idx, dirn[idx], fee)
            enter[idx] = act(obs) == 1

        def exit_fn(g, k, p):
            obs = position_features(T, g, p["dir"], fee, k - p["k"], p["entry_price"], p["entry_dgap"])
            return int(act(obs)[0]) == 1
        return Decider(cand, dirn, lambda g: bool(enter[g]), exit_fn)
    raise ValueError(f"unknown policy kind {kind}")


class SimAdapter:
    """@brief SB3-style predict() for LatencySimEnv, bound to one table window.
    @details The decision second comes from time_remaining_ms; the entry decision
    second and entry price come from holding_time_ms and unrealized_pnl_usd. The
    adapter never reads simulator internals or future rows.
    """

    def __init__(self, decider: Decider, T: dict, e: int, latency_ms: float):
        self.d, self.T, self.e = decider, T, e
        self.g0 = e * EPISODE_SECONDS
        meta = T["meta"]["episodes"][e]
        t0 = int(T["t_ns"][self.g0])
        self.span_ms = (int(meta["last_timestamp_ns"]) - t0) / 1e6
        self.latency_ms = latency_ms

    def predict(self, observation, deterministic: bool = True):
        o = np.asarray(observation, dtype=np.float64)
        elapsed_ms = self.span_ms - o[_OBS["time_remaining_ms"]]
        k = int(round(elapsed_ms / 1000.0))
        # @details Only whole-second decision rows may consult the table; any other
        # row would map to a later second and leak future quotes.
        if abs(elapsed_ms - 1000.0 * k) > 0.5:
            raise ValueError("SimAdapter requires decisions on whole seconds (decision_interval_ms=1000)")
        if not 0 <= k < EPISODE_SECONDS or o[_OBS["pending_action"]] >= 0:
            return int(Action.HOLD), None
        g = self.g0 + k
        inventory = o[_OBS["inventory"]]
        if inventory == 0:
            if self.d.candidates[g] and self.d.entry(g):
                return int(Action.LONG if self.d.direction[g] > 0 else Action.SHORT), None
            return int(Action.HOLD), None
        direction = 1 if inventory > 0 else -1
        holding_ms = o[_OBS["holding_time_ms"]]
        k_entry = int(round(k - (holding_ms + self.latency_ms) / 1000.0))
        mid = o[_OBS["hl_mid_price"]]
        entry_price = mid - o[_OBS["unrealized_pnl_usd"]] / inventory
        pos = {"k": k_entry, "dir": direction, "entry_price": float(entry_price),
               "entry_dgap": float(self.T["dgap"][self.g0 + max(k_entry, 0)])}
        return int(Action.EXIT if self.d.exit(g, k, pos) else Action.HOLD), None
