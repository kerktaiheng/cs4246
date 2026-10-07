"""@file export_policy.py
@brief Export a version 2 PPO actor for a C++ engine: ONNX graph, raw weights, feature spec.
@details The actor is the SB3 MlpPolicy action network: obs(20) -> tanh(64) -> tanh(64) ->
logits(2). The deterministic action is argmax(logits). Observations must be built exactly as
latency_arb.v2.replay.flat_features / position_features (fixed scales, clip to [-10, 10]).
A NumPy forward pass is checked against SB3's own predictions before anything is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from latency_arb.v2.replay import OBS_NAMES, ScreenConfig, _SCALE


class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.body = policy.mlp_extractor.policy_net
        self.head = policy.action_net

    def forward(self, obs):
        return self.head(self.body(obs))


def numpy_forward(weights: dict, obs: np.ndarray) -> np.ndarray:
    h = np.tanh(obs @ weights["w0"].T + weights["b0"])
    h = np.tanh(h @ weights["w1"].T + weights["b1"])
    return h @ weights["w2"].T + weights["b2"]


def export(model_path: str, out_dir: str, check_obs: np.ndarray) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = PPO.load(model_path, device="cpu")
    actor = Actor(model.policy).eval()
    layers = [m for m in actor.body if isinstance(m, torch.nn.Linear)] + [actor.head]
    if [type(m).__name__ for m in actor.body] != ["Linear", "Tanh", "Linear", "Tanh"]:
        raise ValueError("unexpected policy network layout")
    weights = {}
    for i, layer in enumerate(layers):
        weights[f"w{i}"] = layer.weight.detach().numpy().astype(np.float32)
        weights[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    np.savez(out / "actor_weights.npz", **weights)
    sb3_actions, _ = model.predict(check_obs, deterministic=True)
    np_actions = numpy_forward(weights, check_obs).argmax(axis=1)
    mismatches = int((np.asarray(sb3_actions).reshape(-1) != np_actions).sum())
    if mismatches:
        raise AssertionError(f"NumPy actor disagrees with SB3 on {mismatches} observations")
    onnx_status = "written"
    try:
        torch.onnx.export(actor, torch.zeros(1, len(OBS_NAMES)), str(out / "actor.onnx"),
                          input_names=["obs"], output_names=["logits"],
                          dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
    except Exception as error:  # ONNX export needs optional packages in some torch builds
        onnx_status = f"not written: {type(error).__name__}: {error}"
    spec = {"source_checkpoint": str(model_path), "observation_names": OBS_NAMES,
            "observation_scale_divisor": _SCALE.tolist(), "observation_clip": [-10, 10],
            "network": "obs(20) -> Linear(64) -> tanh -> Linear(64) -> tanh -> Linear(2) = logits",
            "action": "argmax(logits); flat: 0 skip, 1 enter in sign(gap - basis); holding: 0 hold, 1 exit",
            "screen": ScreenConfig().__dict__, "basis": "EMA span 60 s of the one-second mid gap, excluding the current second",
            "parity_check": {"observations": int(len(check_obs)), "argmax_mismatches_vs_sb3": mismatches},
            "onnx": onnx_status}
    (out / "actor_spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--table", default="data/v2-tables/validation.npz")
    a = p.parse_args()
    from latency_arb.v2.replay import candidate_mask, flat_features
    from latency_arb.v2.table import load_table
    T = load_table(a.table)
    idx = np.flatnonzero(candidate_mask(T, ScreenConfig()))[:20000]
    rng = np.random.default_rng(0)
    obs = np.concatenate([flat_features(T, idx, np.where(T["dgap"][idx] >= 0, 1, -1), f)
                          for f in (0.5, 1.0, 2.0, 3.5)])
    pos = obs[rng.choice(len(obs), 5000, replace=False)].copy()
    pos[:, 0] = 1.0
    pos[:, 15] = rng.uniform(0, 1, len(pos))
    pos[:, 16] = rng.normal(0, 0.5, len(pos))
    pos[:, 17] = pos[:, 1]
    print(json.dumps(export(a.model, a.out, np.concatenate([obs, pos]).astype(np.float32)), indent=1))


if __name__ == "__main__":
    main()
