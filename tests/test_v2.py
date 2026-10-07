"""Version 2 checks: causal features, simulator equivalence, reward accounting, split guard."""
from pathlib import Path

import numpy as np
import pytest

from latency_arb.data.schema import load_manifest_entry, read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.evaluate import rollout
from latency_arb.v2.policies import SimAdapter, make_decider
from latency_arb.v2.replay import EpisodeRunner, FastLeadLagEnv
from latency_arb.v2.table import _ema_previous, _lag, _rolling_std_previous, build_table

MANIFEST = Path("data/v2-recorded-30d/manifest.json")
needs_data = pytest.mark.skipif(not MANIFEST.exists(), reason="prepared v2 dataset not present")


def _small_table(n=8, day="2026-09-10"):
    _, entries = read_manifest(MANIFEST, split="train")
    ids = sorted((e for e in entries if e["day"] == day), key=lambda e: e["start_timestamp_ns"])
    ids = {e["episode_id"] for e in ids[100:100 + n]}
    return build_table(MANIFEST, "train", episode_ids=ids)


def test_ema_excludes_current_and_future_values():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    seg = np.zeros(200, int)
    seg[120:] = 1
    base = _ema_previous(x, seg, 60)
    for i in (5, 50, 119, 121, 199):
        y = x.copy()
        y[i:] += 100.0
        changed = _ema_previous(y, seg, 60)
        assert np.allclose(changed[:i + 1], base[:i + 1])
    assert base[120] == x[120]  # restart at a segment boundary uses no earlier segment


def test_lag_and_volatility_do_not_cross_segments_or_look_ahead():
    x = np.arange(20, dtype=float)
    age = np.r_[np.arange(10), np.arange(10)]
    lagged = _lag(x, age, 1)
    assert np.isnan(lagged[0]) and np.isnan(lagged[10]) and lagged[11] == 10
    r = np.random.default_rng(1).normal(size=50)
    age = np.arange(50)
    v = _rolling_std_previous(r, age, 30)
    r2 = r.copy()
    r2[40:] += 50
    assert np.allclose(_rolling_std_previous(r2, age, 30)[:40], v[:40])


@needs_data
def test_table_prefix_invariance():
    full = _small_table(8)
    _, entries = read_manifest(MANIFEST, split="train")
    kept = [m["episode_id"] for m in full["meta"]["episodes"][:5]]
    prefix = build_table(MANIFEST, "train", episode_ids=set(kept))
    n = len(prefix["k"])
    for key in ("gap", "basis", "dgap", "bret1", "bret5", "hret1", "hret5", "vol30", "seg_age"):
        assert np.allclose(prefix[key], full[key][:n], equal_nan=True), key


@needs_data
@pytest.mark.parametrize("spec,fee,latency,j", [
    ({"kind": "dgap_rule", "X": 2.5, "H": 2}, 1.0, 150, 1),
    ({"kind": "dgap_rule", "X": 2.0, "H": 45}, 3.5, 450, 3),
    ({"kind": "raw_rule", "X": 2.0}, 0.5, 300, 2),
    ({"kind": "raw_rule", "X": 1.0}, 2.0, 600, 4),
])
def test_fast_replay_matches_simulator(spec, fee, latency, j):
    T = _small_table(8)
    d = make_decider(spec, T, fee)
    _, entries = read_manifest(MANIFEST, split="train")
    by_id = {e["episode_id"]: e for e in entries}
    for e, meta in enumerate(T["meta"]["episodes"]):
        r = EpisodeRunner(T, e, fee, j, d.candidates, d.direction)
        while (nd := r.next_decision()) is not None:
            g, kind = nd
            r.apply(g, kind, d.entry(g) if kind == "entry" else d.exit(g, r.k, r.pos))
        episode = load_manifest_entry(MANIFEST, by_id[meta["episode_id"]])
        sim = rollout(SimAdapter(d, T, e, latency), episode,
                      EnvConfig(fee_bps=fee, latency_ms=latency, decision_interval_ms=1000))
        fast = [t["net_pnl"] for t in r.trades]
        slow = [t["net_pnl"] for t in sim["trades"]]
        assert len(fast) == len(slow)
        assert np.allclose(fast, slow, rtol=0, atol=1e-9)


@needs_data
def test_training_rewards_telescope_to_net_pnl():
    T = _small_table(8)
    env = FastLeadLagEnv(T, fees=(1.0,), j=1, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(5):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, done, _, _ = env.step(int(rng.integers(2)))
            total += reward
        net = sum(t["net_pnl"] for t in env.runner.trades)
        assert np.isclose(total / env.scale, net, atol=1e-9)


@needs_data
def test_training_refuses_non_training_split(tmp_path):
    from latency_arb.v2 import train as train_module
    from latency_arb.v2.table import save_table
    T = build_table(MANIFEST, "validation", episode_ids={read_manifest(MANIFEST, "validation")[1][0]["episode_id"]})
    path = tmp_path / "val.npz"
    save_table(T, path)
    with pytest.raises(ValueError):
        train_module.train(str(path), str(tmp_path / "out"), seed=0, steps=10)


def test_vwap_matches_simulator_on_cancellation_prone_book():
    from latency_arb.v2.table import TableConfig, _vwap
    prices = np.array([[76001.0, 76000.0, 75999.0, 75998.0, 75997.0]])
    sizes = np.array([[7.3e-4, 66.69684, 1.5e-4, 0.85816, 4.1e-4]])
    price, ok, forced = _vwap(prices, sizes, np.array([76001.5]), 0.001, -1, TableConfig())
    expected = (0.00073 * 76001.0 + 0.00027 * 76000.0) / 0.001 * (1 - 0.1 / 1e4)
    assert ok[0] and np.isclose(price[0], expected, rtol=0, atol=1e-9) and np.isclose(forced[0], price[0])


def test_forced_price_penalises_only_the_residual():
    from latency_arb.v2.table import TableConfig, _vwap
    prices = np.array([[100.0, 101.0, 102.0, 103.0, 104.0]])
    sizes = np.array([[2e-4, 2e-4, 2e-4, 2e-4, 1e-4]])
    price, ok, forced = _vwap(prices, sizes, np.array([99.5]), 0.001, 1, TableConfig())
    assert not ok[0] and np.isnan(price[0])
    notional = 2e-4 * (100 + 101 + 102 + 103) + 1e-4 * 104 + 1e-4 * 104 * (1 + 25 / 1e4)
    assert np.isclose(forced[0], notional / 0.001 * (1 + 0.1 / 1e4))


@needs_data
def test_sim_adapter_refuses_sub_second_cadence():
    T = _small_table(2)
    d = make_decider({"kind": "raw_rule", "X": 0.0}, T, 1.0)
    _, entries = read_manifest(MANIFEST, split="train")
    by_id = {e["episode_id"]: e for e in entries}
    episode = load_manifest_entry(MANIFEST, by_id[T["meta"]["episodes"][0]["episode_id"]])
    with pytest.raises(ValueError):
        rollout(SimAdapter(d, T, 0, 150), episode, EnvConfig(fee_bps=1.0, decision_interval_ms=0))
