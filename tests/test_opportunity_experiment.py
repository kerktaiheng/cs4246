"""@file test_opportunity_experiment.py
@brief Protect the fresh-test boundary, economic selection, and saved failures.
@details These orchestration tests use deterministic training/evaluation doubles.
Real PPO fits and economic replay are exercised by their dedicated integration
tests. The doubles make forbidden fresh-data access observable in every branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import latency_arb.opportunity_experiment as experiment


@pytest.fixture
def controlled_study(tmp_path, monkeypatch):
    """@brief Install deterministic stage doubles without opening market archives."""
    manifest = tmp_path / "old.json"
    fresh = tmp_path / "fresh.json"
    manifest.write_text("{}")
    fresh.write_text("{}")
    output = tmp_path / "run"
    calls = []
    pnl = {"warmstart": 0.1, "ppo": 0.2}

    def read(path, split=None):
        """@brief Supply chronology while leaving every simulated archive unopened."""
        if Path(path) == fresh:
            entries = [{"day": "2026-10-03", "split": "test"}]
        else:
            entries = [
                {"day": "2026-09-03", "split": "train"},
                {"day": "2026-09-25", "split": "validation"},
                {"day": "2026-10-02", "split": "test"},
            ]
        return {"episodes": entries}, [e for e in entries if split is None or e["split"] == split]

    def prepare(path, directory, **kwargs):
        """@brief Record train preparation while returning only required metadata."""
        assert Path(path) == manifest
        calls.append("prepare")
        return {"directory": str(directory), "event_count": 10}

    def train(path, events, directory, **kwargs):
        """@brief Materialize distinct checkpoint bytes for actual selection hashes."""
        assert Path(path) == manifest
        result = {}
        for stage in ("warmstart", "ppo"):
            destination = directory / stage
            destination.mkdir(parents=True)
            (destination / "policy.zip").write_bytes(stage.encode())
            result[f"{stage}_dir"] = str(destination)
        calls.append("train")
        return result

    def evaluate(path, split, policies, execution, opportunity, output_dir=None):
        """@brief Enforce frozen selection before any fresh outcome is available."""
        if Path(path) == fresh:
            assert split == "test"
            frozen = output / "selection.json"
            assert frozen.exists()
            selection = json.loads(frozen.read_text())
            assert selection["selected"]["stage"] == "ppo"
            calls.append(("fresh", frozen.read_bytes()))
        else:
            assert Path(path) == manifest
            assert split == "validation"
            calls.append("validation")
        results = {}
        for name in policies:
            if name == "flat":
                value = 0.0
            elif name.startswith("threshold_"):
                value = 0.05
            elif name.endswith("_warmstart"):
                value = pnl["warmstart"]
            elif name.endswith("_ppo"):
                value = pnl["ppo"]
            else:
                value = 0.12
            results[name] = {"aggregate": {"net_pnl": value,
                                          "trade_count": int(value != 0)},
                             "episodes": []}
        return {"policies": results}

    monkeypatch.setattr(experiment, "read_manifest", read)
    monkeypatch.setattr(experiment, "prepare_training_events", prepare)
    monkeypatch.setattr(experiment, "train_opportunity_policy", train)
    monkeypatch.setattr(experiment, "load_opportunity_policy", lambda path: object())
    monkeypatch.setattr(experiment, "evaluate_opportunity_policies", evaluate)
    monkeypatch.setattr(experiment, "source_fingerprint", lambda: {"stub": "source"})
    return manifest, fresh, output, calls, pnl


def test_positive_selection_is_frozen_before_fresh_test(controlled_study):
    """@brief A profitable development model may consume fresh data only once frozen."""
    manifest, fresh, output, calls, _ = controlled_study
    result = experiment.run_opportunity_experiment(
        manifest, fresh, output, seeds=(7,), steps=1, warm_epochs=1,
    )
    access = [item for item in calls if isinstance(item, tuple)]
    assert len(access) == 1
    assert access[0][1] == (output / "selection.json").read_bytes()
    assert result["fresh_test_evaluated"] is True
    assert result["selection"]["selected"]["name"] == "seed7_ppo"
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert result["protocol"]["execution"]["fee_bps"] == 4.5


@pytest.mark.parametrize("warm,ppo", [(0.0, 0.0), (-0.1, -0.2)])
def test_nonpositive_development_preserves_unused_fresh_data(controlled_study, warm, ppo):
    """@brief Both abstention and losing trading fail the economic test-access gate."""
    manifest, fresh, output, calls, pnl = controlled_study
    pnl.update(warmstart=warm, ppo=ppo)
    result = experiment.run_opportunity_experiment(
        manifest, fresh, output, seeds=(7,), steps=1, warm_epochs=1,
    )
    assert not any(isinstance(item, tuple) for item in calls)
    assert result["fresh_test_evaluated"] is False
    assert result["fresh_test"] is None
    assert not (output / "fresh_test").exists()


def test_existing_artifacts_are_preserved(controlled_study):
    """@brief A repeated invocation cannot overwrite previous learning or evidence."""
    manifest, fresh, output, calls, _ = controlled_study
    output.mkdir()
    preserved = output / "partial.zip"
    preserved.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        experiment.run_opportunity_experiment(manifest, fresh, output)
    assert preserved.read_bytes() == b"preserve"
    assert not calls


def test_failure_keeps_partial_model_and_records_error(controlled_study, monkeypatch):
    """@brief A training failure is explicit and never triggers fallback test access."""
    manifest, fresh, output, calls, _ = controlled_study

    def fail(path, events, directory, **kwargs):
        """@brief Mimic an interruption after useful partial checkpoint output exists."""
        directory.mkdir(parents=True)
        (directory / "partial.zip").write_bytes(b"recoverable")
        raise RuntimeError("controlled interruption")

    monkeypatch.setattr(experiment, "train_opportunity_policy", fail)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        experiment.run_opportunity_experiment(
            manifest, fresh, output, seeds=(7,), steps=1, warm_epochs=1,
        )
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["error"]["message"] == "controlled interruption"
    assert (output / "models/seed7/partial.zip").read_bytes() == b"recoverable"
    assert not any(isinstance(item, tuple) for item in calls)


def test_selection_prefers_net_dollars_then_fewer_trades():
    """@brief Profitable trade counts cannot outrank the actual after-cost objective."""
    records = [
        {"name": "many", "aggregate": {"net_pnl": 0.1, "trade_count": 10}},
        {"name": "few", "aggregate": {"net_pnl": 0.1, "trade_count": 1}},
        {"name": "flat", "aggregate": {"net_pnl": 0.0, "trade_count": 0}},
    ]
    assert experiment.choose_candidate(records)["name"] == "few"
    with pytest.raises(ValueError):
        experiment.choose_candidate([])


def test_fresh_dates_must_follow_even_previously_examined_test(monkeypatch):
    """@brief Previously examined test days cannot be relabeled as a fresh holdout."""
    def read(path, split=None):
        """@brief Isolate the metadata chronology guard from archive I/O."""
        entries = [{"day": "2026-10-02", "split": "test"}]
        return {"episodes": entries}, entries

    monkeypatch.setattr(experiment, "read_manifest", read)
    with pytest.raises(ValueError, match="every previously examined day"):
        experiment.validate_fresh_dates("old", "fresh")

