"""@file test_opportunity_value_experiment.py
@brief Verify frozen fee-study boundaries with synthetic stages and real artifact hashes.
@details These tests never open recorded books or claim synthetic returns as evidence.
They make both forbidden holdout access and model mutation observable at the boundary.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

import latency_arb.opportunity_value_experiment as experiment


@pytest.fixture
def study(tmp_path, monkeypatch):
    """@brief Build metadata-valid inputs and deterministic training/replay test doubles."""
    manifest, fresh = tmp_path / "old.json", tmp_path / "fresh.json"
    events, prior, output = tmp_path / "events", tmp_path / "prior", tmp_path / "run"
    manifest.write_text("{}")
    fresh.write_text("{}")
    events.mkdir()
    prior.mkdir()
    source = {"producer": "unchanged"}
    calls = []
    controls = {"pnl": 0.2, "neural_pnl": 0.1, "mutation": None, "fresh_failure": False}
    entries = [
        {"day": "2026-09-03", "split": "train"},
        {"day": "2026-09-25", "split": "validation"},
        {"day": "2026-10-02", "split": "test"},
    ]

    def read(path, split=None):
        """@brief Expose calendar metadata independently of any book-array loader."""
        selected = [{"day": "2026-10-03", "split": "test"}] if Path(path) == fresh else entries
        return {"episodes": selected}, [row for row in selected if split is None or row["split"] == split]

    monkeypatch.setattr(experiment, "read_manifest", read)
    monkeypatch.setattr(experiment, "source_fingerprint", lambda: dict(source))
    (events / "events.npz").write_bytes(b"numeric events")
    (events / "normalization.npz").write_bytes(b"numeric moments")
    metadata = {
        "split": "train", "training_entries": entries[:1],
        "manifest_sha256": experiment._sha(manifest),
        "feature_names": list(experiment.FEATURE_NAMES),
        "future_rows_used_only_for_targets": True,
        "execution_config": asdict(experiment._execution(4.5)),
        "opportunity_config": asdict(experiment.OpportunityConfig()),
        "source_sha256": dict(source),
        "artifact_sha256": {name: experiment._sha(events / name)
                            for name in ("events.npz", "normalization.npz")},
    }
    experiment._save(events / "events.json", metadata)
    experiment._save(prior / "status.json", {"status": "completed", "fresh_test_evaluated": False})
    experiment._save(prior / "summary.json", {"fresh_test_evaluated": False})
    archived = []
    for seed in (7, 17, 27):
        for stage in ("warmstart", "ppo"):
            directory = prior / "models" / f"seed{seed}" / stage
            directory.mkdir(parents=True)
            (directory / "policy.zip").write_bytes(f"{seed}{stage}".encode())
            archived.append({
                "name": f"seed{seed}_{stage}", "seed": seed, "stage": stage,
                "model_dir": str(directory), "model_sha256": experiment._sha(directory / "policy.zip"),
            })
    experiment._save(prior / "candidate_results.json", archived)

    class FixedPolicy:
        """@brief Retain the interface while returning deterministic test-only actions."""
        metadata = {}

        def predict(self, observation, deterministic=True):
            """@brief This fake never fits or inspects a held-out outcome."""
            return 0, None

    def train(directory, destination, **kwargs):
        """@brief Require protocol publication before targets or fitting can be used."""
        assert Path(directory) == events
        protocol = experiment._read(output / "protocol.json")
        assert protocol["learned_candidate_count"] == 10
        assert len(protocol["archived_neural_candidates"]) == 6
        assert kwargs["expected_events_sha256"] == experiment._sha(events / "events.npz")
        calls.append("train")
        candidates = []
        for index in range(4):
            path = destination / f"value{index}"
            path.mkdir(parents=True)
            (path / "model.pkl").write_bytes(f"model{index}".encode())
            candidates.append({
                "id": f"value{index}", "directory": str(path),
                "margin_usd": (0.0, 0.01)[index % 2], "regressor_config": {},
                "model_sha256": experiment._sha(path / "model.pkl"),
            })
        return {"candidates": candidates}

    def evaluate(path, split, policies, execution, opportunity, output_dir=None):
        """@brief Verify frozen selection and exact fee at every held-out replay call."""
        if Path(path) == fresh:
            assert split == "test"
            selection = experiment._read(output / "selection.json")
            assert selection["validation_beats_flat"] is True
            calls.append(("fresh", execution.fee_bps, (output / "selection.json").read_bytes()))
            if controls["fresh_failure"]:
                raise RuntimeError("interrupted fresh replay")
        else:
            assert Path(path) == manifest and split == "validation"
            assert execution.fee_bps == 3.5
            calls.append(("validation", tuple(policies)))
        results = {}
        for name in policies:
            value = (0.0 if name == "flat" else
                     0.05 if name.startswith("threshold") else
                     controls["neural_pnl"] if name.startswith("seed") else controls["pnl"])
            results[name] = {"aggregate": {"net_pnl": value,
                                          "trade_count": int(value != 0), "synthetic": True},
                             "episodes": []}
        if "seed27_ppo" in policies and controls["mutation"]:
            controls["mutation"]()
        return {"policies": results, "split": split, "config": asdict(execution)}

    monkeypatch.setattr(experiment, "train_opportunity_value", train)
    monkeypatch.setattr(experiment, "load_opportunity_value_policy", lambda *args, **kwargs: FixedPolicy())
    monkeypatch.setattr(experiment, "load_opportunity_policy", lambda *args, **kwargs: FixedPolicy())
    monkeypatch.setattr(experiment, "evaluate_opportunity_policies", evaluate)
    return {
        "args": (manifest, fresh, events, prior, output), "calls": calls,
        "controls": controls, "source": source, "entries": entries,
    }


def test_primary_and_stress_share_frozen_selection_and_ten_candidates(study):
    """@brief The fresh day is accessed only after all ten primary-fee candidates freeze."""
    result = experiment.run_opportunity_value_experiment(*study["args"])
    output = study["args"][-1]
    assert len(result["selection"]["candidates"]) == 10
    assert result["selection"]["selected"]["kind"] == "expected_value"
    access = [call for call in study["calls"] if isinstance(call, tuple) and call[0] == "fresh"]
    assert [call[1] for call in access] == [3.5, 4.5]
    assert all(call[2] == (output / "selection.json").read_bytes() for call in access)
    assert result["fresh_test_evaluated"] is True
    assert result["protocol"]["training_target_fee_bps"] == 4.5
    assert (output / "fresh_test/comparison.json").exists()
    assert (output / "fee_stress/comparison.json").exists()


def test_archived_neural_candidate_keeps_honest_method_label(study):
    """@brief A saved warm-start winner is never rebranded as a newly successful PPO."""
    study["controls"].update(pnl=0.1, neural_pnl=0.3)
    result = experiment.run_opportunity_value_experiment(*study["args"])
    assert result["selection"]["selected"]["name"] == "seed7_warmstart"
    assert result["selection"]["selected"]["kind"] == "warmstart"
    assert "warmstart" in result["conclusion"]


@pytest.mark.parametrize("pnl", [0.0, -0.1])
def test_nonpositive_selection_preserves_fresh_day(study, pnl):
    """@brief Neither permanent abstention nor losing trades passes the test-use gate."""
    study["controls"].update(pnl=pnl, neural_pnl=pnl)
    result = experiment.run_opportunity_value_experiment(*study["args"])
    assert result["fresh_test_evaluated"] is False
    assert result["fresh_test"] is None and result["fee_stress"] is None
    assert not any(isinstance(call, tuple) and call[0] == "fresh" for call in study["calls"])


def test_fee_adapter_preserves_inputs_and_training_predictions():
    """@brief Primary and stress observations map to the same frozen training feature."""
    observed = []

    class Policy:
        """@brief Capture feature inputs rather than introducing a fitted predictor."""
        def predict(self, features, deterministic=True):
            """@brief Retain the exact model input for the algebraic invariant check."""
            observed.append(features.copy())
            return int(features[1] > 0), None

        def expected_net_pnl(self, features):
            """@brief Return one cost-proxy value for optional adapter parity checking."""
            return float(features[1])

    original = np.arange(len(experiment.FEATURE_NAMES), dtype=np.float32)
    original[1] = 3.0
    primary = original.copy()
    primary[1] += 2.0
    assert experiment.TrainingCostPolicy(Policy(), 3.5).predict(primary) == \
        experiment.TrainingCostPolicy(Policy(), 4.5).predict(original)
    np.testing.assert_array_equal(observed[0], original)
    np.testing.assert_array_equal(observed[1], original)
    assert primary[1] == 5.0
    assert experiment.TrainingCostPolicy(Policy(), 3.5).expected_net_pnl(primary) == 3.0


@pytest.mark.parametrize("target", ["events", "source", "archived_model"])
def test_mutation_before_boundary_prevents_fresh_access(study, target):
    """@brief Source, events, and saved old checkpoints remain frozen through selection."""
    manifest, fresh, events, prior, output = study["args"]

    def mutate():
        """@brief Inject a change only after the final validation policy has run."""
        if target == "events":
            (events / "events.npz").write_bytes(b"changed")
        elif target == "source":
            study["source"]["producer"] = "changed"
        else:
            (prior / "models/seed7/ppo/policy.zip").write_bytes(b"changed")

    study["controls"]["mutation"] = mutate
    with pytest.raises(ValueError, match="Frozen"):
        experiment.run_opportunity_value_experiment(*study["args"])
    assert not any(isinstance(call, tuple) and call[0] == "fresh" for call in study["calls"])
    assert experiment._read(output / "status.json")["status"] == "failed"


def test_interrupted_fresh_access_is_recorded_as_spent(study):
    """@brief An exception while reading the fresh day cannot make it appear untouched."""
    study["controls"]["fresh_failure"] = True
    with pytest.raises(RuntimeError, match="interrupted fresh replay"):
        experiment.run_opportunity_value_experiment(*study["args"])
    output = study["args"][-1]
    state = experiment._read(output / "status.json")
    assert state["fresh_test_access_started"] is True
    assert state["fresh_test_evaluated"] is False
    assert (output / "selection.json").exists()
    assert (output / "models/value0/model.pkl").exists()


def test_training_failure_preserves_partial_artifacts(study, monkeypatch):
    """@brief A failed fit preserves its saved partial model and never touches fresh data."""
    def fail(events, output, **kwargs):
        """@brief Fail after producing a useful partial recovery artifact."""
        output.mkdir(parents=True)
        (output / "partial.pkl").write_bytes(b"partial")
        raise RuntimeError("fit interrupted")

    monkeypatch.setattr(experiment, "train_opportunity_value", fail)
    with pytest.raises(RuntimeError, match="fit interrupted"):
        experiment.run_opportunity_value_experiment(*study["args"])
    output = study["args"][-1]
    assert (output / "models/partial.pkl").read_bytes() == b"partial"
    assert experiment._read(output / "status.json")["fresh_test_access_started"] is False


def test_existing_output_is_never_overwritten(study):
    """@brief A repeated invocation cannot remove the original experiment evidence."""
    output = study["args"][-1]
    output.mkdir()
    (output / "keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        experiment.run_opportunity_value_experiment(*study["args"])
    assert (output / "keep").read_text() == "preserve"
    assert study["calls"] == []


@pytest.mark.parametrize("violation", ["train_split", "old_test_used", "producer_source", "archive_hash"])
def test_invalid_provenance_fails_before_any_fit(study, violation):
    """@brief Reject held-out targets, prior fresh use, and incompatible event caches."""
    manifest, fresh, events, prior, output = study["args"]
    if violation == "old_test_used":
        experiment._save(prior / "summary.json", {"fresh_test_evaluated": True})
    else:
        metadata = experiment._read(events / "events.json")
        if violation == "train_split":
            metadata["training_entries"][0]["split"] = "test"
        elif violation == "producer_source":
            metadata["source_sha256"]["producer"] = "different"
        else:
            metadata["artifact_sha256"]["events.npz"] = "0" * 64
        experiment._save(events / "events.json", metadata)
    with pytest.raises(ValueError):
        experiment.run_opportunity_value_experiment(*study["args"])
    assert not (output / "protocol.json").exists()
    assert study["calls"] == []


def test_chronology_rejects_reused_test_date(study):
    """@brief Fresh dates must follow even the old already-examined test period."""
    study["entries"][-1]["day"] = "2026-10-03"
    with pytest.raises(ValueError, match="every previously examined day"):
        experiment.run_opportunity_value_experiment(*study["args"])


def test_selection_uses_net_dollars_then_fewer_trades_and_stable_ties():
    """@brief Activity and reported win rate cannot replace after-cost dollar profit."""
    records = [
        {"name": "many", "aggregate": {"net_pnl": 0.1, "trade_count": 9}},
        {"name": "few", "aggregate": {"net_pnl": 0.1, "trade_count": 1}},
        {"name": "tie", "aggregate": {"net_pnl": 0.1, "trade_count": 1}},
    ]
    assert experiment.choose_candidate(records)["name"] == "few"
