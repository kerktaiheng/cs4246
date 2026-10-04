"""@file test_experiment.py
@brief Integration checks for protected, restartable PPO experiment orchestration.

@details Small real PPO fits verify pipeline wiring. Targeted instrumentation
checks the test-access boundary, deterministic completed-stage reuse, lazy archive
loading, the predeclared entropy retry, and preservation of partial model files.
"""

from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path

import pytest

import latency_arb.experiment as experiments
from latency_arb.data.generate_episodes import create_demo
from latency_arb.experiment import ExperimentConfig, LazyEpisodeSequence, run_experiment


@pytest.fixture(scope="module")
def miniature_experiment(tmp_path_factory):
    """@brief Execute two real seeded PPO candidates and a small frozen cost sweep."""
    root = tmp_path_factory.mktemp("experiment")
    manifest = create_demo(root / "data", days=6, steps_per_day=32, seed=11)
    output = root / "results"
    settings = ExperimentConfig(
        steps=256, seeds=(7, 17), n_steps=64, batch_size=32, epochs=1,
        thresholds=(1.0, 7.0, 100.0), fee_grid=(1.0, 3.5),
        latency_grid=(150.0,), spread_grid=(1.0,), evaluate_test=True,
    )
    test_accesses = []
    original = experiments.load_manifest_entry

    def checked_load(path, entry):
        """@brief Require immutable selection to exist before opening test arrays."""
        if entry["split"] == "test":
            frozen = output / "selection.json"
            assert frozen.exists(), "Test archive opened before selection was frozen."
            test_accesses.append(frozen.read_bytes())
        return original(path, entry)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(experiments, "load_manifest_entry", checked_load)
        summary = run_experiment(manifest, output, settings)
    return manifest, output, settings, summary, test_accesses


def test_miniature_experiment_freezes_selection_before_test(miniature_experiment):
    """@brief Real final evaluation and sensitivity share one immutable selection."""
    _, output, _, summary, test_accesses = miniature_experiment
    assert test_accesses
    assert all(value == test_accesses[0] for value in test_accesses)
    assert (output / "selection.json").read_bytes() == test_accesses[0]
    assert summary["test_evaluated"] is True
    assert summary["sensitivity_completed"] is True
    assert summary["synthetic"] is True
    assert set(summary["test"]["policies"]) == {"flat", "threshold", "ppo"}
    assert len(summary["sensitivity"]) == 6
    assert len(summary["fee_break_even"]) == 3
    assert summary["test"]["policies"]["flat"]["aggregate"]["trade_count"] == 0
    assert len(summary["candidates"]) in (2, 3)
    for split in ("validation", "test"):
        metrics = summary[split]["policies"]["ppo"]["aggregate"]
        assert "daily_pnl" in metrics and "daily_pnl_std" in metrics
        assert "abstention_rate" in metrics
    assert (output / "sensitivity.csv").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "completed"


def test_completed_experiment_resume_does_not_retrain_or_reopen_data(
    miniature_experiment, monkeypatch
):
    """@brief Completed compatible requests return saved results without test replay."""
    manifest, output, settings, summary, _ = miniature_experiment

    def forbidden(*args, **kwargs):
        """@brief Fail if completed-stage reuse attempts new learning or archive I/O."""
        raise AssertionError("A completed experiment must not reopen data or retrain.")

    monkeypatch.setattr(experiments, "train_policy", forbidden)
    monkeypatch.setattr(experiments, "load_manifest_entry", forbidden)
    assert run_experiment(manifest, output, settings) == summary


def test_changed_protocol_preserves_existing_experiment(miniature_experiment):
    """@brief A new budget cannot silently reuse models or overwrite frozen results."""
    manifest, output, settings, _, _ = miniature_experiment
    before = (output / "selection.json").read_bytes()
    with pytest.raises(ValueError, match="protocol differs"):
        run_experiment(manifest, output, replace(settings, steps=settings.steps + 64))
    assert (output / "selection.json").read_bytes() == before


def test_validation_only_run_never_opens_test_and_retry_is_predeclared(
    tmp_path, monkeypatch
):
    """@brief Zero validation trades trigger exactly one declared entropy retry.

    @details Test files are physically absent. Instrumentation sets only the two
    PPO selection metrics to zero to exercise this branch deterministically while
    preserving real PPO fits and real evaluator metrics everywhere else.
    """
    manifest = create_demo(tmp_path / "data", days=6, steps_per_day=24, seed=23)
    document = json.loads(manifest.read_text())
    for entry in document["episodes"]:
        if entry["split"] == "test":
            (manifest.parent / entry["path"]).unlink()
    original_compare = experiments.compare_policies

    def zero_trades(episodes, policies, *args, **kwargs):
        """@brief Simulate PPO abstention without replacing baseline evaluation."""
        report = original_compare(episodes, policies, *args, **kwargs)
        if set(policies) == {"ppo"}:
            aggregate = report["policies"]["ppo"]["aggregate"]
            aggregate["trade_count"] = 0
            aggregate["net_pnl"] = 0.0
        return report

    monkeypatch.setattr(experiments, "compare_policies", zero_trades)
    settings = ExperimentConfig(
        steps=64, seeds=(7, 17), n_steps=32, batch_size=16, epochs=1,
        thresholds=(7.0,), skip_sensitivity=True,
    )
    output = tmp_path / "results"
    summary = run_experiment(manifest, output, settings)
    assert summary["test_evaluated"] is False
    assert summary["test"] is None
    assert summary["retry_triggered"] is True
    assert [item["name"] for item in summary["candidates"]] == [
        "ppo_seed7", "ppo_seed17", "ppo_seed7_entropy005"
    ]
    retry = summary["candidates"][-1]
    assert retry["training_config"]["seed"] == 7
    assert retry["training_config"]["ent_coef"] == 0.05
    assert summary["selected_candidate"] == "ppo_seed7"
    declared = json.loads((output / "experiment.json").read_text())["protocol"]
    assert declared["conditional_retry"]["maximum_retries"] == 1
    assert declared["conditional_retry"]["training"]["ent_coef"] == 0.05
    assert not (output / "test").exists()


def test_partial_model_is_preserved_and_failure_is_recorded(tmp_path, monkeypatch):
    """@brief Interrupted model files fail clearly and are never overwritten."""
    manifest = create_demo(tmp_path / "data", days=3, steps_per_day=20, seed=8)
    output = tmp_path / "results"
    settings = ExperimentConfig(
        steps=32, seeds=(7,), n_steps=32, batch_size=16, epochs=1,
        thresholds=(7.0,), skip_sensitivity=True,
    )
    original_train = experiments.train_policy

    def interrupted(manifest_path, directory, *args, **kwargs):
        """@brief Leave a recoverable partial artifact like an interrupted PPO job."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "partial.txt").write_text("preserve me", encoding="utf-8")
        raise RuntimeError("simulated training interruption")

    monkeypatch.setattr(experiments, "train_policy", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        run_experiment(manifest, output, settings)
    first_status = json.loads((output / "status.json").read_text())
    assert first_status["status"] == "failed"
    assert first_status["error"]["type"] == "RuntimeError"
    monkeypatch.setattr(experiments, "train_policy", original_train)
    with pytest.raises(ValueError, match="Partial model directory"):
        run_experiment(manifest, output, settings)
    assert (output / "models" / "ppo_seed7" / "partial.txt").read_text() == "preserve me"
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_lazy_evaluation_sequence_is_reiterable_and_bounded(
    miniature_experiment, monkeypatch
):
    """@brief Evaluation retains at most its configured number of replay archives."""
    manifest, _, _, _, _ = miniature_experiment
    opened = []
    original = experiments.load_manifest_entry

    def counted(path, entry):
        """@brief Record selected archive reads while preserving real verification."""
        opened.append(entry["path"])
        return original(path, entry)

    monkeypatch.setattr(experiments, "load_manifest_entry", counted)
    episodes = LazyEpisodeSequence(manifest, "train", cache_size=1)
    assert opened == []
    first = []
    for episode in episodes:
        first.append(episode.metadata["episode_id"])
        assert len(episodes._cache) <= 1
    second = [episode.metadata["episode_id"] for episode in episodes]
    assert first == second
    assert len(opened) == 2 * len(episodes)
    assert episodes[-1].metadata["episode_id"] == first[-1]
    with pytest.raises(IndexError):
        _ = episodes[len(episodes)]
    episodes.clear()
    assert len(episodes._cache) == 0


@pytest.mark.parametrize(
    "values",
    [
        {"seeds": ()}, {"seeds": (7, 7)}, {"thresholds": ()},
        {"fee_grid": (-1.0,)}, {"spread_grid": (0.5,)},
        {"decision_interval_ms": -1}, {"latency_grid": (40_000.0,)},
    ],
)
def test_experiment_rejects_invalid_protocol(values):
    """@brief Invalid analytical settings fail before any model or output mutation."""
    with pytest.raises(ValueError):
        ExperimentConfig(**values).validate()



def test_completed_model_configuration_is_rechecked_on_resume(
    miniature_experiment, tmp_path
):
    """@brief A completed summary cannot bypass model compatibility verification."""
    manifest, output, settings, _, _ = miniature_experiment
    copied = tmp_path / "copied_results"
    shutil.copytree(output, copied)
    metadata_path = copied / "models" / "ppo_seed7" / "training.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["training_config"]["gamma"] = 0.5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible completed model"):
        run_experiment(manifest, copied, settings)
    assert json.loads((copied / "status.json").read_text())["status"] == "failed"


def test_completed_selection_cannot_change_on_resume(miniature_experiment, tmp_path):
    """@brief Frozen candidate selection is protected even after all stages finish."""
    manifest, output, settings, _, _ = miniature_experiment
    copied = tmp_path / "copied_results"
    shutil.copytree(output, copied)
    selection_path = copied / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_candidate"] = "altered_candidate"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen selection checksum"):
        run_experiment(manifest, copied, settings)


def test_changed_implementation_rejects_resume_without_editing_sources(
    miniature_experiment, monkeypatch
):
    """@brief Core source identity participates in completed-experiment compatibility."""
    manifest, output, settings, _, _ = miniature_experiment
    original = experiments._implementation_fingerprint()
    changed = {**original, "source_sha256": "0" * 64}
    selection_before = (output / "selection.json").read_bytes()
    monkeypatch.setattr(experiments, "_implementation_fingerprint", lambda: changed)
    with pytest.raises(ValueError, match="protocol differs"):
        run_experiment(manifest, output, settings)
    assert (output / "selection.json").read_bytes() == selection_before


def test_source_fingerprint_excludes_reports_and_normalizes_line_endings(tmp_path):
    """@brief Report-only edits preserve compatibility; economic edits invalidate it.

    @details Copy core files into an isolated tree so the live checkout and active
    overnight experiment never change while this compatibility rule is tested.
    """
    root = Path(experiments.__file__).resolve().parents[1]
    original = experiments._implementation_fingerprint()
    for relative in original["files"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(normalized.replace(b"\n", b"\r\n"))
    assert experiments._implementation_fingerprint(tmp_path) == original
    (tmp_path / "latency_arb" / "report.py").write_text(
        "# Cosmetic report-only fixture.\n", encoding="utf-8"
    )
    assert experiments._implementation_fingerprint(tmp_path) == original
    baseline = tmp_path / "latency_arb" / "baseline.py"
    baseline.write_bytes(baseline.read_bytes() + b"# Changed economic implementation.\n")
    assert experiments._implementation_fingerprint(tmp_path)["source_sha256"] != original["source_sha256"]
