"""@file opportunity_value_experiment.py
@brief Freeze a train-only payoff-regression study before spending a fresh day.
@details This experiment follows two failed PPO attempts. It retains their evidence
and evaluates four predeclared supervised entry policies plus six archived neural
checkpoints at the primary fee. Every stage keeps its correct method label. Primary replay uses the user's stated 3.5 bps per side; existing training targets
remain at 4.5 bps per side. A frozen feature adapter preserves the training cost
feature under both primary and stress evaluation. No future outcome selects a policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from latency_arb.agent.opportunity_train import load_opportunity_policy
from latency_arb.agent.opportunity_value import (
    COMMON_SETTINGS, MARGINS_USD, REGRESSOR_CONFIGS,
    load_opportunity_value_policy, train_opportunity_value,
)
from latency_arb.data.schema import read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import FEATURE_NAMES, OpportunityConfig
from latency_arb.opportunity_evaluate import (
    GapThresholdPolicy, SkipPolicy, evaluate_opportunity_policies,
)

TRAINING_FEE_BPS = 4.5
PRIMARY_FEE_BPS = 3.5
THRESHOLDS_BPS = (6.0, 7.0, 9.0, 10.0, 12.0, 15.0, 20.0)


def _now():
    """@brief Return UTC timestamps independently of the user's display timezone."""
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    """@brief Hash exact saved artifact bytes without retaining them in memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _save(path, value):
    """@brief Atomically publish strict JSON while preserving incomplete prior stages."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _read(path):
    """@brief Read one saved JSON artifact, allowing missing/corrupt evidence to fail."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution(fee_bps):
    """@brief Keep execution mechanics identical while changing only the explicit fee."""
    return EnvConfig(fee_bps=fee_bps, latency_ms=150, decision_interval_ms=1000,
                     slippage_bps=0.1, position_size_btc=0.001,
                     max_holding_ms=30_000, reward_scale=1.0, log_history=True)


def source_fingerprint():
    """@brief Bind the study to every economic, feature, fitting, and runner module."""
    root = Path(__file__).resolve().parents[1]
    names = (
        "common/gym_base.py", "latency_arb/data/schema.py",
        "latency_arb/env/latency_sim.py", "latency_arb/opportunity.py",
        "latency_arb/agent/policy.py", "latency_arb/agent/opportunity_train.py",
        "latency_arb/agent/opportunity_value.py",
        "latency_arb/opportunity_evaluate.py",
        "latency_arb/opportunity_value_experiment.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in names}


def choose_candidate(candidates):
    """@brief Select validation net dollars, then fewer trades, preserving declared ties."""
    if not candidates:
        raise ValueError("No candidate validation records were provided.")
    return max(candidates, key=lambda row: (
        row["aggregate"]["net_pnl"], -row["aggregate"]["trade_count"],
    ))


def validate_fresh_dates(manifest, fresh_manifest):
    """@brief Inspect calendar metadata only; never open a fresh market archive here."""
    previous, _ = read_manifest(manifest)
    _, fresh_entries = read_manifest(fresh_manifest, split="test")
    previous_days = sorted({entry["day"] for entry in previous["episodes"]})
    fresh_days = sorted({entry["day"] for entry in fresh_entries})
    if not previous_days or not fresh_days or fresh_days[0] <= previous_days[-1]:
        raise ValueError("Fresh holdout must follow every previously examined day.")
    dates = {
        split: sorted({entry["day"] for entry in previous["episodes"]
                       if entry["split"] == split})
        for split in ("train", "validation")
    }
    if not dates["train"] or not dates["validation"]:
        raise ValueError("Training and validation dates must both exist.")
    dates["fresh_test"] = fresh_days
    dates["previously_examined_test_excluded"] = sorted({
        entry["day"] for entry in previous["episodes"] if entry["split"] == "test"
    })
    return dates


class TrainingCostPolicy:
    """@brief Present the exact training-cost convention to an immutable value policy.

    @details Replay features subtract two contemporaneous fees. Subtracting twice
    (training fee minus evaluation fee) restores the training convention. All other
    raw features remain unchanged; no prediction threshold is retuned in stress.
    Float32 arithmetic can differ by one unit in the final place across conventions.
    """

    def __init__(self, policy, evaluation_fee_bps):
        """@brief Freeze the wrapped policy and the explicit algebraic cost correction."""
        if not np.isfinite(evaluation_fee_bps) or evaluation_fee_bps < 0:
            raise ValueError("Evaluation fee must be finite and nonnegative.")
        self.policy = policy
        self.evaluation_fee_bps = float(evaluation_fee_bps)
        self.metadata = getattr(policy, "metadata", {})

    def _features(self, observation):
        """@brief Copy the input before changing only its named cost-proxy component."""
        features = np.asarray(observation, dtype=np.float32).copy()
        if features.ndim == 0 or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("Observation does not match the frozen feature schema.")
        column = FEATURE_NAMES.index("gap_excess_roundtrip_cost_bps")
        features[..., column] -= np.float32(
            2.0 * (TRAINING_FEE_BPS - self.evaluation_fee_bps)
        )
        return features

    def predict(self, observation, deterministic=True):
        """@brief Delegate frozen actions after correcting the fee convention only."""
        return self.policy.predict(self._features(observation), deterministic=deterministic)

    def expected_net_pnl(self, observation):
        """@brief Expose the unchanged training-fee value estimate for optional audits."""
        return self.policy.expected_net_pnl(self._features(observation))

    def reset(self):
        """@brief Forward any episode-reset hook without fitting or updating statistics."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def validate_inputs(manifest, fresh_manifest, events, prior_ppo, execution, opportunity):
    """@brief Verify chronology, failed-PPO isolation, and training-event provenance.

    @details Only metadata and exact bytes are inspected before protocol publication.
    No model outcomes or market observations are computed. The producer source and
    every saved event-array checksum must match; targets must come from train only.
    """
    dates = validate_fresh_dates(manifest, fresh_manifest)
    prior_status = _read(prior_ppo / "status.json")
    prior_summary = _read(prior_ppo / "summary.json")
    if prior_status.get("status") != "completed":
        raise ValueError("The prior PPO experiment must be completed.")
    if prior_status.get("fresh_test_evaluated") is not False or \
            prior_summary.get("fresh_test_evaluated") is not False:
        raise ValueError("The prior PPO experiment must have left the fresh test unused.")
    metadata = _read(events / "events.json")
    _, train_entries = read_manifest(manifest, split="train")
    if metadata.get("split") != "train" or metadata.get("training_entries") != train_entries:
        raise ValueError("Event provenance must contain exactly the manifest training split.")
    if metadata.get("manifest_sha256") != _sha(manifest):
        raise ValueError("Training events were generated from a different manifest.")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Training-event features differ from the current feature schema.")
    if metadata.get("future_rows_used_only_for_targets") is not True:
        raise ValueError("Training-event metadata must declare causal features.")
    expected = asdict(execution)
    recorded = metadata.get("execution_config", {})
    for key in ("reward_scale", "log_history"):
        expected.pop(key)
        recorded = {name: value for name, value in recorded.items() if name != key}
    if recorded != expected or metadata.get("opportunity_config") != asdict(opportunity):
        raise ValueError("Training targets and declared economic settings differ.")
    current_sources = source_fingerprint()
    producer_sources = metadata.get("source_sha256", {})
    if not producer_sources or any(current_sources.get(name) != digest
                                   for name, digest in producer_sources.items()):
        raise ValueError("Training-event producer source fingerprint has changed.")
    artifact_hashes = metadata.get("artifact_sha256", {})
    if set(artifact_hashes) != {"events.npz", "normalization.npz"}:
        raise ValueError("Training events require both recorded numeric artifact hashes.")
    for name, digest in artifact_hashes.items():
        if _sha(events / name) != digest:
            raise ValueError(f"Training-event artifact checksum mismatch: {name}")
    files = (manifest, fresh_manifest, events / "events.json", events / "events.npz",
             events / "normalization.npz", prior_ppo / "status.json",
             prior_ppo / "summary.json")
    return dates, {str(path): _sha(path) for path in files}


def archived_candidates(prior_ppo):
    """@brief Verify the exact six earlier checkpoints without using their old outcomes."""
    records = _read(prior_ppo / "candidate_results.json")
    expected = [(seed, stage) for seed in (7, 17, 27) for stage in ("warmstart", "ppo")]
    by_key = {(row.get("seed"), row.get("stage")): row for row in records}
    if len(records) != 6 or set(by_key) != set(expected):
        raise ValueError("Prior PPO run must contain exactly six declared checkpoints.")
    result = []
    for seed, stage in expected:
        saved = by_key[(seed, stage)]
        directory = Path(saved["model_dir"]).resolve()
        if not directory.is_relative_to(prior_ppo / "models"):
            raise ValueError("Archived checkpoint lies outside the declared prior run.")
        if _sha(directory / "policy.zip") != saved["model_sha256"]:
            raise ValueError("Archived checkpoint no longer matches its original checksum.")
        result.append({"id": saved["name"], "directory": str(directory),
                       "kind": stage, "stage": stage, "seed": seed,
                       "margin_usd": None, "regressor_config": None,
                       "model_sha256": saved["model_sha256"]})
    return result


def _load_candidate(candidate, event_hash):
    """@brief Select the saved model loader from its predeclared method, never its PnL."""
    if candidate["kind"] == "expected_value":
        return load_opportunity_value_policy(
            candidate["directory"], expected_events_sha256=event_hash,
        )
    return load_opportunity_policy(candidate["directory"])


def _model_files(directory):
    """@brief Hash all selected model artifacts, restricting them to their owned tree."""
    directory = Path(directory).resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Selected model directory contains no saved artifacts.")
    if any(not path.resolve().is_relative_to(directory) for path in files):
        raise ValueError("Selected model artifact escapes its directory.")
    return {str(path): _sha(path) for path in files}


def verify_frozen_boundary(selection_path, selection_sha256):
    """@brief Refuse fresh evaluation if any frozen evidence or selected weight changed.

    @param selection_path Saved declaration written before the first fresh replay.
    @param selection_sha256 In-memory fingerprint captured at publication.
    @details This check also runs between primary and fee-stress replay. Changed
    artifacts fail explicitly instead of silently evaluating a different hypothesis.
    """
    if _sha(selection_path) != selection_sha256:
        raise ValueError("Frozen selection changed before fresh evaluation.")
    selection = _read(selection_path)
    boundary = selection["boundary_hashes"]
    for path, expected in boundary["artifacts"].items():
        if _sha(path) != expected:
            raise ValueError(f"Frozen artifact changed before fresh evaluation: {path}")
    if source_fingerprint() != boundary["sources"]:
        raise ValueError("Frozen source changed before fresh evaluation.")


def run_opportunity_value_experiment(manifest, fresh_manifest, event_data_dir,
                                     prior_ppo_run, output_dir):
    """@brief Compare four new value policies and six archived neural policies, then test once.

    @details All overlapping counterfactual labels remain training targets only.
    Realized metrics come from complete continuous portfolio replay. A nonpositive
    validation policy leaves the fresh day unused. A positive single fresh day is
    provisional evidence, not a guarantee of future profit or a claim of PPO success.
    """
    manifest, fresh_manifest = Path(manifest).resolve(), Path(fresh_manifest).resolve()
    events, prior_ppo = Path(event_data_dir).resolve(), Path(prior_ppo_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new empty output directory; preserve prior experiments.")
    execution, stress_execution = _execution(PRIMARY_FEE_BPS), _execution(TRAINING_FEE_BPS)
    opportunity = OpportunityConfig(min_gap_bps=6.0, exit_gap_bps=0.5, min_remaining_ms=31_000)
    dates, input_hashes = validate_inputs(
        manifest, fresh_manifest, events, prior_ppo, stress_execution, opportunity,
    )
    archived = archived_candidates(prior_ppo)
    input_hashes[str(prior_ppo / "candidate_results.json")] = _sha(prior_ppo / "candidate_results.json")
    for candidate in archived:
        input_hashes.update(_model_files(candidate["directory"]))
    protocol = {
        "version": 1, "created_at_utc": _now(), "method": "Frozen fee comparison of supervised value, warm-start, and PPO entry policies",
        "manifest": str(manifest), "fresh_manifest": str(fresh_manifest),
        "event_data_dir": str(events), "prior_ppo_run": str(prior_ppo), "dates": dates,
        "execution": asdict(execution), "stress_execution": asdict(stress_execution),
        "opportunity": asdict(opportunity), "training_target_fee_bps": TRAINING_FEE_BPS,
        "candidate_grid": {"regressors": list(REGRESSOR_CONFIGS),
                           "margins_usd": list(MARGINS_USD), "common_settings": COMMON_SETTINGS},
        "archived_neural_candidates": archived,
        "learned_candidate_count": 10,
        "baseline_thresholds_bps": list(THRESHOLDS_BPS),
        "selection": "Maximum validation net USD; fewer trades on ties; stable declared ordering.",
        "fresh_test_gate": "Selected learned validation net PnL > 0 with at least one executed trade.",
        "cost_feature_adapter": "Subtract 2 * (4.5 - evaluation fee bps) from the raw cost-excess feature.",
        "input_sha256": input_hashes, "source_fingerprint": source_fingerprint(),
        "training_labels": "Train-only counterfactual payoffs at 4.5 bps per side; overlapping labels are never portfolio returns.",
        "limitations": "One fresh day; development reuses known validation; no profit guarantee; distinguish warm-start, PPO, and value-regression winners.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "protocol.json", protocol)
    started = time.perf_counter()

    def status(stage, **extra):
        """@brief Save real completed progress without inventing metrics or completion."""
        value = {"status": "running", "stage": stage, "updated_at_utc": _now(), **extra}
        _save(output / "status.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        # @details Bound fitting and all per-opportunity predictions to one native
        # worker. This prevents tiny histograms from oversubscribing the machine.
        with threadpool_limits(limits=1):
            status("training_value_models")
            training = train_opportunity_value(
                events, output / "models",
                expected_manifest_sha256=input_hashes[str(manifest)],
                expected_events_sha256=input_hashes[str(events / "events.npz")],
            )
            _save(output / "training.json", training)
            trained = training["candidates"]
            if len(trained) != len(REGRESSOR_CONFIGS) * len(MARGINS_USD):
                raise ValueError("Trainer did not return exactly the predeclared policy grid.")
            status("validation_baselines")
            policies = {"flat": SkipPolicy()}
            policies.update({f"threshold_{int(value)}": GapThresholdPolicy(value)
                             for value in THRESHOLDS_BPS})
            baseline = evaluate_opportunity_policies(
                manifest, "validation", policies, execution, opportunity,
                output_dir=output / "validation" / "baselines",
            )
            _save(output / "validation" / "baselines.json", baseline)
            chosen_baseline = choose_candidate([
                {"name": name, "aggregate": result["aggregate"],
                 "threshold_bps": float(name.removeprefix("threshold_"))}
                for name, result in baseline["policies"].items() if name != "flat"
            ])
            # @details Neural checkpoints are reused unchanged, not trained on
            # the new fee or on validation. Their saved failures at 9 bps remain.
            declared = [
                {**candidate, "kind": "expected_value", "stage": "expected_value", "seed": None}
                for candidate in trained
            ] + archived
            candidates = []
            for candidate in declared:
                name, directory = candidate["id"], Path(candidate["directory"]).resolve()
                if candidate["kind"] == "expected_value" and not directory.is_relative_to(output / "models"):
                    raise ValueError("Trainer returned a model outside the current study.")
                model_file = "model.pkl" if candidate["kind"] == "expected_value" else "policy.zip"
                if _sha(directory / model_file) != candidate["model_sha256"]:
                    raise ValueError("Trained model checksum differs from its declaration.")
                status("validating_candidate", candidate=name)
                policy = TrainingCostPolicy(
                    _load_candidate(candidate, input_hashes[str(events / "events.npz")]),
                    PRIMARY_FEE_BPS,
                )
                report = evaluate_opportunity_policies(
                    manifest, "validation", {name: policy}, execution, opportunity,
                    output_dir=output / "validation" / name,
                )
                _save(output / "validation" / name / "comparison.json", report)
                record = {
                    "name": name, "model_dir": str(directory),
                    "kind": candidate["kind"], "stage": candidate["stage"], "seed": candidate["seed"],
                    "model_sha256": candidate["model_sha256"],
                    "margin_usd": candidate["margin_usd"],
                    "regressor_config": candidate["regressor_config"],
                    "aggregate": report["policies"][name]["aggregate"],
                }
                candidates.append(record)
                _save(output / "candidate_results.json", candidates)
                status("candidate_complete", candidate=name, validation=record["aggregate"])

            selected = choose_candidate(candidates)
            positive = selected["aggregate"]["net_pnl"] > 0 and selected["aggregate"]["trade_count"] > 0
            boundary_files = {**input_hashes, **_model_files(selected["model_dir"]),
                              str(output / "protocol.json"): _sha(output / "protocol.json")}
            selection = {
                "frozen_at_utc": _now(), "selection_split": "validation",
                "selected": selected, "baseline": chosen_baseline, "candidates": candidates,
                "validation_beats_flat": positive,
                "protocol_sha256": _sha(output / "protocol.json"),
                "boundary_hashes": {"artifacts": boundary_files,
                                    "sources": protocol["source_fingerprint"]},
            }
            _save(output / "selection.json", selection)
            selection_hash = _sha(output / "selection.json")
            summary = {
                "protocol": protocol, "selection": selection,
                "fresh_test_evaluated": False, "fresh_test_access_started": False,
                "fresh_test": None, "fee_stress": None,
                "conclusion": "No learned validation candidate beat cash; fresh test remains unused.",
            }
            if positive:
                # @details This is the only fresh-market access boundary. Persist
                # and verify model, settings, source, event, and manifest identities
                # before opening books. Fee stress changes execution cost only.
                for key, fee, config, folder in (
                    ("fresh_test", PRIMARY_FEE_BPS, execution, "fresh_test"),
                    ("fee_stress", TRAINING_FEE_BPS, stress_execution, "fee_stress"),
                ):
                    verify_frozen_boundary(output / "selection.json", selection_hash)
                    status(key, candidate=selected["name"], selection_sha256=selection_hash)
                    frozen_policy = TrainingCostPolicy(
                        _load_candidate(
                            {**selected, "directory": selected["model_dir"]},
                            input_hashes[str(events / "events.npz")],
                        ), fee,
                    )
                    summary["fresh_test_access_started"] = True
                    _save(output / "summary.partial.json", summary)
                    result = evaluate_opportunity_policies(
                        fresh_manifest, "test",
                        {"flat": SkipPolicy(),
                         "threshold": GapThresholdPolicy(chosen_baseline["threshold_bps"]),
                         "learned": frozen_policy},
                        config, opportunity, output_dir=output / folder,
                    )
                    _save(output / folder / "comparison.json", result)
                    summary[key] = result
                    summary["fresh_test_evaluated"] = True
                    _save(output / "summary.partial.json", summary)
                learned = summary["fresh_test"]["policies"]["learned"]["aggregate"]
                summary["conclusion"] = (
                    f"Frozen {selected['kind']} policy was net positive on the fresh day at the stated fee; one day and baseline comparison limit any edge claim."
                    if learned["net_pnl"] > 0 and learned["trade_count"] > 0
                    else f"Frozen {selected['kind']} policy did not establish a positive fresh-day result."
                )
            summary["elapsed_seconds"] = time.perf_counter() - started
            _save(output / "summary.json", summary)
            _save(output / "status.json", {
                "status": "completed", "stage": "completed", "updated_at_utc": _now(),
                "fresh_test_evaluated": summary["fresh_test_evaluated"],
                "selected_candidate": selected["name"], "conclusion": summary["conclusion"],
            })
            print(json.dumps({"status": "completed", "conclusion": summary["conclusion"]}), flush=True)
            return summary
    except Exception as error:
        # @details Never retry with different parameters or erase a partial model.
        # A saved partial fresh result records that its day has already been spent.
        partial = _read(output / "summary.partial.json") if (output / "summary.partial.json").exists() else {}
        _save(output / "status.json", {
            "status": "failed", "updated_at_utc": _now(),
            "fresh_test_evaluated": partial.get("fresh_test_evaluated", False),
            "fresh_test_access_started": partial.get("fresh_test_access_started", False),
            "error": {"type": type(error).__name__, "message": str(error)},
        })
        raise


def main(argv=None):
    """@brief Run one explicitly located value study without resuming or overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--prior-ppo-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_opportunity_value_experiment(
        args.manifest, args.fresh_manifest, args.events, args.prior_ppo_run, args.output,
    )


if __name__ == "__main__":
    main()
