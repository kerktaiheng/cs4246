"""@file experiment.py
@brief Predeclared, restartable offline PPO experiments with protected test selection.

@details Training and threshold tuning use their chronological splits exclusively.
All PPO seeds, the single conditional entropy retry, and cost scenarios are fixed
before training. Selection is saved before any test archive is opened. Resuming
reuses only compatible complete models and frozen results; it never tunes on test
returns. Test access requires an explicit --evaluate-test invocation.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterator, Sequence
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from latency_arb.agent.policy import load_policy
from latency_arb.agent.train import TrainingConfig, train_policy
from latency_arb.baseline import FlatPolicy, ThresholdPolicy
from latency_arb.data.schema import ReplayEpisode, load_manifest_entry, read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.evaluate import compare_policies, cost_sweep, fee_break_even


@dataclass(frozen=True)
class ExperimentConfig:
    """@brief Predeclared candidate search and sensitivity controls.

    @details evaluate_test and skip_sensitivity control which frozen stages may
    execute; they do not change the analytical protocol. This permits a later
    explicit test invocation after a completed validation-only run.
    """

    steps: int = 1_000_000
    seeds: tuple[int, ...] = (7, 17, 27)
    n_steps: int = 4096
    batch_size: int = 256
    epochs: int = 5
    gamma: float = 0.999
    decision_interval_ms: float = 1000.0
    max_holding_ms: float = 30_000.0
    thresholds: tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 100.0)
    fee_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 3.5)
    latency_grid: tuple[float, ...] = (150.0, 300.0)
    spread_grid: tuple[float, ...] = (1.0, 1.5)
    evaluate_test: bool = False
    skip_sensitivity: bool = False

    def validate(self) -> None:
        """@brief Validate the entire declared search before creating artifacts."""
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct.")
        for seed in self.seeds:
            self.training(seed).validate()
        if not np.isfinite([self.decision_interval_ms, self.max_holding_ms]).all():
            raise ValueError("Decision cadence and holding limit must be finite.")
        if self.decision_interval_ms < 0 or self.max_holding_ms <= 0:
            raise ValueError("Decision cadence must be nonnegative and holding positive.")
        for threshold in self.thresholds:
            ThresholdPolicy(threshold)
        for name in ("thresholds", "fee_grid", "latency_grid", "spread_grid"):
            values = getattr(self, name)
            if not values or not np.isfinite(values).all() or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct finite values.")
        if min(self.fee_grid) < 0 or max(self.fee_grid) >= 10_000:
            raise ValueError("fee_grid values must be in [0, 10000).")
        if min(self.latency_grid) < 0 or max(self.latency_grid) > self.max_holding_ms:
            raise ValueError("latency_grid must fit within max_holding_ms.")
        if min(self.spread_grid) < 1:
            raise ValueError("spread_grid cannot improve the displayed spread.")

    def training(self, seed: int, entropy: float = 0.01) -> TrainingConfig:
        """@brief Build the identical PPO configuration except declared seed/entropy."""
        return TrainingConfig(
            total_timesteps=self.steps, seed=seed, n_steps=self.n_steps,
            batch_size=self.batch_size, n_epochs=self.epochs, gamma=self.gamma,
            ent_coef=entropy,
        )


class LazyEpisodeSequence(Sequence[ReplayEpisode]):
    """@brief Reiterable, chronological evaluation split with a small archive cache.

    @details Iterating again replays the same declared split in the same order.
    Only selected archives are opened, and every cache miss verifies their hashes.
    Integer lookup is lazy; an explicit slice materializes only that requested slice.
    """

    def __init__(self, manifest: str | Path, split: str, cache_size: int = 2) -> None:
        """@brief Read split metadata without loading any book arrays."""
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer.")
        self.manifest = Path(manifest).resolve()
        _, self.entries = read_manifest(self.manifest, split=split)
        self.split = split
        self.cache_size = cache_size
        self._cache: OrderedDict[int, ReplayEpisode] = OrderedDict()

    def __len__(self) -> int:
        """@brief Return selected archive count using metadata alone."""
        return len(self.entries)

    def __getitem__(self, index):
        """@brief Open one verified archive and evict the least recently used entry."""
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Episode indices must be integers or slices.")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._cache:
            self._cache[index] = load_manifest_entry(self.manifest, self.entries[index])
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._cache[index]

    def __iter__(self) -> Iterator[ReplayEpisode]:
        """@brief Yield chronological archives without retaining the complete split."""
        for index in range(len(self)):
            yield self[index]

    def clear(self) -> None:
        """@brief Release retained books between experiment phases."""
        self._cache.clear()


def _utc_now() -> str:
    """@brief Produce an unambiguous UTC timestamp for local audit artifacts."""
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    """@brief Serialize the analytical protocol consistently for compatibility hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_file(path: Path) -> str:
    """@brief Hash local artifacts in bounded memory without exposing file contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _implementation_fingerprint(source_root: Path | None = None) -> dict[str, Any]:
    """@brief Fingerprint economic/data/training source independently of report styling.

    @param source_root Optional repository root used for isolated verification.
    @return Per-file SHA256 hashes and a deterministic aggregate source digest.
    @details Hash relative filenames and LF-normalized bytes so the same checkout
    has one identity across platforms. All Python files under env and agent are
    included; report.py is deliberately excluded because cosmetic reports cannot
    change model selection or execution economics.
    """
    root = source_root or Path(__file__).resolve().parents[1]
    relative_files = {
        "common/__init__.py", "common/gym_base.py", "latency_arb/__init__.py",
        "latency_arb/data/__init__.py", "latency_arb/data/schema.py",
        "latency_arb/data/generate_episodes.py", "latency_arb/data/recorded.py",
        "latency_arb/baseline.py", "latency_arb/evaluate.py",
        "latency_arb/experiment.py",
    }
    for package in ("latency_arb/agent", "latency_arb/env"):
        relative_files.update(
            path.relative_to(root).as_posix()
            for path in (root / package).rglob("*.py")
        )
    hashes = {
        relative: hashlib.sha256(
            (root / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for relative in sorted(relative_files)
    }
    return {
        "source_sha256": hashlib.sha256(_canonical(hashes)).hexdigest(),
        "files": hashes,
    }


def _git_commit() -> str | None:
    """@brief Read optional local commit provenance without affecting compatibility.

    @details Git HEAD is deliberately outside the protocol fingerprint: committing
    report-only edits may change HEAD while the economic implementation remains
    identical. Uncommitted core changes are still detected by the source hashes.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    if result.returncode == 0 and len(value) in (40, 64) and all(
        char in "0123456789abcdef" for char in value
    ):
        return value
    return None


def _write_json(path: Path, value: Any) -> None:
    """@brief Atomically replace a complete local JSON stage artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """@brief Read a completed object artifact; malformed partial files fail clearly."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Incomplete or invalid JSON artifact: {path.name}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object in artifact: {path.name}")
    return result


def _compatible_result(path: Path, protocol_hash: str, **identities: str) -> dict | None:
    """@brief Reuse a stage only when its immutable protocol and inputs agree."""
    if not path.exists():
        return None
    result = _read_json(path)
    for key, expected in {"protocol_sha256": protocol_hash, **identities}.items():
        if result.get(key) != expected:
            raise ValueError(f"Incompatible completed stage {path.name}: {key} differs.")
    return result


def _ensure_model(
    manifest: Path, directory: Path, training: TrainingConfig, execution: EnvConfig,
    manifest_hash: str,
):
    """@brief Train a fresh candidate or load an exact compatible completed run.

    @details A nonempty directory lacking training.json is deliberately not
    overwritten or treated as resumable. Partial PPO optimizer recovery is outside
    this protocol; the existing files remain available for inspection.
    """
    metadata_path = directory / "training.json"
    if directory.exists() and any(directory.iterdir()):
        if not metadata_path.exists():
            raise ValueError(f"Partial model directory cannot be resumed: {directory}")
        metadata = _read_json(metadata_path)
        expected = {
            "manifest_sha256": manifest_hash,
            "training_config": asdict(training),
            "environment_config": asdict(replace(execution, log_history=False)),
            "training_split": "train",
            "held_out_data_loaded": False,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Incompatible completed model {directory.name}: {key} differs.")
    else:
        metadata = train_policy(manifest, directory, training, execution)
    return load_policy(directory), metadata


def run_experiment(
    manifest_path: str | Path, output_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """@brief Train candidates, freeze validation selection, and optionally test once.

    @param manifest_path Checked chronological train/validation/test manifest.
    @param output_dir Dedicated experiment directory; compatible completed stages resume.
    @param config Predeclared seeds, budgets, and explicit test execution permission.
    @return Portable summary including daily variation, abstention, and cost sensitivity.
    @throws ValueError For incompatible protocols or incomplete model directories.
    @details Reinvocation may finish an interrupted frozen test/sensitivity stage,
    but never changes the selected policy or learns from its test outcomes.
    """
    settings = config or ExperimentConfig()
    settings.validate()
    manifest = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    manifest_hash = _hash_file(manifest)
    document, train_entries = read_manifest(manifest, split="train")
    execution = EnvConfig(
        fee_bps=3.5, latency_ms=150.0, slippage_bps=0.1,
        position_size_btc=0.001, decision_interval_ms=settings.decision_interval_ms,
        max_holding_ms=settings.max_holding_ms, reward_scale=100.0, log_history=False,
    )
    evaluation = replace(execution, reward_scale=1.0, log_history=True)

    ## @details Execution permission is separate from the analytical protocol:
    # a later --evaluate-test can authorize the already frozen final comparison.
    declared = asdict(settings)
    declared.pop("evaluate_test")
    declared.pop("skip_sensitivity")
    protocol = {
        "version": 1, "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "implementation": _implementation_fingerprint(),
        "config": declared, "training_environment": asdict(execution),
        "evaluation_environment": asdict(evaluation),
        "initial_candidates": [
            {"name": f"ppo_seed{seed}", "training": asdict(settings.training(seed))}
            for seed in settings.seeds
        ],
        "conditional_retry": {
            "condition": "all initial PPO candidates execute zero trades on validation",
            "name": "ppo_seed7_entropy005", "training": asdict(settings.training(7, 0.05)),
            "maximum_retries": 1,
        },
        "ppo_selection": "validation net_pnl, then fewer trades, then declared candidate order",
        "threshold_selection": "validation net_pnl, then fewer trades, then larger threshold",
        "flat_control": "An independent always-flat policy; high thresholds do not imply flat.",
        "test_rule": "Freeze selection before test access; no test-driven changes.",
    }
    protocol = json.loads(_canonical(protocol))
    protocol_hash = hashlib.sha256(_canonical(protocol)).hexdigest()
    plan_path = destination / "experiment.json"
    if plan_path.exists():
        previous = _read_json(plan_path)
        if previous.get("protocol_sha256") != protocol_hash or previous.get("protocol") != protocol:
            raise ValueError("Experiment protocol differs; use a new output directory.")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Nonempty experiment directory lacks a complete experiment.json.")
        _write_json(plan_path, {
            "created_at_utc": _utc_now(), "protocol_sha256": protocol_hash, "protocol": protocol,
            "git_commit": _git_commit(),
        })

    plan_provenance = _read_json(plan_path)
    stage = "initializing"

    def status(state: str, next_stage: str, **extra: Any) -> None:
        """@brief Save and print a compact stage transition without credential output."""
        nonlocal stage
        stage = next_stage
        record = {
            "status": state, "stage": stage, "updated_at_utc": _utc_now(),
            "protocol_sha256": protocol_hash, "evaluate_test": settings.evaluate_test,
            "skip_sensitivity": settings.skip_sensitivity, **extra,
        }
        _write_json(destination / "status.json", record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    validation: LazyEpisodeSequence | None = None
    testing: LazyEpisodeSequence | None = None
    try:
        ## @details A fully completed compatible request returns its existing result
        # without reopening test archives. Missing optional sensitivity can proceed
        # later using the same predeclared scenario grid and frozen selection.
        prior_summary = _compatible_result(destination / "summary.json", protocol_hash)
        if prior_summary is not None:
            enough_test = not settings.evaluate_test or prior_summary.get("test_evaluated", False)
            enough_sensitivity = (
                not settings.evaluate_test or settings.skip_sensitivity
                or prior_summary.get("sensitivity_completed", False)
            )
            if enough_test and enough_sensitivity:
                ## @details Reusing a report must not silently bless altered model
                # metadata or a changed selection. Check trusted artifact hashes
                # and declared settings without reopening any replay archives.
                frozen_path = destination / "selection.json"
                if _hash_file(frozen_path) != prior_summary.get("selection_sha256"):
                    raise ValueError("Frozen selection checksum differs from completed summary.")
                declared_training = {
                    item["name"]: TrainingConfig(**item["training"])
                    for item in protocol["initial_candidates"]
                }
                retry = protocol["conditional_retry"]
                declared_training[retry["name"]] = TrainingConfig(**retry["training"])
                for item in prior_summary["candidates"]:
                    name = item["name"]
                    if name not in declared_training:
                        raise ValueError("Completed summary contains an undeclared candidate.")
                    model_dir = destination / "models" / name
                    if not (model_dir / "training.json").exists():
                        raise ValueError(f"Completed model artifact is missing: {name}")
                    _ensure_model(
                        manifest, model_dir, declared_training[name], execution, manifest_hash
                    )
                status("completed", "completed", reused_completed_experiment=True)
                return prior_summary

        status("running", "validation_baselines")
        validation = LazyEpisodeSequence(manifest, "validation")
        baseline_path = destination / "validation" / "baselines.json"
        baseline_record = _compatible_result(baseline_path, protocol_hash)
        if baseline_record is None:
            thresholds = sorted(settings.thresholds)
            baseline_policies = {"flat": FlatPolicy(), **{
                f"threshold_{index}": ThresholdPolicy(threshold)
                for index, threshold in enumerate(thresholds)
            }}
            baseline_report = compare_policies(validation, baseline_policies, evaluation, seed=7)
            candidates = [
                {"policy_name": f"threshold_{index}", "entry_threshold_bps": threshold,
                 **baseline_report["policies"][f"threshold_{index}"]["aggregate"]}
                for index, threshold in enumerate(thresholds)
            ]
            best = max(candidates, key=lambda row: (
                row["net_pnl"], -row["trade_count"], row["entry_threshold_bps"]
            ))
            baseline_record = {
                "protocol_sha256": protocol_hash, "report": baseline_report,
                "selection": {
                    "selection_split": "validation",
                    "entry_threshold_bps": best["entry_threshold_bps"],
                    "policy_name": best["policy_name"], "candidates": candidates,
                    "flat_control": baseline_report["policies"]["flat"]["aggregate"],
                    "selection_objective": protocol["threshold_selection"],
                },
            }
            _write_json(baseline_path, baseline_record)
        status("running", "validation_baselines_complete")

        candidates: list[dict[str, Any]] = []

        def candidate(name: str, training: TrainingConfig) -> None:
            """@brief Fit or reuse one declared model, then evaluate validation only."""
            status("running", "training_candidate", candidate=name,
                   candidates_completed=len(candidates))
            model_dir = destination / "models" / name
            policy, metadata = _ensure_model(
                manifest, model_dir, training, execution, manifest_hash
            )
            status("running", "validating_candidate", candidate=name)
            model_hash = metadata["artifact_sha256"]["policy.zip"]
            result_path = destination / "validation" / f"{name}.json"
            record = _compatible_result(result_path, protocol_hash, model_sha256=model_hash)
            if record is None:
                report = compare_policies(validation, {"ppo": policy}, evaluation, seed=7)
                record = {
                    "protocol_sha256": protocol_hash, "name": name,
                    "model_sha256": model_hash, "model_dir": str(model_dir),
                    "training_config": asdict(training), "report": report,
                }
                _write_json(result_path, record)
            candidates.append(record)
            status("running", "candidate_complete", candidate=name,
                   candidates_completed=len(candidates),
                   validation=record["report"]["policies"]["ppo"]["aggregate"])

        for seed in settings.seeds:
            candidate(f"ppo_seed{seed}", settings.training(seed))
        retry_triggered = all(
            item["report"]["policies"]["ppo"]["aggregate"]["trade_count"] == 0
            for item in candidates
        )
        if retry_triggered:
            candidate("ppo_seed7_entropy005", settings.training(7, 0.05))
        selected = max(candidates, key=lambda item: (
            item["report"]["policies"]["ppo"]["aggregate"]["net_pnl"],
            -item["report"]["policies"]["ppo"]["aggregate"]["trade_count"],
        ))
        selection_payload = {
            "protocol_sha256": protocol_hash, "selection_split": "validation",
            "selected_candidate": selected["name"], "model_dir": selected["model_dir"],
            "model_sha256": selected["model_sha256"],
            "entry_threshold_bps": baseline_record["selection"]["entry_threshold_bps"],
            "retry_triggered": retry_triggered,
            "ppo_selection_objective": protocol["ppo_selection"],
            "threshold_selection": baseline_record["selection"],
            "candidates": [
                {"name": item["name"], "model_sha256": item["model_sha256"],
                 "training_config": item["training_config"],
                 "aggregate": item["report"]["policies"]["ppo"]["aggregate"]}
                for item in candidates
            ],
        }

        ## @details This write is the test-access boundary. Existing selection is
        # immutable; even compatible resumption refuses altered validation results.
        selection_path = destination / "selection.json"
        if selection_path.exists():
            frozen = _read_json(selection_path)
            if {key: value for key, value in frozen.items() if key != "frozen_at_utc"} != selection_payload:
                raise ValueError("Frozen selection disagrees; refusing test-driven reselection.")
        else:
            _write_json(selection_path, {"frozen_at_utc": _utc_now(), **selection_payload})
        selection_hash = _hash_file(selection_path)
        status("running", "selection_frozen", selected_candidate=selected["name"])
        chosen_threshold_name = baseline_record["selection"]["policy_name"]
        validation_comparison = {
            "config": asdict(evaluation), "seed": 7, "split": "validation",
            "policies": {
                "flat": baseline_record["report"]["policies"]["flat"],
                "threshold": baseline_record["report"]["policies"][chosen_threshold_name],
                "ppo": selected["report"]["policies"]["ppo"],
            },
        }
        _write_json(destination / "validation" / "comparison.json", validation_comparison)
        validation.clear()

        test_report, sensitivity, break_even = None, None, None
        if settings.evaluate_test:
            ## @details Only frozen models and thresholds reach this branch.
            # No later result is fed back to training, candidate selection, or costs.
            policies = {
                "flat": FlatPolicy(),
                "threshold": ThresholdPolicy(selection_payload["entry_threshold_bps"]),
                "ppo": load_policy(selected["model_dir"]),
            }
            status("running", "test_comparison", selection_sha256=selection_hash)
            test_record_path = destination / "test_result.json"
            test_record = _compatible_result(
                test_record_path, protocol_hash, selection_sha256=selection_hash
            )
            if test_record is None:
                testing = LazyEpisodeSequence(manifest, "test")
                test_report = compare_policies(
                    testing, policies, evaluation, output_dir=destination / "test", seed=7
                )
                test_report["split"] = "test"
                _write_json(destination / "test" / "comparison.json", test_report)
                test_record = {
                    "protocol_sha256": protocol_hash, "selection_sha256": selection_hash,
                    "report": test_report,
                }
                _write_json(test_record_path, test_record)
            test_report = test_record["report"]
            status("running", "test_comparison_complete")

            if not settings.skip_sensitivity:
                sensitivity = []
                scenarios = [
                    (fee, latency, spread)
                    for fee in settings.fee_grid for latency in settings.latency_grid
                    for spread in settings.spread_grid
                ]
                for index, (fee, latency, spread) in enumerate(scenarios):
                    status("running", "cost_sensitivity", scenario=index + 1,
                           scenario_count=len(scenarios), fee_bps=fee,
                           latency_ms=latency, spread_multiplier=spread)
                    scenario_path = destination / "sensitivity" / f"scenario_{index:03d}.json"
                    scenario_record = _compatible_result(
                        scenario_path, protocol_hash, selection_sha256=selection_hash
                    )
                    if scenario_record is None:
                        if testing is None:
                            testing = LazyEpisodeSequence(manifest, "test")
                        rows = cost_sweep(
                            testing, policies, evaluation, [fee], [latency], [spread], seed=7
                        )
                        scenario_record = {
                            "protocol_sha256": protocol_hash,
                            "selection_sha256": selection_hash, "rows": rows,
                        }
                        _write_json(scenario_path, scenario_record)
                    sensitivity.extend(scenario_record["rows"])
                break_even = fee_break_even(sensitivity)
                _write_json(destination / "sensitivity.json", sensitivity)
                _write_json(destination / "fee_break_even.json", break_even)
                with (destination / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(sensitivity[0]))
                    writer.writeheader()
                    writer.writerows(sensitivity)
                status("running", "cost_sensitivity_complete")

        ## @details Preserve the evaluator's economic, daily-variation, and abstention
        # measures verbatim instead of inventing an orchestration-specific metric.
        summary = {
            "protocol_sha256": protocol_hash, "selection_sha256": selection_hash,
            "completed_at_utc": _utc_now(), "manifest_sha256": manifest_hash,
            "implementation_source_sha256": protocol["implementation"]["source_sha256"],
            "git_commit": plan_provenance.get("git_commit"),
            "synthetic": all(bool(entry.get("synthetic", False)) for entry in train_entries),
            "selected_candidate": selected["name"], "selected_model_dir": selected["model_dir"],
            "threshold_bps": selection_payload["entry_threshold_bps"],
            "retry_triggered": retry_triggered,
            "selected_ppo_zero_validation_trades": (
                validation_comparison["policies"]["ppo"]["aggregate"]["trade_count"] == 0
            ),
            "candidates": selection_payload["candidates"],
            "validation": validation_comparison, "test": test_report,
            "test_evaluated": test_report is not None,
            "sensitivity_completed": sensitivity is not None,
            "sensitivity": sensitivity, "fee_break_even": break_even,
            "interpretation": (
                "Synthetic data test the pipeline only. A zero-trade policy does not "
                "demonstrate a profitable executable edge. Test outcomes never alter selection."
            ),
        }
        _write_json(destination / "summary.json", summary)
        status("completed", "completed", selected_candidate=selected["name"],
               test_evaluated=summary["test_evaluated"],
               sensitivity_completed=summary["sensitivity_completed"])
        return summary
    except Exception as error:
        ## @details Persist a concise failure without stack traces or environment
        # variables; preserve all candidate files and the frozen selection for review.
        status("failed", stage, error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if validation is not None:
            validation.clear()
        if testing is not None:
            testing.clear()
        torch.set_num_threads(previous_threads)


def _seeds(text: str) -> tuple[int, ...]:
    """@brief Parse the predeclared comma-separated integer seed list."""
    try:
        return tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from error


def main(argv: Sequence[str] | None = None) -> int:
    """@brief Run a protected experiment; test access is disabled unless requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=_seeds, default=(7, 17, 27))
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--decision-interval-ms", type=float, default=1000.0)
    parser.add_argument("--max-holding-ms", type=float, default=30_000.0)
    parser.add_argument("--skip-sensitivity", action="store_true")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--evaluate-test", action="store_true")
    test_group.add_argument("--no-test", dest="evaluate_test", action="store_false")
    parser.set_defaults(evaluate_test=False)
    args = parser.parse_args(argv)
    summary = run_experiment(
        args.manifest, args.output,
        ExperimentConfig(
            steps=args.steps, seeds=args.seeds, n_steps=args.n_steps,
            batch_size=args.batch_size, epochs=args.epochs,
            decision_interval_ms=args.decision_interval_ms,
            max_holding_ms=args.max_holding_ms, evaluate_test=args.evaluate_test,
            skip_sensitivity=args.skip_sensitivity,
        ),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected_candidate": summary["selected_candidate"],
        "test_evaluated": summary["test_evaluated"],
        "sensitivity_completed": summary["sensitivity_completed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
