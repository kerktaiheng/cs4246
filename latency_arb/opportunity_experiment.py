"""@file opportunity_experiment.py
@brief Run a protected entry-selection study with fresh, unused final dates.
@details The earlier experiment is preserved. Development uses the old training
and validation splits only; previously examined test dates never enter fitting or
selection. A fresh test is opened only after a profitable validation candidate is
selected and its exact weights and settings are saved. An edge is not guaranteed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from latency_arb.agent.opportunity_train import (
    prepare_training_events, train_opportunity_policy, load_opportunity_policy,
)
from latency_arb.data.schema import read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import OpportunityConfig
from latency_arb.opportunity_evaluate import (
    GapThresholdPolicy, SkipPolicy, evaluate_opportunity_policies,
)


def _now():
    """@brief Return explicit UTC audit time rather than a host-local timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    """@brief Fingerprint an artifact in bounded memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _save(path, value):
    """@brief Atomically persist one complete, strict JSON stage artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose_candidate(candidates):
    """@brief Select maximum validation dollars, then fewer trades, with stable ties.
    @details No test outcome or training exploration reward participates. A zero
    return is preserved as abstention and cannot trigger fresh-test consumption.
    """
    if not candidates:
        raise ValueError("No candidate validation records were provided.")
    return max(candidates, key=lambda item: (
        item["aggregate"]["net_pnl"], -item["aggregate"]["trade_count"],
    ))


def validate_fresh_dates(manifest, fresh_manifest):
    """@brief Check calendar isolation using manifest metadata, without opening books.
    @return The train, validation, and fresh test UTC dates for the saved protocol.
    @details Fresh days must be later than ALL dates in the previously examined
    manifest, including its old test split. Structural quality inspection is not
    policy evaluation and is the only permitted access before frozen selection.
    """
    document, _ = read_manifest(manifest)
    fresh, entries = read_manifest(fresh_manifest, split="test")
    previous_days = sorted({entry["day"] for entry in document["episodes"]})
    fresh_days = sorted({entry["day"] for entry in entries})
    if fresh_days[0] <= previous_days[-1]:
        raise ValueError("Fresh holdout must follow every previously examined day.")
    return {
        "train": sorted({entry["day"] for entry in document["episodes"] if entry["split"] == "train"}),
        "validation": sorted({entry["day"] for entry in document["episodes"] if entry["split"] == "validation"}),
        "fresh_test": fresh_days,
        "previously_examined_test_excluded": sorted({entry["day"] for entry in document["episodes"] if entry["split"] == "test"}),
    }


def source_fingerprint():
    """@brief Bind results to normalized bytes of every economic and learning module."""
    root = Path(__file__).resolve().parents[1]
    names = (
        "common/gym_base.py", "latency_arb/data/schema.py",
        "latency_arb/env/latency_sim.py", "latency_arb/opportunity.py",
        "latency_arb/agent/opportunity_train.py",
        "latency_arb/opportunity_evaluate.py", "latency_arb/opportunity_experiment.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in names}


def run_opportunity_experiment(manifest, fresh_manifest, output_dir, *,
                               seeds=(7, 17, 27), steps=65_536, warm_epochs=50):
    """@brief Prepare train targets, fit declared candidates, freeze selection, then test.
    @param manifest Existing 22/4/4-day manifest; only train/validation are replayed.
    @param fresh_manifest Later, previously unevaluated complete UTC day(s).
    @param output_dir New experiment directory; existing results are never overwritten.
    @param seeds Three independent neural initializations by default.
    @param steps Number of genuine PPO opportunity decisions requested per seed.
    @param warm_epochs Fixed cost-regret-weighted actor initialization passes.
    @return Full study summary, or a development-only negative result if no learned
        validation policy beats the always-flat control.
    @details Positive validation is necessary to spend the fresh test, but it is
    not proof of an edge. All selected policies are frozen before that boundary.
    """
    manifest, fresh_manifest = Path(manifest).resolve(), Path(fresh_manifest).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; preserve prior experiments.")
    if not seeds or len(set(seeds)) != len(seeds) or steps < 1 or warm_epochs < 1:
        raise ValueError("Distinct seeds and positive training budgets are required.")
    date_split = validate_fresh_dates(manifest, fresh_manifest)
    execution = EnvConfig(fee_bps=4.5, latency_ms=150, decision_interval_ms=1000,
                          slippage_bps=0.1, position_size_btc=0.001, max_holding_ms=30_000,
                          reward_scale=1.0, log_history=True)
    opportunity = OpportunityConfig(min_gap_bps=6.0, exit_gap_bps=0.5, min_remaining_ms=31_000)
    thresholds = (6.0, 7.0, 9.0, 10.0, 12.0, 15.0, 20.0)
    protocol = {
        "version": 1, "created_at_utc": _now(), "manifest": str(manifest),
        "manifest_sha256": _sha(manifest), "fresh_manifest": str(fresh_manifest),
        "fresh_manifest_sha256": _sha(fresh_manifest), "dates": date_split,
        "execution": asdict(execution), "opportunity": asdict(opportunity),
        "seeds": list(seeds), "ppo_steps_per_seed": steps, "warm_epochs": warm_epochs,
        "candidate_stages": ["warmstart", "ppo"],
        "baseline_thresholds_bps": list(thresholds),
        "selection": "Maximum validation net USD; fewer trades on ties; stable declared ordering.",
        "fresh_test_gate": "Learned validation net PnL must exceed zero and execute at least one trade.",
        "reward": "Unchanged net marked equity differences; gamma=1 per opportunity; fixed positive scaling only.",
        "source_fingerprint": source_fingerprint(),
    }
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "protocol.json", protocol)
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def status(stage, **extra):
        """@brief Record progress without implying that uncompleted results exist."""
        value = {"stage": stage, "status": "running", "updated_at_utc": _now(), **extra}
        _save(output / "status.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        # @details Training targets are generated strictly from the training split.
        # They include losing and future-rejected entries rather than an oracle filter.
        status("preparing_training_events")
        event_data = prepare_training_events(
            manifest, output / "events", execution=execution, opportunity=opportunity,
        )
        _save(output / "event_preparation.json", event_data)
        status("validation_baselines", events=event_data["event_count"])
        baseline_policies = {"flat": SkipPolicy()}
        baseline_policies.update({f"threshold_{int(value)}": GapThresholdPolicy(value)
                                  for value in thresholds})
        baseline = evaluate_opportunity_policies(
            manifest, "validation", baseline_policies, execution, opportunity,
            output_dir=output / "validation" / "baselines",
        )
        threshold_records = [
            {"name": name, "aggregate": result["aggregate"],
             "threshold_bps": float(name.removeprefix("threshold_"))}
            for name, result in baseline["policies"].items() if name != "flat"
        ]
        chosen_baseline = choose_candidate(threshold_records)
        candidates = []

        # @details Both warm-start and PPO checkpoints are predeclared candidates.
        # If imitation wins, the report must not credit PPO fine-tuning for that gain.
        for seed in seeds:
            status("training_candidate", seed=seed, candidates_completed=len(candidates))
            metadata = train_opportunity_policy(
                manifest, output / "events", output / "models" / f"seed{seed}",
                seed=seed, total_timesteps=steps, warm_epochs=warm_epochs,
                execution=execution, opportunity=opportunity,
            )
            _save(output / "models" / f"seed{seed}" / "run_metadata.json", metadata)
            for stage in ("warmstart", "ppo"):
                name = f"seed{seed}_{stage}"
                directory = Path(metadata[f"{stage}_dir"])
                status("validating_candidate", candidate=name)
                report = evaluate_opportunity_policies(
                    manifest, "validation", {name: load_opportunity_policy(directory)},
                    execution, opportunity, output_dir=output / "validation" / name,
                )
                record = {
                    "name": name, "stage": stage, "seed": seed,
                    "model_dir": str(directory), "model_sha256": _sha(directory / "policy.zip"),
                    "aggregate": report["policies"][name]["aggregate"],
                }
                candidates.append(record)
                _save(output / "candidate_results.json", candidates)
                status("candidate_complete", candidate=name, validation=record["aggregate"])

        selected = choose_candidate(candidates)
        positive = selected["aggregate"]["net_pnl"] > 0 and selected["aggregate"]["trade_count"] > 0
        selection = {
            "frozen_at_utc": _now(), "selection_split": "validation",
            "selected": selected, "baseline": chosen_baseline, "candidates": candidates,
            "validation_beats_flat": positive, "protocol_sha256": _sha(output / "protocol.json"),
        }
        _save(output / "selection.json", selection)
        _save(output / "validation" / "baselines.json", baseline)
        summary = {
            "protocol": protocol, "selection": selection,
            "fresh_test_evaluated": False, "fresh_test": None,
            "conclusion": "No learned validation candidate beat cash; fresh test remains unused.",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if positive:
            # @details This is the sole fresh-test access boundary. Selection is
            # saved first; no outcome below is used to retrain or choose a checkpoint.
            status("selection_frozen", candidate=selected["name"])
            policies = {
                "flat": SkipPolicy(),
                "threshold": GapThresholdPolicy(chosen_baseline["threshold_bps"]),
                "learned": load_opportunity_policy(selected["model_dir"]),
            }
            status("fresh_test", candidate=selected["name"], selection_sha256=_sha(output / "selection.json"))
            summary["fresh_test"] = evaluate_opportunity_policies(
                fresh_manifest, "test", policies, execution, opportunity,
                output_dir=output / "fresh_test",
            )
            summary["fresh_test_evaluated"] = True
            learned = summary["fresh_test"]["policies"]["learned"]["aggregate"]
            summary["conclusion"] = (
                "Frozen learned policy was net positive on fresh data; sample size and baseline comparison still limit any edge claim."
                if learned["net_pnl"] > 0 and learned["trade_count"] > 0
                else "Frozen learned policy did not establish a positive fresh-data result."
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
        # @details Preserve partial models and evidence; failures never consume the
        # test by fallback or silently restart a different training specification.
        _save(output / "status.json", {
            "status": "failed", "updated_at_utc": _now(),
            "error": {"type": type(error).__name__, "message": str(error)},
        })
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    """@brief Expose one explicit study invocation with auditable training budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=65_536)
    parser.add_argument("--warm-epochs", type=int, default=50)
    parser.add_argument("--seeds", default="7,17,27")
    args = parser.parse_args(argv)
    run_opportunity_experiment(
        args.manifest, args.fresh_manifest, args.output,
        steps=args.steps, warm_epochs=args.warm_epochs,
        seeds=tuple(int(value) for value in args.seeds.split(",")),
    )


if __name__ == "__main__":
    main()
