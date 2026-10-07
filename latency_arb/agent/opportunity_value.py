"""@file opportunity_value.py
@brief Optional train-only expected-net-dollar regression comparator.

@details Two predeclared histogram boosting regressors learn directly from every
cached causal training feature and its unchanged counterfactual net-dollar target.
Each regressor yields policies with fixed margins of zero and one cent. There is
no class weighting, payoff clipping, outcome filtering, validation fitting, or PPO
reward modification. Counterfactual labels overlap and are not portfolio returns.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from latency_arb.opportunity import FEATURE_NAMES


# @details This deliberately small search space is declared before any development
# evaluation. The margin is strict: a prediction exactly at the margin skips.
REGRESSOR_CONFIGS = (
    {"id": "leaves7_min200", "max_leaf_nodes": 7, "min_samples_leaf": 200},
    {"id": "leaves15_min50", "max_leaf_nodes": 15, "min_samples_leaf": 50},
)
MARGINS_USD = (0.0, 0.01)
COMMON_SETTINGS = {
    "loss": "squared_error", "learning_rate": 0.05, "max_iter": 200,
    "l2_regularization": 1.0, "early_stopping": False, "random_state": 7,
}
_PRODUCER_FILES = (
    "latency_arb/agent/opportunity_train.py", "latency_arb/agent/policy.py",
    "latency_arb/opportunity.py", "latency_arb/env/latency_sim.py",
    "latency_arb/data/schema.py",
)


def _hash(path: Path) -> str:
    """@brief Stream a local artifact checksum without interpreting its contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _source_hashes(include_comparator: bool = False) -> dict[str, str]:
    """@brief Match the frozen event producer and optionally this comparator source.

    @details Source files, rather than any market archive or manifest, establish
    compatibility. LF normalization matches the original event producer protocol.
    """
    root = Path(__file__).resolve().parents[2]
    files = _PRODUCER_FILES + (("latency_arb/agent/opportunity_value.py",)
                               if include_comparator else ())
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in files}


def _write(path: Path, document: dict) -> None:
    """@brief Atomically publish finite, human-readable provenance within new output."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _versions() -> dict[str, str]:
    """@brief Record exact numerical and serialization dependency versions."""
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")}


def _candidate_id(configuration: dict, margin: float) -> str:
    """@brief Give the fixed zero/one-cent decisions stable, reviewable identities."""
    return configuration["id"] + ("_margin0" if margin == 0 else "_margin001")


def _settings(configuration: dict) -> dict:
    """@brief Construct a fresh numerical configuration from the declared grid."""
    return {**COMMON_SETTINGS, **{key: value for key, value in configuration.items()
                                  if key != "id"}}


def _load_events(directory: Path, *, expected_manifest_sha256: str | None = None,
                 expected_events_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """@brief Read only the cached training event arrays and their provenance JSON.

    @details The source manifest, normalization file, and all train/validation/test
    books are deliberately unnecessary. Every stored row is retained in its original
    order and dtype. These checks detect accidental source/metadata/file mismatch;
    they cannot authenticate an adversarially rewritten, internally consistent cache.
    """
    metadata = json.loads((directory / "events.json").read_text(encoding="utf-8"))
    if (metadata.get("event_version") != 1 or metadata.get("split") != "train"
            or metadata.get("future_rows_used_only_for_targets") is not True):
        raise ValueError("Expected train-only counterfactual event metadata.")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Event feature names do not match the causal feature contract.")
    if metadata.get("source_sha256") != _source_hashes():
        raise ValueError("Event producer source fingerprint mismatch.")
    manifest_hash = metadata.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("Event manifest provenance must contain a SHA-256 checksum.")
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Event manifest checksum does not match the expected study.")
    event_hash = _hash(directory / "events.npz")
    if event_hash != metadata.get("artifact_sha256", {}).get("events.npz"):
        raise ValueError("Training events checksum mismatch.")
    if expected_events_sha256 is not None and event_hash != expected_events_sha256:
        raise ValueError("Training events checksum does not match the expected study.")
    entries = metadata.get("training_entries")
    if (not isinstance(entries, list) or not entries
            or any(not isinstance(entry, dict) or entry.get("split") != "train"
                   for entry in entries)):
        raise ValueError("Event provenance must contain exclusively training entries.")
    economics = dict(metadata.get("execution_config", {}))
    economics.pop("reward_scale", None)
    economics.pop("log_history", None)
    if not economics or economics != metadata.get("economic_config"):
        raise ValueError("Event execution/economic metadata disagree.")
    if not isinstance(metadata.get("opportunity_config"), dict):
        raise ValueError("Event opportunity metadata is missing.")

    # @details Only numeric arrays are deserialized. Future target values never
    # determine row eligibility, sample weights, or a feature transformation.
    with np.load(directory / "events.npz", allow_pickle=False) as archive:
        features = archive["features"]
        targets = archive["net_payoffs"]
        episode_indices = archive["episode_indices"]
        candidate_indices = archive["candidate_indices"]
    count = metadata.get("event_count")
    if (type(count) is not int or count < 1
            or features.shape != (count, len(FEATURE_NAMES))
            or features.dtype != np.float32 or targets.shape != (count,)
            or targets.dtype != np.float64):
        raise ValueError("Event feature/target shape, dtype, or count mismatch.")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Event features and net-dollar targets must be finite.")
    for values in (episode_indices, candidate_indices):
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError("Event row provenance indices are invalid.")
    if (episode_indices >= len(entries)).any():
        raise ValueError("An event references an unavailable training entry.")
    eligible = sorted(set(map(int, episode_indices)))
    if (metadata.get("eligible_episode_indices") != eligible
            or metadata.get("eligible_episode_count") != len(eligible)):
        raise ValueError("Event eligible-episode metadata disagree with stored rows.")
    for field, condition in (("positive_target_count", targets > 0),
                             ("negative_target_count", targets < 0),
                             ("zero_target_count", targets == 0)):
        if metadata.get(field) != int(condition.sum()):
            raise ValueError("Event target-count metadata disagree with stored targets.")
    return features, targets, metadata


class OpportunityValuePolicy:
    """@brief Frozen expected-profit regressor with a predeclared trade/skip margin."""

    def __init__(self, model: Any, metadata: dict) -> None:
        """@brief Bind one saved regressor to its raw-feature and margin contracts."""
        self.model, self.metadata = model, metadata
        self.feature_names = tuple(metadata["feature_names"])
        self.margin_usd = float(metadata["margin_usd"])

    def expected_net_pnl(self, raw_features):
        """@brief Predict net trade dollars without refitting or changing raw features.

        @return A float for one feature vector, or an array for a feature matrix.
        @details These are supervised model estimates, not realized portfolio PnL
        or calibrated probabilities. Actual economic evaluation requires replay.
        """
        from threadpoolctl import threadpool_limits

        values = np.asarray(raw_features)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError("Expected finite raw causal opportunity features.")
        # @details A single numerical worker avoids competition with other study
        # processes and preserves the same deterministic computation on reload.
        with threadpool_limits(limits=1):
            result = self.model.predict(values)
        return float(result[0]) if single else result

    def predict(self, raw_features, deterministic: bool = True):
        """@brief Return the SB3-compatible action/state tuple using a strict USD gate.

        @param deterministic Accepted for evaluator compatibility; regression has
        no sampling branch, so both values produce the same frozen decisions.
        """
        estimates = self.expected_net_pnl(raw_features)
        actions = np.asarray(estimates > self.margin_usd, dtype=np.int64)
        return actions, None


def train_opportunity_value(
    event_data_dir: str | Path, output_dir: str | Path, *,
    expected_manifest_sha256: str | None = None,
    expected_events_sha256: str | None = None,
) -> dict[str, Any]:
    """@brief Fit exactly two fixed regressors and publish four fixed-margin policies.

    @param event_data_dir Existing trusted train-only events.json/events.npz cache.
    @param output_dir New or empty directory; nonempty outputs are preserved.
    @details No replay, archive loading, normalization fitting, train/validation
    split, internal early stopping, or evaluation happens here. Squared-error loss
    estimates conditional net dollars from unchanged natural-frequency samples.
    The two margins reuse identical weights, preventing redundant refitting.
    """
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    directory = Path(event_data_dir).resolve()
    features, targets, event_metadata = _load_events(
        directory, expected_manifest_sha256=expected_manifest_sha256,
        expected_events_sha256=expected_events_sha256,
    )
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ImportError("Optional comparator requires scikit-learn in its isolated runtime.") from error
    versions = _versions()
    provenance = {
        "artifact_version": 1, "policy_family": "opportunity_expected_net_value",
        "training_split": "train", "held_out_data_loaded": False,
        "feature_names": list(FEATURE_NAMES), "event_count": len(targets),
        "events_sha256": _hash(directory / "events.npz"),
        "events_metadata_sha256": _hash(directory / "events.json"),
        "manifest_sha256": event_metadata["manifest_sha256"],
        "execution_config": event_metadata["execution_config"],
        "economic_config": event_metadata["economic_config"],
        "opportunity_config": event_metadata["opportunity_config"],
        "source_sha256": _source_hashes(include_comparator=True),
        "producer_source_sha256": event_metadata["source_sha256"],
        "versions": versions,
        "training_rule": "All cached rows, original order/dtypes, raw features and unchanged net-dollar targets.",
        "sample_weighting": "none", "target_transformation": "none",
        "normalization": "none", "internal_validation": False,
        "counterfactual_warning": "Overlapping labels are not a realizable portfolio.",
    }
    # @details Publish the predeclared search space before fitting; a partial run
    # cannot silently become a different search or overwrite prior completed files.
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {**provenance, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "regressor_configurations": [dict(c) for c in REGRESSOR_CONFIGS],
                "common_settings": dict(COMMON_SETTINGS), "margins_usd": list(MARGINS_USD)}
    _write(destination / "protocol.json", protocol)
    candidates = []
    with threadpool_limits(limits=1):
        for configuration in REGRESSOR_CONFIGS:
            settings = _settings(configuration)
            model = HistGradientBoostingRegressor(**settings)
            # @details Deliberately supply no sample_weight or held-out data. Raw
            # array values and all negative/zero targets reach fit unchanged.
            model.fit(features, targets)
            if model.n_iter_ != COMMON_SETTINGS["max_iter"]:
                raise RuntimeError("Comparator did not complete the declared 200 iterations.")
            serialized = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            model_hash = hashlib.sha256(serialized).hexdigest()
            for margin in MARGINS_USD:
                identifier = _candidate_id(configuration, margin)
                policy_dir = destination / identifier
                policy_dir.mkdir()
                (policy_dir / "model.pkl").write_bytes(serialized)
                metadata = {
                    **provenance, "candidate_id": identifier,
                    "regressor_id": configuration["id"], "regressor_config": settings,
                    "margin_usd": margin, "decision_rule": "trade iff expected net USD > margin_usd",
                    "actual_iterations": int(model.n_iter_),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "artifact_sha256": {"model.pkl": model_hash},
                }
                _write(policy_dir / "training.json", metadata)
                _write(policy_dir / "checksums.json", {
                    name: _hash(policy_dir / name) for name in ("model.pkl", "training.json")
                })
                candidates.append({"id": identifier, "directory": str(policy_dir),
                                   "margin_usd": margin, "regressor_config": settings,
                                   "model_sha256": model_hash})
    result = {
        "directory": str(destination), "candidates": candidates,
        "policy_dirs": {candidate["id"]: candidate["directory"] for candidate in candidates},
        "regressors_fitted": len(REGRESSOR_CONFIGS), "candidate_count": len(candidates),
        "event_count": len(targets), "held_out_data_loaded": False,
        "events_sha256": provenance["events_sha256"], "versions": versions,
    }
    _write(destination / "run.json", result)
    return result


def load_opportunity_value_policy(
    model_dir: str | Path, *, expected_events_sha256: str | None = None,
) -> OpportunityValuePolicy:
    """@brief Load one frozen margin policy after provenance and checksum checks.

    @warning Pickle deserialization executes trusted Python objects. Only load
    locally created, trusted artifacts; checksums detect accidental mismatch and
    do not authenticate hostile files. Exact sklearn version matching is required.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    directory = Path(model_dir)
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    for name in ("model.pkl", "training.json"):
        if checksums.get(name) != _hash(directory / name):
            raise ValueError(f"Value policy artifact checksum mismatch: {name}")
    metadata = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    if (metadata.get("artifact_version") != 1
            or metadata.get("policy_family") != "opportunity_expected_net_value"
            or metadata.get("training_split") != "train"
            or metadata.get("held_out_data_loaded") is not False):
        raise ValueError("Unsupported or non-training-only value policy metadata.")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Value policy feature contract mismatch.")
    if metadata.get("source_sha256") != _source_hashes(include_comparator=True):
        raise ValueError("Value policy source fingerprint mismatch.")
    if metadata.get("versions", {}).get("scikit-learn") != importlib.metadata.version("scikit-learn"):
        raise ValueError("Value policy requires the saved scikit-learn version.")
    if (expected_events_sha256 is not None
            and metadata.get("events_sha256") != expected_events_sha256):
        raise ValueError("Value policy event checksum does not match the expected study.")
    configurations = {c["id"]: c for c in REGRESSOR_CONFIGS}
    configuration = configurations.get(metadata.get("regressor_id"))
    margin = metadata.get("margin_usd")
    if (configuration is None or margin not in MARGINS_USD
            or metadata.get("regressor_config") != _settings(configuration)
            or metadata.get("candidate_id") != _candidate_id(configuration, margin)):
        raise ValueError("Value policy does not match a predeclared configuration/margin.")
    if metadata.get("artifact_sha256", {}).get("model.pkl") != checksums["model.pkl"]:
        raise ValueError("Value policy model provenance checksum mismatch.")
    # @details Deserialize only after the trusted-local artifact passes all numeric
    # contract checks, then cross-check actual fitted settings against its metadata.
    with (directory / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, HistGradientBoostingRegressor):
        raise ValueError("Value policy artifact is not the declared regressor type.")
    if (model.n_features_in_ != len(FEATURE_NAMES)
            or model.n_iter_ != COMMON_SETTINGS["max_iter"]
            or any(model.get_params()[k] != v for k, v in _settings(configuration).items())):
        raise ValueError("Fitted value policy does not match its saved configuration.")
    return OpportunityValuePolicy(model, metadata)


def main(argv: list[str] | None = None) -> int:
    """@brief Explicitly fit the fixed comparator grid from an existing training cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = train_opportunity_value(args.events, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
