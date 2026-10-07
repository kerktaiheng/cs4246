"""@file test_opportunity_value.py
@brief Synthetic verification of the isolated expected-net-dollar comparator.

@details The fixture fits the two declared histogram boosting regressors to known
synthetic outcomes. No recorded market data, validation archive, or fresh holdout
is opened. These checks establish implementation integrity, never trading edge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="Value comparator uses its optional isolated runtime.")
from sklearn.ensemble import HistGradientBoostingRegressor

import latency_arb.agent.opportunity_value as value
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import FEATURE_NAMES, OpportunityConfig


def _save_json(path: Path, document: dict) -> None:
    """@brief Save finite fixture metadata without borrowing production validation."""
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")


def _checksum(path: Path) -> str:
    """@brief Independently compute the complete artifact hash used by fixture edits."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_fixture(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """@brief Build natural-frequency synthetic labels with unavailable book archives.

    @details Positive, negative and zero dollar outcomes all remain in the cache.
    Deliberately nonexistent train/validation/test paths prove that fitting needs
    only this already-produced cache and does not reopen the market dataset.
    """
    from dataclasses import asdict

    directory.mkdir()
    generator = np.random.default_rng(71)
    features = generator.normal(size=(1200, len(FEATURE_NAMES))).astype(np.float32)
    targets = (0.025 * features[:, 0] - 0.013 * features[:, 1]).astype(np.float64)
    targets[::37] = 0.0
    episode_indices = np.repeat(np.arange(3, dtype=np.int64), 400)
    candidate_indices = np.tile(np.arange(400, dtype=np.int64), 3)
    np.savez(directory / "events.npz", features=features, net_payoffs=targets,
             episode_indices=episode_indices, candidate_indices=candidate_indices)
    execution = asdict(EnvConfig(fee_bps=4.5, decision_interval_ms=1000.0))
    economics = {key: val for key, val in execution.items()
                 if key not in ("reward_scale", "log_history")}
    metadata = {
        "event_version": 1, "split": "train",
        "future_rows_used_only_for_targets": True,
        "feature_names": list(FEATURE_NAMES),
        "source_sha256": value._source_hashes(),
        "manifest_sha256": "a" * 64,
        "manifest_path": str(directory / "unavailable_manifest.json"),
        "artifact_sha256": {"events.npz": _checksum(directory / "events.npz")},
        "training_entries": [{"split": "train", "path": f"unavailable_train_{i}.npz"}
                             for i in range(3)],
        "execution_config": execution, "economic_config": economics,
        "opportunity_config": asdict(OpportunityConfig()),
        "event_count": len(targets), "eligible_episode_indices": [0, 1, 2],
        "eligible_episode_count": 3,
        "positive_target_count": int((targets > 0).sum()),
        "negative_target_count": int((targets < 0).sum()),
        "zero_target_count": int((targets == 0).sum()),
    }
    _save_json(directory / "events.json", metadata)
    return features, targets


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """@brief Execute both fixed real fits while auditing exact inputs and read paths."""
    root = tmp_path_factory.mktemp("opportunity_value")
    directory = root / "events"
    features, targets = _event_fixture(directory)
    original_fit = HistGradientBoostingRegressor.fit
    original_open = Path.open
    calls, event_reads = [], []

    def checked_fit(model, x, y, *args, **kwargs):
        """@brief Observe real training without replacing its numerical implementation."""
        np.testing.assert_array_equal(x, features)
        np.testing.assert_array_equal(y, targets)
        assert x.dtype == np.float32 and y.dtype == np.float64
        assert not args and not kwargs
        calls.append(model.get_params())
        return original_fit(model, x, y)

    def checked_open(path, mode="r", *args, **kwargs):
        """@brief Fail any attempted dataset access beyond the two training-cache files."""
        resolved = path.resolve()
        if resolved.is_relative_to(directory):
            assert resolved.name in {"events.json", "events.npz"}
            assert "w" not in mode and "a" not in mode
            event_reads.append(resolved.name)
        return original_open(path, mode, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HistGradientBoostingRegressor, "fit", checked_fit)
        patch.setattr(Path, "open", checked_open)
        result = value.train_opportunity_value(
            directory, root / "models", expected_manifest_sha256="a" * 64,
            expected_events_sha256=_checksum(directory / "events.npz"),
        )
    assert set(event_reads) == {"events.json", "events.npz"}
    return root, directory, features, targets, result, calls


def test_exact_two_real_fits_generate_four_fixed_policies(fitted):
    """@brief Confirm all rows and targets reached exactly the predeclared real fits."""
    _, _, _, targets, result, calls = fitted
    assert len(calls) == result["regressors_fitted"] == 2
    assert result["candidate_count"] == len(result["candidates"]) == 4
    assert result["event_count"] == len(targets)
    assert result["held_out_data_loaded"] is False
    assert [(params["max_leaf_nodes"], params["min_samples_leaf"]) for params in calls] == [
        (7, 200), (15, 50),
    ]
    for settings in calls:
        for key, expected in value.COMMON_SETTINGS.items():
            assert settings[key] == expected
    candidates = result["candidates"]
    assert [candidate["margin_usd"] for candidate in candidates] == [0.0, 0.01, 0.0, 0.01]
    assert candidates[0]["model_sha256"] == candidates[1]["model_sha256"]
    assert candidates[2]["model_sha256"] == candidates[3]["model_sha256"]
    for candidate in candidates:
        policy = value.load_opportunity_value_policy(candidate["directory"])
        assert policy.model.n_iter_ == 200
        assert policy.metadata["sample_weighting"] == "none"
        assert policy.metadata["target_transformation"] == "none"
        assert policy.metadata["internal_validation"] is False


def test_saved_predictions_reproduce_native_model_and_never_mutate(fitted):
    """@brief Reloaded inference preserves predictions, raw inputs and model artifacts."""
    _, directory, features, _, result, _ = fitted
    for candidate in result["candidates"]:
        model_dir = Path(candidate["directory"])
        checksum_before = _checksum(model_dir / "model.pkl")
        policy = value.load_opportunity_value_policy(
            model_dir, expected_events_sha256=_checksum(directory / "events.npz"),
        )
        reloaded = value.load_opportunity_value_policy(model_dir)
        inputs = features[:25].copy()
        original = inputs.copy()
        predictions = policy.expected_net_pnl(inputs)
        np.testing.assert_array_equal(predictions, reloaded.expected_net_pnl(inputs))
        np.testing.assert_array_equal(predictions, policy.model.predict(inputs))
        assert policy.expected_net_pnl(inputs[0]) == predictions[0]
        decisions, state = policy.predict(inputs)
        np.testing.assert_array_equal(decisions, predictions > candidate["margin_usd"])
        assert state is None
        assert policy.predict(inputs[0])[0].shape == ()
        np.testing.assert_array_equal(inputs, original)
        assert _checksum(model_dir / "model.pkl") == checksum_before
        assert np.any(predictions > 0) and np.any(predictions < 0)


def test_margin_is_strict_and_invalid_features_are_rejected():
    """@brief A prediction equal to the safety margin skips without random sampling."""
    class FixedModel:
        """@brief Supply controlled predictions solely to inspect the public USD gate."""
        def predict(self, features):
            """@brief Expose the first feature as an exact boundary value."""
            return features[:, 0]

    policy = value.OpportunityValuePolicy(
        FixedModel(), {"feature_names": list(FEATURE_NAMES), "margin_usd": 0.01},
    )
    rows = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float64)
    rows[:, 0] = [0.009, 0.01, 0.011]
    np.testing.assert_array_equal(policy.predict(rows)[0], [0, 0, 1])
    np.testing.assert_array_equal(policy.predict(rows, deterministic=False)[0], [0, 0, 1])
    for malformed in (np.zeros(2), np.zeros((1, 2)), np.full_like(rows, np.nan)):
        with pytest.raises(ValueError, match="finite raw"):
            policy.expected_net_pnl(malformed)


def test_output_preservation_precedes_any_input_read(fitted, tmp_path):
    """@brief Refitting cannot erase an existing checkpoint or even inspect new input."""
    sentinel = tmp_path / "preserve.txt"
    sentinel.write_text("original result")
    with pytest.raises(FileExistsError, match="not empty"):
        value.train_opportunity_value(tmp_path / "does_not_exist", tmp_path)
    assert sentinel.read_text() == "original result"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preserve.txt"]


@pytest.mark.parametrize("mutation,match", [
    ("split", "train-only"),
    ("producer", "source fingerprint"),
    ("mixed_entry", "exclusively training"),
    ("feature_names", "feature names"),
    ("economics", "economic metadata"),
    ("target_counts", "target-count"),
    ("eligible", "eligible-episode"),
    ("event_hash", "checksum"),
])
def test_event_contract_rejects_corruption_before_fitting(fitted, tmp_path, mutation, match):
    """@brief Mislabelled splits, economics, provenance and artifacts fail before fitting."""
    _, directory, _, _, _, _ = fitted
    cache = tmp_path / "cache"
    shutil.copytree(directory, cache)
    metadata = json.loads((cache / "events.json").read_text())
    if mutation == "split":
        metadata["split"] = "validation"
    elif mutation == "producer":
        metadata["source_sha256"] = {}
    elif mutation == "mixed_entry":
        metadata["training_entries"][0]["split"] = "test"
    elif mutation == "feature_names":
        metadata["feature_names"] = list(reversed(FEATURE_NAMES))
    elif mutation == "economics":
        metadata["economic_config"]["fee_bps"] = 3.5
    elif mutation == "target_counts":
        metadata["positive_target_count"] += 1
    elif mutation == "eligible":
        metadata["eligible_episode_indices"] = [0]
    elif mutation == "event_hash":
        with (cache / "events.npz").open("ab") as handle:
            handle.write(b"corrupted artifact")
    _save_json(cache / "events.json", metadata)
    with pytest.raises(ValueError, match=match):
        value.train_opportunity_value(cache, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("field", ["expected_manifest_sha256", "expected_events_sha256"])
def test_expected_study_hash_prevents_using_another_training_cache(fitted, tmp_path, field):
    """@brief Explicit study boundaries reject a valid cache from a different protocol."""
    _, directory, _, _, _, _ = fitted
    with pytest.raises(ValueError, match="expected study"):
        value.train_opportunity_value(directory, tmp_path / "rejected", **{field: "b" * 64})
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("filename", ["model.pkl", "training.json"])
def test_policy_checksums_reject_modified_artifacts(fitted, tmp_path, filename):
    """@brief Checkpoint corruption fails before trusted-local pickle deserialization."""
    _, _, _, _, result, _ = fitted
    candidate = result["candidates"][0]
    directory = tmp_path / "policy"
    shutil.copytree(candidate["directory"], directory)
    with (directory / filename).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        value.load_opportunity_value_policy(directory)


@pytest.mark.parametrize("mutation,match", [
    ("source", "source fingerprint"),
    ("version", "scikit-learn version"),
    ("margin", "predeclared"),
    ("settings", "predeclared"),
    ("split", "non-training-only"),
])
def test_policy_contract_rejects_internally_rehashed_metadata(fitted, tmp_path, mutation, match):
    """@brief Consistent file hashes cannot override declared source/model/split contracts."""
    _, _, _, _, result, _ = fitted
    directory = tmp_path / "policy"
    shutil.copytree(result["candidates"][0]["directory"], directory)
    metadata = json.loads((directory / "training.json").read_text())
    if mutation == "source":
        metadata["source_sha256"] = {}
    elif mutation == "version":
        metadata["versions"]["scikit-learn"] = "different"
    elif mutation == "margin":
        metadata["margin_usd"] = -0.01
    elif mutation == "settings":
        metadata["regressor_config"]["max_iter"] = 10
    elif mutation == "split":
        metadata["training_split"] = "validation"
    _save_json(directory / "training.json", metadata)
    checksums = json.loads((directory / "checksums.json").read_text())
    checksums["training.json"] = _checksum(directory / "training.json")
    _save_json(directory / "checksums.json", checksums)
    with pytest.raises(ValueError, match=match):
        value.load_opportunity_value_policy(directory)


def test_loaded_policy_must_belong_to_expected_event_cache(fitted):
    """@brief Frozen inference rejects the wrong expected training provenance."""
    _, _, _, _, result, _ = fitted
    with pytest.raises(ValueError, match="expected study"):
        value.load_opportunity_value_policy(
            result["candidates"][0]["directory"], expected_events_sha256="b" * 64,
        )
