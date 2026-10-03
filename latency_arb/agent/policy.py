"""@file policy.py
@brief Deterministic inference for PPO artifacts produced by the offline trainer.

@details Only open policy directories created by this project or another trusted
source. Stable-Baselines3 model archives contain serialized Python/PyTorch objects.
Observation statistics are additionally exported as a non-pickled NumPy archive so
inference never creates an environment or updates statistics on evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


@dataclass(frozen=True)
class FrozenNormalizer:
    """@brief Read-only copy of training-only VecNormalize observation statistics.

    @details The transformation matches Stable-Baselines3 VecNormalize exactly:
    subtract the running mean, divide by sqrt(variance + epsilon), clip, then cast
    to float32. There is deliberately no update method, reward normalization, or
    dependency on evaluation transitions.
    """

    mean: np.ndarray
    variance: np.ndarray
    count: float
    epsilon: float
    clip_obs: float

    def __post_init__(self) -> None:
        """@brief Validate and privately copy statistics before accepting an artifact.

        @throws ValueError If statistics are malformed or non-finite.
        @details Read-only copies prevent callers from inadvertently changing the
        training distribution when evaluating multiple policies or cost settings.
        """
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        variance = np.array(self.variance, dtype=np.float64, copy=True)
        if mean.ndim != 1 or mean.size == 0 or variance.shape != mean.shape:
            raise ValueError("Normalizer mean and variance must be matching vectors.")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Normalizer statistics must be finite.")
        if (variance < 0).any():
            raise ValueError("Normalizer variance must be non-negative.")
        if not np.isfinite([self.count, self.epsilon, self.clip_obs]).all():
            raise ValueError("Normalizer scalar parameters must be finite.")
        if self.count <= 0 or self.epsilon <= 0 or self.clip_obs <= 0:
            raise ValueError("Normalizer count, epsilon, and clipping must be positive.")
        mean.setflags(write=False)
        variance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        """@brief Transform one raw observation or a batch without learning.

        @param observation Raw environment features, shaped (features,) or
        (batch, features), in the feature order recorded in training.json.
        @return A new float32 array suitable for PPO.predict.
        @throws ValueError If the shape or feature values are invalid.
        """
        values = np.asarray(observation)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.mean):
            raise ValueError(
                f"Expected {len(self.mean)} observation features, got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Inference observations must be finite.")
        normalized = (values - self.mean) / np.sqrt(self.variance + self.epsilon)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)


class FrozenPolicy:
    """@brief Bind a trained PPO model to its immutable observation transform.

    @details Consumers provide the raw observation returned by LatencySimEnv.
    Returning the standard (action, recurrent_state) tuple keeps the adapter
    compatible with callers that already use Stable-Baselines3 predict.
    """

    def __init__(
        self,
        model: PPO,
        normalizer: FrozenNormalizer,
        metadata: dict[str, Any],
    ) -> None:
        """@brief Check model/normalizer compatibility and retain provenance.

        @param model Trusted PPO model loaded on the CPU.
        @param normalizer Statistics fitted exclusively on training observations.
        @param metadata Saved configuration, feature order, and artifact hashes.
        """
        if model.observation_space.shape != normalizer.mean.shape:
            raise ValueError("Policy and normalization feature shapes do not match.")
        self.model = model
        self.normalizer = normalizer
        self.metadata = metadata
        self.observation_names = tuple(metadata["observation_names"])

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        """@brief Select actions after applying frozen training statistics.

        @param observation Raw single observation or vectorized batch.
        @param deterministic Use the highest-probability PPO action by default.
        @return Standard Stable-Baselines3 (actions, recurrent_state) tuple.
        """
        return self.model.predict(
            self.normalizer.normalize(observation), deterministic=deterministic
        )


def load_policy(model_dir: str | Path) -> FrozenPolicy:
    """@brief Load and verify a locally trained policy for held-out evaluation.

    @param model_dir Directory written by train_policy.
    @return A deterministic-by-default adapter accepting raw environment features.
    @throws ValueError If artifact hashes, schema, or feature order disagree.
    @warning Policy archives deserialize Python/PyTorch objects; use trusted local
    artifacts only. Hashes detect accidental mismatches, not malicious provenance.
    """
    directory = Path(model_dir)
    metadata = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != 1:
        raise ValueError("Unsupported training artifact version.")

    ## @details Verify the exact weights and observation statistics before loading
    # either, so mixing files from different runs cannot silently change behavior.
    for filename in ("policy.zip", "observation_normalization.npz"):
        expected = metadata.get("artifact_sha256", {}).get(filename)
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if expected is None or actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")

    ## @details Reject a changed observation contract even when its length matches:
    # exchanging two same-shaped features would otherwise produce plausible actions.
    from latency_arb.env.latency_sim import OBSERVATION_NAMES

    if tuple(metadata.get("observation_names", ())) != tuple(OBSERVATION_NAMES):
        raise ValueError("Policy observation names do not match this environment.")
    with np.load(directory / "observation_normalization.npz", allow_pickle=False) as data:
        normalizer = FrozenNormalizer(
            mean=data["mean"],
            variance=data["variance"],
            count=float(data["count"].item()),
            epsilon=float(data["epsilon"].item()),
            clip_obs=float(data["clip_obs"].item()),
        )

    ## @details The environment is intentionally absent at load time. Prediction
    # cannot reset it, collect evaluation observations, or refit normalization.
    model = PPO.load(directory / "policy.zip", device="cpu")
    model.policy.set_training_mode(False)
    return FrozenPolicy(model, normalizer, metadata)
