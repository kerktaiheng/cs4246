"""@file baseline.py
@brief Deterministic policies used as common-cost research controls.
@details Policies consume the same causal observations as PPO. They cannot inspect
future replay rows, and all execution and risk decisions remain in the environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_arb.env.latency_sim import OBSERVATION_NAMES


@dataclass(frozen=True)
class ThresholdPolicy:
    """@brief Enter only when the signed cross-venue gap passes a fixed threshold.
    @param entry_threshold_bps Minimum absolute gap required to request an entry.
    @param exit_threshold_bps Convergence boundary at which to request liquidation.
    @details Positive gaps request a Hyperliquid long; negative gaps request a short.
    A pending order suppresses new requests. Existing positions never reverse in one
    action, so both fees and the environment's execution delay remain unavoidable.
    """

    entry_threshold_bps: float = 8.0
    exit_threshold_bps: float = 0.5

    def __post_init__(self) -> None:
        """@brief Reject nonfinite or overlapping entry/exit boundaries."""
        if not np.isfinite([self.entry_threshold_bps, self.exit_threshold_bps]).all():
            raise ValueError("Thresholds must be finite.")
        if not 0 <= self.exit_threshold_bps < self.entry_threshold_bps:
            raise ValueError("Require 0 <= exit threshold < entry threshold.")

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """@brief Return an SB3-compatible action tuple using only present features.
        @param observation One raw, unnormalized environment observation.
        @param deterministic Accepted for a uniform evaluation-policy interface.
        @return Integer action and unused recurrent state.
        """
        del deterministic
        values = dict(zip(OBSERVATION_NAMES, np.asarray(observation), strict=True))
        gap, inventory = float(values["gap_bps"]), float(values["inventory"])
        # @details HOLD preserves a pending instruction rather than resubmitting it.
        if values["pending_action"] >= 0:
            return 0, None
        if inventory > 0:
            return (3 if gap <= self.exit_threshold_bps else 0), None
        if inventory < 0:
            return (3 if gap >= -self.exit_threshold_bps else 0), None
        # @details Flat states use symmetric boundaries and never force a trade.
        if gap >= self.entry_threshold_bps:
            return 1, None
        if gap <= -self.entry_threshold_bps:
            return 2, None
        return 0, None


class FlatPolicy:
    """@brief Always abstain, providing the essential zero-exposure control."""

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """@brief Request HOLD without inspecting the book or using future data."""
        del observation, deterministic
        return 0, None
