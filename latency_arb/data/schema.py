"""@file schema.py
@brief Portable, validated arrays shared by offline preparation and replay.
@details NPZ files contain numeric arrays plus JSON metadata only. Loading never
enables pickle, and timestamps remain signed 64-bit nanoseconds throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReplayEpisode:
    """@brief A contiguous single-day stream of causally available market states.
    @details Rows are ordered by local receive time. Funding rates are signed
    settlement rates at their event row only; zero means no settlement. Quote
    ages use venue clocks unless metadata explicitly records receive-age fallback.
    All prices/sizes are finite, and each book contains the configured depth.
    """

    timestamp_ns: np.ndarray
    binance_bid: np.ndarray
    binance_ask: np.ndarray
    hl_bid_prices: np.ndarray
    hl_bid_sizes: np.ndarray
    hl_ask_prices: np.ndarray
    hl_ask_sizes: np.ndarray
    binance_imbalance: np.ndarray
    hl_imbalance: np.ndarray
    volatility_bps: np.ndarray
    hl_quote_age_ms: np.ndarray
    hl_received_age_ms: np.ndarray
    funding_rate: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """@brief Return the number of decision/event rows in this episode."""
        return len(self.timestamp_ns)

    @property
    def hl_bid(self) -> np.ndarray:
        """@brief Return Hyperliquid best bid prices without copying book depth."""
        return self.hl_bid_prices[:, 0]

    @property
    def hl_ask(self) -> np.ndarray:
        """@brief Return Hyperliquid best ask prices without copying book depth."""
        return self.hl_ask_prices[:, 0]

    @property
    def hl_mid(self) -> np.ndarray:
        """@brief Return contemporaneous Hyperliquid midprices."""
        return (self.hl_bid + self.hl_ask) / 2.0

    @property
    def binance_mid(self) -> np.ndarray:
        """@brief Return contemporaneous Binance midprices."""
        return (self.binance_bid + self.binance_ask) / 2.0

    @property
    def gap_bps(self) -> np.ndarray:
        """@brief Return Binance minus Hyperliquid midprice in basis points."""
        return (self.binance_mid / self.hl_mid - 1.0) * 10_000.0

    def validate(self) -> None:
        """@brief Reject malformed shapes, timestamps, prices, sizes, and features.
        @throws ValueError If the episode violates the common replay contract.
        @details Equal timestamps are legal: separate records can arrive during
        the same nanosecond. Their source order remains causal and deterministic.
        """
        #! @details Require integer timestamps rather than silently rounding float
        #! nanoseconds, since float64 loses individual nanoseconds in modern dates.
        ts = np.asarray(self.timestamp_ns)
        if ts.ndim != 1 or ts.dtype != np.dtype("int64") or len(ts) < 2:
            raise ValueError("timestamp_ns must be int64[n] with at least two rows")
        if np.any(ts < 0) or np.any(ts[1:] < ts[:-1]):
            raise ValueError("timestamps must be nonnegative and receive-time ordered")
        n = len(ts)
        depth = None
        #! @details All market arrays use one shape/dtype contract, preventing
        #! broadcasting and object arrays from entering the execution simulator.
        for item in fields(self):
            if item.name in {"timestamp_ns", "metadata"}:
                continue
            value = np.asarray(getattr(self, item.name))
            is_book = item.name in {
                "hl_bid_prices", "hl_bid_sizes", "hl_ask_prices", "hl_ask_sizes"
            }
            if value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise ValueError(f"{item.name} must contain finite float64 values")
            if is_book:
                if value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1:
                    raise ValueError(f"{item.name} must have shape [n, depth]")
                if depth is None:
                    depth = value.shape[1]
                if value.shape[1] != depth:
                    raise ValueError("all Hyperliquid books must have identical depth")
            elif value.shape != (n,):
                raise ValueError(f"{item.name} must have shape [n]")
        #! @details Enforce economically valid ordered books before fills occur.
        if np.any(self.binance_bid <= 0) or np.any(self.binance_ask <= self.binance_bid):
            raise ValueError("Binance best prices must be positive and uncrossed")
        if np.any(self.hl_bid_prices <= 0) or np.any(self.hl_ask_prices <= 0):
            raise ValueError("Hyperliquid prices must be positive")
        if np.any(self.hl_bid_sizes <= 0) or np.any(self.hl_ask_sizes <= 0):
            raise ValueError("Hyperliquid sizes must be positive")
        if np.any(self.hl_bid >= self.hl_ask):
            raise ValueError("Hyperliquid best prices must be uncrossed")
        if np.any(np.diff(self.hl_bid_prices, axis=1) >= 0) or np.any(
            np.diff(self.hl_ask_prices, axis=1) <= 0
        ):
            raise ValueError("book levels must have strictly ordered prices")
        for name in ("binance_imbalance", "hl_imbalance"):
            if np.any(np.abs(getattr(self, name)) > 1.0 + 1e-12):
                raise ValueError(f"{name} must be between -1 and 1")
        for name in ("volatility_bps", "hl_quote_age_ms", "hl_received_age_ms"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} cannot be negative")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        json.dumps(self.metadata, allow_nan=False)

    def save(self, path: str | Path) -> Path:
        """@brief Validate and save a compressed, non-pickled replay archive.
        @param path Exact output filename; no automatic extension is appended.
        @return The written path.
        """
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            item.name: getattr(self, item.name)
            for item in fields(self) if item.name != "metadata"
        }
        arrays["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        )
        with destination.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ReplayEpisode":
        """@brief Load and validate an archive with object deserialization disabled.
        @throws ValueError If required fields or metadata are malformed.
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            names = [item.name for item in fields(cls) if item.name != "metadata"]
            missing = set(names + ["metadata_json"]) - set(archive.files)
            if missing:
                raise ValueError(f"episode archive lacks fields: {sorted(missing)}")
            episode = cls(
                **{name: archive[name].copy() for name in names},
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
        episode.validate()
        return episode



def read_manifest(path: str | Path, split: str | None = None) -> tuple[dict, list[dict]]:
    """@brief Validate manifest metadata without opening any episode archives.
    @details This permits training to keep only one or two episodes resident.
    Paths, day/split chronology, nonoverlapping declared intervals, and hash syntax
    are checked here. Actual file hashes and array bounds are checked on loading.
    """
    if split not in {None, "train", "validation", "test"}:
        raise ValueError("split must be train, validation, test, or None")
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    entries = manifest.get("episodes", [])
    if not entries:
        raise ValueError("manifest contains no usable episodes")
    ranks = {"train": 0, "validation": 1, "test": 2}
    day_splits: dict[str, str] = {}
    root = manifest_path.parent.resolve()
    selected: list[dict] = []
    previous_start: int | None = None
    previous_end: int | None = None
    for entry in entries:
        if entry.get("split") not in ranks:
            raise ValueError("manifest contains an unknown split")
        day = entry["day"]
        if day in day_splits and day_splits[day] != entry["split"]:
            raise ValueError("one UTC day cannot appear in multiple splits")
        day_splits[day] = entry["split"]
        if split is not None and entry["split"] != split:
            continue
        candidate = (root / entry["path"]).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("episode path must remain inside the manifest directory")
        expected_hash = entry.get("content_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
            value not in "0123456789abcdef" for value in expected_hash.lower()
        ):
            raise ValueError("manifest episode lacks a SHA256 content hash")
        start = entry.get("start_timestamp_ns")
        end = entry.get("end_timestamp_ns")
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("manifest episode requires exact integer timestamp bounds")
        if previous_start is not None and start < previous_start:
            raise ValueError("selected episodes must be ordered by receive timestamp")
        if previous_end is not None and start < previous_end:
            raise ValueError("selected episodes must not overlap")
        if int(entry.get("rows", 0)) < 2:
            raise ValueError("manifest episode requires at least two rows")
        previous_start, previous_end = start, end
        selected.append(entry)
    ordered_ranks = [ranks[day_splits[day]] for day in sorted(day_splits)]
    if ordered_ranks != sorted(ordered_ranks):
        raise ValueError("manifest splits must be chronological")
    if not selected:
        raise ValueError(f"manifest contains no episodes for split {split}")
    return manifest, selected


def load_manifest_entry(path: str | Path, entry: dict) -> ReplayEpisode:
    """@brief Verify and load one selected archive without touching held-out files.
    @details Validates path containment independently, verifies content SHA256,
    then cross-checks schema, actual day, row count, split and timestamp bounds.
    """
    root = Path(path).parent.resolve()
    candidate = (root / entry["path"]).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("episode path must remain inside the manifest directory")
    with candidate.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != entry.get("content_sha256"):
        raise ValueError("episode content hash disagrees with manifest")
    episode = ReplayEpisode.load(candidate)
    start, end = int(episode.timestamp_ns[0]), int(episode.timestamp_ns[-1])
    if entry.get("start_timestamp_ns") != start or entry.get("end_timestamp_ns") != end:
        raise ValueError("episode timestamp bounds disagree with manifest")
    if entry.get("rows") != len(episode):
        raise ValueError("episode row count disagrees with manifest")
    actual_days = episode.timestamp_ns // 86_400_000_000_000
    if np.any(actual_days != actual_days[0]):
        raise ValueError("an episode crosses a UTC day boundary")
    actual_day = str(np.datetime64(int(actual_days[0]), "D"))
    if actual_day != entry["day"]:
        raise ValueError("episode timestamp day disagrees with manifest")
    if episode.metadata.get("split", entry["split"]) != entry["split"]:
        raise ValueError("episode split metadata disagrees with manifest")
    return episode


def load_manifest(path: str | Path, split: str | None = None) -> list[ReplayEpisode]:
    """@brief Eagerly load selected episodes; use read_manifest for bounded memory."""
    _, entries = read_manifest(path, split)
    return [load_manifest_entry(path, entry) for entry in entries]
