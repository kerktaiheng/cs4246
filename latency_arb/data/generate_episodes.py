"""@file generate_episodes.py
@brief Prepare strictly causal local replay episodes or a labeled synthetic demo.
@details The CLI never connects to an exchange or database. Input JSONL and
Parquet contain venue snapshots in ascending local receive time; multiple files
are merged in that same time domain. Day splits and quality exclusions are fixed
before any policy is trained or evaluated.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .schema import ReplayEpisode, load_manifest

DAY_NS = 86_400_000_000_000
VENUE_NAMES = {
    "binance": "binance", "binance_futures": "binance",
    "hyperliquid": "hyperliquid", "hl": "hyperliquid",
}


@dataclass(frozen=True)
class QualityConfig:
    """@brief Predeclared exclusions used identically across all dataset splits.
    @param clock_policy reject rejects all output if any source clock is ahead;
    receive explicitly replaces quote age with elapsed local receive age.
    @details Gaps and invalid states terminate episodes; no replay episode jumps
    across an excluded interval. The default depth matches the proposal.
    """

    depth: int = 10
    max_received_age_ms: float = 1000.0
    max_quote_age_ms: float = 2000.0
    max_gap_ms: float = 1000.0
    max_spread_bps: float = 100.0
    max_cross_venue_gap_bps: float = 250.0
    min_episode_steps: int = 32
    max_episode_steps: int = 100_000
    volatility_window: int = 20
    clock_policy: str = "reject"

    def validate(self) -> None:
        """@brief Reject undefined or permissive-by-accident quality thresholds."""
        if self.max_episode_steps < self.min_episode_steps:
            raise ValueError("max_episode_steps must be at least min_episode_steps")
        if self.clock_policy not in {"reject", "receive"}:
            raise ValueError("clock_policy must be reject or receive")
        if self.depth < 1 or self.min_episode_steps < 2 or self.volatility_window < 2:
            raise ValueError("depth >= 1, min_episode_steps >= 2, volatility_window >= 2")
        for name in (
            "max_received_age_ms", "max_quote_age_ms", "max_gap_ms",
            "max_spread_bps", "max_cross_venue_gap_bps",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _timestamp(value: Any, name: str) -> int:
    """@brief Parse nanoseconds exactly and reject lossy floats or booleans.
    @details Integer decimal strings are accepted for JSON exporters that avoid
    Javascript's integer precision limit. Float timestamps are never rounded.
    """
    if isinstance(value, (bool, float, np.floating)) or value is None:
        raise ValueError(f"{name} must be exact integer nanoseconds")
    if not isinstance(value, (int, np.integer, str)):
        raise ValueError(f"{name} must be exact integer nanoseconds")
    parsed = int(value)
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ValueError(f"{name} must be a decimal integer")
    if parsed < 0 or parsed > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside nonnegative int64 range")
    return parsed


def _read_file(path: Path) -> Iterator[dict[str, Any]]:
    """@brief Stream JSONL or Parquet rows, checking exact receive-time ordering.
    @details Arrow converts integer columns directly to Python integers; no
    pandas iterrows conversion can accidentally turn timestamps into float64.
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected JSON object")
                    yield value
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet input requires the pyarrow package") from exc
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=65536):
            yield from batch.to_pylist()
    else:
        raise ValueError(f"unsupported input extension: {path}")


def _ordered_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """@brief Refuse out-of-order source data instead of silently reshuffling it."""
    previous = -1
    for record in _read_file(path):
        timestamp = _timestamp(record.get("local_ts_ns"), "local_ts_ns")
        if timestamp < previous:
            raise ValueError(f"{path}: records are not ordered by local_ts_ns")
        previous = timestamp
        yield timestamp, record


def read_records(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """@brief Merge already ordered files using backward receive-time semantics.
    @details Heap merge keeps the original order within each file. Ties across
    files follow the supplied filename order, a deterministic assumption recorded
    in the manifest; no later receive event is available to an earlier row.
    """
    ordered = [_ordered_file(Path(path)) for path in paths]
    for _, record in heapq.merge(*ordered, key=lambda item: item[0]):
        yield record


def _book(record: dict[str, Any], depth: int) -> tuple[np.ndarray, ...]:
    """@brief Normalize pair-list and separate-array books, requiring sound depth."""
    if record.get("bids") is not None and record.get("asks") is not None:
        bids = np.asarray(record["bids"], dtype=np.float64)
        asks = np.asarray(record["asks"], dtype=np.float64)
        if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] != 2 or asks.shape[1] != 2:
            raise ValueError("book sides must be lists of [price, size] pairs")
        bid_prices, bid_sizes = bids[:, 0], bids[:, 1]
        ask_prices, ask_sizes = asks[:, 0], asks[:, 1]
    else:
        bid_prices, bid_sizes, ask_prices, ask_sizes = (
            np.asarray(record.get(name, []), dtype=np.float64)
            for name in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes")
        )
    arrays = (bid_prices, bid_sizes, ask_prices, ask_sizes)
    if any(array.ndim != 1 or len(array) < depth for array in arrays):
        raise ValueError("insufficient book depth")
    if len(bid_prices) != len(bid_sizes) or len(ask_prices) != len(ask_sizes):
        raise ValueError("book prices and sizes have unequal lengths")
    arrays = tuple(array[:depth].copy() for array in arrays)
    if any(not np.isfinite(array).all() or np.any(array <= 0) for array in arrays):
        raise ValueError("book prices and sizes must be finite and positive")
    bp, _, ap, _ = arrays
    if bp[0] >= ap[0] or np.any(np.diff(bp) >= 0) or np.any(np.diff(ap) <= 0):
        raise ValueError("book is crossed, locked, or unsorted")
    return arrays


def _imbalance(book: tuple[np.ndarray, ...]) -> float:
    """@brief Compute depth-weighted quantity imbalance using currently known sizes."""
    bid_volume = float(book[1].sum())
    ask_volume = float(book[3].sum())
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)


def _episode(rows: list[dict[str, Any]], metadata: dict[str, Any],
             window: int) -> ReplayEpisode:
    """@brief Convert one contiguous segment into arrays with trailing volatility.
    @details Volatility is the population standard deviation of at most window
    observed log-midprice returns. It uses the current and preceding rows only,
    resets after gaps and day boundaries, and requires no global normalization.
    """
    columns = {name: [row[name] for row in rows] for name in rows[0]}
    timestamp = np.asarray(columns.pop("timestamp_ns"), dtype=np.int64)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in columns.items()}
    mid = (arrays["binance_bid"] + arrays["binance_ask"]) / 2.0
    volatility = np.zeros(len(mid), dtype=np.float64)
    returns: deque[float] = deque(maxlen=window)
    for index in range(1, len(mid)):
        returns.append(float(np.log(mid[index] / mid[index - 1]) * 10_000.0))
        volatility[index] = float(np.std(returns)) if len(returns) >= 2 else 0.0
    result = ReplayEpisode(
        timestamp_ns=timestamp, volatility_bps=volatility, metadata=metadata, **arrays
    )
    result.validate()
    return result


def build_episodes(
    records: Iterable[dict[str, Any]], config: QualityConfig | None = None,
    *, synthetic: bool = False, source_files: list[str] | None = None,
    episode_sink: Callable[[ReplayEpisode], None] | None = None,
) -> tuple[list[ReplayEpisode], dict[str, Any]]:
    """@brief Backward-align receive events and report every exclusion category.
    @details The most recently received valid book on each venue is the only
    available state. Invalid updates invalidate that venue's cached book. Negative
    clock ages are counted here and cause prepare() to fail under reject policy.
    Explicit funding settlement events appear once at their receive timestamp;
    carried/predicted funding-rate fields on ordinary book updates are ignored.
    """
    config = config or QualityConfig()
    config.validate()
    counts: Counter[str] = Counter()
    venues: Counter[str] = Counter()
    symbols: dict[str, set[str]] = {"binance": set(), "hyperliquid": set()}
    books: dict[str, tuple[int, int, tuple[np.ndarray, ...]]] = {}
    episodes: list[ReplayEpisode] = []
    rows: list[dict[str, Any]] = []
    previous_time: int | None = None
    previous_day: int | None = None
    seen_funding: set[tuple[Any, ...]] = set()
    max_negative_clock_ms = 0.0
    largest_gap_ms = 0.0
    observed_days: set[int] = set()

    def finish() -> None:
        """@brief Close a segment and record short-segment exclusions explicitly."""
        nonlocal rows
        if not rows:
            return
        if len(rows) >= config.min_episode_steps:
            day = str(np.datetime64(rows[0]["timestamp_ns"] // DAY_NS, "D"))
            episode = _episode(rows, {
                "day": day, "synthetic": synthetic, "clock_policy": config.clock_policy,
                "source_files": source_files or [], "quality_config": asdict(config),
                "funding_semantics": "settlement_events_only",
                "quote_age_semantics": (
                    "local_receive_age" if config.clock_policy == "receive"
                    else "local_time_minus_exchange_time"
                ),
            }, config.volatility_window)
            counts["usable_episodes"] += 1
            counts["usable_rows"] += len(episode)
            if episode_sink is None:
                episodes.append(episode)
            else:
                episode_sink(episode)
        else:
            counts["short_segments"] += 1
            counts["short_segment_rows"] += len(rows)
        rows = []

    for record in records:
        counts["input_records"] += 1
        now = _timestamp(record.get("local_ts_ns"), "local_ts_ns")
        if previous_time is not None and now < previous_time:
            raise ValueError("input records must be ordered by local receive time")
        day = now // DAY_NS
        observed_days.add(day)
        #! @details Day boundaries and receive-stream outages invalidate cached
        #! quotes, ensuring segments never bridge overnight or missing intervals.
        if previous_day is not None and day != previous_day:
            finish()
            books.clear()
        if previous_time is not None:
            gap_ms = (now - previous_time) / 1_000_000.0
            largest_gap_ms = max(largest_gap_ms, gap_ms)
            if gap_ms > config.max_gap_ms:
                counts["receive_gaps"] += 1
                finish()
                books.clear()
            elif gap_ms == 0:
                counts["duplicate_receive_timestamps"] += 1
        previous_time, previous_day = now, day
        #! @details Older ClickHouse Arrow exports may encode String as binary;
        #! decode only this venue identifier without touching integer timestamps.
        raw_venue = record.get("exchange", "")
        venue_name = raw_venue.decode("utf-8") if isinstance(raw_venue, bytes) else str(raw_venue)
        venue = VENUE_NAMES.get(venue_name.lower())
        if venue is None:
            counts["unknown_venue_records"] += 1
            continue
        venues[venue] += 1
        if record.get("symbol") is not None:
            symbols[venue].add(str(record["symbol"]))
            if len(symbols[venue]) > 1:
                raise ValueError(f"multiple symbols for {venue}; prepare one market pair")
        event = str(record.get("event_type", record.get("type", "book"))).lower()
        funding = 0.0
        is_funding = event in {"funding", "funding_settlement"}
        if is_funding:
            #! @details A settlement never updates quote timestamps or refreshes
            #! stale liquidity. Duplicate settlement IDs/timestamps are excluded.
            if venue != "hyperliquid":
                counts["ignored_non_hl_funding_events"] += 1
                continue
            try:
                funding = float(record["funding_rate"])
                if not np.isfinite(funding):
                    raise ValueError("nonfinite funding settlement")
            except (KeyError, TypeError, ValueError):
                counts["invalid_funding_events"] += 1
                finish()
                continue
            identity = (record.get("event_id"), now) if record.get("event_id") is None else (
                record["event_id"],
            )
            if identity in seen_funding:
                counts["duplicate_funding_events"] += 1
                continue
            seen_funding.add(identity)
            counts["funding_events"] += 1
        else:
            #! @details Book validity is checked before cached liquidity is
            #! replaced. A corrupt update removes the old quote immediately.
            if record.get("funding_rate") is not None:
                counts["ignored_carried_funding_rates"] += 1
            try:
                exchange_time = _timestamp(record.get("exchange_ts_ns"), "exchange_ts_ns")
                book = _book(record, config.depth)
                spread_bps = (book[2][0] - book[0][0]) / (
                    (book[2][0] + book[0][0]) / 2.0
                ) * 10_000.0
                if spread_bps > config.max_spread_bps:
                    counts["excessive_spread_records"] += 1
                    raise ValueError("spread exceeds quality threshold")
            except (TypeError, ValueError, OverflowError):
                counts["invalid_book_records"] += 1
                books.pop(venue, None)
                finish()
                continue
            clock_age = (now - exchange_time) / 1_000_000.0
            if clock_age < 0:
                counts["negative_clock_age_records"] += 1
                max_negative_clock_ms = max(max_negative_clock_ms, -clock_age)
                if config.clock_policy == "reject":
                    books.pop(venue, None)
                    finish()
                    continue
            books[venue] = (now, exchange_time, book)
            counts["valid_book_records"] += 1
        if len(books) != 2:
            counts["unpaired_rows"] += 1
            if is_funding:
                counts["unusable_funding_events"] += 1
            continue
        b_received, b_exchange, binance = books["binance"]
        h_received, h_exchange, hyperliquid = books["hyperliquid"]
        b_receive_age = (now - b_received) / 1_000_000.0
        h_receive_age = (now - h_received) / 1_000_000.0
        b_quote_age = b_receive_age if config.clock_policy == "receive" else (
            now - b_exchange
        ) / 1_000_000.0
        h_quote_age = h_receive_age if config.clock_policy == "receive" else (
            now - h_exchange
        ) / 1_000_000.0
        #! @details Both venues must remain fresh. Excluded intervals terminate
        #! the episode so execution cannot interpolate across a data-quality gap.
        if max(b_receive_age, h_receive_age) > config.max_received_age_ms:
            counts["stale_receive_rows"] += 1
            finish()
            if is_funding:
                counts["unusable_funding_events"] += 1
            continue
        if max(b_quote_age, h_quote_age) > config.max_quote_age_ms:
            counts["stale_exchange_rows"] += 1
            finish()
            if is_funding:
                counts["unusable_funding_events"] += 1
            continue
        binance_mid = (binance[0][0] + binance[2][0]) / 2.0
        hl_mid = (hyperliquid[0][0] + hyperliquid[2][0]) / 2.0
        if abs(binance_mid / hl_mid - 1.0) * 10_000.0 > config.max_cross_venue_gap_bps:
            counts["excessive_cross_venue_gap_rows"] += 1
            finish()
            if is_funding:
                counts["unusable_funding_events"] += 1
            continue
        rows.append({
            "timestamp_ns": now,
            "binance_bid": binance[0][0], "binance_ask": binance[2][0],
            "hl_bid_prices": hyperliquid[0], "hl_bid_sizes": hyperliquid[1],
            "hl_ask_prices": hyperliquid[2], "hl_ask_sizes": hyperliquid[3],
            "binance_imbalance": _imbalance(binance),
            "hl_imbalance": _imbalance(hyperliquid),
            "hl_quote_age_ms": h_quote_age, "hl_received_age_ms": h_receive_age,
            "funding_rate": funding,
        })
        counts["aligned_rows"] += 1
        #! @details Bound live preprocessing memory by emitting fixed-length
        #! segments. Each segment restarts its trailing features and replay state.
        if len(rows) >= config.max_episode_steps:
            finish()
    finish()
    return episodes, {
        "counts": dict(counts), "venue_counts": dict(venues),
        "symbols": {key: sorted(value) for key, value in symbols.items()},
        "observed_utc_days": [
            str(np.datetime64(day, "D")) for day in sorted(observed_days)
        ],
        "max_negative_clock_age_ms": max_negative_clock_ms,
        "largest_receive_gap_ms": largest_gap_ms,
        "quality_config": asdict(config), "usable_episodes": counts["usable_episodes"],
        "usable_rows": counts["usable_rows"],
        "synthetic": synthetic,
    }


def _day_splits(days: Iterable[str], train_fraction: float,
                validation_fraction: float) -> dict[str, str]:
    """@brief Assign complete UTC days to nonempty chronological held-out splits."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train and validation fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train plus validation fraction must leave test data")
    days = sorted(set(days))
    if len(days) < 3:
        raise ValueError(
            f"need at least 3 usable UTC days for train/validation/test; found {len(days)}"
        )
    train_count = max(1, min(len(days) - 2, int(len(days) * train_fraction)))
    validation_count = max(1, min(
        len(days) - train_count - 1, int(len(days) * validation_fraction)
    ))
    return {
        day: "train" if index < train_count else (
            "validation" if index < train_count + validation_count else "test"
        ) for index, day in enumerate(days)
    }


def write_dataset(
    episodes: Iterable[ReplayEpisode], report: dict[str, Any], output_dir: str | Path,
    *, train_fraction: float = 0.7, validation_fraction: float = 0.15,
    days: Iterable[str] | None = None,
) -> Path:
    """@brief Write checked archives, hashes, provenance, and one split manifest.
    @details The quality report is written even when chronology or clock checks
    reject dataset creation. Existing manifests are never silently overwritten.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"dataset already exists: {manifest_path}")
    (destination / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if report["quality_config"]["clock_policy"] == "reject" and report["counts"].get(
        "negative_clock_age_records", 0
    ):
        raise ValueError(
            "negative exchange-clock ages detected; inspect quality_report.json. "
            "Use --clock-policy receive only to explicitly select local receive ages."
        )
    if days is None:
        episodes = list(episodes)
        days = [episode.metadata["day"] for episode in episodes]
    splits = _day_splits(days, train_fraction, validation_fraction)
    entries = []
    for index, episode in enumerate(episodes):
        day = episode.metadata["day"]
        split = splits[day]
        episode_id = f"{day}-{index:05d}"
        episode.metadata.update({"split": split, "episode_id": episode_id})
        relative_path = Path("episodes") / f"{episode_id}.npz"
        saved = episode.save(destination / relative_path)
        with saved.open("rb") as handle:
            content_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        entries.append({
            "path": relative_path.as_posix(), "day": day, "split": split,
            "episode_id": episode_id, "rows": len(episode),
            "start_timestamp_ns": int(episode.timestamp_ns[0]),
            "end_timestamp_ns": int(episode.timestamp_ns[-1]),
            "synthetic": bool(episode.metadata.get("synthetic", False)),
            "content_sha256": content_hash,
        })
    manifest = {
        "schema_version": 1, "synthetic": bool(report["synthetic"]),
        "purpose": (
            "SYNTHETIC SOFTWARE DEMONSTRATION; NOT EVIDENCE OF PROFITABILITY"
            if report["synthetic"] else "offline historical replay research"
        ),
        "split_method": "whole UTC days in chronological order",
        "requested_train_fraction": train_fraction,
        "requested_validation_fraction": validation_fraction,
        "day_splits": splits, "quality_report": "quality_report.json",
        "quality_config": report["quality_config"], "episodes": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def prepare(
    inputs: Iterable[str | Path], output_dir: str | Path,
    *, config: QualityConfig | None = None, train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> Path:
    """@brief Prepare one market pair from explicitly supplied local data files."""
    paths = [Path(path) for path in inputs]
    if not paths:
        raise ValueError("at least one input file is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"dataset already exists: {destination / 'manifest.json'}")
    staging = destination / "_preparation"
    staging.mkdir(exist_ok=False)
    staged: list[tuple[Path, str]] = []

    def save_segment(episode: ReplayEpisode) -> None:
        """@brief Spill each bounded segment immediately; retain only path metadata."""
        target = staging / f"{len(staged):06d}.npz"
        episode.save(target)
        staged.append((target, episode.metadata["day"]))

    def reload_segments() -> Iterator[ReplayEpisode]:
        """@brief Finalize one staged archive at a time after day splits are known."""
        for target, _ in staged:
            yield ReplayEpisode.load(target)
            target.unlink()

    try:
        _, report = build_episodes(
            read_records(paths), config,
            source_files=[str(path.resolve()) for path in paths],
            episode_sink=save_segment,
        )
        return write_dataset(
            reload_segments(), report, destination, train_fraction=train_fraction,
            validation_fraction=validation_fraction, days=[day for _, day in staged],
        )
    finally:
        #! @details Delete only exact temporary filenames created in this call;
        #! retain the quality report and any finalized output for inspection.
        for target, _ in staged:
            if target.exists():
                target.unlink()
        staging.rmdir()


def _synthetic_records(days: int, steps_per_day: int, seed: int,
                       depth: int) -> Iterator[dict[str, Any]]:
    """@brief Produce deterministic artificial books with a delayed follower.
    @details This process is deliberately simple and is solely for exercising
    the offline pipeline. Its edge, quote frequency, and volume distribution
    are invented and cannot substantiate real-market performance claims.
    """
    rng = np.random.default_rng(seed)
    first_day = date(2026, 1, 1)
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        day_ns = int(np.datetime64(day.isoformat(), "D").astype(np.int64)) * DAY_NS
        mids = 60_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.00012, steps_per_day)))
        for step in range(steps_per_day):
            now = day_ns + 12 * 3_600_000_000_000 + step * 100_000_000
            for venue, mid, receive_offset in (
                ("binance", mids[step], 0),
                ("hyperliquid", mids[max(0, step - 3)], 20_000_000),
            ):
                half_spread = float(mid) * 0.000025
                distances = np.arange(depth, dtype=np.float64) * float(mid) * 0.00002
                bid_sizes = rng.uniform(0.4, 2.0, depth)
                ask_sizes = rng.uniform(0.4, 2.0, depth)
                yield {
                    "exchange": venue, "local_ts_ns": now + receive_offset,
                    "exchange_ts_ns": now + receive_offset - (
                        20_000_000 if venue == "binance" else 285_000_000
                    ),
                    "bids": np.column_stack((mid - half_spread - distances, bid_sizes)).tolist(),
                    "asks": np.column_stack((mid + half_spread + distances, ask_sizes)).tolist(),
                }


def create_demo(output_dir: str | Path, *, days: int = 6,
                steps_per_day: int = 300, seed: int = 7, depth: int = 10) -> Path:
    """@brief Save a reproducible, conspicuously synthetic multi-day dataset."""
    if days < 3 or steps_per_day < 20:
        raise ValueError("demo requires at least 3 days and 20 steps per day")
    config = QualityConfig(depth=depth)
    episodes, report = build_episodes(
        _synthetic_records(days, steps_per_day, seed, depth), config, synthetic=True
    )
    report["seed"] = seed
    for episode in episodes:
        episode.metadata["synthetic_seed"] = seed
        episode.metadata["warning"] = "Artificial demo; no real-market profitability evidence"
    return write_dataset(episodes, report, output_dir)


def main(argv: list[str] | None = None) -> None:
    """@brief Parse explicit offline prepare/demo commands and print their manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="create clearly labeled synthetic data")
    demo.add_argument("--output", "--output-dir", dest="output", required=True)
    demo.add_argument("--days", type=int, default=6)
    demo.add_argument("--steps-per-day", type=int, default=300)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--depth", type=int, default=10)
    prep = commands.add_parser("prepare", help="check local JSONL/Parquet and split by UTC day")
    prep.add_argument("inputs", nargs="+")
    prep.add_argument("--output", "--output-dir", dest="output", required=True)
    prep.add_argument("--clock-policy", choices=("reject", "receive"), default="reject")
    prep.add_argument("--depth", type=int, default=10)
    prep.add_argument("--min-episode-steps", type=int, default=32)
    prep.add_argument("--max-episode-steps", type=int, default=100_000)
    prep.add_argument("--max-received-age-ms", type=float, default=1000.0)
    prep.add_argument("--max-quote-age-ms", type=float, default=2000.0)
    prep.add_argument("--max-gap-ms", type=float, default=1000.0)
    prep.add_argument("--max-spread-bps", type=float, default=100.0)
    prep.add_argument("--max-cross-venue-gap-bps", type=float, default=250.0)
    prep.add_argument("--volatility-window", type=int, default=20)
    prep.add_argument("--train-fraction", type=float, default=0.7)
    prep.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = create_demo(
                args.output, days=args.days, steps_per_day=args.steps_per_day,
                seed=args.seed, depth=args.depth,
            )
        else:
            config = QualityConfig(**{
                field: getattr(args, field) for field in QualityConfig.__dataclass_fields__
            })
            result = prepare(
                args.inputs, args.output, config=config,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(result)


if __name__ == "__main__":
    main()
