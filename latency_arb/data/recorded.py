"""@file recorded.py
@brief Bounded, resumable preparation of the recorded September research dataset.
@details Explicit client calls export one venue and UTC day at a time. Local
NumPy searchsorted performs backward receive-time alignment at exact decision
and execution timestamps. A bucket's final observation is never relabeled as
its start. Credentials, clients, and network connections are never created here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np

from .schema import ReplayEpisode

DAY_NS = 86_400_000_000_000
SECOND_NS = 1_000_000_000
PROJECTION_VERSION = 2
Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RecordedConfig:
    """@brief Fixed sampling and quality rules declared before policy fitting.
    @details Negative source clock ages invalidate that quote's interval only.
    Funding is explicitly unavailable and assumed zero. Five Hyperliquid levels
    match the recorded source; the ten-level proposal assumption is not invented.
    """

    depth: int = 5
    execution_offsets_ms: tuple[int, ...] = (150, 300)
    max_received_age_ms: float = 2000.0
    max_quote_age_ms: float = 3000.0
    max_spread_bps: float = 100.0
    max_cross_venue_gap_bps: float = 250.0
    episode_seconds: int = 300
    max_episode_rows: int = 10800
    require_complete_windows: bool = True
    min_decision_steps: int = 32
    volatility_window: int = 20

    def validate(self) -> None:
        """@brief Reject ambiguous grids and unusable segmentation thresholds."""
        if self.depth != 5:
            raise ValueError("recorded Hyperliquid data has exactly five levels")
        offsets = self.execution_offsets_ms
        if not offsets or tuple(sorted(set(offsets))) != tuple(offsets) or any(
            type(value) is not int or not 0 < value < 1000 for value in offsets
        ):
            raise ValueError("execution offsets must be unique sorted integer milliseconds in (0,1000)")
        if self.episode_seconds < 1 or self.min_decision_steps < 2:
            raise ValueError("episode_seconds >= 1 and min_decision_steps >= 2 required")
        if self.max_episode_rows < (len(offsets) + 1) * self.min_decision_steps:
            raise ValueError("max_episode_rows is too small for minimum decision count")
        if 86400 % self.episode_seconds:
            raise ValueError("episode_seconds must divide a UTC day")
        if self.require_complete_windows and (
            self.episode_seconds * (len(offsets) + 1) > self.max_episode_rows
        ):
            raise ValueError("a complete fixed window must fit max_episode_rows")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least two")
        for field in ("max_received_age_ms", "max_quote_age_ms", "max_spread_bps",
                      "max_cross_venue_gap_bps"):
            if not np.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")


def _sha256(path: Path) -> str:
    """@brief Hash one file using bounded reads without retaining its bytes."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """@brief Replace a module-owned JSON checkpoint only after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _day_ns(day: str) -> int:
    """@brief Convert an ISO UTC day to exact integer nanoseconds without floats."""
    parsed = date.fromisoformat(day)
    return (parsed - date(1970, 1, 1)).days * DAY_NS


def fixed_split(day: str) -> str:
    """@brief Apply the predeclared 22/4/4 chronological research split."""
    if "2026-09-03" <= day <= "2026-09-24":
        return "train"
    if "2026-09-25" <= day <= "2026-09-28":
        return "validation"
    if "2026-09-29" <= day <= "2026-10-02":
        return "test"
    raise ValueError(f"{day} is outside the predeclared clean research range")


def export_recorded_day(
    client: Any, day: str, cache_dir: str | Path, *,
    table: str = "order_book_states", progress: Progress | None = None,
) -> dict[str, Any]:
    """@brief Explicitly stream and cache one day, one venue query at a time.
    @param client Already authorized ClickHouse client supplied by the caller.
    @details Each query binds exchange and exact day bounds, caps server threads
    at two, memory at one GiB, and execution at 90 seconds. Binance retains scalar
    top prices and full-source-depth imbalance; Hyperliquid retains five levels.
    Existing raw cache files are reused only after their content hashes verify.
    """
    import pyarrow.parquet as parquet

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError("table must be a simple optional database-qualified identifier")
    fixed_split(day)
    start = _day_ns(day)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    venues = {}
    for venue in ("binance", "hyperliquid"):
        target = cache / f"{day}.{venue}.parquet"
        checkpoint = target.with_suffix(".json")
        if checkpoint.exists() and target.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("day") == day and saved.get("venue") == venue and saved.get("table") == table and saved.get("projection_version") == PROJECTION_VERSION and (
                saved.get("content_sha256") == _sha256(target)
            ):
                venues[venue] = saved
                if progress:
                    progress({"stage": "export_cached", "day": day, "venue": venue,
                              "rows": saved["rows"]})
                continue
        if client is None:
            raise ValueError(
                f"{day} {venue} requires a complete verified projection-v2 cache "
                "or an explicitly supplied database client"
            )
        #! @details Validate full source arrays before projecting Binance depth
        #! away; positive aggregate volume cannot conceal negative individual
        #! quantities or out-of-order deeper prices in its imbalance feature.
        source_validity_sql = (
            "(length(bid_prices) > 0 AND length(ask_prices) > 0 "
            "AND length(bid_prices)=length(bid_sizes) "
            "AND length(ask_prices)=length(ask_sizes) "
            "AND arrayAll(x -> isFinite(x) AND x>0, bid_prices) "
            "AND arrayAll(x -> isFinite(x) AND x>0, ask_prices) "
            "AND arrayAll(x -> isFinite(x) AND x>0, bid_sizes) "
            "AND arrayAll(x -> isFinite(x) AND x>0, ask_sizes) "
            "AND arrayAll((x,y) -> x>y, arrayPopBack(bid_prices), arrayPopFront(bid_prices)) "
            "AND arrayAll((x,y) -> x<y, arrayPopBack(ask_prices), arrayPopFront(ask_prices)) "
            "AND bid_prices[1]<ask_prices[1]) AS full_source_valid"
        )
        common = (
            "toInt64(local_ts_ns) AS local_ts_ns, "
            "toInt64(exchange_ts_ns) AS exchange_ts_ns, "
            "length(bid_prices) AS bid_depth, length(ask_prices) AS ask_depth, "
            "length(bid_sizes) AS bid_size_depth, length(ask_sizes) AS ask_size_depth"
        )
        if venue == "binance":
            specific = (
                "bid_prices[1] AS bid, ask_prices[1] AS ask, "
                "arraySum(bid_sizes) AS bid_volume, arraySum(ask_sizes) AS ask_volume"
            )
        else:
            specific = (
                "arraySlice(bid_prices,1,5) AS bid_prices, "
                "arraySlice(bid_sizes,1,5) AS bid_sizes, "
                "arraySlice(ask_prices,1,5) AS ask_prices, "
                "arraySlice(ask_sizes,1,5) AS ask_sizes"
            )
        query = (
            f"SELECT {common}, {source_validity_sql}, {specific} FROM {table} "
            "WHERE exchange = {venue:String} AND local_ts_ns >= {start:Int64} "
            "AND local_ts_ns < {end:Int64} ORDER BY local_ts_ns"
        )
        temporary = target.with_suffix(".parquet.partial")
        writer = None
        count = 0
        minimum = None
        maximum = None
        try:
            with client.query_arrow_stream(
                query, parameters={"venue": venue, "start": start, "end": start + DAY_NS},
                settings={"max_threads": 2, "max_execution_time": 90,
                          "max_memory_usage": 1_073_741_824, "max_block_size": 65536},
                use_strings=True,
            ) as batches:
                for batch in batches:
                    if writer is None:
                        writer = parquet.ParquetWriter(temporary, batch.schema, compression="zstd")
                    writer.write_batch(batch)
                    timestamps = batch.column("local_ts_ns").to_numpy()
                    if len(timestamps):
                        minimum = int(timestamps[0]) if minimum is None else minimum
                        maximum = int(timestamps[-1])
                    count += batch.num_rows
            if writer is None:
                raise ValueError(f"no {venue} records found for {day}")
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(target)
        saved = {
            "day": day, "venue": venue, "path": str(target.resolve()), "rows": count,
            "start_timestamp_ns": minimum, "end_timestamp_ns": maximum,
            "content_sha256": _sha256(target), "table": table,
            "projection": "scalar_binance_full_depth_imbalance_or_five_level_hyperliquid",
            "projection_version": PROJECTION_VERSION,
        }
        _write_json(checkpoint, saved)
        venues[venue] = saved
        if progress:
            progress({"stage": "exported", "day": day, "venue": venue, "rows": count})
    return {"day": day, "venues": venues}


def _columns(path: Path, venue: str, depth: int) -> dict[str, np.ndarray]:
    """@brief Read one projected venue day into compact numeric arrays.
    @details Regular Arrow lists flatten directly to numeric matrices. The rare
    malformed-depth fallback pads only that field with NaN, making its source row
    fail validity checks instead of crashing or manufacturing tradable depth.
    """
    import pyarrow as arrow
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    table = parquet.read_table(path)
    result = {}
    for name in table.column_names:
        column = table[name].combine_chunks()
        if arrow.types.is_list(column.type) or arrow.types.is_large_list(column.type):
            lengths = compute.list_value_length(column).to_numpy(zero_copy_only=False)
            if len(lengths) and np.all(lengths == depth):
                result[name] = column.values.to_numpy(zero_copy_only=False).astype(
                    np.float64, copy=False
                ).reshape(-1, depth)
            else:
                padded = np.full((len(column), depth), np.nan)
                for index, values in enumerate(column.to_pylist()):
                    if values is not None and len(values) >= depth:
                        padded[index] = values[:depth]
                result[name] = padded
        else:
            values = column.to_numpy(zero_copy_only=False)
            if name.endswith("_ts_ns"):
                if values.dtype.kind not in "iu":
                    raise ValueError("raw cache timestamps must be integer nanoseconds")
                values = values.astype(np.int64, copy=False)
            result[name] = values
    received = result["local_ts_ns"]
    if len(received) == 0 or np.any(received < 0) or np.any(received[1:] < received[:-1]):
        raise ValueError(f"{venue} cache must contain ordered nonnegative receive times")
    return result


def _source_validity(columns: dict[str, np.ndarray], venue: str,
                     config: RecordedConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """@brief Compute per-source-row exclusions before any as-of lookup.
    @details An invalid latest row invalidates the grid interval until another
    update arrives; searchsorted never skips a bad row to resurrect older quotes.
    """
    local = columns["local_ts_ns"]
    exchange = columns["exchange_ts_ns"]
    clock_ok = (exchange >= 0) & (exchange <= local)
    depth_ok = (
        (columns["bid_depth"] >= config.depth) &
        (columns["ask_depth"] >= config.depth) &
        (columns["bid_size_depth"] == columns["bid_depth"]) &
        (columns["ask_size_depth"] == columns["ask_depth"])
    )
    if venue == "binance":
        bid, ask = columns["bid"], columns["ask"]
        volumes = np.column_stack((columns["bid_volume"], columns["ask_volume"]))
        book_ok = np.isfinite(volumes).all(axis=1) & (volumes > 0).all(axis=1)
    else:
        bp, bs, ap, ass = (
            columns[name] for name in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes")
        )
        bid, ask = bp[:, 0], ap[:, 0]
        book_ok = np.ones(len(local), dtype=bool)
        for values in (bp, bs, ap, ass):
            book_ok &= np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        book_ok &= (np.diff(bp, axis=1) < 0).all(axis=1)
        book_ok &= (np.diff(ap, axis=1) > 0).all(axis=1)
    price_ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (ask - bid) / ((ask + bid) / 2.0) * 10000.0
    spread_ok = np.isfinite(spread) & (spread <= config.max_spread_bps)
    if "full_source_valid" not in columns:
        raise ValueError("raw cache lacks full-source validity; re-export projection version 2")
    source_ok = columns["full_source_valid"].astype(bool)
    valid = clock_ok & depth_ok & book_ok & price_ok & spread_ok & source_ok
    gaps = np.diff(local)
    return valid, {
        "rows": len(local), "invalid_rows": int((~valid).sum()),
        "negative_clock_age_rows": int((exchange > local).sum()),
        "invalid_depth_rows": int((~depth_ok).sum()),
        "invalid_full_source_book_rows": int((~source_ok).sum()),
        "invalid_book_rows": int((~(book_ok & price_ok)).sum()),
        "excessive_spread_rows": int((~spread_ok).sum()),
        "bid_depth_min": int(columns["bid_depth"].min()),
        "bid_depth_max": int(columns["bid_depth"].max()),
        "receive_gaps_over_2s": int((gaps > 2 * SECOND_NS).sum()),
        "largest_receive_gap_ms": float(gaps.max() / 1_000_000.0) if len(gaps) else 0.0,
    }


def _trailing_volatility(mid: np.ndarray, window: int) -> np.ndarray:
    """@brief Compute causal rolling return volatility using cumulative moments."""
    returns = np.zeros(len(mid), dtype=np.float64)
    returns[1:] = np.log(mid[1:] / mid[:-1]) * 10000.0
    values = returns[1:]
    total = np.concatenate(([0.0], np.cumsum(values)))
    squares = np.concatenate(([0.0], np.cumsum(values * values)))
    end = np.arange(1, len(mid))
    start = np.maximum(0, end - window)
    count = end - start
    means = (total[end] - total[start]) / count
    variance = (squares[end] - squares[start]) / count - means * means
    result = np.zeros(len(mid), dtype=np.float64)
    result[1:] = np.sqrt(np.maximum(variance, 0.0))
    return result


def _segments(grid: np.ndarray, valid: np.ndarray,
              config: RecordedConfig) -> Iterable[np.ndarray]:
    """@brief Yield fixed complete windows or explicitly requested contiguous slices.
    @details The default accepts a fixed five-minute window only when every row
    is valid, preventing future outages from becoming known terminal times.
    Optional partial-window runs break at every invalid grid row. Chunking
    respects whole seconds so a decision and its execution snapshots stay together.
    A short fragment is reported by the caller rather than joined across a hole.
    """
    if config.require_complete_windows:
        #! @details Window start and terminal time are fixed before inspecting
        #! data. A single bad grid row rejects the entire window; never tell the
        #! policy in advance where a subsequently discovered outage will begin.
        width = config.episode_seconds * (len(config.execution_offsets_ms) + 1)
        groups = np.arange(len(grid), dtype=np.int64).reshape(-1, width)
        for group in groups[valid.reshape(-1, width).all(axis=1)]:
            yield group
        return
    indices = np.flatnonzero(valid)
    if not len(indices):
        return
    hour = grid // (config.episode_seconds * SECOND_NS)
    boundaries = np.flatnonzero(
        (np.diff(indices) != 1) | (hour[indices[1:]] != hour[indices[:-1]])
    ) + 1
    rows_per_second = len(config.execution_offsets_ms) + 1
    seconds_per_chunk = max(1, config.max_episode_rows // rows_per_second)
    for run in np.split(indices, boundaries):
        decisions = np.flatnonzero(grid[run] % SECOND_NS == 0)
        if not len(decisions):
            continue
        run = run[decisions[0]:]
        #! @details Bound each chunk by elapsed whole decision seconds, not merely
        #! by available rows; this preserves the declared one-second decision rate.
        while len(run):
            cutoff = grid[run[0]] + seconds_per_chunk * SECOND_NS
            stop = int(np.searchsorted(grid[run], cutoff, side="left"))
            yield run[:stop]
            run = run[stop:]


def prepare_recorded_day(
    cache: dict[str, Any], output_dir: str | Path, *,
    config: RecordedConfig | None = None, progress: Progress | None = None,
) -> dict[str, Any]:
    """@brief Align a cached day causally and checkpoint verified replay archives.
    @details Memory is bounded by one projected raw day plus its sampled arrays.
    Daily checkpoints are resumed only if configuration, raw hashes, and every
    output episode hash agree. Funding remains zero with an explicit limitation.
    """
    config = config or RecordedConfig()
    config.validate()
    day = cache["day"]
    split = fixed_split(day)
    destination = Path(output_dir)
    checkpoint = destination / "days" / f"{day}.json"
    signature_data = {
        "config": asdict(config), "raw_hashes": {
            venue: value["content_sha256"] for venue, value in cache["venues"].items()
        }, "pipeline_version": 1,
    }
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("signature") == signature and all(
            (destination / entry["path"]).exists() and
            _sha256(destination / entry["path"]) == entry["content_sha256"]
            for entry in saved.get("episodes", [])
        ):
            if progress:
                progress({"stage": "prepared_cached", "day": day,
                          "episodes": len(saved["episodes"]),
                          "decision_rows": saved["report"]["retained_decision_rows"]})
            return saved
    source = {}
    source_valid = {}
    source_reports = {}
    for venue in ("binance", "hyperliquid"):
        raw = cache["venues"][venue]
        path = Path(raw["path"])
        if _sha256(path) != raw["content_sha256"]:
            raise ValueError(f"{day} {venue} raw cache hash mismatch")
        source[venue] = _columns(path, venue, config.depth)
        source_valid[venue], source_reports[venue] = _source_validity(
            source[venue], venue, config
        )
    day_start = _day_ns(day)
    offsets = np.asarray((0,) + config.execution_offsets_ms, dtype=np.int64) * 1_000_000
    seconds = day_start + np.arange(86400, dtype=np.int64) * SECOND_NS
    grid = (seconds[:, None] + offsets[None, :]).reshape(-1)
    paired = np.ones(len(grid), dtype=bool)
    valid = np.ones(len(grid), dtype=bool)
    matched = {}
    receive_ages = {}
    quote_ages = {}
    exclusion = {}
    #! @details right-side search selects the final event received at or before
    #! each exact grid timestamp. Negative indices are masked before use.
    for venue in ("binance", "hyperliquid"):
        columns = source[venue]
        positions = np.searchsorted(columns["local_ts_ns"], grid, side="right") - 1
        available = positions >= 0
        paired &= available
        safe_positions = np.maximum(positions, 0)
        matched[venue] = safe_positions
        receive_age = (grid - columns["local_ts_ns"][safe_positions]) / 1_000_000.0
        quote_age = (grid - columns["exchange_ts_ns"][safe_positions]) / 1_000_000.0
        receive_ages[venue], quote_ages[venue] = receive_age, quote_age
        latest_valid = source_valid[venue][safe_positions]
        fresh_receive = (receive_age >= 0) & (receive_age <= config.max_received_age_ms)
        fresh_quote = (quote_age >= 0) & (quote_age <= config.max_quote_age_ms)
        valid &= available & latest_valid & fresh_receive & fresh_quote
        exclusion[venue] = {
            "unavailable_grid_rows": int((~available).sum()),
            "invalid_latest_quote_grid_rows": int((available & ~latest_valid).sum()),
            "stale_receive_grid_rows": int((available & ~fresh_receive).sum()),
            "stale_exchange_grid_rows": int((available & ~fresh_quote).sum()),
        }
    b = source["binance"]
    h = source["hyperliquid"]
    bi, hi = matched["binance"], matched["hyperliquid"]
    bmid = (b["bid"][bi] + b["ask"][bi]) / 2.0
    hmid = (h["bid_prices"][hi, 0] + h["ask_prices"][hi, 0]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gaps = (bmid / hmid - 1.0) * 10000.0
    good_gap = np.isfinite(gaps) & (np.abs(gaps) <= config.max_cross_venue_gap_bps)
    valid &= good_gap
    entries = []
    short_rows = 0
    retained_rows = 0
    retained_decisions = 0
    split_dir = destination / "episodes" / day
    split_dir.mkdir(parents=True, exist_ok=True)
    for indices in _segments(grid, valid, config):
        decisions = np.flatnonzero(grid[indices] % SECOND_NS == 0)
        if len(decisions) < config.min_decision_steps:
            short_rows += len(indices)
            continue
        bidx, hidx = bi[indices], hi[indices]
        bvolume = b["bid_volume"][bidx] + b["ask_volume"][bidx]
        hbidvol = h["bid_sizes"][hidx].sum(axis=1)
        haskvol = h["ask_sizes"][hidx].sum(axis=1)
        episode_id = f"{day}-{len(entries):04d}"
        episode = ReplayEpisode(
            timestamp_ns=grid[indices].copy(),
            binance_bid=b["bid"][bidx].astype(np.float64),
            binance_ask=b["ask"][bidx].astype(np.float64),
            hl_bid_prices=h["bid_prices"][hidx].copy(),
            hl_bid_sizes=h["bid_sizes"][hidx].copy(),
            hl_ask_prices=h["ask_prices"][hidx].copy(),
            hl_ask_sizes=h["ask_sizes"][hidx].copy(),
            binance_imbalance=((b["bid_volume"][bidx] - b["ask_volume"][bidx]) / bvolume).astype(np.float64),
            hl_imbalance=((hbidvol - haskvol) / (hbidvol + haskvol)).astype(np.float64),
            volatility_bps=_trailing_volatility(bmid[indices], config.volatility_window),
            hl_quote_age_ms=quote_ages["hyperliquid"][indices].astype(np.float64),
            hl_received_age_ms=receive_ages["hyperliquid"][indices].astype(np.float64),
            funding_rate=np.zeros(len(indices), dtype=np.float64),
            metadata={
                "day": day, "split": split, "episode_id": episode_id, "synthetic": False,
                "decision_interval_ms": 1000, "decision_indices": decisions.tolist(),
                "execution_offsets_ms": list(config.execution_offsets_ms),
                "source": "recorded_clickhouse_order_book_states",
                "alignment": "backward_asof_local_receive_time",
                "clock_policy": "exclude_negative_source_age_intervals",
                "quote_age_semantics": "local_time_minus_exchange_time",
                "funding_semantics": "unavailable_zero_assumption",
                "funding_unavailable_zero_assumption": True,
                "hl_recorded_depth": 5, "proposal_depth": 10,
                "binance_imbalance_depth": "all_recorded_levels",
                                "quality_config": asdict(config), "raw_cache_hashes": signature_data["raw_hashes"],
                "episode_boundary_policy": (
                    "fixed_utc_windows_complete_quality" if config.require_complete_windows
                    else "quality_selected_horizon_contains_boundary_hindsight"
                ),
                "scheduled_window_seconds": config.episode_seconds,
            },
        )
        target = split_dir / f"{episode_id}.npz"
        episode.save(target)
        entry = {
            "path": target.relative_to(destination).as_posix(), "day": day, "split": split,
            "episode_id": episode_id, "rows": len(episode),
            "decision_rows": len(decisions), "synthetic": False,
            "start_timestamp_ns": int(episode.timestamp_ns[0]),
            "end_timestamp_ns": int(episode.timestamp_ns[-1]),
            "content_sha256": _sha256(target),
        }
        entries.append(entry)
        retained_rows += len(episode)
        retained_decisions += len(decisions)
    window_width = config.episode_seconds * len(offsets)
    complete_windows = valid.reshape(-1, window_width).all(axis=1)
    report = {
        "total_fixed_windows": len(complete_windows),
        "valid_fixed_windows": int(complete_windows.sum()),
        "rejected_fixed_windows": int((~complete_windows).sum()),
        "retained_seconds": retained_decisions,
        "fixed_window_seconds": config.episode_seconds,
        "require_complete_windows": config.require_complete_windows,
        "day": day, "split": split, "source": source_reports, "grid_exclusions": exclusion,
        "grid_rows": len(grid), "paired_grid_rows": int(paired.sum()),
        "valid_grid_rows": int(valid.sum()), "retained_rows": retained_rows,
        "retained_decision_rows": retained_decisions, "short_segment_rows": short_rows,
        "cross_venue_gap_excluded_rows": int((paired & ~good_gap).sum()),
        "episodes": len(entries), "clock_policy": "exclude_negative_source_age_intervals",
        "funding": "unavailable: zero settlement-rate assumption, not measured",
        "quality_config": asdict(config), "synthetic": False,
    }
    result = {"signature": signature, "day": day, "episodes": entries, "report": report}
    _write_json(checkpoint, result)
    if progress:
        progress({"stage": "prepared", "day": day, "episodes": len(entries),
                  "decision_rows": retained_decisions})
    return result


def prepare_recorded_dataset(
    client: Any, output_dir: str | Path, *, cache_dir: str | Path | None = None,
    config: RecordedConfig | None = None, start: str = "2026-09-03",
    end: str = "2026-10-02", table: str = "order_book_states",
    progress: Progress | None = None, source_inventory: dict | None = None,
) -> Path:
    """@brief Export/prepare an explicit date range and write one resumable manifest.
    @details Defaults cover the predeclared 22 training, four validation and four
    test days. A partial range is allowed for staged execution and never changes
    those labels. No live credentials, connections, or training are created here.
    """
    config = config or RecordedConfig()
    config.validate()
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    fixed_split(start)
    fixed_split(end)
    if first > last:
        raise ValueError("start must be no later than end")
    destination = Path(output_dir)
    raw_cache = Path(cache_dir) if cache_dir is not None else destination / "raw"
    entries = []
    daily_reports = []
    raw_rows = {"binance": 0, "hyperliquid": 0}
    current = first
    while current <= last:
        day = current.isoformat()
        cache = export_recorded_day(client, day, raw_cache, table=table, progress=progress)
        result = prepare_recorded_day(cache, destination, config=config, progress=progress)
        entries.extend(result["episodes"])
        daily_reports.append(result["report"])
        for venue in raw_rows:
            raw_rows[venue] += cache["venues"][venue]["rows"]
        #! @details Daily progress is atomically checkpointed even if a later
        #! network request fails. Reruns verify raw/output hashes before reuse.
        _write_json(destination / "progress.json", {
            "last_completed_day": day, "start": start, "end": end,
            "completed_days": len(daily_reports), "episodes": len(entries),
            "raw_rows": raw_rows,
        })
        current += timedelta(days=1)
    if not entries:
        raise ValueError("no valid recorded episodes survived declared quality checks")
    quality = {
        "synthetic": False, "day_reports": daily_reports, "raw_rows": raw_rows,
        "retained_rows": sum(entry["rows"] for entry in entries),
        "retained_decision_rows": sum(entry["decision_rows"] for entry in entries),
        "source_inventory": source_inventory or {},
        "limitations": [
            "Hyperliquid records contain five levels, versus ten in the proposal.",
            "Funding settlement rates are unavailable; all funding is assumed zero.",
            "One-second decisions plus fixed execution offsets approximate event replay.",
        ],
    }
    _write_json(destination / "quality_report.json", quality)
    manifest = {
        "schema_version": 1, "synthetic": False,
        "purpose": "recorded offline cross-venue replay research",
        "split_method": "predeclared fixed UTC dates: 22 train / 4 validation / 4 test",
        "day_splits": {report["day"]: report["split"] for report in daily_reports},
        "quality_report": "quality_report.json", "quality_config": asdict(config),
        "source_inventory": source_inventory or {}, "raw_rows": raw_rows,
        "decision_interval_ms": 1000, "execution_offsets_ms": list(config.execution_offsets_ms),
        "funding_unavailable_zero_assumption": True,
        "recorded_hl_depth": 5, "proposal_depth": 10, "episodes": entries,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
