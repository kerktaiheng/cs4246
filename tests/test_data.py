"""@file test_data.py
@brief Behavioral checks for causal replay preparation and data integrity.
@details Tests use artificial books and temporary files only; no external client
or network is involved. They target leakage, clock semantics, gaps, funding, and
held-out split integrity rather than reproducing implementation details.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_arb.data.generate_episodes import (
    DAY_NS, QualityConfig, build_episodes, create_demo, prepare, read_records,
)
from latency_arb.data.schema import ReplayEpisode, load_manifest


def records(days: int = 1, steps: int = 8) -> list[dict]:
    """@brief Create known two-level books with exact nanosecond event times."""
    base = 1_767_225_600_000_000_003
    output = []
    for day in range(days):
        for step in range(steps):
            for venue, offset in (("binance", 0), ("hyperliquid", 10_000_000)):
                now = base + day * DAY_NS + step * 100_000_000 + offset
                mid = 100.0 + step * 0.001
                output.append({
                    "exchange": venue, "local_ts_ns": now,
                    "exchange_ts_ns": now - 20_000_000,
                    "bids": [[mid - 0.01, 2.0], [mid - 0.02, 3.0]],
                    "asks": [[mid + 0.01, 2.5], [mid + 0.02, 2.0]],
                })
    return output


def config(**changes) -> QualityConfig:
    """@brief Keep fixtures small while retaining all production quality checks."""
    return QualityConfig(depth=2, min_episode_steps=2, **changes)


def write_jsonl(path: Path, values: list[dict]) -> None:
    """@brief Preserve integer timestamps when writing local fixture records."""
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n")


def test_backward_alignment_and_future_changes_do_not_leak():
    """@brief Future updates cannot change an earlier state or volatility feature."""
    original = records()
    revised = copy.deepcopy(original)
    for record in revised[8:]:
        for side in ("bids", "asks"):
            for level in record[side]:
                level[0] += 0.1
    first, _ = build_episodes(original, config())
    second, _ = build_episodes(revised, config())
    cutoff = original[8]["local_ts_ns"]
    mask = first[0].timestamp_ns < cutoff
    for name in ("binance_bid", "hl_bid_prices", "volatility_bps", "gap_bps"):
        np.testing.assert_array_equal(getattr(first[0], name)[mask], getattr(second[0], name)[mask])
    #! @details The first Binance update after pairing still sees the preceding
    #! Hyperliquid bid; the subsequent Hyperliquid event makes its new bid visible.
    assert first[0].hl_bid[1] == original[1]["bids"][0][0]
    assert first[0].hl_bid[2] == original[3]["bids"][0][0]


def test_nanoseconds_roundtrip_and_float_rejected(tmp_path):
    """@brief Archive and parsing paths retain nanoseconds beyond float precision."""
    episodes, _ = build_episodes(records(), config())
    path = episodes[0].save(tmp_path / "episode.npz")
    loaded = ReplayEpisode.load(path)
    np.testing.assert_array_equal(loaded.timestamp_ns, episodes[0].timestamp_ns)
    assert loaded.timestamp_ns[0] % 10 == 3
    bad = records()
    bad[0]["local_ts_ns"] = float(bad[0]["local_ts_ns"])
    with pytest.raises(ValueError, match="exact integer"):
        build_episodes(bad, config())


def test_negative_clock_rejects_and_explicit_receive_fallback(tmp_path):
    """@brief Negative venue clocks cannot be silently clipped into valid quote ages."""
    source = records(days=3)
    source[1]["exchange_ts_ns"] = source[1]["local_ts_ns"] + 17_000_000
    path = tmp_path / "records.jsonl"
    write_jsonl(path, source)
    with pytest.raises(ValueError, match="negative exchange-clock"):
        prepare([path], tmp_path / "reject", config=config())
    report = json.loads((tmp_path / "reject" / "quality_report.json").read_text())
    assert report["counts"]["negative_clock_age_records"] == 1
    assert report["max_negative_clock_age_ms"] == 17
    assert not (tmp_path / "reject" / "manifest.json").exists()
    manifest = prepare([path], tmp_path / "receive", config=config(clock_policy="receive"))
    for episode in load_manifest(manifest):
        np.testing.assert_array_equal(episode.hl_quote_age_ms, episode.hl_received_age_ms)
        assert episode.metadata["quote_age_semantics"] == "local_receive_age"


def test_invalid_book_ends_segment_and_invalidates_cached_quote():
    """@brief A corrupt update cannot leave its previous liquidity executable."""
    source = records(steps=10)
    source[7]["bids"][0][0] = source[7]["asks"][0][0] + 1
    episodes, report = build_episodes(source, config())
    assert report["counts"]["invalid_book_records"] == 1
    assert len(episodes) == 2
    assert episodes[0].timestamp_ns[-1] < source[7]["local_ts_ns"]
    assert episodes[1].timestamp_ns[0] >= source[9]["local_ts_ns"]


def test_receive_gaps_split_episodes_and_require_fresh_pair():
    """@brief Long outages terminate replay instead of carrying liquidity across them."""
    source = records(steps=10)
    for record in source[10:]:
        record["local_ts_ns"] += 5_000_000_000
        record["exchange_ts_ns"] += 5_000_000_000
    episodes, report = build_episodes(source, config())
    assert len(episodes) == 2
    assert report["counts"]["receive_gaps"] == 1
    assert episodes[1].timestamp_ns[0] == source[11]["local_ts_ns"]


def test_funding_is_settlement_only_and_duplicate_events_are_excluded():
    """@brief Ordinary carried rates never become repeated portfolio charges."""
    source = records()
    for record in source:
        record["funding_rate"] = 0.02
    settlement = {
        "exchange": "hyperliquid", "event_type": "funding_settlement",
        "event_id": "one-settlement", "local_ts_ns": source[5]["local_ts_ns"] + 1,
        "funding_rate": 0.0001,
    }
    source.extend([settlement, dict(settlement)])
    source.sort(key=lambda record: record["local_ts_ns"])
    episodes, report = build_episodes(source, config())
    rates = np.concatenate([episode.funding_rate for episode in episodes])
    assert np.count_nonzero(rates) == 1
    assert rates.sum() == pytest.approx(0.0001)
    assert report["counts"]["duplicate_funding_events"] == 1
    assert report["counts"]["ignored_carried_funding_rates"] == 16


def test_chronological_whole_day_splits_and_synthetic_labels(tmp_path):
    """@brief Reproducible demo data has disjoint ordered days and explicit provenance."""
    path = create_demo(tmp_path / "first", days=6, steps_per_day=32, seed=5, depth=2)
    second = create_demo(tmp_path / "second", days=6, steps_per_day=32, seed=5, depth=2)
    train = load_manifest(path, "train")
    validation = load_manifest(path, "validation")
    test = load_manifest(path, "test")
    assert train[-1].timestamp_ns[-1] < validation[0].timestamp_ns[0]
    assert validation[-1].timestamp_ns[-1] < test[0].timestamp_ns[0]
    assert all(episode.metadata["synthetic"] for episode in train + validation + test)
    np.testing.assert_array_equal(train[0].hl_mid, load_manifest(second, "train")[0].hl_mid)


def test_parquet_and_jsonl_preserve_same_exact_replay(tmp_path):
    """@brief Both supported file types produce the same receive-time ordered arrays."""
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    source = records(days=3)
    json_path = tmp_path / "input.jsonl"
    parquet_path = tmp_path / "input.parquet"
    write_jsonl(json_path, source)
    parquet.write_table(arrow.Table.from_pylist(source), parquet_path)
    left = prepare([json_path], tmp_path / "json", config=config())
    right = prepare([parquet_path], tmp_path / "parquet", config=config())
    for lhs, rhs in zip(load_manifest(left), load_manifest(right)):
        np.testing.assert_array_equal(lhs.timestamp_ns, rhs.timestamp_ns)
        np.testing.assert_array_equal(lhs.hl_bid_prices, rhs.hl_bid_prices)


def test_short_dataset_reports_coverage_and_refuses_fake_holdout(tmp_path):
    """@brief One observed day cannot be advertised as three held-out periods."""
    source = tmp_path / "one-day.jsonl"
    write_jsonl(source, records())
    with pytest.raises(ValueError, match="at least 3 usable UTC days"):
        prepare([source], tmp_path / "out", config=config())
    assert (tmp_path / "out" / "quality_report.json").exists()
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_source_order_is_checked_before_merging(tmp_path):
    """@brief Source reversals are rejected rather than hidden by global sorting."""
    source = records()
    source[2], source[3] = source[3], source[2]
    path = tmp_path / "unsorted.jsonl"
    write_jsonl(path, source)
    with pytest.raises(ValueError, match="not ordered"):
        list(read_records([path]))


def test_streaming_bounds_and_staging_cleanup(tmp_path):
    """@brief Production preparation caps segment size and removes only its staging files."""
    source = tmp_path / "source.jsonl"
    write_jsonl(source, records(days=3, steps=25))
    path = prepare([source], tmp_path / "out", config=config(max_episode_steps=10))
    episodes = load_manifest(path)
    assert all(len(episode) <= 10 for episode in episodes)
    assert not (tmp_path / "out" / "_preparation").exists()


def test_manifest_checks_hash_paths_and_empty_split(tmp_path):
    """@brief Loading detects changed archives, path traversal, and absent split data."""
    path = create_demo(tmp_path / "demo", days=3, steps_per_day=24, depth=2)
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["episodes"][0]["path"] = "../outside.npz"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inside"):
        load_manifest(path, "train")
    changed = copy.deepcopy(original)
    changed["episodes"] = [entry for entry in changed["episodes"] if entry["split"] != "test"]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="no episodes"):
        load_manifest(path, "test")
    path.write_text(json.dumps(original))
    archive = path.parent / original["episodes"][0]["path"]
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path, "train")


def test_manifest_rejects_reordered_and_overlapping_episodes(tmp_path):
    """@brief Split labels alone do not excuse overlapping/reversed replay intervals."""
    path = create_demo(tmp_path / "demo", days=6, steps_per_day=24, depth=2)
    document = json.loads(path.read_text())
    document["episodes"][0], document["episodes"][1] = (
        document["episodes"][1], document["episodes"][0]
    )
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="ordered"):
        load_manifest(path, "train")


def test_clickhouse_import_and_construction_are_offline(monkeypatch):
    """@brief Accidental imports cannot inspect a server or require any credentials."""
    for name in ("CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    module = importlib.import_module("common.clickhouse_client")
    client = module.ClickHouseClient()
    client.close()
    with pytest.raises(ValueError, match="Missing explicit"):
        module.client_from_environment()


def test_separate_array_books_and_arrow_binary_venue_names(tmp_path):
    """@brief Existing database exports retain venue identity and array-book support."""
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    source = records(days=3)
    for record in source:
        record["exchange"] = record["exchange"].encode("utf-8")
        bids, asks = record.pop("bids"), record.pop("asks")
        record["bid_prices"] = [level[0] for level in bids]
        record["bid_sizes"] = [level[1] for level in bids]
        record["ask_prices"] = [level[0] for level in asks]
        record["ask_sizes"] = [level[1] for level in asks]
    path = tmp_path / "existing-export.parquet"
    parquet.write_table(arrow.Table.from_pylist(source), path)
    manifest = prepare([path], tmp_path / "out", config=config())
    assert len(load_manifest(manifest)) == 3


def recorded_batch(venue: str, start_ns: int, seconds: int = 610, bad_second=None):
    """@brief Build source updates just after integer seconds to detect bucket lookahead."""
    import pyarrow as arrow

    values = []
    for second in range(seconds):
        now = start_ns + second * 1_000_000_000 + (
            50_000_003 if venue == "binance" else 100_000_007
        )
        mid = 100.0 + second * 0.0001
        exchange = now - 20_000_000
        if bad_second == second:
            exchange = now + 1
        row = {
            "local_ts_ns": now, "exchange_ts_ns": exchange,
            "full_source_valid": 1,
            "bid_depth": 20 if venue == "binance" else 5,
            "ask_depth": 20 if venue == "binance" else 5,
            "bid_size_depth": 20 if venue == "binance" else 5,
            "ask_size_depth": 20 if venue == "binance" else 5,
        }
        if venue == "binance":
            row.update({"bid": mid - 0.01, "ask": mid + 0.01,
                        "bid_volume": 20.0, "ask_volume": 25.0})
        else:
            row.update({
                "bid_prices": [mid - 0.01 - index * 0.01 for index in range(5)],
                "ask_prices": [mid + 0.01 + index * 0.01 for index in range(5)],
                "bid_sizes": [1.0] * 5, "ask_sizes": [1.5] * 5,
            })
        values.append(row)
    return arrow.RecordBatch.from_pylist(values)


class RecordedTestClient:
    """@brief Offline stand-in for verifying safe day-scoped streaming SQL calls."""

    def __init__(self, bad_second=None):
        """@brief Retain call evidence and an optional clock anomaly fixture."""
        self.calls = []
        self.bad_second = bad_second

    def query_arrow_stream(self, query, *, parameters, settings, use_strings):
        """@brief Yield one local Arrow batch through the driver's context protocol."""
        from contextlib import contextmanager

        self.calls.append((query, parameters, settings))
        assert "exchange = {venue:String}" in query
        assert "local_ts_ns >= {start:Int64}" in query
        assert "local_ts_ns < {end:Int64}" in query
        assert settings["max_threads"] == 2
        assert settings["max_memory_usage"] <= 1_073_741_824
        assert settings["max_execution_time"] == 90
        assert "AS full_source_valid" in query
        assert use_strings
        batch = recorded_batch(
            parameters["venue"], parameters["start"],
            bad_second=self.bad_second if parameters["venue"] == "hyperliquid" else None,
        )

        @contextmanager
        def stream():
            """@brief Mimic context-managed Arrow stream without any networking."""
            yield iter([batch])

        return stream()


def test_recorded_grid_is_backward_asof_with_fixed_complete_windows(tmp_path):
    """@brief Decision seconds never use the later source record inside their bucket."""
    from latency_arb.data.recorded import (
        RecordedConfig, export_recorded_day, prepare_recorded_day, _day_ns,
    )

    client = RecordedTestClient()
    cache = export_recorded_day(client, "2026-09-03", tmp_path / "raw")
    assert len(client.calls) == 2
    result = prepare_recorded_day(cache, tmp_path / "dataset")
    assert result["report"]["valid_fixed_windows"] == 1
    assert result["report"]["rejected_fixed_windows"] == 287
    entry = result["episodes"][0]
    episode = ReplayEpisode.load(tmp_path / "dataset" / entry["path"])
    start = _day_ns("2026-09-03")
    assert episode.timestamp_ns[0] == start + 300_000_000_000
    assert episode.timestamp_ns[-1] == start + 599_300_000_000
    assert episode.binance_bid[0] == pytest.approx(100 + 299 * 0.0001 - 0.01)
    assert episode.binance_bid[1] == pytest.approx(100 + 300 * 0.0001 - 0.01)
    assert np.all(episode.timestamp_ns[np.asarray(episode.metadata["decision_indices"])] %
                  1_000_000_000 == 0)
    assert len(episode) == 900
    assert episode.metadata["episode_boundary_policy"] == "fixed_utc_windows_complete_quality"
    assert episode.metadata["funding_unavailable_zero_assumption"]
    assert np.all(episode.funding_rate == 0)
    #! @details Validated day checkpoints avoid both repeated source queries and
    #! regenerating archives during a resumed full-month preparation run.
    export_recorded_day(client, "2026-09-03", tmp_path / "raw")
    repeated = prepare_recorded_day(cache, tmp_path / "dataset", config=RecordedConfig())
    assert len(client.calls) == 2
    assert repeated["signature"] == result["signature"]


def test_recorded_clock_anomaly_rejects_entire_fixed_window(tmp_path):
    """@brief A bad clock does not expose a data-dependent episode end in advance."""
    from latency_arb.data.recorded import export_recorded_day, prepare_recorded_day

    cache = export_recorded_day(
        RecordedTestClient(bad_second=400), "2026-09-03", tmp_path / "raw"
    )
    result = prepare_recorded_day(cache, tmp_path / "dataset")
    assert result["report"]["source"]["hyperliquid"]["negative_clock_age_rows"] == 1
    assert result["report"]["valid_fixed_windows"] == 0
    assert result["episodes"] == []


def test_recorded_dataset_uses_fixed_labels_and_resumable_progress(tmp_path):
    """@brief Staged date exports keep predeclared split labels and provenance."""
    from latency_arb.data.recorded import prepare_recorded_dataset, fixed_split

    path = prepare_recorded_dataset(
        RecordedTestClient(), tmp_path / "dataset",
        start="2026-09-25", end="2026-09-25",
        source_inventory={"recorded_rows": 32_888_724, "clean_range_rows": 28_917_283},
    )
    episodes = load_manifest(path, "validation")
    assert len(episodes) == 1
    assert episodes[0].metadata["split"] == "validation"
    progress = json.loads((path.parent / "progress.json").read_text())
    assert progress["last_completed_day"] == "2026-09-25"
    manifest = json.loads(path.read_text())
    assert manifest["recorded_hl_depth"] == 5
    assert manifest["proposal_depth"] == 10
    assert fixed_split("2026-09-24") == "train"
    assert fixed_split("2026-09-29") == "test"
    with pytest.raises(ValueError, match="predeclared"):
        fixed_split("2026-09-02")



def test_recorded_projection_checks_full_source_depth_before_discarding_it():
    """@brief A malformed deep Binance book cannot be hidden by valid top/sum fields."""
    from latency_arb.data.recorded import _source_validity, RecordedConfig

    columns = {
        "local_ts_ns": np.array([100_000_000, 200_000_000], dtype=np.int64),
        "exchange_ts_ns": np.array([90_000_000, 190_000_000], dtype=np.int64),
        "bid_depth": np.array([20, 20]), "ask_depth": np.array([20, 20]),
        "bid_size_depth": np.array([20, 20]), "ask_size_depth": np.array([20, 20]),
        "bid": np.array([99.0, 99.0]), "ask": np.array([99.1, 99.1]),
        "bid_volume": np.array([20.0, 20.0]), "ask_volume": np.array([25.0, 25.0]),
        "full_source_valid": np.array([1, 0], dtype=np.uint8),
    }
    valid, report = _source_validity(columns, "binance", RecordedConfig())
    assert valid.tolist() == [True, False]
    assert report["invalid_full_source_book_rows"] == 1
    columns.pop("full_source_valid")
    with pytest.raises(ValueError, match="projection version 2"):
        _source_validity(columns, "binance", RecordedConfig())
