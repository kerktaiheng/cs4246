"""@file test_report.py
@brief Check that rendered reports preserve evidence, omissions, and interpretations.
@details Small synthetic JSON fixtures test reporting decisions independently from
PPO and execution. No model is trained and no market or network connection is used.
"""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from latency_arb.report import collect_report_data, generate_report


def write_json(path, value):
    """@brief Save a test-owned artifact without changing any experiment fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def aggregate(pnl=0., trades=0):
    """@brief Define explicit measured outputs, including undefined zero-trade metrics."""
    return {"synthetic": True, "net_pnl": pnl, "net_return": pnl / 10000,
            "max_drawdown_usd": abs(min(pnl, 0)), "trade_count": trades,
            "win_rate": .5 if trades else None, "trade_selectivity": trades / 4,
            "daily_pnl": {"2026-01-05": pnl}, "abstention_rate": 1. if trades == 0 else .5,
            "hold_action_rate": .9, "flat_exposure_rate": .8, "flat_entry_rate": trades / 10}


@pytest.fixture
def saved_run(tmp_path):
    """@brief Build a clearly synthetic run with flat best baseline and abstaining PPO."""
    run = tmp_path / "run"
    comparison = {"config": {"position_size_btc": .001, "fee_bps": 3.5, "latency_ms": 150,
                               "slippage_bps": .1, "decision_interval_ms": 1000,
                               "max_holding_ms": 30000, "max_drawdown_usd": 100},
                  "policies": {"flat": {"aggregate": aggregate()},
                               "threshold": {"aggregate": aggregate(-.25, 3)},
                               "ppo": {"aggregate": aggregate()}}}
    candidates = [{"name": "ppo_seed7", "training_config": {"seed": 7, "ent_coef": .01},
                   "aggregate": aggregate()},
                  {"name": "ppo_seed17", "training_config": {"seed": 17, "ent_coef": .01},
                   "aggregate": aggregate(-2, 8)}]
    sensitivity = [{"policy": name, "fee_bps": fee, "latency_ms": delay,
                    "spread_multiplier": spread, "net_pnl": -fee if name == "threshold" else 0,
                    "trade_count": 3 if name == "threshold" else 0}
                   for fee in (.5, 1, 2, 3.5) for delay in (150, 300) for spread in (1, 1.5)
                   for name in ("flat", "threshold", "ppo")]
    brackets = [{"policy": "ppo", "latency_ms": 150, "spread_multiplier": 1,
                 "status": "no_trades", "highest_profitable_sampled_fee_bps": None,
                 "next_nonpositive_sampled_fee_bps": None}]
    write_json(run / "summary.json", {"synthetic": True, "selected_candidate": "ppo_seed7",
               "threshold_bps": 8, "retry_triggered": False, "candidates": candidates,
               "validation": comparison, "test": comparison, "sensitivity": sensitivity,
               "fee_break_even": brackets, "selected_ppo_zero_validation_trades": True})
    write_json(run / "test" / "comparison.json", {"report": comparison, "protocol_sha256": "fixture"})
    write_json(run / "models" / "ppo_seed7" / "training.json", {"actual_timesteps": 1024})
    (run / "models" / "ppo_seed7" / "training_episodes.monitor.csv").write_text(
        '#{"fixture":true}\nr,l,t,trade_count,pnl,fees_paid\n0,30,1,0,0,0\n-1,30,2,3,-1,.1\n', encoding="utf-8")
    return run


def test_report_distinguishes_abstention_from_a_profitable_strategy(saved_run):
    """@brief Zero test trades must remain an explicit failure to establish executable edge."""
    data = collect_report_data(saved_run)
    text = " ".join(data["executive"])
    assert data["synthetic"] is True
    assert data["evidence_kind"] == "SYNTHETIC PIPELINE CHECK"
    assert "best held-out baseline is flat" in text
    assert "PPO made zero test trades" in text
    assert "not convergence" in text
    assert "not evidence of market profitability" in text
    assert "also made zero validation trades" in text
    diagnostics = data["diagnostics_table"]["rows"]
    assert diagnostics[0][1:] == ["1,024", "2", "3", "50.0%"]
    assert diagnostics[1][1:] == ["N/A", "N/A", "N/A", "N/A"]


def test_missing_test_and_sensitivity_are_not_invented(saved_run):
    """@brief An unfinished report cannot promote validation performance to held-out evidence."""
    summary_path = saved_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["test"] = None
    summary["sensitivity"] = None
    write_json(summary_path, summary)
    (saved_run / "test" / "comparison.json").unlink()
    data = collect_report_data(saved_run)
    assert data["test_available"] is False
    assert data["test"] is None
    assert data["sensitivity"] == []
    assert any("report is incomplete" in text for text in data["executive"])
    assert not any("best held-out baseline" in text for text in data["executive"])


def test_provenance_counts_coverage_and_fixed_windows_use_saved_values(saved_run, tmp_path):
    """@brief Inventory, clean input, and retained decisions remain separate measured counts."""
    manifest = tmp_path / "data" / "manifest.json"
    write_json(manifest, {"synthetic": True, "source_inventory": {"venue_totals": [{"rows": 15}, {"rows": 4}]},
                         "raw_rows": {"binance": 10, "hyperliquid": 2}, "recorded_hl_depth": 5,
                         "funding_unavailable_zero_assumption": True, "execution_offsets_ms": [150, 300],
                         "quality_config": {"require_complete_windows": True, "episode_seconds": 300},
                         "quality_report": "quality_report.json",
                         "day_splits": {"2026-01-01": "train", "2026-01-02": "validation", "2026-01-03": "test"}})
    write_json(manifest.parent / "quality_report.json", {"retained_decision_rows": 900,
               "day_reports": [{"total_fixed_windows": 4, "valid_fixed_windows": 3}]})
    data = collect_report_data(saved_run, manifest)
    observed = dict(data["data_table"]["rows"])
    assert observed["Source inventory raw rows"] == "19"
    assert observed["Selected clean-range raw rows"] == "12"
    assert observed["Retained decision rows"] == "900"
    assert observed["Window retention"] == "75.00%"
    assert any("Funding is assumed zero" in text for text in data["limitations"])
    assert any("20 Binance levels" in text for text in data["limitations"])
    assert any("fixed UTC 300 second windows" in text for text in data["method"])
    assert any("+150, 300 ms" in text for text in data["method"])
    assert all(len(source["sha256"]) == 64 for source in data["sources"])
    with pytest.raises(FileNotFoundError, match="Explicit report manifest"):
        collect_report_data(saved_run, tmp_path / "missing.json")


def test_renderer_generates_real_charts_and_preserves_metric_meanings(saved_run):
    """@brief An end-to-end synthetic report includes four nonempty, correctly labeled figures."""
    result = generate_report(saved_run, include_pdf=False)
    assert result["pdf"] is None
    assert result["pdf_status"] == "not_requested"
    assert len(result["charts"]) == 4
    assert all(Path(path).stat().st_size > 5000 for path in result["charts"])
    markdown = (saved_run / "report.md").read_text()
    assert "SYNTHETIC PIPELINE CHECK" in markdown
    assert "Abstention is the fraction of eligible flat/no-pending decisions" in markdown
    assert "Flat exposure is elapsed replay time" in markdown
    assert "observed fee-grid brackets only" in markdown
    assert "no_trades" in markdown
    assert "report_data.json" not in result["charts"]


def test_saved_source_hash_changes_when_actual_evidence_changes(saved_run):
    """@brief The report provenance must identify the exact inputs, not just their filenames."""
    first = collect_report_data(saved_run)
    path = saved_run / "summary.json"
    value = json.loads(path.read_text())
    value["threshold_bps"] = 12
    write_json(path, value)
    second = collect_report_data(saved_run)
    first_hash = next(row["sha256"] for row in first["sources"] if row["path"] == str(path))
    second_hash = next(row["sha256"] for row in second["sources"] if row["path"] == str(path))
    assert first_hash != second_hash
    assert second["selected_threshold_bps"] == 12



def test_legacy_hold_fraction_is_not_relabelled_as_current_abstention(saved_run):
    """@brief Old artifacts without metric-contract fields must show unknown abstention."""
    from latency_arb.report import _sections
    data = collect_report_data(saved_run)
    legacy = data["test"]["policies"]["ppo"]["aggregate"]
    legacy.pop("hold_action_rate")
    legacy["abstention_rate"] = .93
    activity = next(section for section in _sections(data) if section["title"] == "Abstention and exposure")
    ppo = next(row for row in activity["tables"][0]["rows"] if row[0] == "ppo")
    assert ppo[1] == "N/A"


def test_report_preserves_complete_threshold_search_and_roundtrip_fee_definition(saved_run):
    """@brief Show every validation threshold rather than only a favorable winner."""
    from latency_arb.report import _sections
    candidates = [{"entry_threshold_bps": threshold, **aggregate(pnl, trades)}
                  for threshold, pnl, trades in [(1, -12, 80), (7, -1, 9), (10, .1, 2), (100, 0, 0)]]
    write_json(saved_run / "validation" / "baselines.json",
               {"selection": {"candidates": candidates, "entry_threshold_bps": 10}})
    data = collect_report_data(saved_run)
    assert [row[0] for row in data["threshold_table"]["rows"]] == ["1.00", "7.00", "10.00", "100.00"]
    assert any("nominal two-sided fee burden is 7.00 bps before spread and slippage" in text for text in data["method"])
    assert any("may omit volatile or stale periods" in text for text in data["limitations"])
    assert any(section["title"] == "Validation threshold search" for section in _sections(data))
    assert any(row["path"].endswith("training_episodes.monitor.csv") for row in data["sources"])
