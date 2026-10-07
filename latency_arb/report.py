"""@file report.py
@brief Render saved experiment evidence as Markdown, Matplotlib figures, and PDF.
@details This local-only renderer never trains, selects, or evaluates a policy.
Missing results remain unavailable, and synthetic results remain conspicuously
labeled. --pdf-only uses ReportLab and existing PNGs without importing Matplotlib.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

COLORS = {"flat": "#718096", "threshold": "#d18b31", "ppo": "#186c7a"}


def _read(path: Path, default=None):
    """@brief Read present JSON strictly while preserving absent artifacts as absent."""
    if not path.exists():
        return default
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _unwrap(value):
    """@brief Accept direct evaluator JSON and protocol-bound report wrappers."""
    return value.get("report", value) if isinstance(value, dict) else value


def _num(value, digits=4, percent=False):
    """@brief Format measured values without replacing unknown numbers with zero."""
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * (100 if percent else 1):,.{digits}f}" + ("%" if percent else "")


def _count(value):
    """@brief Display exact integer counts and explicit unavailable values."""
    return "N/A" if value is None else f"{int(value):,}"


def _table(headers, rows):
    """@brief Share a small, serializable table contract between output formats."""
    return {"headers": headers, "rows": rows}


def _comparison(report):
    """@brief Present identical economic columns across policies and held-out splits."""
    rows = []
    for name, values in (report or {}).get("policies", {}).items():
        item = values["aggregate"]
        rows.append([name, _num(item.get("net_pnl")), _num(item.get("net_return"), 4, True),
                     _num(item.get("max_drawdown_usd")), _count(item.get("trade_count")),
                     _num(item.get("win_rate"), 1, True), _num(item.get("trade_selectivity"), 3)])
    return _table(["Policy", "Net P&L USD", "Return", "Max DD USD", "Trades", "Win rate", "Selectivity"], rows)


def _diagnostics(run, candidates):
    """@brief Read actual training steps and episode trade counts from saved artifacts.
    @details Requested budgets are never substituted for completed training steps.
    Monitor rows measure training exploration, separately from deterministic inference.
    """
    results = []
    for candidate in candidates:
        name = candidate.get("name", "candidate")
        folder = run / "models" / name
        metadata = _read(folder / "training.json", {})
        path = folder / "training_episodes.monitor.csv"
        episodes = trades = zero = 0
        if path.exists():
            with path.open(encoding="utf-8-sig") as handle:
                for row in csv.DictReader(line for line in handle if not line.startswith("#")):
                    if row.get("trade_count") not in (None, ""):
                        count = int(float(row["trade_count"]))
                        trades += count
                        zero += count == 0
                    episodes += 1
        results.append([name, _count(metadata.get("actual_timesteps")),
                        _count(episodes if path.exists() else None),
                        _count(trades if path.exists() else None),
                        _num(zero / episodes, 1, True) if episodes else "N/A"])
    return results


def collect_report_data(run_dir, manifest_path=None):
    """@brief Collect evidence without loading market archives or inferring absent results.
    @param run_dir Existing experiment output directory.
    @param manifest_path Explicit manifest, otherwise experiment.protocol.manifest.
    @return A JSON-serializable content model and source checksums.
    """
    run = Path(run_dir).resolve()
    summary, selection = _read(run / "summary.json", {}), _read(run / "selection.json", {})
    experiment = _read(run / "experiment.json", {})
    protocol = experiment.get("protocol", {})
    reference = manifest_path or protocol.get("manifest")
    manifest_file = Path(reference) if reference else None
    if manifest_path is not None and not manifest_file.exists():
        raise FileNotFoundError(f"Explicit report manifest does not exist: {manifest_file}")
    manifest = _read(manifest_file, {}) if manifest_file else {}
    quality = _read(manifest_file.parent / manifest["quality_report"], {}) if manifest_file and manifest.get("quality_report") else {}
    validation = _unwrap(_read(run / "validation" / "comparison.json", summary.get("validation")))
    testing = _unwrap(_read(run / "test" / "comparison.json", summary.get("test")))
    # @details Legacy pipeline checks retain their actual split; no validation file
    # is promoted to a test result merely to populate the final report.
    legacy = _unwrap(_read(run / "evaluation" / "comparison.json"))
    if not validation and not testing and legacy:
        if legacy.get("split") == "test":
            testing = legacy
        else:
            validation = legacy
    candidates = summary.get("candidates") or selection.get("candidates") or []
    baseline_record = _read(run / "validation" / "baselines.json", {})
    threshold_selection = selection.get("threshold_selection") or baseline_record.get("selection", {})
    threshold_candidates = threshold_selection.get("candidates", [])
    threshold_rows = [[_num(row.get("entry_threshold_bps"), 2), _num(row.get("net_pnl")),
                       _count(row.get("trade_count")), _num(row.get("fees_paid")),
                       _num(row.get("max_drawdown_usd")), _num(row.get("win_rate"), 1, True)]
                      for row in threshold_candidates]
    sensitivity = _read(run / "sensitivity.json", summary.get("sensitivity"))
    if sensitivity is None:
        sensitivity = _read(run / "evaluation" / "sensitivity.json", [])
    if isinstance(sensitivity, dict):
        sensitivity = sensitivity.get("rows", [])
    brackets = _read(run / "fee_break_even.json", summary.get("fee_break_even"))
    if brackets is None:
        brackets = _read(run / "evaluation" / "fee_break_even.json", [])
    flags = [value["aggregate"].get("synthetic", False) for report in (validation, testing) if report
             for value in report.get("policies", {}).values()]
    synthetic = bool(summary.get("synthetic") or manifest.get("synthetic") or any(flags))
    label = "SYNTHETIC PIPELINE CHECK" if synthetic else "RECORDED OFFLINE RESEARCH" if manifest else "OFFLINE EXPERIMENT - DATA PROVENANCE UNAVAILABLE"
    config = (testing or validation or {}).get("config", protocol.get("evaluation_environment", {}))
    selected = summary.get("selected_candidate", selection.get("selected_candidate", "Not recorded"))
    executive = []
    if synthetic:
        executive.append("These results use synthetic data and verify the software pipeline only. They are not evidence of market profitability.")
    if not testing:
        executive.append("Final held-out test results are unavailable. This report is incomplete and reports validation evidence only where present.")
    else:
        policies = testing.get("policies", {})
        baselines = [(name, value["aggregate"]) for name, value in policies.items() if name in {"flat", "threshold"}]
        if baselines:
            name, baseline = max(baselines, key=lambda pair: (pair[1]["net_pnl"], pair[0] == "flat"))
            executive.append(f"The best held-out baseline is {name}, with USD {_num(baseline['net_pnl'])} net P&L and {_count(baseline['trade_count'])} completed trades.")
            ppo = policies.get("ppo", {}).get("aggregate")
            if ppo is not None:
                difference = float(ppo["net_pnl"]) - float(baseline["net_pnl"])
                relation = "ties" if abs(difference) <= 1e-9 else "exceeds" if difference > 0 else "falls below"
                executive.append(f"Selected PPO ({selected}) {relation} that baseline by USD {_num(abs(difference))}; PPO net P&L is USD {_num(ppo['net_pnl'])} across {_count(ppo['trade_count'])} trades.")
                if ppo["trade_count"] == 0:
                    executive.append("PPO made zero test trades. Zero P&L establishes abstention, not convergence to an executable profitable strategy.")
        if not sensitivity:
            executive.append("The saved fee/delay/spread sensitivity sweep is unavailable; no break-even conclusion is supported.")
    if summary.get("selected_ppo_zero_validation_trades"):
        executive.append("The selected PPO policy also made zero validation trades. Read training trade-count diagnostics alongside rewards.")
    # @details Distinguish capture inventory, selected clean-range raw input, and
    # retained decisions. Counts are read from provenance, not proposal estimates.
    inventory = manifest.get("source_inventory") or quality.get("source_inventory") or {}
    totals = inventory.get("venue_totals", [])
    full_raw = sum(int(row["rows"]) for row in totals) if totals else None
    raw = manifest.get("raw_rows") or quality.get("raw_rows") or {}
    clean_raw = sum(int(value) for value in raw.values()) if raw else None
    days = manifest.get("day_splits") or {entry["day"]: entry["split"] for entry in manifest.get("episodes", [])}
    split_rows = []
    for split in ("train", "validation", "test"):
        dates = sorted(day for day, value in days.items() if value == split)
        split_rows.append([split, dates[0] if dates else "N/A", dates[-1] if dates else "N/A", _count(len(dates) if days else None)])
    reports = quality.get("day_reports", [])
    total_windows = sum(int(row.get("total_fixed_windows", 0)) for row in reports)
    kept_windows = sum(int(row.get("valid_fixed_windows", 0)) for row in reports)
    retained = quality.get("retained_decision_rows")
    if retained is None and reports:
        retained = sum(int(row.get("retained_decision_rows", 0)) for row in reports)
    coverage = [["Source inventory raw rows", _count(full_raw)], ["Selected clean-range raw rows", _count(clean_raw)],
                ["Binance clean-range rows", _count(raw.get("binance"))], ["Hyperliquid clean-range rows", _count(raw.get("hyperliquid"))],
                ["Retained decision rows", _count(retained)],
                ["Retained / candidate fixed windows", f"{kept_windows:,} / {total_windows:,}" if total_windows else "N/A"],
                ["Window retention", _num(kept_windows / total_windows, 2, True) if total_windows else "N/A"],
                ["Recorded Hyperliquid depth", _count(manifest.get("recorded_hl_depth"))]]
    quality_config = manifest.get("quality_config", {})
    fee_note = (f"At {_num(config.get('fee_bps'), 2)} bps per side, the nominal two-sided fee burden is {_num(2 * config['fee_bps'], 2)} bps before spread and slippage; actual fees use each fill's notional."
                if config.get("fee_bps") is not None else "Per-side and round-trip fee assumptions are unavailable.")
    methods = [
        fee_note,
        "Binance supplies a cross-venue signal; the strategy holds one directional Hyperliquid position. This is not a simultaneously hedged, risk-free two-leg portfolio.",
        f"Shared costs: {_num(config.get('position_size_btc'), 6)} BTC fixed size, {_num(config.get('fee_bps'), 2)} bps fee per side, {_num(config.get('latency_ms'), 0)} ms submission delay, and {_num(config.get('slippage_bps'), 2)} bps extra adverse slippage.",
        f"Policy cadence: {_num(config.get('decision_interval_ms'), 0)} ms (zero means event cadence). Every available intermediate replay row is processed. Entry and exit pay the displayed spread and consumed depth VWAP.",
        "Reward is the change in marked equity, scaled explicitly for PPO. Reported dollar P&L reconciles to completed trades after terminal liquidation; reward scaling does not change economics.",
        f"Holding deadline: {_num(config.get('max_holding_ms'), 0)} ms. Drawdown trigger: USD {_num(config.get('max_drawdown_usd'), 2)}. Risk exits pay delay and can overshoot the trigger. Forced exits beyond recorded depth use an adverse residual-price penalty.",
        "Normalization is fitted on training observations only. Threshold and PPO selection use validation results; frozen policies are then compared on test. Fee sensitivity replays the frozen policies under changed costs.",
    ]
    if manifest.get("execution_offsets_ms"):
        offsets = ", ".join(map(str, manifest["execution_offsets_ms"]))
        methods.append(f"Recorded replay uses integer UTC decision seconds plus exact +{offsets} ms execution samples. Each uses the latest locally received quote at or before its timestamp; last-in-bucket quotes are never relabeled as bucket starts.")
    if quality_config.get("require_complete_windows"):
        methods.append(f"Episodes are complete fixed UTC {quality_config.get('episode_seconds')} second windows. Any invalid grid row rejects the whole window before fitting. Terminal times are preset, avoiding advance knowledge of subsequently discovered outages.")
    limits = [
        "Replay does not measure private fills, queue priority, market impact, or the market's response to orders. Displayed liquidity and adverse residual pricing are assumptions, not execution guarantees.",
        "Risk and drawdown observe sampled replay rows. Unsampled intrasecond price excursions are invisible; deadlines between samples execute on the next available sample.",
        "Full-window quality exclusions condition coverage on the selected clean windows and may omit volatile or stale periods. Retention is reported; results do not establish performance during excluded malformed, stale, or outage intervals.",
        "Each episode resets capital and uses fixed BTC size. Aggregate P&L is a research accounting curve without reinvestment, live margin dynamics, or annualized claims.",
        "Generic variable, quality-selected episode boundaries can reveal their ending retrospectively. Recorded complete fixed windows avoid that boundary approximation.",
        "Fee-grid sign changes supply observed brackets only, not exact break-even fees. No-trade policies supply no evidence of a tradable edge.",
    ]
    if manifest.get("funding_unavailable_zero_assumption"):
        limits.insert(0, "The recorded source has no funding settlement history. Funding is assumed zero; results are not fully adjusted for measured funding costs.")
    if manifest.get("recorded_hl_depth") == 5:
        limits.insert(1, "The source has five Hyperliquid levels and 20 Binance levels. Five Hyperliquid levels replace the proposal's ten-level assumption; missing levels are not invented.")
    candidate_rows = []
    for row in candidates:
        training, aggregate = row.get("training_config", {}), row.get("aggregate", {})
        candidate_rows.append([row.get("name", "N/A"), _count(training.get("seed")), _num(training.get("ent_coef"), 3),
                               _num(aggregate.get("net_pnl")), _count(aggregate.get("trade_count")), _num(aggregate.get("max_drawdown_usd"))])
    values = [float(row["aggregate"]["net_pnl"]) for row in candidates if row.get("aggregate", {}).get("net_pnl") is not None]
    variation = (f"Across {len(values)} saved validation candidates, net P&L ranges from USD {_num(min(values))} to USD {_num(max(values))}; population standard deviation is USD {_num(statistics.pstdev(values))}. Entropy retries may differ from original seeds; this is descriptive variation, not a confidence interval."
                 if values else "No completed PPO candidate validation results are available.")
    bracket_rows = [[row.get("policy", "N/A"), _num(row.get("latency_ms"), 0), _num(row.get("spread_multiplier"), 2),
                     row.get("status", "N/A"), _num(row.get("highest_profitable_sampled_fee_bps"), 2),
                     _num(row.get("next_nonpositive_sampled_fee_bps"), 2)] for row in brackets or []]
    paths = [run / "experiment.json", run / "summary.json", run / "selection.json", run / "validation" / "comparison.json",
             run / "test" / "comparison.json", run / "validation" / "baselines.json", run / "sensitivity.json", run / "fee_break_even.json"]
    for candidate in candidates:
        folder = run / "models" / candidate.get("name", "candidate")
        paths.extend([folder / "training.json", folder / "training_episodes.monitor.csv"])
    if manifest_file:
        paths.append(manifest_file)
        if manifest.get("quality_report"):
            paths.append(manifest_file.parent / manifest["quality_report"])
    sources = []
    for path in paths:
        if path.exists():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            sources.append({"path": str(path), "sha256": digest})
    return {"report_version": 1, "title": "Reinforcement Learning for Cross-Venue Latency Arbitrage", "subtitle": "Cross-venue latency arbitrage after execution costs",
            "generated_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "evidence_kind": label, "synthetic": synthetic, "test_available": bool(testing), "executive": executive,
            "data_table": _table(["Measure", "Observed value"], coverage),
            "split_table": _table(["Split", "First UTC day", "Last UTC day", "Days"], split_rows),
            "candidate_table": _table(["Candidate", "Seed", "Entropy", "Validation P&L USD", "Trades", "Max DD USD"], candidate_rows),
            "diagnostics_table": _table(["Candidate", "Actual steps", "Episodes", "Training trades", "Zero-trade episodes"], _diagnostics(run, candidates)),
            "threshold_table": _table(["Entry gap bps", "Validation P&L USD", "Trades", "Fees USD", "Max DD USD", "Win rate"], threshold_rows),
            "candidate_variation": variation, "selected_candidate": selected,
            "selected_threshold_bps": summary.get("threshold_bps", selection.get("entry_threshold_bps", threshold_selection.get("entry_threshold_bps"))),
            "retry_triggered": summary.get("retry_triggered", selection.get("retry_triggered")),
            "validation": validation, "test": testing, "sensitivity": sensitivity or [], "break_even": brackets or [],
            "break_even_table": _table(["Policy", "Delay ms", "Spread x", "Observed status", "Profitable fee", "Next nonpositive"], bracket_rows),
            "method": methods, "limitations": limits, "sources": sources, "candidates": candidates, "charts": [], "config": config}


def _charts(data, output):
    """@brief Draw standard, exportable figures from saved measurements only.
    @details Missing result families omit their figure. The lazy Matplotlib import
    keeps --pdf-only usable in a separate, smaller document-rendering runtime.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    folder = output / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    charts = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.labelcolor": "#253447", "text.color": "#253447"})

    def save(figure, filename, title, caption):
        """@brief Save a sharp PNG with consistent margins, then release its figure."""
        figure.tight_layout(pad=1.6)
        figure.savefig(folder / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        charts.append({"path": f"charts/{filename}", "title": title, "caption": caption})

    report = data["test"] or data["validation"]
    split = "Test" if data["test"] else "Validation"
    if report and report.get("policies"):
        policies, names = report["policies"], list(report["policies"])
        figure, axes = plt.subplots(1, 2, figsize=(10.4, 4))
        colors = [COLORS.get(name, "#826b9f") for name in names]
        axes[0].bar(names, [policies[name]["aggregate"]["net_pnl"] for name in names], color=colors)
        axes[0].axhline(0, color="#536176", linewidth=.7)
        axes[0].set(title=f"{split}: net P&L", ylabel="USD after modeled costs")
        counts = [policies[name]["aggregate"]["trade_count"] for name in names]
        axes[1].bar(names, counts, color=colors)
        axes[1].set(title=f"{split}: completed trades", ylabel="Trades")
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        for index, count in enumerate(counts):
            axes[1].annotate(f"{count:,}", (index, count), xytext=(0, 5), textcoords="offset points", ha="center")
        axes[1].set_ylim(0, max(counts + [1]) * 1.2)
        save(figure, "policy_comparison.png", "Policy outcomes and activity",
             f"{split} policies share execution assumptions. Zero trades indicate abstention, not an executable edge.")
        days = sorted({day for value in policies.values() for day in value["aggregate"].get("daily_pnl", {})})
        if days:
            figure, axis = plt.subplots(figsize=(10.4, 3.7))
            for name, value in policies.items():
                pnl = value["aggregate"].get("daily_pnl", {})
                axis.plot(days, [pnl.get(day, float("nan")) for day in days], marker="o", label=name,
                          color=COLORS.get(name), linewidth=1.8, markersize=4)
            axis.axhline(0, color="#536176", linewidth=.7)
            axis.set(title=f"{split}: daily net P&L", ylabel="USD", xlabel="UTC day")
            axis.tick_params(axis="x", rotation=20)
            axis.legend(frameon=False, ncol=min(len(policies), 4))
            save(figure, "daily_pnl.png", "Daily variation", "Observed daily variation is not annualized or treated as independent profit guarantees.")
    candidates = [item for item in data["candidates"] if item.get("aggregate")]
    if candidates:
        figure, axes = plt.subplots(2, 1, figsize=(10.4, 5.1), sharex=True)
        labels = [item["name"].replace("ppo_", "") for item in candidates]
        colors = ["#186c7a" if item["name"] == data["selected_candidate"] else "#abbfca" for item in candidates]
        axes[0].bar(labels, [item["aggregate"]["net_pnl"] for item in candidates], color=colors)
        axes[0].axhline(0, color="#536176", linewidth=.7)
        axes[0].set(title="PPO candidate variation on validation", ylabel="Net P&L USD")
        axes[1].bar(labels, [item["aggregate"]["trade_count"] for item in candidates], color=colors)
        axes[1].set(ylabel="Completed trades", xlabel="Saved candidate (seed / retry)")
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].tick_params(axis="x", rotation=15)
        save(figure, "validation_candidates.png", "Validation seed and candidate comparison",
             "The selected candidate is highlighted. Entropy retries are separate candidates; test outcomes are not used in these bars.")
    rows = data["sensitivity"]
    if rows:
        scenarios = sorted({(row["latency_ms"], row["spread_multiplier"]) for row in rows})
        columns = min(2, len(scenarios))
        figure, axes = plt.subplots(math.ceil(len(scenarios) / columns), columns,
                                   figsize=(10.4, 3.15 * math.ceil(len(scenarios) / columns)), squeeze=False)
        for axis, (latency, spread) in zip(axes.flat, scenarios):
            scenario = [row for row in rows if row["latency_ms"] == latency and row["spread_multiplier"] == spread]
            for name in sorted({row["policy"] for row in scenario}):
                values = sorted((row for row in scenario if row["policy"] == name), key=lambda item: item["fee_bps"])
                axis.plot([row["fee_bps"] for row in values], [row["net_pnl"] for row in values],
                          marker="o", label=name, color=COLORS.get(name), linewidth=1.7, markersize=4)
            axis.axhline(0, color="#536176", linewidth=.7)
            axis.set(title=f"Delay {latency:g} ms / spread {spread:g}x", xlabel="Fee per side (bps)", ylabel="Net P&L USD")
            axis.legend(frameon=False, fontsize=8)
        for axis in list(axes.flat)[len(scenarios):]:
            axis.set_visible(False)
        save(figure, "fee_sensitivity.png", "Fee, delay, and spread sensitivity",
             "Each point is a separate replay of frozen policies. Connecting lines describe sampled scenarios, not an exact break-even estimate.")
    return charts


def _sections(data):
    """@brief Share content order and metric definitions between Markdown and PDF."""
    activity_rows = []
    for name, value in (data["test"] or data["validation"] or {}).get("policies", {}).items():
        item = value["aggregate"]
        activity_rows.append([name, _num(item.get("abstention_rate") if "hold_action_rate" in item else None, 1, True),
                              _num(item.get("hold_action_rate"), 1, True),
                              _num(item.get("flat_exposure_rate"), 1, True),
                              _num(item.get("flat_entry_rate"), 3)])
    return [
        {"title": "Executive assessment", "paragraphs": data["executive"]},
        {"title": "Dataset and coverage", "paragraphs": ["Counts distinguish source capture, selected raw dates, and retained windows. Missing provenance is N/A, never estimated from the proposal."], "tables": [data["data_table"], data["split_table"]]},
        {"title": "Selection and training activity", "paragraphs": [f"Selected candidate: {data['selected_candidate']}. Selected threshold: {_num(data['selected_threshold_bps'], 2)} bps. Higher-entropy retry triggered: {data['retry_triggered'] if data['retry_triggered'] is not None else 'not recorded'}.", data["candidate_variation"], "Training trade counts measure exploration. Validation/test counts measure the frozen deterministic policy. Zero reward with zero trades is not evidence of convergence."], "tables": [data["candidate_table"], data["diagnostics_table"]]},
        {"title": "Validation threshold search", "paragraphs": ["Every saved threshold candidate is shown under identical costs. The threshold is selected on validation only; the flat strategy remains an explicit zero-exposure control. Sparse trading at high thresholds does not establish a robust edge."], "tables": [data.get("threshold_table", _table([], []))]},
        {"title": "Validation comparison", "paragraphs": [] if data["validation"] else ["No validation comparison artifact is available."], "tables": [_comparison(data["validation"])]},
        {"title": "Held-out test comparison", "paragraphs": ["Return is cumulative dollar P&L divided by reference initial cash, without reinvestment. Selectivity is trades divided by full-grid opportunities at absolute gap >=7 bps. It can exceed one; N/A means no reference opportunities."] if data["test"] else ["The held-out test comparison has not been completed or its artifact is absent."], "tables": [_comparison(data["test"])]},
        {"title": "Abstention and exposure", "paragraphs": ["Abstention is the fraction of eligible flat/no-pending decisions that do not request an entry. HOLD share is action 0 across all decisions. Flat exposure is elapsed replay time without a position, including intermediate fills. Flat entry rate is completed entries per eligible flat decision. Missing legacy fields remain N/A."], "tables": [_table(["Policy", "Abstention", "HOLD share", "Flat exposure", "Flat entry rate"], activity_rows)]},
        {"title": "Sensitivity and observed fee brackets", "paragraphs": ["These are observed fee-grid brackets only. A no-trade result cannot identify a tradable fee threshold."] if data["sensitivity"] else ["No completed sensitivity artifact is available."], "tables": [data["break_even_table"]]},
        {"title": "Method and cost accounting", "paragraphs": data["method"]},
        {"title": "Limitations and interpretation", "paragraphs": data["limitations"]},
    ]


def _markdown(data):
    """@brief Write a readable report with portable chart links and source checksums."""
    lines = [f"# {data['title']}", "", data["subtitle"], "", f"**{data['evidence_kind']}**", "",
             f"Generated {data['generated_at_singapore']} (Singapore).", ""]
    for section in _sections(data):
        lines.extend([f"## {section['title']}", ""])
        for paragraph in section["paragraphs"]:
            lines.extend([paragraph, ""])
        for table in section.get("tables", []):
            if table["rows"]:
                lines.append("| " + " | ".join(table["headers"]) + " |")
                lines.append("| " + " | ".join("---" for _ in table["headers"]) + " |")
                for row in table["rows"]:
                    lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
                lines.append("")
    if data["charts"]:
        lines.extend(["## Figures", ""])
        for chart in data["charts"]:
            lines.extend([f"### {chart['title']}", "", f"![{chart['title']}]({chart['path']})", "", chart["caption"], ""])
    lines.extend(["## Source artifacts", "", "Checksums identify the saved evidence used to generate this report.", ""])
    for source in data["sources"]:
        lines.append(f"- `{source['path']}` - SHA256 `{source['sha256']}`")
    return "\n".join(lines) + "\n"


def render_pdf(data, output_dir):
    """@brief Create a paginated ReportLab PDF from saved report data and existing charts.
    @details PDF export does not establish visual QA; callers must render and inspect
    the pages before reporting the layout as verified. ImportError leaves Markdown
    and charts usable when ReportLab is unavailable in a training runtime.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

    output, width, height = Path(output_dir), 595.28, 841.89
    navy, teal = colors.HexColor("#183047"), colors.HexColor("#186c7a")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleR", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=navy, spaceAfter=13))
    styles.add(ParagraphStyle("SubtitleR", fontName="Helvetica", fontSize=12, leading=17, textColor=teal, spaceAfter=18))
    styles.add(ParagraphStyle("BodyR", fontName="Helvetica", fontSize=9.5, leading=14, spaceAfter=8, textColor=navy))
    styles.add(ParagraphStyle("HeadingR", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=teal, spaceBefore=14, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle("CaptionR", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#526578"), spaceAfter=12))
    styles.add(ParagraphStyle("CellR", fontName="Helvetica", fontSize=7.4, leading=10, textColor=navy))
    styles.add(ParagraphStyle("CellHeadR", parent=styles["CellR"], fontName="Helvetica-Bold", textColor=colors.white))

    def para(text, style="BodyR"):
        """@brief Escape source text so artifact fields cannot become PDF markup."""
        return Paragraph(escape(str(text)), styles[style])

    def footer(canvas, document):
        """@brief Repeat provenance and unobtrusive page numbers consistently."""
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#ced9e0"))
        canvas.line(44, 42, width - 44, 42)
        canvas.setFillColor(colors.HexColor("#647789"))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(44, 28, "OFFLINE RESEARCH | " + data["evidence_kind"])
        canvas.drawRightString(width - 44, 28, str(document.page))
        canvas.restoreState()

    destination = output / "report.pdf"
    document = BaseDocTemplate(str(destination), pagesize=(width, height), leftMargin=44, rightMargin=44,
                               topMargin=40, bottomMargin=55, title=data["title"], author="Offline research experiment")
    frame = Frame(44, 55, width - 88, height - 95, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=footer))
    story = [para(data["evidence_kind"], "CaptionR"), para(data["title"], "TitleR"), para(data["subtitle"], "SubtitleR"),
             para("Generated " + data["generated_at_singapore"] + " (Singapore)", "CaptionR")]
    for section in _sections(data):
        story.append(para(section["title"], "HeadingR"))
        story.extend(para(text) for text in section["paragraphs"])
        for table in section.get("tables", []):
            if not table["rows"]:
                continue
            rows = [[para(value, "CellHeadR") for value in table["headers"]]]
            rows += [[para(value, "CellR") for value in row] for row in table["rows"]]
            count = len(table["headers"])
            weights = [.57, .43] if count == 2 else [.23] + [.77 / (count - 1)] * (count - 1)
            rendered = Table(rows, colWidths=[(width - 88) * value for value in weights], repeatRows=1, hAlign="LEFT")
            rendered.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f6")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([rendered, Spacer(1, 10)])
    if data["charts"]:
        story.append(PageBreak())
    for chart in data["charts"]:
        # @details Resolve image dimensions before constructing the flowable.
        # ReportLab lazily initializes images and can overwrite drawWidth if it is
        # assigned before imageWidth is first read, producing a cropped oversized plot.
        path = str(output / chart["path"])
        pixel_width, pixel_height = ImageReader(path).getSize()
        draw_width = width - 88
        draw_height = pixel_height / pixel_width * draw_width
        if draw_height > 540:
            draw_height = 540
            draw_width = pixel_width / pixel_height * draw_height
        image = Image(path, width=draw_width, height=draw_height)
        story.append(KeepTogether([para(chart["title"], "HeadingR"), image,
                                   para(chart["caption"], "CaptionR")]))
    story.extend([PageBreak(), para("Evidence and reproducibility", "HeadingR"),
                  para("Complete paths and SHA256 hashes are preserved in report_data.json and report.md. These artifacts supplied the report:")])
    for source in data["sources"]:
        story.append(para(str(Path(source["path"]).parent.name + "/" + Path(source["path"]).name) + " | SHA256 " + source["sha256"], "CaptionR"))
    document.build(story)
    return destination


def generate_report(run_dir, manifest_path=None, *, output_dir=None, include_pdf=True):
    """@brief Create report.md, report_data.json, charts, and optional report.pdf.
    @param run_dir Existing experiment directory, which remains unchanged otherwise.
    @param manifest_path Optional explicit dataset manifest for coverage/provenance.
    @param output_dir Defaults to the experiment directory.
    @param include_pdf Attempt PDF generation if ReportLab is available locally.
    @return Output paths plus explicit PDF availability and verification status.
    """
    output = Path(output_dir or run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = collect_report_data(run_dir, manifest_path)
    data["charts"] = _charts(data, output)
    (output / "report_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(data), encoding="utf-8")
    result = {"markdown": str(output / "report.md"), "report_data": str(output / "report_data.json"),
              "charts": [str(output / item["path"]) for item in data["charts"]], "pdf": None,
              "pdf_status": "not_requested", "synthetic": data["synthetic"], "test_available": data["test_available"]}
    if include_pdf:
        try:
            result["pdf"] = str(render_pdf(data, output))
            result["pdf_status"] = "created_unverified_layout"
        except ImportError:
            result["pdf_status"] = "unavailable_reportlab; use --pdf-only with a ReportLab runtime"
    (output / "report_artifacts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    """@brief Render a complete report or finish PDF output in a separate runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args(argv)
    if args.pdf_only:
        output = args.output_dir or args.run_dir
        data = _read(output / "report_data.json")
        if data is None:
            parser.error("--pdf-only requires an existing report_data.json")
        target = render_pdf(data, output)
        result = _read(output / "report_artifacts.json", {})
        result.update({"pdf": str(target), "pdf_status": "created_unverified_layout"})
        (output / "report_artifacts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        result = generate_report(args.run_dir, args.manifest, output_dir=args.output_dir, include_pdf=not args.no_pdf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



