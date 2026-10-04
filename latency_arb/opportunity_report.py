"""@file opportunity_report.py
@brief Render a saved opportunity study without policies, training, or market access.
@details The only input files opened are protocol.json, selection.json,
candidate_results.json, and summary.json in the requested run directory. A final
summary is required. Callers must wait for the experiment's completed status
before invoking this renderer. Saved model or manifest paths are never followed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_STAGE_LABELS = {
    "warmstart": "Supervised warm-start",
    "ppo": "PPO fine-tuned",
}
_STAGE_COLORS = {"warmstart": "#3176A8", "ppo": "#BD6B32"}
_INPUT_NAMES = ("protocol.json", "selection.json", "candidate_results.json", "summary.json")


def _reject_constant(value: str) -> None:
    """@brief Reject nonstandard JSON NaN and Infinity rather than plotting them."""
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def _read_saved(run_dir: Path) -> tuple[dict, dict, list[dict], dict]:
    """@brief Read only four allowlisted result files and verify their consistency.
    @details Requiring the runner's final summary prevents partial candidate lists
    from masquerading as a final experiment. No status, checkpoint, dataset,
    observation, trade-log, or policy file is opened by this renderer.
    """
    content = {}
    raw_protocol = None
    for name in _INPUT_NAMES:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"Final saved study artifact is missing: {name}")
        raw = path.read_bytes()
        content[name] = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
        if name == "protocol.json":
            raw_protocol = raw
    protocol, selection = content["protocol.json"], content["selection.json"]
    candidates, summary = content["candidate_results.json"], content["summary.json"]
    if not all(isinstance(value, dict) for value in (protocol, selection, summary)):
        raise ValueError("protocol, selection and summary must be JSON objects")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("a final study requires a nonempty saved candidate list")
    if summary.get("protocol") != protocol or summary.get("selection") != selection:
        raise ValueError("summary disagrees with the separately saved protocol or selection")
    if selection.get("candidates") != candidates or selection.get("selected") not in candidates:
        raise ValueError("saved candidate results disagree with frozen selection")
    if selection.get("selection_split") != "validation" or not selection.get("frozen_at_utc"):
        raise ValueError("a frozen validation selection is required")
    if selection.get("protocol_sha256") != hashlib.sha256(raw_protocol).hexdigest():
        raise ValueError("protocol content hash disagrees with frozen selection")
    if type(summary.get("fresh_test_evaluated")) is not bool:
        raise ValueError("summary must explicitly state whether fresh test was evaluated")
    fresh = summary.get("fresh_test")
    if summary["fresh_test_evaluated"]:
        if not isinstance(fresh, dict) or fresh.get("split") != "test":
            raise ValueError("fresh evaluation must be a saved test-split report")
        if not {"flat", "threshold", "learned"} <= set(fresh.get("policies", {})):
            raise ValueError("fresh comparison lacks one of the frozen policy controls")
    elif fresh is not None:
        raise ValueError("unevaluated fresh test must not contain an outcome report")
    for candidate in candidates:
        if candidate.get("stage") not in _STAGE_LABELS:
            raise ValueError("candidate stage must be warmstart or ppo")
        _validate_aggregate(candidate.get("aggregate"))
    _validate_aggregate(selection["baseline"]["aggregate"])
    if fresh is not None:
        for policy in fresh["policies"].values():
            _validate_aggregate(policy.get("aggregate"))
    return protocol, selection, candidates, summary


def _validate_aggregate(aggregate: Any) -> None:
    """@brief Require finite PnL and an exact nonnegative completed-trade count."""
    if not isinstance(aggregate, dict):
        raise ValueError("candidate/policy aggregate must be a JSON object")
    pnl, count = aggregate.get("net_pnl"), aggregate.get("trade_count")
    if isinstance(pnl, bool) or not isinstance(pnl, (float, int)) or not math.isfinite(pnl):
        raise ValueError("aggregate net_pnl must be a finite number")
    if type(count) is not int or count < 0:
        raise ValueError("aggregate trade_count must be a nonnegative integer")


def _text(value: Any) -> str:
    """@brief Keep saved labels on one Markdown table line without adding markup."""
    return str(value).replace("|", r"\|").replace("\n", " ").replace("\r", " ")


def _money(value: Any) -> str:
    """@brief Preserve small research PnL values with six decimal USD precision."""
    if value is None:
        return "not recorded"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("report dollar amounts must be finite")
    return f"{number:,.6f}"


def _percent(value: Any) -> str:
    """@brief Distinguish undefined ratios from measured zero performance."""
    return "undefined" if value is None else f"{100 * float(value):.2f}%"


def _dates(days: list[str] | None) -> str:
    """@brief Format only the calendar dates already saved in the experiment protocol."""
    if not days:
        return "not recorded"
    ordered = sorted(set(days))
    return ordered[0] if len(ordered) == 1 else f"{ordered[0]} to {ordered[-1]} ({len(ordered)} days)"


def _provenance(candidates: list[dict], selection: dict, summary: dict) -> tuple[str, str]:
    """@brief Never infer recorded-data provenance merely from realistic-looking dates."""
    aggregates = [item["aggregate"] for item in candidates] + [selection["baseline"]["aggregate"]]
    if summary["fresh_test_evaluated"]:
        aggregates.extend(item["aggregate"] for item in summary["fresh_test"]["policies"].values())
    flags = [item.get("synthetic") for item in aggregates]
    if all(flag is True for flag in flags):
        return "SYNTHETIC TEST FIXTURE", (
            "**Synthetic fixture:** these saved results are for software verification, "
            "not evidence about recorded-market performance."
        )
    if all(flag is False for flag in flags):
        return "RECORDED-DATA STUDY", (
            "All saved aggregates identify their inputs as recorded data. "
            "This renderer does not independently inspect the source dataset."
        )
    return "PROVENANCE UNVERIFIED OR MIXED", (
        "**Provenance is unverified or mixed:** saved synthetic flags do not consistently "
        "identify one data type. Do not present this report as a recorded-market result."
    )


def _policy_label(name: str, selection: dict) -> str:
    """@brief Label fresh learned results by the actual selected training stage."""
    if name == "flat":
        return "Always skip"
    if name == "threshold":
        threshold = selection["baseline"].get("threshold_bps", "unrecorded")
        return f"Fixed threshold ({threshold} bps)"
    if name == "learned":
        chosen = selection["selected"]
        return f"{_STAGE_LABELS[chosen['stage']]} (seed {chosen.get('seed', '?')})"
    return name


def _plot_validation(candidates: list[dict], selected: dict, badge: str, path: Path) -> None:
    """@brief Plot saved validation dollars with visibly different warm-start/PPO colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = [f"Seed {item.get('seed', '?')} · {_STAGE_LABELS[item['stage']]}" +
              (" [selected]" if item["name"] == selected["name"] else "") for item in candidates]
    values = [item["aggregate"]["net_pnl"] for item in candidates]
    figure, axis = plt.subplots(figsize=(10, max(4.3, 0.55 * len(candidates) + 1.5)))
    bars = axis.barh(labels, values, color=[_STAGE_COLORS[item["stage"]] for item in candidates])
    for bar, candidate in zip(bars, candidates):
        if candidate["name"] == selected["name"]:
            bar.set_edgecolor("#172C3C")
            bar.set_linewidth(2)
    axis.invert_yaxis()
    axis.axvline(0, color="#344451", linewidth=0.9)
    axis.set_xlabel("Validation net PnL (USD; saved execution costs)")
    axis.set_title("Validation candidates")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(handles=[Patch(color=color, label=_STAGE_LABELS[stage])
                         for stage, color in _STAGE_COLORS.items()], loc="best", fontsize=8)
    figure.suptitle(badge, fontsize=10, color="#344451")
    figure.text(0.02, 0.015, "Warm-start results are not PPO gains. Validation was used for selection.",
                fontsize=8, color="#344451")
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _plot_fresh(fresh: dict, selection: dict, badge: str, path: Path) -> None:
    """@brief Show saved fresh-policy net dollars and actual completed-trade counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    names = [name for name in ("flat", "threshold", "learned") if name in fresh["policies"]]
    labels = [_policy_label(name, selection) for name in names]
    aggregates = [fresh["policies"][name]["aggregate"] for name in names]
    colors = ["#82929B", "#78814C", _STAGE_COLORS[selection["selected"]["stage"]]][:len(names)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].barh(labels, [item["net_pnl"] for item in aggregates], color=colors)
    axes[1].barh(labels, [item["trade_count"] for item in aggregates], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Fresh net PnL (USD)")
    axes[1].set_xlabel("Completed trades")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes:
        axis.axvline(0, color="#344451", linewidth=0.9)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(f"Frozen fresh-policy comparison · {badge}", fontsize=11)
    figure.text(0.02, 0.015, "A single fresh day cannot establish a repeatable trading edge.",
                fontsize=9, color="#344451")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _assumptions(protocol: dict, summary: dict) -> list[str]:
    """@brief Describe saved economics and identify provenance that the allowed files omit."""
    execution, opportunity = protocol.get("execution", {}), protocol.get("opportunity", {})
    descriptions = {
        "fee_bps": "Taker fee per side (bps)", "latency_ms": "Submission delay (ms)",
        "slippage_bps": "Additional adverse slippage per fill (bps)",
        "spread_multiplier": "Displayed-spread multiplier",
        "position_size_btc": "Fixed position size (BTC)",
        "max_holding_ms": "Maximum holding time (ms)",
        "max_drawdown_usd": "Drawdown trigger (USD)",
        "initial_cash": "Fixed reference cash (USD)",
        "decision_interval_ms": "Underlying decision cadence (ms)",
        "max_quote_age_ms": "Entry exchange-age guard (ms)",
        "max_received_age_ms": "Entry receive-age guard (ms)",
        "max_replay_gap_ms": "Maximum permitted replay gap (ms)",
        "reward_scale": "Saved execution reward scale",
        "forced_liquidity_penalty_bps": "Forced-exit residual-depth penalty (bps)",
    }
    lines = ["## Execution and selection assumptions", "",
             "| Saved setting | Value |", "|---|---:|"]
    for name, label in descriptions.items():
        lines.append(f"| {label} | {_text(execution.get(name, 'not recorded'))} |")
    for name, label in (
        ("min_gap_bps", "Current absolute-gap screen (bps)"),
        ("exit_gap_bps", "Fixed convergence-exit gap (bps)"),
        ("min_remaining_ms", "Known episode-end reserve (ms)"),
    ):
        lines.append(f"| {label} | {_text(opportunity.get(name, 'not recorded'))} |")
    lines += [
        "",
        f"- Seeds: {_text(protocol.get('seeds', []))}; "
        f"supervised warm-start epochs: {_text(protocol.get('warm_epochs', 'not recorded'))}; "
        f"requested PPO decisions per seed: {_text(protocol.get('ppo_steps_per_seed', 'not recorded'))}.",
        f"- Selection: {_text(protocol.get('selection', 'not recorded'))}",
        f"- Fresh-test gate: {_text(protocol.get('fresh_test_gate', 'not recorded'))}",
        f"- Reward: {_text(protocol.get('reward', 'not recorded'))}",
        "- All compared policies use the saved opportunity screen, execution costs, and fixed exit rules. "
        "Independent candidate trade labels are not summed as a portfolio.",
        "- Binance supplies the reference signal for one fixed-size Hyperliquid position. "
        "This is directional execution without a simultaneous second-venue hedge.",
        "- Orders cross displayed depth using VWAP and additional adverse slippage. Fees apply "
        "to executed notional on each side. Forced exits beyond displayed depth use the recorded "
        "worst level plus the saved penalty; that extrapolation is an assumption.",
        "- Risk exits incur execution delay and can exceed the drawdown trigger before closing.",
        "- Returns use one fixed reference cash amount with no reinvestment. Drawdown uses sampled "
        "marked equity, including intermediate fills; movements between replay samples are unobserved.",
        "- Episodes begin flat and have known terminal times. Quality filtering can favor better-captured "
        "periods; results do not cover excluded intervals or uninterrupted deployment.",
    ]
    if summary["fresh_test_evaluated"] and summary["fresh_test"].get(
        "funding_unavailable_zero_assumption", False
    ):
        lines.append("- **Funding rates were unavailable and assumed zero.** Reported fresh PnL is "
                     "not fully adjusted for measured funding payments.")
    else:
        lines.append("- Funding availability is not specified in the allowed final artifacts for this "
                     "result. A reported zero funding charge does not prove funding data were available.")
    lines += [
        "- Source book depth and detailed preparation thresholds are not stored in these four report "
        "artifacts. Consult the dataset provenance separately; this renderer never opens market files.",
        "- Margin rules, liquidation tiers, market impact, private fills, counterfactual queue dynamics, "
        "and the market's response to the strategy are outside this replay.",
    ]
    return lines


def _markdown(protocol: dict, selection: dict, candidates: list[dict],
              summary: dict, provenance: str) -> str:
    """@brief Build a concise factual report while keeping validation and fresh results separate."""
    chosen = selection["selected"]
    label = _STAGE_LABELS[chosen["stage"]]
    dates = protocol.get("dates", {})
    lines = [
        "# Opportunity-policy research report", "", provenance, "",
        f"Selected candidate: **{_text(chosen['name'])} — {label}**.",
        f"Selection was frozen at {_text(selection['frozen_at_utc'])} using validation only.",
        "",
        f"- Training: {_dates(dates.get('train'))} UTC.",
        f"- Validation: {_dates(dates.get('validation'))} UTC.",
        f"- Fresh holdout: {_dates(dates.get('fresh_test'))} UTC.",
        f"- Previously examined test dates excluded from fitting/selection: "
        f"{_dates(dates.get('previously_examined_test_excluded'))} UTC.",
        "",
    ]
    if chosen["stage"] == "warmstart":
        lines += ["**The selected checkpoint is the supervised warm-start. Its result must not be "
                  "credited to PPO fine-tuning.**", ""]
    else:
        lines += ["The selected checkpoint includes PPO fine-tuning after supervised initialization; "
                  "the separate warm-start validation results remain visible below.", ""]
    lines += ["## Validation: development evidence", "",
              "| Candidate | Stage | Net PnL (USD) | Trades | Fees (USD) | Max drawdown (USD) |",
              "|---|---|---:|---:|---:|---:|"]
    for candidate in candidates:
        metrics = candidate["aggregate"]
        lines.append(f"| {_text(candidate['name'])} | {_STAGE_LABELS[candidate['stage']]} | "
                     f"{_money(metrics['net_pnl'])} | {metrics['trade_count']} | "
                     f"{_money(metrics.get('fees_paid'))} | {_money(metrics.get('max_drawdown_usd'))} |")
    baseline = selection["baseline"]
    lines += [
        "",
        f"Selected fixed baseline: {_text(baseline.get('threshold_bps', 'unrecorded'))} bps; "
        f"validation net PnL {_money(baseline['aggregate']['net_pnl'])} USD across "
        f"{baseline['aggregate']['trade_count']} trades.",
        "",
        "![Validation candidate PnL](validation_candidates.png)", "",
        "Validation selects candidates and therefore does not provide an independent performance test.",
        "",
    ]
    if summary["fresh_test_evaluated"]:
        fresh = summary["fresh_test"]
        lines += ["## Frozen fresh evaluation", "",
                  "| Policy | Net PnL (USD) | Trades | Win rate | Fees (USD) | Funding paid (USD) | Max drawdown (USD) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("flat", "threshold", "learned"):
            metrics = fresh["policies"][name]["aggregate"]
            lines.append(f"| {_text(_policy_label(name, selection))} | {_money(metrics['net_pnl'])} | "
                         f"{metrics['trade_count']} | {_percent(metrics.get('win_rate'))} | "
                         f"{_money(metrics.get('fees_paid'))} | {_money(metrics.get('funding_paid'))} | "
                         f"{_money(metrics.get('max_drawdown_usd'))} |")
        learned = fresh["policies"]["learned"]["aggregate"]
        lines += [
            "",
            f"Fresh coverage in the saved learned-policy aggregate: "
            f"{_text(learned.get('episode_count', 'not recorded'))} complete episodes, "
            f"{_text(learned.get('day_count', 'not recorded'))} days, "
            f"{_text(learned.get('macro_decision_count', 'not recorded'))} eligible macro decisions; "
            f"{_text(learned.get('entry_request_count', 'not recorded'))} entry requests, "
            f"{_text(learned.get('rejection_count', 'not recorded'))} rejections, and "
            f"{_text(learned.get('cancellation_count', 'not recorded'))} cancellations.",
            "",
            "![Fresh-policy PnL and completed trades](fresh_policy_comparison.png)", "",
            "**This evaluation does not establish a repeatable trading edge.** One fresh day, "
            "a small trade count, validation selection, and replay assumptions limit the inference. "
            "Repeated untouched periods are required before making a stronger performance claim.",
            "",
        ]
    else:
        lines += ["## Fresh holdout remains unused", ""]
        if selection.get("validation_beats_flat") is False:
            lines.append("No learned candidate cleared the saved positive-validation gate. "
                         "This is a development-only negative result; no fresh policy result exists.")
        else:
            lines.append("The saved summary contains no fresh evaluation. "
                         "Do not substitute validation or training results for a fresh-test result.")
        lines += ["The holdout was not consumed by this renderer.", ""]
    lines += _assumptions(protocol, summary)
    lines += [
        "", "## Saved-artifact audit", "",
        f"- Protocol SHA256: {_text(selection['protocol_sha256'])}.",
        f"- Selected model SHA256, as saved: {_text(chosen.get('model_sha256', 'not recorded'))}.",
        "- Inputs: protocol.json, selection.json, candidate_results.json, summary.json only. "
        "The renderer checks their consistency, never loads a model, and never reads or evaluates market data.",
        "- Model hashes are reported from the saved selection; no policy file is opened to recompute them.",
        "",
    ]
    return "\n".join(lines)


def render_opportunity_report(run_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    """@brief Render Markdown and PNGs from a finalized study's four saved JSON artifacts.
    @param run_dir Directory whose experiment has reached completed status.
    @param output_dir Optional report folder; defaults to run_dir/report.
    @return Output paths and the saved fresh-evaluation/provenance flags.
    @details This function neither opens status.json nor polls a running experiment.
    The caller must check completion first. A missing/inconsistent final summary
    fails before rendering. Only fixed report filenames are written.
    """
    run = Path(run_dir)
    protocol, selection, candidates, summary = _read_saved(run)
    badge, provenance = _provenance(candidates, selection, summary)
    destination = Path(output_dir) if output_dir is not None else run / "report"
    destination.mkdir(parents=True, exist_ok=True)
    validation_chart = destination / "validation_candidates.png"
    _plot_validation(candidates, selection["selected"], badge, validation_chart)
    charts = [str(validation_chart)]
    if summary["fresh_test_evaluated"]:
        fresh_chart = destination / "fresh_policy_comparison.png"
        _plot_fresh(summary["fresh_test"], selection, badge, fresh_chart)
        charts.append(str(fresh_chart))
    report_path = destination / "report.md"
    report_path.write_text(_markdown(protocol, selection, candidates, summary, provenance),
                           encoding="utf-8")
    return {"report": str(report_path), "charts": charts,
            "fresh_test_evaluated": summary["fresh_test_evaluated"], "provenance": badge}


def main(argv: list[str] | None = None) -> None:
    """@brief Render an explicitly requested completed saved study without any training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_opportunity_report(args.run_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
