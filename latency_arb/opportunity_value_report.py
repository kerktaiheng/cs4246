"""@file opportunity_value_report.py
@brief Render a complete saved research record without training or market access.
@details Only four allowlisted value-study JSON files and the explicitly supplied
prior summary.json files are opened. Checkpoints and datasets are never loaded.
Original PPO failure, revised PPO failure at its historical cost, and the new
primary-fee comparison remain distinct; selected warm-start or regression gains
must not be credited to PPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_INPUTS = ("protocol.json", "selection.json", "candidate_results.json", "summary.json")
_LABELS = {"expected_value": "Expected-profit regression",
           "warmstart": "Archived supervised warm-start", "ppo": "Archived PPO"}
_COLORS = {"expected_value": "#247B8B", "warmstart": "#397BB0", "ppo": "#B36D42"}


def _reject_constant(value: str) -> None:
    """@brief Refuse NaN and Infinity before generating a report."""
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _read(path: Path, audit: list[dict]) -> Any:
    """@brief Read one fixed saved-result path and record its exact content hash."""
    if not path.is_file():
        raise ValueError(f"Final saved result is missing: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    audit.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return value


def _aggregate(value: Any) -> dict:
    """@brief Require finite dollars and exact nonnegative completed-trade counts."""
    if not isinstance(value, dict):
        raise ValueError("A saved aggregate must be an object")
    pnl, count = value.get("net_pnl"), value.get("trade_count")
    if isinstance(pnl, bool) or not isinstance(pnl, (int, float)) or not math.isfinite(pnl):
        raise ValueError("net_pnl must be finite")
    if type(count) is not int or count < 0:
        raise ValueError("trade_count must be a nonnegative integer")
    for field in ("fees_paid", "funding_paid", "max_drawdown_usd"):
        amount = value.get(field)
        if amount is not None and (isinstance(amount, bool) or
                                   not isinstance(amount, (int, float)) or
                                   not math.isfinite(amount)):
            raise ValueError(f"{field} must be finite when recorded")
    return value


def _kind(candidate: dict) -> str:
    """@brief Preserve the exact algorithm family rather than infer it from a filename."""
    kind = candidate.get("kind", candidate.get("stage", "expected_value"))
    if kind not in _LABELS:
        raise ValueError(f"Unrecognized learned candidate kind: {kind}")
    return kind


def _comparison(value: Any, expected_fee: float) -> None:
    """@brief Validate complete saved fresh controls and the declared scenario fee."""
    if not isinstance(value, dict) or value.get("split") != "test":
        raise ValueError("Fresh comparison must identify the test split")
    if value.get("config", {}).get("fee_bps") != expected_fee:
        raise ValueError("Fresh comparison fee disagrees with the protocol")
    policies = value.get("policies", {})
    if not {"flat", "threshold", "learned"} <= set(policies):
        raise ValueError("Fresh comparison lacks learned/threshold/flat controls")
    for item in policies.values():
        _aggregate(item.get("aggregate"))


def _inputs(run: Path, prior: Path, original: Path | None) -> dict:
    """@brief Reject partial or contradictory studies before any output is written.
    @details The frozen selection must agree with all saved candidate results and
    the protocol hash. An unused holdout cannot contain a primary or stressed
    outcome. Prior model paths are retained as text and never dereferenced.
    """
    audit: list[dict] = []
    files = {name: _read(run / name, audit) for name in _INPUTS}
    protocol, selection = files["protocol.json"], files["selection.json"]
    candidates, summary = files["candidate_results.json"], files["summary.json"]
    if not all(isinstance(item, dict) for item in (protocol, selection, summary)):
        raise ValueError("Protocol, selection and summary must be objects")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Final candidate results are required")
    if summary.get("protocol") != protocol or summary.get("selection") != selection:
        raise ValueError("Summary disagrees with saved protocol or selection")
    if selection.get("candidates") != candidates or selection.get("selected") not in candidates:
        raise ValueError("Selection disagrees with saved candidates")
    if not selection.get("frozen_at_utc") or selection.get("selection_split", "validation") != "validation":
        raise ValueError("A frozen validation selection is required")
    if selection.get("protocol_sha256") != audit[0]["sha256"]:
        raise ValueError("Frozen protocol hash disagrees with saved protocol")
    if type(summary.get("fresh_test_evaluated")) is not bool:
        raise ValueError("Summary must explicitly identify fresh evaluation status")
    if type(selection.get("validation_beats_flat")) is not bool:
        raise ValueError("Selection must explicitly identify its positive-profit gate")
    for candidate in candidates:
        _aggregate(candidate.get("aggregate"))
        _kind(candidate)
    _aggregate(selection.get("baseline", {}).get("aggregate"))
    if (protocol.get("execution", {}).get("fee_bps") != 3.5 or
            protocol.get("stress_execution", {}).get("fee_bps") != 4.5 or
            protocol.get("training_target_fee_bps") != 4.5):
        raise ValueError("Report requires the declared 3.5/4.5 per-side fee protocol")
    if summary["fresh_test_evaluated"]:
        if not selection["validation_beats_flat"]:
            raise ValueError("Fresh data were evaluated without passing the validation gate")
        _comparison(summary.get("fresh_test"), 3.5)
        _comparison(summary.get("fee_stress"), 4.5)
    elif summary.get("fresh_test") is not None or summary.get("fee_stress") is not None:
        raise ValueError("Unused fresh holdout cannot contain performance results")

    # @brief Read completed historical summaries without following any data paths.
    prior_summary = _read(prior / "summary.json", audit)
    if not isinstance(prior_summary, dict):
        raise ValueError("Revised PPO summary must be an object")
    previous = prior_summary.get("selection", {}).get("candidates")
    if not isinstance(previous, list) or not previous:
        raise ValueError("Revised PPO summary lacks candidates")
    for candidate in previous:
        if candidate.get("stage") not in ("ppo", "warmstart"):
            raise ValueError("Historical revised candidate stage is unrecognized")
        _aggregate(candidate.get("aggregate"))
    original_summary = _read(original / "summary.json", audit) if original else None
    if original_summary is not None:
        if not isinstance(original_summary, dict) or not original_summary.get("candidates"):
            raise ValueError("Original PPO summary lacks candidates")
        for candidate in original_summary["candidates"]:
            _aggregate(candidate.get("aggregate"))
        if original_summary.get("test_evaluated"):
            for item in original_summary.get("test", {}).get("policies", {}).values():
                _aggregate(item.get("aggregate"))
    return {"protocol": protocol, "selection": selection, "candidates": candidates,
            "summary": summary, "prior": prior_summary, "original": original_summary, "audit": audit}


def _text(value: Any) -> str:
    """@brief Keep saved labels within one Markdown table cell."""
    return str(value).replace("|", r"\|").replace("\n", " ").replace("\r", " ")


def _money(value: Any) -> str:
    """@brief Show small outcomes without rounding a small loss to zero."""
    return "not recorded" if value is None else f"{float(value):,.6f}"


def _dates(days: list[str] | None) -> str:
    """@brief Format only UTC dates already present in saved metadata."""
    if not days:
        return "not recorded"
    ordered = sorted(set(days))
    return ordered[0] if len(ordered) == 1 else f"{ordered[0]} to {ordered[-1]} ({len(ordered)} days)"


def _provenance(data: dict) -> tuple[str, str]:
    """@brief Distinguish synthetic verification from recorded-market evidence."""
    metrics = [item["aggregate"] for item in data["candidates"]]
    metrics.append(data["selection"]["baseline"]["aggregate"])
    for key in ("fresh_test", "fee_stress"):
        report = data["summary"].get(key)
        if report is not None:
            metrics += [item["aggregate"] for item in report["policies"].values()]
    flags = [item.get("synthetic") for item in metrics]
    if all(flag is True for flag in flags):
        return "SYNTHETIC TEST FIXTURE", (
            "**Synthetic fixture:** new-study values verify software only, not recorded-market performance."
        )
    if all(flag is False for flag in flags):
        return "RECORDED-DATA STUDY", (
            "Saved new-study aggregates identify recorded data. This renderer does not inspect market files."
        )
    return "PROVENANCE UNVERIFIED OR MIXED", (
        "**Unverified or mixed provenance:** inconsistent synthetic flags prevent a recorded-market claim."
    )


def _header() -> list[str]:
    """@brief Keep profit, completed activity, cost and risk in every result table."""
    return ["| Policy/checkpoint | Net PnL (USD) | Trades | Fees (USD) | Max drawdown (USD) |",
            "|---|---:|---:|---:|---:|"]


def _row(label: str, metrics: dict) -> str:
    """@brief Render one saved aggregate without recomputing trading results."""
    return (f"| {_text(label)} | {_money(metrics['net_pnl'])} | {metrics['trade_count']} | "
            f"{_money(metrics.get('fees_paid'))} | {_money(metrics.get('max_drawdown_usd'))} |")


def _history(data: dict) -> list[str]:
    """@brief Preserve both PPO failures and separate warm-start results from PPO.
    @details Failure here means failure to demonstrate profitable learned trading,
    not a software crash. Historical costs remain attached to their own results;
    they cannot be compared with a new fee assumption as a controlled improvement.
    """
    lines = ["## Preserved failed PPO attempts", "",
             "Failure means failure to demonstrate the requested profitable learned trading. "
             "A completed training job and zero dollars from no trades do not meet that objective.", ""]
    original = data["original"]
    if original is None:
        lines += ["The original summary was not supplied. Its outcome is not reconstructed here.", ""]
    else:
        fee = original.get("test", {}).get("config", {}).get("fee_bps")
        lines += ["### First attempt: original PPO", "",
                  f"Saved fee per side: {_text(fee)} bps. Every declared validation checkpoint, "
                  "including the entropy retry, is retained.", ""] + _header()
        for item in original["candidates"]:
            lines.append(_row(item.get("name", "unnamed PPO"), item["aggregate"]))
        if all(item["aggregate"]["net_pnl"] <= 0 for item in original["candidates"]):
            lines += ["", "**Outcome: FAILED to demonstrate a profitable PPO policy.**", ""]
        if original.get("test_evaluated"):
            lines += ["Original previously examined holdout:", ""] + _header()
            policies = original["test"]["policies"]
            for key, label in (("ppo", "Original selected PPO"),
                               ("threshold", "Original fixed threshold"), ("flat", "Original always skip")):
                if key in policies:
                    lines.append(_row(label, policies[key]["aggregate"]))
            if policies.get("ppo", {}).get("aggregate", {}).get("trade_count") == 0:
                lines += ["", "Original PPO completed **zero held-out trades**. Positive threshold "
                          "returns belong to the baseline, not PPO."]
            lines += ["", "Those test dates were already examined. They are excluded from new fitting "
                      "and selection and cannot count as another independent test.", ""]
    prior = data["prior"]
    fee = prior.get("protocol", {}).get("execution", {}).get("fee_bps")
    previous = prior["selection"]["candidates"]
    lines += ["### Second attempt: opportunity PPO", "",
              f"Historical fee per side: {_text(fee)} bps. Take-or-skip decisions, fixed exits, "
              "supervised initialization and PPO fine-tuning were evaluated. The warm-starts "
              "are supervised checkpoints, not PPO performance.", ""] + _header()
    for item in previous:
        label = "PPO fine-tuned" if item["stage"] == "ppo" else "supervised warm-start"
        lines.append(_row(f"{item.get('name', '?')} ({label})", item["aggregate"]))
    ppo = [item for item in previous if item["stage"] == "ppo"]
    if ppo and all(item["aggregate"]["net_pnl"] <= 0 for item in ppo):
        lines += ["", "**Outcome: FAILED to demonstrate a profitable PPO policy at its saved fee.**"]
    if ppo and all(item["aggregate"]["trade_count"] == 0 for item in ppo):
        lines += ["Every PPO fine-tuned checkpoint completed zero validation trades."]
    if prior.get("fresh_test_evaluated") is False:
        lines += ["The positive-validation gate failed, so this attempt left the fresh holdout unused."]
    lines += ["", "The new primary-fee comparison re-evaluates archived checkpoints at 7 bps round trip. "
              "It does not erase their historical results at 9 bps. The studies differ in cost and "
              "policy setup; raw totals do not isolate an algorithmic improvement.", ""]
    return lines


def _fresh_table(report: dict, selection: dict, title: str) -> list[str]:
    """@brief Show the same frozen controls in one named fee scenario."""
    selected = selection["selected"]
    labels = {"flat": "Always skip",
              "threshold": f"Fixed threshold ({selection['baseline'].get('threshold_bps', '?')} bps)",
              "learned": f"{_LABELS[_kind(selected)]} ({selected.get('name', '?')})"}
    return [title, ""] + _header() + [
        _row(labels[key], report["policies"][key]["aggregate"])
        for key in ("learned", "threshold", "flat")
    ] + [""]


def _markdown(data: dict, provenance: str, comparison: str) -> str:
    """@brief Report actual outcomes, all failures and independent-test limitations."""
    protocol, selection, summary = data["protocol"], data["selection"], data["summary"]
    selected = selection["selected"]
    kind = _kind(selected)
    dates = protocol.get("dates", {})
    lines = ["# Trading-policy research record", "", provenance, "",
             f"Selected candidate: **{_text(selected.get('name', '?'))} — {_LABELS[kind]}**.", "",
             "The new study compares expected-profit regression with archived supervised and PPO "
             "checkpoints. **Expected-profit regression is not PPO.** It predicts net trade dollars "
             "and trades only when the frozen prediction clears a selected margin.", "",
             "**Primary fee: 7 bps round trip (3.5 per side). Fee stress: 9 bps round trip "
             "(4.5 per side).** These are study scenarios, not an assertion about an account's fee tier.", "",
             "Regression training targets used the conservative 4.5 bps per side. The model's cost "
             "feature stays at its training value of 4.5 during both fee scenarios. Archived neural "
             "checkpoints also retain their training-fee feature. The stress changes replay charges "
             "without refitting the policy.", ""]
    if kind == "warmstart":
        lines += ["**Any selected-policy gain belongs to the supervised warm-start, not PPO fine-tuning.**", ""]
    elif kind == "expected_value":
        lines += [f"Selected predicted-profit margin: {_money(selected.get('margin_usd'))} USD.", ""]
    else:
        lines += ["The selected policy is an archived PPO checkpoint evaluated under the newly "
                  "declared fee assumptions; it was not retrained by this comparison.", ""]
    lines += [f"- Training: {_dates(dates.get('train'))} UTC.",
              f"- Validation used for selection: {_dates(dates.get('validation'))} UTC.",
              f"- Fresh holdout: {_dates(dates.get('fresh_test'))} UTC.",
              f"- Previously examined test excluded: "
              f"{_dates(dates.get('previously_examined_test_excluded'))} UTC.", ""]
    if summary["fresh_test_evaluated"]:
        learned = summary["fresh_test"]["policies"]["learned"]["aggregate"]
        baseline = summary["fresh_test"]["policies"]["threshold"]["aggregate"]
        if learned["net_pnl"] > 0 and learned["trade_count"] > 0:
            lines += [f"**Positive fresh result: {_money(learned['net_pnl'])} USD over "
                      f"{learned['trade_count']} completed trades.** This single saved period does "
                      "not establish a repeatable edge."]
        else:
            lines += [f"**The selected method did not demonstrate profitable fresh trading:** "
                      f"{_money(learned['net_pnl'])} USD over {learned['trade_count']} completed trades."]
        lines += [f"The frozen fixed-threshold baseline returned {_money(baseline['net_pnl'])} USD "
                  f"over {baseline['trade_count']} trades on the same primary-cost replay.", ""]
    else:
        lines += ["**No fresh result exists.** The learned candidates failed the positive-validation "
                  "gate. The fresh holdout remains unused. This is a development failure, not a "
                  "successful edge discovery.", ""]
    lines += ["## New validation: development evidence", "",
              "This period has been reused across research iterations. Its results select candidates "
              "and carry selection bias; they are not independent proof of profitability.", ""] + _header()
    for item in data["candidates"]:
        label = f"{item.get('name', '?')} ({_LABELS[_kind(item)]})"
        lines.append(_row(label + (" [selected]" if item == selected else ""), item["aggregate"]))
    lines.append(_row("Selected fixed-threshold baseline", selection["baseline"]["aggregate"]))
    lines += ["", "![New validation candidates](validation_candidates.png)", ""]
    if summary["fresh_test_evaluated"]:
        lines += _fresh_table(summary["fresh_test"], selection, "## Fresh primary result: 7 bps round trip")
        lines += _fresh_table(summary["fee_stress"], selection, "## Fresh fee stress: 9 bps round trip")
        primary = summary["fresh_test"]["policies"]["learned"]["aggregate"]
        stress = summary["fee_stress"]["policies"]["learned"]["aggregate"]
        lines += [f"The selected-policy net change under the higher fee is "
                  f"{_money(stress['net_pnl'] - primary['net_pnl'])} USD. This is a sensitivity "
                  "check on the same fresh period, not a second independent test.", "",
                  f"![Frozen fee comparison]({comparison})", ""]
        daily = primary.get("daily_pnl", {})
        if daily:
            lines += ["| Fresh UTC date | Selected-policy net PnL (USD) | Trades |", "|---|---:|---:|"]
            for day, pnl in sorted(daily.items()):
                count = primary.get("daily_trades", {}).get(day, "not recorded")
                lines.append(f"| {_text(day)} | {_money(pnl)} | {_text(count)} |")
            lines += [""]
        lines += [f"Saved coverage: {_text(primary.get('episode_count', 'not recorded'))} episodes, "
                  f"{_text(primary.get('day_count', 'not recorded'))} days; "
                  f"{_text(primary.get('entry_request_count', 'not recorded'))} entry requests and "
                  f"{_text(primary.get('rejection_count', 'not recorded'))} rejected requests.", ""]
    else:
        lines += ["## Fresh holdout not evaluated", "",
                  "No validation result or fee scenario is substituted for a fresh test.", "",
                  f"![Preserved PPO validation history]({comparison})", ""]
    lines += _history(data)
    execution, opportunity = protocol["execution"], protocol.get("opportunity", {})
    lines += ["## Method and limitations", "",
              f"- Fixed size: {_text(execution.get('position_size_btc'))} BTC; delay: "
              f"{_text(execution.get('latency_ms'))} ms; extra adverse slippage: "
              f"{_text(execution.get('slippage_bps'))} bps per fill; max holding time: "
              f"{_text(execution.get('max_holding_ms'))} ms.",
              f"- Candidate gap screen: {_text(opportunity.get('min_gap_bps', 'not recorded'))} bps; "
              f"convergence exit: {_text(opportunity.get('exit_gap_bps', 'not recorded'))} bps. "
              "The gap screen is not a guaranteed executable profit margin.",
              "- Independent candidate outcomes are training targets only. Overlapping candidate "
              "profits are not summed into portfolio returns. Reported policies replay one position at a time.",
              "- Expected profit is not a promise that every trade wins. Fees, delayed fills, "
              "rejections, price changes and model error can turn a candidate into a loss.",
              "- Binance supplies a reference signal for a Hyperliquid position; there is no "
              "simultaneous hedge on the reference venue.",
              "- Replay uses visible recorded depth and simulated delay/slippage. Market impact, "
              "private fills, queue dynamics and market response are not modeled.",
              "- Five-minute episodes reset flat with known endings. Quality-filtered coverage "
              "excludes rejected intervals and is not uninterrupted deployment.",
              "- Recorded funding data were unavailable and assumed zero. Zero funding charges "
              "do not mean funding costs were fully measured.",
              "- Drawdown uses sampled marked equity. Fixed-size dollars do not assume reinvestment "
              "and do not describe a different capital amount or order size.",
              "- Validation success cannot establish an edge. One fresh day and a small trade count "
              "are also insufficient to establish persistent profitability. Further untouched "
              "periods are required before stronger performance or deployment claims.", "",
              "## Saved-artifact audit", "",
              f"- Selection frozen at: {_text(selection['frozen_at_utc'])}.",
              f"- Protocol SHA256: {_text(selection['protocol_sha256'])}.",
              f"- Selected checkpoint hash, as saved: {_text(selected.get('model_sha256', 'not recorded'))}.",
              "- Only saved JSON is read. No training, policy loading, market access or new evaluation "
              "occurs. Checkpoint paths/hashes are reported, never dereferenced.", "",
              "| Input artifact | SHA256 |", "|---|---|"]
    for source in data["audit"]:
        lines.append(f"| {_text(source['path'])} | {_text(source['sha256'])} |")
    return "\n".join(lines) + "\n"


def _plot_validation(data: dict, badge: str, path: Path) -> None:
    """@brief Plot every candidate with algorithm color and completed-trade count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, selected = data["candidates"], data["selection"]["selected"]
    labels = [f"{item.get('name', '?')} ({item['aggregate']['trade_count']} trades)" +
              (" *" if item == selected else "") for item in rows]
    figure, axis = plt.subplots(figsize=(11, max(4.8, len(rows) * 0.38 + 2)))
    bars = axis.barh(labels, [item["aggregate"]["net_pnl"] for item in rows],
                     color=[_COLORS[_kind(item)] for item in rows])
    for bar, item in zip(bars, rows):
        if item == selected:
            bar.set_edgecolor("#172C3C")
            bar.set_linewidth(2)
    axis.invert_yaxis()
    axis.axvline(0, color="#374A57", linewidth=0.9)
    axis.axvline(data["selection"]["baseline"]["aggregate"]["net_pnl"],
                 color="#7A8349", linestyle="--", linewidth=1.3)
    axis.set_xlabel("Validation net PnL (USD; 7 bps round-trip fee)")
    axis.set_title("New candidate comparison")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    handles = [Patch(color=_COLORS[key], label=label) for key, label in _LABELS.items()
               if any(_kind(item) == key for item in rows)]
    handles.append(Patch(color="#7A8349", label="Dashed line: selected fixed baseline"))
    axis.legend(handles=handles, fontsize=8, loc="best")
    figure.suptitle(badge, fontsize=10)
    figure.text(0.02, 0.012, "* Selected on reused validation data; this chart does not establish an edge.",
                fontsize=8)
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _plot_comparison(data: dict, badge: str, path: Path) -> None:
    """@brief Plot actual fresh scenarios, or history when the fresh holdout is unused."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fresh = data["summary"]["fresh_test_evaluated"]
    if fresh:
        keys = ("learned", "threshold", "flat")
        labels = (_LABELS[_kind(data["selection"]["selected"])], "Fixed threshold", "Always skip")
        figure, axis = plt.subplots(figsize=(11, 4.8))
        values = []
        for key, shift, color, label in (("fresh_test", -0.18, "#247B8B", "7 bps round trip"),
                                          ("fee_stress", 0.18, "#A07836", "9 bps round trip")):
            metrics = [data["summary"][key]["policies"][name]["aggregate"] for name in keys]
            values += [item["net_pnl"] for item in metrics]
            bars = axis.barh([i + shift for i in range(3)],
                             [item["net_pnl"] for item in metrics], height=0.34, color=color, label=label)
            for bar, item in zip(bars, metrics):
                positive = bar.get_width() >= 0
                axis.annotate(f" {item['net_pnl']:+.6f} USD / {item['trade_count']} trades ",
                              (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                              ha="left" if positive else "right", va="center", fontsize=8)
        axis.set_yticks(range(3), labels)
        span = max(max(values) - min(values), max(abs(value) for value in values), 0.1)
        axis.set_xlim(min(min(values), 0) - span * 0.75, max(max(values), 0) + span * 0.75)
        axis.legend(fontsize=8, loc="best")
        axis.set_title("Same frozen policies and fresh period, two fee scenarios")
        axis.set_xlabel("Fresh net PnL (USD)")
        note = "Fee stress is not a second independent test; one fresh period does not prove a repeatable edge."
    else:
        rows = []
        if data["original"] is not None:
            rows += [(f"Original / {item.get('name', '?')}", item["aggregate"])
                     for item in data["original"]["candidates"]]
        rows += [(f"Revised / {item.get('name', '?')}", item["aggregate"])
                 for item in data["prior"]["selection"]["candidates"]]
        labels = [f"{name} ({metrics['trade_count']} trades)" for name, metrics in rows]
        values = [metrics["net_pnl"] for _, metrics in rows]
        figure, axis = plt.subplots(figsize=(11, max(4.8, len(rows) * 0.4 + 1.5)))
        axis.barh(labels, values, color="#A47B6C")
        axis.scatter(values, list(range(len(rows))), color="#64433B", s=18, zorder=3)
        axis.set_title("Preserved PPO history — fresh holdout remains unused")
        axis.set_xlabel("Historical validation net PnL (USD; each study's saved fees)")
        note = "Original and revised fees differ. Supervised warm-start outcomes are not PPO outcomes."
    axis.invert_yaxis()
    axis.axvline(0, color="#374A57", linewidth=0.9)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.suptitle(badge, fontsize=10)
    figure.text(0.02, 0.015, note, fontsize=8)
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def render_value_report(run_dir: str | Path, prior_ppo_run_dir: str | Path,
                        output_dir: str | Path | None = None,
                        original_ppo_run_dir: str | Path | None = None) -> dict:
    """@brief Create Markdown and two PNG charts from finalized saved result files.
    @param run_dir Completed new primary-fee study directory.
    @param prior_ppo_run_dir Completed revised PPO directory.
    @param output_dir Optional report destination, defaulting to run_dir/report.
    @param original_ppo_run_dir Optional original PPO directory for full history.
    @return Explicit output paths and saved fresh-evaluation/provenance flags.
    @details The caller confirms completion first. This function neither polls
    status nor attempts to finish training. Validation precedes output creation.
    """
    run = Path(run_dir)
    data = _inputs(run, Path(prior_ppo_run_dir),
                   Path(original_ppo_run_dir) if original_ppo_run_dir is not None else None)
    badge, provenance = _provenance(data)
    destination = Path(output_dir) if output_dir is not None else run / "report"
    destination.mkdir(parents=True, exist_ok=True)
    validation = destination / "validation_candidates.png"
    comparison = destination / ("fresh_fee_comparison.png" if
                                data["summary"]["fresh_test_evaluated"] else "ppo_failure_history.png")
    _plot_validation(data, badge, validation)
    _plot_comparison(data, badge, comparison)
    report = destination / "report.md"
    report.write_text(_markdown(data, provenance, comparison.name), encoding="utf-8")
    return {"report": str(report), "charts": [str(validation), str(comparison)],
            "fresh_test_evaluated": data["summary"]["fresh_test_evaluated"], "provenance": badge}


def main(argv: list[str] | None = None) -> None:
    """@brief Render an explicitly supplied final study with no training side effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prior-ppo-run-dir", required=True, type=Path)
    parser.add_argument("--original-ppo-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_value_report(args.run_dir, args.prior_ppo_run_dir,
                                        args.output_dir, args.original_ppo_run_dir), indent=2))


if __name__ == "__main__":
    main()
