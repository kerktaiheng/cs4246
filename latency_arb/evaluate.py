"""@file evaluate.py
@brief Evaluate frozen policies on complete held-out replay episodes.
@details This module never fits policy parameters or observation normalization.
Validation-only threshold selection is a separate command. Test evaluation requires
an explicit --split test argument; cost sweeps replay every policy under identical
assumptions and record both trades and abstentions for audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from latency_arb.baseline import FlatPolicy, ThresholdPolicy
from latency_arb.data.schema import ReplayEpisode, load_manifest
from latency_arb.env.latency_sim import EnvConfig, LatencySimEnv, OBSERVATION_NAMES


def _json_value(value: Any) -> Any:
    """@brief Convert NumPy scalars and arrays to strict portable JSON values.
    @details Nonfinite floats are rejected by json.dump rather than silently writing
    nonstandard NaN tokens. The evaluator's own empty-denominator metrics use None.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    """@brief Save a readable audit artifact, creating its local parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_value, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """@brief Stream independent decision/trade records without a large JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_value, allow_nan=False) + "\n")


def maximum_drawdown(equity: Sequence[float]) -> float:
    """@brief Compute the largest peak-to-subsequent-trough loss in dollars.
    @param equity Chronological portfolio values including the pre-trade initial value.
    @return Zero for empty/flat input; otherwise a nonnegative dollar loss.
    """
    values = np.asarray(equity, dtype=np.float64)
    if not len(values):
        return 0.0
    if not np.isfinite(values).all():
        raise ValueError("Equity history must be finite.")
    return float(np.max(np.maximum.accumulate(values) - values))


def rollout(policy: Any, episode: ReplayEpisode, config: EnvConfig | None = None,
            seed: int = 0) -> dict[str, Any]:
    """@brief Replay one full episode and collect economics independently of reward scaling.
    @param policy Object exposing predict(raw_observation, deterministic=True).
    @param episode One checked, chronological held-out segment.
    @param config Identical economic assumptions for every compared policy.
    @param seed Environment seed, recorded by the containing experiment.
    @return Metrics, fills, trades, decisions, and full equity history.
    @details Selectivity is completed entries divided by full-episode scheduled
    decisions whose absolute gap is at least seven basis points. It can exceed one
    for policies trading below that reference. flat_entry_rate separately measures
    entries divided by flat decisions with no pending order; neither denominator
    is a guarantee that the visible opportunity could be profitably executed.
    No fixed five-step cutoff or repeated reset can hide later transaction costs.
    """
    chosen = replace(config or EnvConfig(), log_history=True)
    env = LatencySimEnv(episode, config=chosen)
    observation, info = env.reset(seed=seed)
    equity = [float(info.get("equity", chosen.initial_cash))]
    decisions, rewards = [], []
    flat_decisions, entry_requests = 0, 0
    inventory_index = OBSERVATION_NAMES.index("inventory")
    pending_index = OBSERVATION_NAMES.index("pending_action")
    # @details Follow the environment's termination, including delayed risk exits.
    while True:
        prediction = policy.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        action = int(np.asarray(action).item())
        flat = observation[inventory_index] == 0 and observation[pending_index] < 0
        flat_decisions += int(flat)
        entry_requests += int(flat and action in (1, 2))
        decision = {"decision": len(decisions), "action": action,
                    "inventory_before": float(observation[inventory_index]),
                    "pending_action_before": int(observation[pending_index]),
                    "timestamp_ns": int(info["timestamp_ns"]),
                    "gap_bps": float(observation[OBSERVATION_NAMES.index("gap_bps")])}
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        equity.append(float(info["equity"]))
        decision.update({"reward": float(reward), "equity_after": equity[-1],
                         "terminated": bool(terminated), "truncated": bool(truncated)})
        decisions.append(decision)
        if terminated or truncated:
            break
    # @details Include every internal fill/mark, not only one-second policy endpoints.
    equity = [float(row["equity"]) for row in env.equity_history]
    # @details Win rate is undefined with no closed trades; JSON null preserves that.
    trades = list(env.trades)
    # @details Count the complete scheduled decision grid independently of a policy's
    # early risk stop. Ratios may exceed one when a policy trades below the reference.
    reference_gap_bps = 7.0
    if chosen.decision_interval_ms > 0:
        interval = int(round(chosen.decision_interval_ms * 1_000_000))
        targets = np.arange(int(episode.timestamp_ns[0]), int(episode.timestamp_ns[-1]),
                            interval, dtype=np.int64)
        indices = np.unique(np.searchsorted(episode.timestamp_ns, targets, side="left"))
        indices = indices[indices < len(episode) - 1]
    else:
        indices = np.arange(len(episode) - 1)
    opportunity_count = int(np.count_nonzero(np.abs(episode.gap_bps[indices]) >= reference_gap_bps))
    profits = [float(trade["net_pnl"]) for trade in trades]
    final_pnl = equity[-1] - chosen.initial_cash
    total_fees = float(info.get("fees_paid", 0.0))
    funding_paid = float(info.get("funding_paid", 0.0))
    metrics = {
        "episode_id": episode.metadata.get("episode_id", "episode"),
        "day": episode.metadata.get("day", "unknown"),
        "synthetic": bool(episode.metadata.get("synthetic", False)),
        "initial_cash": chosen.initial_cash,
        "net_pnl": final_pnl,
        "net_return": final_pnl / chosen.initial_cash,
        "max_drawdown_usd": maximum_drawdown(equity),
        "trade_count": len(trades),
        "win_rate": float(np.mean(np.asarray(profits) > 0)) if profits else None,
        "fees_paid": total_fees,
        "funding_paid": funding_paid,
        "pnl_before_fees_and_funding": final_pnl + total_fees + funding_paid,
        "decision_count": len(decisions),
        "flat_decision_count": flat_decisions,
        "entry_request_count": entry_requests,
        "trade_selectivity": len(trades) / opportunity_count if opportunity_count else None,
        "reference_opportunity_count": opportunity_count,
        "reference_gap_bps": reference_gap_bps,
        "flat_entry_rate": len(trades) / flat_decisions if flat_decisions else 0.0,
        "abstention_rate": (sum(row["action"] == 0 for row in decisions)
                            / len(decisions) if decisions else 0.0),
        "rejection_count": int(info.get("rejection_count", 0)),
        "cancellation_count": int(info.get("cancellation_count", 0)),
        "scaled_reward_sum": float(sum(rewards)),
        "termination_reason": info.get("termination_reason"),
        "start_timestamp_ns": int(episode.timestamp_ns[0]),
        "end_timestamp_ns": int(info["timestamp_ns"]),
        "scheduled_end_timestamp_ns": int(episode.timestamp_ns[-1]),
    }
    # @details A self-financing replay must reconcile ledger trades with final equity.
    if not np.isclose(sum(profits), final_pnl, rtol=1e-7, atol=1e-7):
        raise AssertionError("Completed trade ledger does not reconcile with terminal equity.")
    env.close()
    return {"metrics": metrics, "trades": trades, "fills": list(env.fills),
            "decisions": decisions, "environment_actions": list(env.action_history),
            "equity": equity}


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """@brief Combine independent fixed-size episodes without pretending to compound capital.
    @details The cumulative return divides summed dollar PnL by one reference initial
    balance. Each episode resets its account and retains fixed BTC size, so this is a
    research accounting curve, not a reinvested or margin-aware live portfolio.
    """
    if not results:
        raise ValueError("At least one episode is required.")
    metrics = [result["metrics"] for result in results]
    capital = float(metrics[0]["initial_cash"])
    curve, cumulative = [capital], 0.0
    for result in results:
        # @details Stitch only incremental episode PnL, preventing reset discontinuities.
        start = float(result["metrics"]["initial_cash"])
        curve.extend(capital + cumulative + float(value) - start
                     for value in result["equity"][1:])
        cumulative += float(result["metrics"]["net_pnl"])
    profits = [float(trade["net_pnl"]) for result in results for trade in result["trades"]]
    flat_count = sum(item["flat_decision_count"] for item in metrics)
    opportunity_count = sum(item["reference_opportunity_count"] for item in metrics)
    decision_count = sum(item["decision_count"] for item in metrics)
    day_pnl: dict[str, float] = {}
    for item in metrics:
        day_pnl[item["day"]] = day_pnl.get(item["day"], 0.0) + item["net_pnl"]
    # @details Daily variation is reported directly rather than extrapolated to a year.
    return {
        "episode_count": len(results), "day_count": len(day_pnl),
        "synthetic": all(item["synthetic"] for item in metrics),
        "net_pnl": cumulative, "net_return": cumulative / capital,
        "max_drawdown_usd": maximum_drawdown(curve), "trade_count": len(profits),
        "win_rate": float(np.mean(np.asarray(profits) > 0)) if profits else None,
        "trade_selectivity": len(profits) / opportunity_count if opportunity_count else None,
        "reference_opportunity_count": opportunity_count,
        "reference_gap_bps": metrics[0]["reference_gap_bps"],
        "flat_entry_rate": len(profits) / flat_count if flat_count else 0.0,
        "decision_count": decision_count, "flat_decision_count": flat_count,
        "entry_request_count": sum(item["entry_request_count"] for item in metrics),
        "abstention_rate": (sum(item["abstention_rate"] * item["decision_count"]
                               for item in metrics) / decision_count if decision_count else 0.0),
        "fees_paid": sum(item["fees_paid"] for item in metrics),
        "funding_paid": sum(item["funding_paid"] for item in metrics),
        "rejection_count": sum(item["rejection_count"] for item in metrics),
        "cancellation_count": sum(item["cancellation_count"] for item in metrics),
        "daily_pnl": day_pnl,
        "daily_pnl_std": float(np.std(list(day_pnl.values()))),
        "accounting": "Fixed BTC size; episode resets; cumulative dollar PnL, no reinvestment.",
    }


def compare_policies(episodes: Sequence[ReplayEpisode], policies: dict[str, Any],
                     config: EnvConfig | None = None, output_dir: Path | None = None,
                     seed: int = 0) -> dict[str, Any]:
    """@brief Evaluate all policies on identical episodes and save optional audit trails.
    @param episodes Checked segments from exactly one declared chronological split.
    @param policies Named frozen policies; insertion order only controls report order.
    @param config Shared transaction costs and environment limits.
    @param output_dir Optional experiment directory for metrics and JSONL audit logs.
    @param seed Repeatable reset seed applied identically for every policy.
    """
    report = {"config": asdict(config or EnvConfig()), "seed": seed, "policies": {}}
    for name, policy in policies.items():
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
            raise ValueError("Policy names must be safe local filename components.")
        # @details Retain compact economic curves, while audit records are written
        # one episode at a time so an entire multi-day decision log never lives in RAM.
        results, handles = [], {}
        try:
            if output_dir is not None:
                directory = Path(output_dir) / name
                directory.mkdir(parents=True, exist_ok=True)
                handles = {key: (directory / f"{key}.jsonl").open("w", encoding="utf-8")
                           for key in ("trades", "fills", "decisions", "environment_actions")}
            for index, episode in enumerate(episodes):
                result = rollout(policy, episode, config, seed + index)
                for key, handle in handles.items():
                    for row in result[key]:
                        handle.write(json.dumps({"episode": index, **row},
                                     default=_json_value, allow_nan=False) + "\n")
                results.append({key: result[key] for key in ("metrics", "trades", "equity")})
        finally:
            for handle in handles.values():
                handle.close()
        report["policies"][name] = {"aggregate": aggregate_results(results),
                                   "episodes": [result["metrics"] for result in results]}
        if output_dir is not None:
            _write_json(Path(output_dir) / name / "equity.json",
                        [result["equity"] for result in results])
    if output_dir is not None:
        _write_json(Path(output_dir) / "comparison.json", report)
    return report


def select_threshold(manifest_path: str | Path, thresholds: Sequence[float],
                     config: EnvConfig | None = None, seed: int = 0) -> dict[str, Any]:
    """@brief Select the fixed entry threshold using validation data exclusively.
    @details Ties favor fewer trades and then a larger threshold. No test file is
    loaded by this function, and all candidates share the same transaction costs.
    @return Chosen boundary plus the complete validation candidate table.
    """
    if not thresholds:
        raise ValueError("Provide at least one threshold candidate.")
    episodes = load_manifest(Path(manifest_path), "validation")
    candidates = []
    for threshold in sorted(set(thresholds)):
        result = compare_policies(episodes, {"threshold": ThresholdPolicy(threshold)},
                                  config, seed=seed)["policies"]["threshold"]["aggregate"]
        candidates.append({"entry_threshold_bps": threshold, **result})
    best = max(candidates, key=lambda item: (item["net_pnl"], -item["trade_count"],
                                           item["entry_threshold_bps"]))
    return {"selection_split": "validation", "entry_threshold_bps": best["entry_threshold_bps"],
            "selection_objective": "net_pnl, then fewer trades, then larger threshold",
            "candidates": candidates}


def cost_sweep(episodes: Sequence[ReplayEpisode], policies: dict[str, Any],
               config: EnvConfig, fee_grid: Sequence[float], latency_grid: Sequence[float],
               spread_grid: Sequence[float], seed: int = 0) -> list[dict[str, Any]]:
    """@brief Re-evaluate frozen policies under a Cartesian grid of worse/better costs.
    @details This is a sensitivity study, not a policy retraining or test-set selection
    procedure. Changed fees can change risk exits; therefore results are recomputed,
    not adjusted by simply subtracting a constant fee from an old trade list.
    """
    rows = []
    for fee, latency, spread in itertools.product(fee_grid, latency_grid, spread_grid):
        chosen = replace(config, fee_bps=fee, latency_ms=latency, spread_multiplier=spread)
        comparison = compare_policies(episodes, policies, chosen, seed=seed)
        for name, result in comparison["policies"].items():
            aggregate = result["aggregate"]
            rows.append({"policy": name, "fee_bps": fee, "latency_ms": latency,
                         "spread_multiplier": spread,
                         **{key: aggregate[key] for key in
                            ("net_pnl", "net_return", "max_drawdown_usd", "trade_count",
                             "win_rate", "trade_selectivity", "fees_paid", "funding_paid")}})
    return rows


def fee_break_even(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """@brief Report observed fee-grid sign changes without asserting exact break-even.
    @details Policies with no trades carry no evidence of an executable edge. A bracket
    records the highest sampled profitable fee and next higher nonpositive fee; no
    interpolation is justified because policy paths and risk exits may differ.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["policy"], row["latency_ms"], row["spread_multiplier"])
        groups.setdefault(key, []).append(row)
    result = []
    for (policy, latency, spread), values in groups.items():
        values.sort(key=lambda item: item["fee_bps"])
        profitable = [item for item in values if item["net_pnl"] > 0 and item["trade_count"] > 0]
        highest = max((item["fee_bps"] for item in profitable), default=None)
        next_loss = next((item["fee_bps"] for item in values if highest is not None
                          and item["fee_bps"] > highest and item["net_pnl"] <= 0), None)
        status = ("no_trades" if not any(item["trade_count"] for item in values)
                  else "nonpositive_at_all_sampled_fees" if highest is None
                  else "bracket_observed" if next_loss is not None
                  else "positive_at_highest_sampled_fee")
        result.append({"policy": policy, "latency_ms": latency, "spread_multiplier": spread,
                       "status": status, "highest_profitable_sampled_fee_bps": highest,
                       "next_nonpositive_sampled_fee_bps": next_loss})
    return result


def _grid(value: str) -> list[float]:
    """@brief Parse a nonempty comma-separated numeric cost grid for the CLI."""
    values = [float(part) for part in value.split(",") if part.strip()]
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("Provide finite comma-separated values.")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    """@brief Run held-out comparisons, validation selection, and optional cost sweeps.
    @details The default validation split protects the unseen test period during
    routine development. Passing --split test constitutes the explicit final evaluation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold-bps", type=float, default=8.0)
    parser.add_argument("--select-thresholds", type=_grid)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fee-bps", type=float, default=3.5)
    parser.add_argument("--latency-ms", type=float, default=150.0)
    parser.add_argument("--slippage-bps", type=float, default=0.1)
    parser.add_argument("--decision-interval-ms", type=float, default=0.0)
    parser.add_argument("--position-size-btc", type=float, default=0.001)
    parser.add_argument("--fee-grid", type=_grid)
    parser.add_argument("--latency-grid", type=_grid, default=[150.0, 300.0])
    parser.add_argument("--spread-grid", type=_grid, default=[1.0, 1.5])
    args = parser.parse_args(argv)
    config = EnvConfig(fee_bps=args.fee_bps, latency_ms=args.latency_ms,
                       slippage_bps=args.slippage_bps,
                       decision_interval_ms=args.decision_interval_ms,
                       position_size_btc=args.position_size_btc)
    # @details Validate-only selection remains a separate phase even for final test runs.
    if args.select_thresholds:
        selection = select_threshold(args.manifest, args.select_thresholds, config, args.seed)
        args.threshold_bps = selection["entry_threshold_bps"]
        _write_json(args.output / "threshold_selection.json", selection)
    policies = {"flat": FlatPolicy(), "threshold": ThresholdPolicy(args.threshold_bps)}
    if args.model:
        from latency_arb.agent.policy import load_policy
        policies["ppo"] = load_policy(args.model)
    episodes = load_manifest(args.manifest, args.split)
    report = compare_policies(episodes, policies, config, args.output, args.seed)
    # @details Hash input manifest bytes for an auditable link to data preparation.
    report.update({"split": args.split, "manifest": str(args.manifest.resolve()),
                   "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   "threshold_bps": args.threshold_bps,
                   "scope": "Offline replay; synthetic results are pipeline checks, not market evidence."})
    _write_json(args.output / "comparison.json", report)
    if args.fee_grid:
        rows = cost_sweep(episodes, policies, config, args.fee_grid,
                          args.latency_grid, args.spread_grid, args.seed)
        _write_json(args.output / "sensitivity.json", rows)
        _write_json(args.output / "fee_break_even.json", fee_break_even(rows))
        with (args.output / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"split": args.split, "output": str(args.output),
                      "policies": {name: result["aggregate"] for name, result in report["policies"].items()}},
                     indent=2, default=_json_value, allow_nan=False))


if __name__ == "__main__":
    main()
