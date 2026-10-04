"""@file opportunity_evaluate.py
@brief Frozen opportunity-policy evaluation over complete selected episodes.
@details No training, normalization fitting, or counterfactual-label aggregation
occurs here. All clips, including those without candidates, contribute to the
report. Economic histories are kept for one archive at a time.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, TextIO

import numpy as np

from latency_arb.data.schema import ReplayEpisode, load_manifest_entry, read_manifest
from latency_arb.env.latency_sim import EnvConfig
from latency_arb.opportunity import FEATURE_NAMES, OpportunityConfig, OpportunityReplayEnv


class SkipPolicy:
    """@brief Decline every exposed opportunity without fitting any parameters."""

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """@brief Return the SB3-compatible binary skip prediction."""
        return 0, None


class GapThresholdPolicy:
    """@brief Enter above a fixed raw abs_gap_bps threshold under the shared wrapper."""

    def __init__(self, threshold_bps: float = 6.0) -> None:
        """@brief Validate the additional threshold without using evaluation outcomes."""
        if not np.isfinite(threshold_bps) or threshold_bps < 0:
            raise ValueError("threshold_bps must be finite and nonnegative")
        self.threshold_bps = float(threshold_bps)
        self._index = FEATURE_NAMES.index("abs_gap_bps")

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """@brief Compare the current feature only; automatic exits remain shared."""
        values = np.asarray(observation)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("predict requires one finite raw feature vector")
        return int(values[self._index] >= self.threshold_bps), None


def _json_default(value: Any) -> Any:
    """@brief Convert numerical audit fields into strict portable JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _emit(handle: TextIO | None, record: dict) -> None:
    """@brief Write one audit row only when the caller requested file output."""
    if handle is not None:
        handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")


def _reset_policy(policy: Any) -> None:
    """@brief Clear optional recurrent state at each clip, without fitting anything."""
    reset = getattr(policy, "reset", None)
    if not callable(reset):
        reset = getattr(policy, "reset_state", None)
    if callable(reset):
        reset()


def _predict(policy: Any, observation: np.ndarray) -> int:
    """@brief Request deterministic raw-feature inference and validate a scalar action."""
    result = policy.predict(observation, deterministic=True)
    action = result[0] if isinstance(result, tuple) else result
    values = np.asarray(action)
    if values.size != 1 or values.item() not in (0, 1):
        raise ValueError("policy must return one binary action: 0 skip or 1 enter")
    return int(values.item())


def _rollout(episode: ReplayEpisode, policy: Any, execution: EnvConfig,
             opportunity: OpportunityConfig, handles: dict[str, TextIO],
             number: int) -> tuple[dict, list[dict]]:
    """@brief Replay a complete clip, including initial waiting and all automatic exits.
    @details Dummy terminal steps for clips with no candidate do not invoke the
    policy or count as macro skips. Ledger, equity, fees, funding, and rewards
    reconcile before results enter the aggregate.
    """
    env = OpportunityReplayEnv(episode, execution, opportunity)
    context = {"episode": number, "episode_id": episode.metadata.get("episode_id", str(number)),
               "day": episode.metadata["day"]}
    decisions = skips = requests = 0
    rewards = 0.0
    try:
        _reset_policy(policy)
        observation, info = env.reset(seed=0)
        while True:
            has_opportunity = bool(info["has_opportunity"])
            action = _predict(policy, observation) if has_opportunity else 0
            if has_opportunity:
                row = {**context, "macro_decision": decisions, "action": action,
                       "timestamp_ns": int(info["timestamp_ns"]),
                       "features": np.asarray(observation).tolist()}
                decisions += 1
                skips += int(action == 0)
                requests += int(action == 1)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards += float(reward)
            if has_opportunity:
                row.update({"next_timestamp_ns": int(info["timestamp_ns"]),
                            "reward": float(reward), "equity_after": float(info["equity"]),
                            "underlying_steps": int(info.get("macro_underlying_steps", 0)),
                            "terminated": bool(terminated), "truncated": bool(truncated)})
                _emit(handles.get("decisions"), row)
            if terminated or truncated:
                break
        trades, fills, history = env.base.trades, env.base.fills, env.base.equity_history
        if not history:
            raise AssertionError("evaluation requires full sampled equity history")
        equity = np.asarray([row["equity"] for row in history], dtype=np.float64)
        if not np.isfinite(equity).all():
            raise AssertionError("equity history contains nonfinite values")
        pnl = float(info["pnl"])
        profits = np.asarray([row["net_pnl"] for row in trades], dtype=np.float64)
        #! @details Economics derive from the unchanged simulator, not from a
        #! policy's reward interpretation or the sum of overlapping trial labels.
        checks = [float(info["equity"]) - execution.initial_cash,
                  float(equity[-1]) - execution.initial_cash, float(profits.sum())]
        if any(not np.isclose(value, pnl, rtol=1e-7, atol=1e-7) for value in checks):
            raise AssertionError("trade ledger and terminal equity do not reconcile")
        if not np.isclose(rewards, pnl * execution.reward_scale, rtol=1e-7, atol=1e-7):
            raise AssertionError("macro rewards do not reconcile with terminal PnL")
        if info["inventory"] != 0 or int(info["trade_count"]) != len(trades):
            raise AssertionError("terminal position or ledger count is inconsistent")
        previous_exit = None
        for trade in trades:
            if previous_exit is not None and trade["entry_timestamp_ns"] < previous_exit:
                raise AssertionError("opportunity replay produced overlapping trades")
            previous_exit = trade["exit_timestamp_ns"]
        fees, funding = float(info["fees_paid"]), float(info["funding_paid"])
        if not np.isclose(sum(row["fee"] for row in fills), fees, rtol=1e-7, atol=1e-7):
            raise AssertionError("fill fees disagree with terminal accounting")
        if not np.isclose(sum(row["funding_paid"] for row in trades), funding,
                          rtol=1e-7, atol=1e-7):
            raise AssertionError("trade funding disagrees with terminal accounting")
        timestamps = np.asarray([row["timestamp_ns"] for row in history], dtype=np.int64)
        durations = np.diff(timestamps)
        if np.any(durations < 0):
            raise AssertionError("equity marks are not chronological")
        inventory = np.asarray([row["inventory"] for row in history[:-1]])
        invested_ns = int(durations[inventory != 0].sum())
        scheduled_ns = int(episode.timestamp_ns[-1]) - int(episode.timestamp_ns[0])
        if invested_ns > scheduled_ns:
            raise AssertionError("exposure exceeds the selected episode horizon")
        #! @details A risk stop leaves the remaining scheduled interval flat;
        #! exposure denominators still include the complete selected clip.
        metrics = {
            **context, "synthetic": bool(episode.metadata.get("synthetic", False)),
            "initial_cash": execution.initial_cash, "net_pnl": pnl,
            "ledger_net_pnl": float(profits.sum()), "net_return": pnl / execution.initial_cash,
            "max_drawdown_usd": float(np.max(np.maximum.accumulate(equity) - equity)),
            "trade_count": len(trades), "winning_trade_count": int((profits > 0).sum()),
            "win_rate": float((profits > 0).mean()) if len(profits) else None,
            "fees_paid": fees, "funding_paid": funding,
            "pnl_before_fees_and_funding": pnl + fees + funding,
            "macro_decision_count": decisions, "decision_count": decisions,
            "macro_skip_count": skips, "skip_rate": skips / decisions if decisions else None,
            "entry_request_count": requests, "rejection_count": int(info["rejection_count"]),
            "cancellation_count": int(info["cancellation_count"]),
            "no_opportunity": decisions == 0,
            "exposure_duration_ms": invested_ns / 1e6,
            "flat_duration_ms": (scheduled_ns - invested_ns) / 1e6,
            "replay_duration_ms": scheduled_ns / 1e6,
            "exposure_rate": invested_ns / scheduled_ns if scheduled_ns else None,
            "scaled_reward_sum": rewards,
            "underlying_step_count": int(info.get("underlying_step_count", 0)),
            "termination_reason": info.get("termination_reason"),
            "start_timestamp_ns": int(episode.timestamp_ns[0]),
            "end_timestamp_ns": int(info["timestamp_ns"]),
            "scheduled_end_timestamp_ns": int(episode.timestamp_ns[-1]),
        }
        for row in trades:
            _emit(handles.get("trades"), {**context, **row})
        for row in fills:
            _emit(handles.get("fills"), {**context, **row})
        return metrics, history
    finally:
        env.close()


def _accumulate(totals: dict, metrics: dict, history: list[dict],
                curve: TextIO | None) -> None:
    """@brief Stitch full equity marks online without an artificial reset recovery.
    @details Optional compact curves retain at most six informative marks per clip.
    Exact drawdown always uses every original mark, including pre/post-fill costs.
    """
    values = np.asarray([row["equity"] for row in history], dtype=np.float64)
    stitched = totals["initial_cash"] + totals["net_pnl"] + values - metrics["initial_cash"]
    peaks = np.maximum.accumulate(np.r_[totals["peak"], stitched])[1:]
    drawdown = peaks - stitched
    totals["max_drawdown_usd"] = max(totals["max_drawdown_usd"], float(drawdown.max()))
    totals["peak"] = float(peaks[-1])
    if curve is not None:
        trough = int(np.argmax(drawdown))
        peak = int(np.argmax(stitched[:trough + 1]))
        retained = sorted({0, len(values) - 1, int(np.argmin(stitched)),
                           int(np.argmax(stitched)), trough, peak})
        for index in retained:
            _emit(curve, {"episode": metrics["episode"], "episode_id": metrics["episode_id"],
                         "day": metrics["day"], "timestamp_ns": int(history[index]["timestamp_ns"]),
                         "stitched_equity": float(stitched[index]),
                         "episode_equity": float(values[index]),
                         "sampled_drawdown_usd": float(drawdown[index])})
        if metrics["end_timestamp_ns"] < metrics["scheduled_end_timestamp_ns"]:
            _emit(curve, {"episode": metrics["episode"], "episode_id": metrics["episode_id"],
                         "day": metrics["day"], "timestamp_ns": metrics["scheduled_end_timestamp_ns"],
                         "stitched_equity": float(stitched[-1]), "episode_equity": float(values[-1]),
                         "sampled_drawdown_usd": float(drawdown[-1]),
                         "flat_tail_after_termination": True})
    totals["net_pnl"] += metrics["net_pnl"]
    day = metrics["day"]
    totals["daily_pnl"][day] = totals["daily_pnl"].get(day, 0.0) + metrics["net_pnl"]
    totals["daily_trades"][day] = totals["daily_trades"].get(day, 0) + metrics["trade_count"]


def _aggregate(totals: dict, episodes: list[dict]) -> dict:
    """@brief Summarize compact per-episode metrics using one fixed cash reference."""
    def total(name: str):
        """@brief Sum an explicitly named numerical metric across every selected clip."""
        return sum(row[name] for row in episodes)
    decisions = total("macro_decision_count")
    trades = total("trade_count")
    wins = total("winning_trade_count")
    duration = total("replay_duration_ms")
    result = {key: value for key, value in totals.items() if key != "peak"}
    result.update({
        "episode_count": len(episodes), "day_count": len(totals["daily_pnl"]),
        "synthetic": all(row["synthetic"] for row in episodes),
        "net_return": totals["net_pnl"] / totals["initial_cash"],
        "trade_count": trades, "winning_trade_count": wins,
        "win_rate": wins / trades if trades else None,
        "daily_pnl_std": float(np.std(list(totals["daily_pnl"].values()))),
        "no_opportunity_episode_count": sum(row["no_opportunity"] for row in episodes),
        "skip_rate": total("macro_skip_count") / decisions if decisions else None,
        "exposure_rate": total("exposure_duration_ms") / duration if duration else None,
        "accounting": "Fixed BTC size; cumulative USD PnL over reference cash; no reinvestment.",
        "drawdown_basis": "Every sampled base-simulator equity mark; no reset recovery.",
    })
    for field in ("macro_decision_count", "decision_count", "macro_skip_count",
                  "entry_request_count", "rejection_count", "cancellation_count",
                  "fees_paid", "funding_paid", "pnl_before_fees_and_funding",
                  "exposure_duration_ms", "flat_duration_ms", "replay_duration_ms",
                  "scaled_reward_sum", "underlying_step_count"):
        result[field] = total(field)
    return result


def evaluate_opportunity_policies(
    manifest: str | Path, split: str, policies: dict[str, Any], execution: EnvConfig,
    opportunity: OpportunityConfig, output_dir: str | Path | None = None,
) -> dict:
    """@brief Evaluate all and only selected split archives with frozen predictors.
    @param policies Names mapped to predict(raw_features, deterministic=True) objects.
    @param execution Shared execution economics; history logging is enabled for audit.
    @param opportunity Shared present-information gate and automatic exit rule.
    @param output_dir Optional new destination for comparison.json and streamed JSONL.
    @return config/opportunity_config/split/policies with aggregates and episode metrics.
    @details reset/reset_state is honored per clip. The report retains only compact
    metrics; one archive and its base history are resident during each rollout.
    Unselected archive contents, training functions, and fitting routines are unused.
    """
    if not policies:
        raise ValueError("at least one frozen policy is required")
    for name, policy in policies.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("policy names must be safe nonempty filename components")
        if not callable(getattr(policy, "predict", None)):
            raise ValueError(f"policy {name} does not expose predict")
    path = Path(manifest)
    document, entries = read_manifest(path, split)
    chosen = replace(execution, log_history=True)
    destination = Path(output_dir) if output_dir is not None else None
    artifacts = ("trades", "fills", "decisions", "equity_curve")
    if destination is not None:
        targets = [destination / "comparison.json"] + [
            destination / name / f"{artifact}.jsonl" for name in policies for artifact in artifacts]
        if any(target.exists() for target in targets):
            raise FileExistsError("opportunity evaluation audit files already exist")
    with path.open("rb") as handle:
        manifest_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"config": asdict(chosen), "opportunity_config": asdict(opportunity),
              "split": split, "manifest_sha256": manifest_hash, "feature_names": list(FEATURE_NAMES),
              "selected_episode_count": len(entries),
              "funding_unavailable_zero_assumption": bool(
                  document.get("funding_unavailable_zero_assumption", False)),
              "curve_sampling": "Episode extrema/endpoints/drawdown pair; statistics use every mark.",
              "policies": {}}
    for name, policy in policies.items():
        totals = {"initial_cash": chosen.initial_cash, "net_pnl": 0.0,
                  "peak": chosen.initial_cash, "max_drawdown_usd": 0.0,
                  "daily_pnl": {}, "daily_trades": {}}
        metrics = []
        handles = {}
        try:
            if destination is not None:
                directory = destination / name
                directory.mkdir(parents=True, exist_ok=True)
                for artifact in artifacts:
                    handles[artifact] = (directory / f"{artifact}.jsonl").open("x", encoding="utf-8")
            for number, entry in enumerate(entries):
                #! @details Hash verification and loading happen only after split
                #! selection; no full manifest's collection is eagerly materialized.
                episode = load_manifest_entry(path, entry)
                row, history = _rollout(episode, policy, chosen, opportunity, handles, number)
                _accumulate(totals, row, history, handles.get("equity_curve"))
                metrics.append(row)
                del history, episode
        finally:
            for handle in handles.values():
                handle.close()
        report["policies"][name] = {"aggregate": _aggregate(totals, metrics), "episodes": metrics}
    if destination is not None:
        (destination / "comparison.json").write_text(
            json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n",
            encoding="utf-8")
    return report
