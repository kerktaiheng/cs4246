# Implementation and overnight experiment handoff

Updated 4 October 2026, Singapore time. Branch: **codex/proposal-offline-rl**.

## Status

The original placeholder data generator, simulator, trainer, and evaluator have
been replaced. Synthetic PPO training and frozen-policy evaluation have run.
The recorded 30-day export and preparation are in progress; **real-data PPO
results are not yet available**. This status is a handoff checkpoint, not a claim
that the experiment has finished.

The current plan is three independent PPO seeds (7, 17, 27), one million policy
decisions per seed, validation-only threshold and checkpoint selection, then
one frozen final comparison on the held-out four days. A predeclared higher
entropy retry is allowed if every initial candidate executes no validation trades.
The experiment status and exact configuration will be saved under
runs/recorded-ppo-2026-10-04/.

A one-time 8:00 a.m. Singapore follow-up is scheduled in this chat.

## Verified data

The existing SSH tunnel on local port 8123 reaches the collector's ClickHouse
instance. Its credentials are used in memory and never written into the repository,
logs, experiment metadata, or a query URL.

The 4 October 00:30 Singapore inventory query found **32,888,724 rows**:
27,464,104 Binance and 5,424,620 Hyperliquid. The collector remains active, so later
counts naturally increase. Its first receive timestamp is 27 August 2026
02:31:57 UTC; the queried latest timestamp was 3 October 16:30:13 UTC.

The predeclared clean range, **3 September through 2 October UTC**, contains
**28,917,283 raw rows**: 24,146,574 Binance and 4,770,709 Hyperliquid.

| Split | UTC dates | Days |
|---|---|---:|
| Train | 3-24 September | 22 |
| Validation | 25-28 September | 4 |
| Test | 29 September-2 October | 4 |

The 30 August outage, absent 31 August and 1 September, partial 2 September, and
current incomplete capture day are excluded. Full inventory queries and per-day
counts are saved in dataset_inventory_2026-10-04.json.

The raw books contain **20 Binance levels and five Hyperliquid levels**, verified
against the source. Five Hyperliquid levels replace the proposal's ten-level
assumption. The raw table has **no funding-rate history**.

The saved guide's 15 September measurement query was independently rerun:
86,192 aligned seconds, 0.2181177% above seven basis points, and maximum absolute
gap 51.68379 bps. Exact local quantiles were median 0.94046 and p99 4.36124 bps;
these differ slightly from the guide's 0.956 and 4.30. The independently measured
values are preserved in reference_check_2026-09-15.json rather than silently
treated as identical.

## Causal preparation

Exports are sequential per venue and UTC day, with two server threads, a 1 GiB
query memory bound, and a 90-second query limit. Local Parquet caches have checksums
and can be reused after interrupted runs. The server is read only.

A full-source validity flag checks finite positive prices and sizes, consistent
depth, strictly ordered sides, and an uncrossed best bid/ask before Binance is
reduced to best prices and all-depth imbalance. Hyperliquid retains all five levels.

Decisions occur at integer UTC seconds. Execution replay also samples 150 ms and
300 ms after each second. Each sample uses the latest received quote at or before
that exact timestamp, with integer nanosecond clocks. No last-quote-in-a-bucket
observation is mislabeled as the start of the bucket.

Predeclared exclusions include receive ages above two seconds, exchange quote ages
above three seconds, invalid books, negative source-clock ages, spreads above
100 bps, and cross-venue gaps above 250 bps. Exclusions are measured and reported;
they are not adjusted according to policy profit.

Episodes are complete **fixed UTC five-minute windows**. If any grid row in a
window fails quality checks, the entire window is rejected. This avoids giving the
policy advance knowledge of an outage by turning an unexpected gap into a known
terminal time. Retained windows end at their predeclared final execution sample.

On 3 September, the first prepared day retained **273 of 288 windows (94.79%)**,
or 81,900 scheduled decision seconds. Full-month retention will be in the final
quality report. Rejected periods remain excluded from all policies equally.

The generic JSONL preparation command also splits at gaps, but its variable,
quality-selected terminal boundaries are a research approximation. The recorded
experiment uses the stricter fixed-window path.

## Simulator and economics

Four actions are HOLD, LONG, SHORT, and EXIT. HOLD preserves an existing position;
LONG and SHORT can open only one fixed-size Hyperliquid position; there is no
pyramiding or one-action reversal. Binance supplies information rather than a
simultaneously hedged second leg. This is directional slow-venue execution, not a
risk-free two-leg arbitrage portfolio.

The shared base configuration uses 0.001 BTC, 150 ms submission delay, 3.5 bps
taker fee per side, displayed-depth VWAP, and an additional 0.1 bps adverse
slippage per fill. Spread stress increases each level's distance from the midpoint.
Fees are computed on actual fill notional.

One-second decisions do not skip intermediate execution samples. A submitted order
fills only on the first eligible replay sample; it never receives the decision-time
price retroactively. Pending entries may be canceled, and entry quotes must pass
both one-second receive-age and exchange-age guards at submission and fill.

Funding accounting is implemented and tested, with positive funding paid by longs
and received by shorts. A settlement applies only to inventory carried into that
event. **Recorded experiments assume zero funding because the source lacks rates**;
their P&L must not be described as a fully funding-adjusted result.

Reward is the change in marked equity, scaled explicitly for PPO. It contains no
trade bonus and reconciles to final net P&L after liquidation. Reward scaling
does not change reported dollar returns.

Position size, a 30-second holding deadline, and a USD100 drawdown trigger are
enforced by the environment. Risk exits pay delay and can exceed the trigger loss.
Mandatory exits beyond displayed depth price the residual at the worst recorded
level plus a 25 bps adverse penalty; extrapolated quantity is logged explicitly.
Terminal liquidation is scheduled against the known fixed episode horizon.

The simulator records pre-fill and post-fill marked equity so same-row costs and
intermediate sampled drawdowns are preserved. It cannot observe price excursions
between the replay samples. Risk exits whose due times lack an exact sample round
forward to the next available sample, making their effective delay longer.

## PPO and evaluation

Stable-Baselines3 PPO runs on CPU with a seeded two-layer 64-unit policy/value MLP.
Observation normalization fits training observations only and is frozen for every
evaluation. Rewards use an explicit scale without reward normalization or artificial
profit shaping. Full configurations, library versions, manifest and artifact hashes,
and the observation order are saved.

Training loads one archive at reset and keeps an LRU cache of two. Each completed
episode logs actual trades, P&L, fees, and archive identity from the first completion.
Rollout logs distinguish HOLD, LONG, SHORT, and EXIT requests and report cumulative
zero-trade episode frequency. A zero return with no executed trades is reported as
inaction, not evidence of profitable learning or convergence.

Validation chooses the fixed entry threshold from 1, 2, 4, 7, 10, 20, and 100 bps.
A genuinely always-flat policy remains a separate control: a large threshold does
not guarantee abstention. PPO seed/checkpoint selection uses validation P&L, then
fewer trades on ties. The final test is loaded only after selection is saved.

All policies use identical fees, depth, delay, position size, risk limits, and
episode exclusions. The cost grid is 0.5, 1, 2, and 3.5 bps per side; delays 150 and
300 ms; spread multipliers 1.0 and 1.5. Policies stay frozen across the sensitivity
study. Break-even is reported only as an observed fee-grid bracket, not a precise
estimated fee when the evidence does not support one.

Metrics include net P&L, fixed-capital return, marked drawdown, win rate, trade
count, daily variation, requested/rejected/canceled orders, and abstention.
Reference selectivity divides completed trades by scheduled decisions with absolute
gap at least seven bps. It is null when there are no such opportunities and can
exceed one when a policy trades below that reference. A separate flat-entry rate
uses flat decisions without pending orders as its denominator.

Returns stitch fixed-size episode P&L without reinvestment. They do not simulate
margin, liquidation tiers, changing capital allocation, private fills, market
impact, counterfactual queue dynamics, or the market's reaction to this agent.

## Verification and artifacts

Tests cover exact fees/spreads for long and short positions, depth rejection,
latency, funding timing/signs, risk and terminal exits, internal-event drawdown,
no-lookahead prefixes, timestamp precision, stale/bad data, chronological splits,
hash/path integrity, bounded caches, real short PPO fits, seed reproducibility,
and saved-model/normalization equivalence.

The latest completed combined run had **91 passing tests** before additional
report/orchestration and evaluator regression tests were added. A final complete
count will be recorded after integration.

Reproduction starts in README.md. Generated raw data, archives, model checkpoints,
and detailed logs live in root-level data/ and runs/ and are excluded from Git.
The proposal and pre-existing data guide are preserved. No orders are sent to an
exchange, and the capture service and existing SSH tunnel are left running.
