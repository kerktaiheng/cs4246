# Version 2: basis-adjusted lead-lag PPO

Started 5 October 2026 (Singapore). Code in `latency_arb/v2/`, tests in
`tests/test_v2.py`, run directory `runs/v2-leadlag-ppo/`.

## Why the first two PPO attempts abstained

Both attempts measured the opportunity as the raw mid-price gap between Binance and
Hyperliquid (HL). On training days that gap is dominated by a slow-moving basis: the
two venues trade at a persistent offset of about one basis point, and for six days
(18-23 September) HL was persistently several basis points more expensive. A large
raw gap therefore usually means "the basis is large", not "HL has not caught up yet",
and it does not converge within the 30-second holding limit.

Measured on training days only (3-24 September), with `docs/report_v2/analysis/train_signal_analysis.py`.
Each trade enters at the recorded HL book 150 ms after the decision second and exits 3 s later. Both fills are priced at depth VWAP plus 0.1 bps slippage, crossing the spread both ways. The basis must have 30 s of history, and the HL quote must be fresh at the fill.

| Signal threshold | Raw gap: gross bps per trade | Gap minus 60 s EMA basis: gross bps per trade | Signals/day (de-meaned) |
|---|---:|---:|---:|
| 3 bps | -0.22 | 2.91 | 570 |
| 5 bps | -0.29 | 4.99 | 99 |
| 8 bps | -0.26 | 8.12 | 15 |
| 10 bps | -0.12 | 10.76 | 6 |

A raw-gap trade loses before fees at every threshold, so after fees the best a policy
can do is not trade. That is what PPO learned. Subtracting a causal estimate of the
basis (an exponential moving average of the gap over the previous 60 seconds,
excluding the current second) isolates the part of the gap that does converge. An
earlier scratch version of this table used top-of-book prices without these filters
and gave slightly larger numbers (3.30, 5.53, 8.76 and 10.96 bps). The script's numbers
are the reproducible ones.

The edge decays quickly after the signal. With the de-meaned gap at 3 bps or more,
the mean gross edge was 2.91 bps when filled at the book recorded 150 ms later, 2.43 bps
at 300 ms, 2.05 bps at 450 ms, 1.58 bps at 600 ms, 0.28 bps at 1,150 ms and -0.15 bps at
2,150 ms. HL's public book
arrives about 285 ms after HL's own timestamp, so a fill priced from the book recorded
at t+150 ms may be optimistic. Version 2 keeps the project's 150 ms convention for
comparability with version 1 and reports fills at 300, 450 and 600 ms as stress tests.

## Data

`data/v2-recorded-30d/` is the same causal preparation as version 1 (unchanged code in
`latency_arb/data/recorded.py`) with execution samples at 150, 300, 450 and 600 ms after
each decision second instead of 150 and 300 ms. The extra samples add quality checks,
so 8,227 windows are retained instead of 8,245 (5,970 train, 1,117 validation, 1,140
test). `data/v2-fresh-2026-10-03/` prepares 3 October the same way (285 windows).
`runs/v2_prepare_data.py` reproduces both.

`latency_arb/v2/table.py` turns each split into a per-second table: the raw gap, the
basis and de-meaned gap, 1 s and 5 s Binance and HL returns, 30 s realised volatility,
HL spread, quote ages, book imbalances, and precomputed fill prices at every execution
offset (depth VWAP, spread multiplier and slippage, the same arithmetic as the
simulator). Rolling quantities use only earlier seconds and restart wherever retained
windows are not adjacent, so nothing crosses an excluded outage.

## Decision process

- The agent is consulted only (a) when flat and the de-meaned gap is at least 2 bps,
  the basis has at least 30 s of history and the HL quote is fresh, and (b) every
  second while holding a position. All other seconds are automatic "stay flat".
  This is the action-masking remedy listed in the data guide.
- Two actions. Flat: skip, or enter one 0.001 BTC HL position in the direction of the
  de-meaned gap. Holding: keep holding, or exit.
- Observations are sign-normalised by the trade direction: de-meaned gap, raw gap,
  basis, recent returns on both venues, volatility, HL spread, quote ages, both book
  imbalances, the fee per side, holding time, unrealised P&L at the decision-row exit
  price, the de-meaned gap at entry, time left in the window and history length.
  Scales are fixed constants, not fitted to any split.
- The fee per side is drawn per training window from {0.5, 1, 1.5, 2, 2.5, 3, 3.5} bps
  and is part of the observation. One network must learn how selective to be at each
  cost level.
- Reward is the change in marked equity between decision points in basis points of
  notional. It sums exactly to the window's net P&L (tested). There is no trade bonus.
- Execution follows `LatencySimEnv`: fills 150 ms after the decision second at depth
  VWAP plus 0.1 bps slippage, a taker fee on fill notional, entries must be fresh at
  decision and fill, positions are force-closed 30 s after the entry fill or at the
  window's last sample.

## Training and evaluation

`latency_arb/v2/replay.py` replays the table at roughly 1,500 PPO steps per second per
process; the unchanged simulator runs about 3,500 one-second steps per second, too slow
for PPO to make many passes over 22 days. The fast replay reproduces the simulator
trade for trade (`tests/test_v2.py` checks rule policies at 150, 300, 450 and 600 ms,
including forced exits). Reported validation and test numbers come from the unchanged
simulator through `latency_arb.evaluate.rollout` and `aggregate_results`, the same code
as version 1.

PPO (Stable-Baselines3 2.9, CPU): 8 parallel environments, 512 steps each per update,
batch 512, 10 epochs, learning rate 3e-4, gamma 0.99, GAE 0.95, clip 0.2, 64x64 policy
and value networks, entropy coefficient 0.01 or 0.03, seeds 7, 17 and 27, 2 million
steps, checkpoints every 250,000 steps. Six runs in total.

The protocol was written to `runs/v2-leadlag-ppo/protocol.json` before any validation
evaluation of a version 2 checkpoint. Selection uses validation days only and is frozen
in `selection.json` with its hash before the simulator is run on test days.

Baselines, all in the same simulator with the same windows and costs:

- always flat;
- the version 1 raw-gap threshold rule (enter at |gap| >= X, exit when the directional
  gap is back to 0.5 bps), X tuned per fee on validation;
- a de-meaned-gap rule with the same screen as PPO (enter at |gap - basis| >= X, exit
  after H seconds), X and H tuned per fee on validation. This is the ablation that
  separates what the new feature contributes from what learning contributes.

## Commands

~~~bash
.venv/bin/python runs/v2_prepare_data.py data/v2-recorded-30d 2026-09-03 2026-10-02
.venv/bin/python runs/v2_prepare_data.py data/v2-fresh-2026-10-03 2026-10-03 2026-10-03
.venv/bin/python -c "from latency_arb.v2.table import build_table, save_table; \
  [save_table(build_table('data/v2-recorded-30d/manifest.json', s), f'data/v2-tables/{n}.npz') \
   for s, n in [('train','train'),('validation','validation'),('test','test')]]; \
  save_table(build_table('data/v2-fresh-2026-10-03/manifest.json','test'), 'data/v2-tables/fresh_oct03.npz')"
.venv/bin/python -m latency_arb.v2.train --table data/v2-tables/train.npz \
  --out runs/v2-leadlag-ppo/models/seed7_ent001 --seed 7 --steps 2000000 --ent-coef 0.01
.venv/bin/python -m latency_arb.v2.experiment select --run-dir runs/v2-leadlag-ppo
.venv/bin/python -m latency_arb.v2.experiment evaluate --run-dir runs/v2-leadlag-ppo
~~~

## Protocol history

All files are in `runs/v2-leadlag-ppo/` (copies in `runs/v2-leadlag-ppo-lat450/`).

| File | Registered | Content |
|---|---|---|
| protocol.json | before any v2 validation result | 150 ms group: six PPO runs, sum-of-USD checkpoint selection on validation, raw-gap and fixed-hold de-meaned rules tuned per fee, success criteria S1-S3 |
| protocol_amendment_1.json | before any v2 validation result | Sub-second analysis showed received-book fills overstate the edge; add a 450 ms group (three PPO runs trained, selected and evaluated at 450 ms) |
| protocol_amendment_2.json | after the 150 ms validation selection, before any test result | PPO beat the fixed-hold rule 2-4x on validation; add a convergence-exit rule to test whether PPO only learned a better exit. Grid widened (2b) because the best exit level sat on the grid edge |
| protocol_amendment_3.json | before any test result | Review found a floating-point error in the table's depth VWAP and a harsher-than-simulator forced-exit price in the fast replay. Fixed, tables rebuilt, validation selection re-run: same choices |
| protocol_amendment_4.json | before the 450 ms selection result and any 450 ms test result; after part of the 150 ms test_A results | 450 ms group is primary; per-fee rank-sum selection for that group; primary endpoint is the paired per-window difference against the strongest de-meaned rule with hour-block bootstrap intervals; inconclusive below 100 trades or when the interval contains zero; HL-clock fill stress; test sets labelled as previously examined |

An independent code and protocol review (three reviewers, each finding checked by a
skeptic) found no look-ahead in features, screens or decisions. Its confirmed findings
are listed in the amendments above. Two further points are limitations rather than
fixes: windows are kept only if all five minutes pass quality checks (inherited from
version 1), and rules are tuned on four validation days while PPO is fitted on 22
training days.

## Deployment export

`latency_arb/v2/export_policy.py` writes the selected actor as ONNX (`actor.onnx`),
raw float32 weights (`actor_weights.npz`) and a feature specification
(`actor_spec.json`) to `runs/v2-leadlag-ppo/export/`. The network is
obs(20) -> 64 tanh -> 64 tanh -> 2 logits; the action is the argmax. ONNX Runtime,
a NumPy forward pass and Stable-Baselines3 agree on every checked observation.

## Formal model (for the report)

**Semi-Markov decision process.** Decisions happen at irregular event times, so version 2
is a semi-MDP (SMDP), not a fixed-step MDP. Each five-minute window is a finite-horizon
episode.

- **Decision epochs:** the whole seconds where the agent is consulted, t_0 < t_1 < ... < t_K.
  A flat agent is consulted when |gap - basis| >= 2 bps, the basis has 30 s of history and
  the HL quote is fresh. A holding agent is consulted every second. Every other second is
  an automatic "stay as you are".
- **State:** s = (x, z, f).
  - x is the exogenous market state at the epoch: de-meaned gap, raw gap, basis estimate,
    1 s and 5 s returns on both venues, 30 s volatility, HL spread, quote and receive ages,
    book imbalances, time left in the window, history length.
  - z is the endogenous position state: flat, or (direction, seconds held, entry price,
    de-meaned gap at entry).
  - f is the taker fee per side, fixed for the episode. It is a context variable, so this
    is a contextual MDP family indexed by f.
- **Actions** depend on the state: A(s) = {skip, enter} when flat; A(s) = {hold, exit}
  when holding. Restricting decision epochs and actions this way is formal action masking.
  Entering means taking the de-meaned gap direction.
- **Transitions** factor into two parts:
  - P(x' | x) does not depend on the action, because a 0.001 BTC order is assumed not to
    move either venue's book.
  - z' follows deterministically from z, the action and the recorded fills: the order
    fills L ms later at depth VWAP, the position is force-closed 30 s after entry or at
    the episode end, and entries are rejected on stale quotes.
- **Reward:** r_k = (E(t_{k+1}) - E(t_k)) / (notional x 1e-4), the change in marked equity
  between epochs in basis points, with fees and slippage charged at fills. Rewards sum to
  the episode's net P&L.
- **Objective:** maximise E[sum_k gamma^k r_k] within an episode. PPO and DQN use
  gamma = 0.99 per epoch (approximately the undiscounted SMDP objective, since trades last
  a few epochs). FQI uses the undiscounted objective.

**Why version 1 was a POMDP.** The cross-venue gap is the sum of a slowly drifting basis
b_t (a persistent price offset between the venues, probably funding-related) and a
transient lead-lag component. Only the transient converges within 30 s. Version 1
observed the raw gap, which mixes the two, so its observation was not Markov with
respect to the quantity that determines trade outcomes. For that observation, the best
response is to stay flat. Version 2 adds b_hat_t, an exponential moving average of the
past 60 s of gaps, which acts as a filtered (belief-state) estimate of b_t. With it, the
observation becomes approximately Markov again.

**What exogenous dynamics imply.** Because P(x' | x) does not depend on the action:

1. The return of any policy on recorded data can be computed exactly, by counterfactual
   replay, so off-policy evaluation has no distribution-shift problem.
2. The holding phase is an optimal-stopping problem, V(s) = max(exit payoff, E[V(s')]),
   which fitted-Q iteration solves by backward bootstrapping without exploration.
3. The entry decision is close to a contextual bandit. It is not exact, because holding a
   position blocks later entries and skipping keeps the option to enter later.

## Algorithms compared (amendment 5, 450 ms group)

| Algorithm | Family | What it assumes / tests |
|---|---|---|
| PPO (Schulman 2017) | on-policy actor-critic | General policy-gradient learner; needs exploration and many samples |
| Double DQN (van Hasselt 2016) | off-policy value-based | Replay buffer reuse; the Double-DQN target reduces the max-operator overestimation that would favour noisy entries |
| Fitted-Q iteration (Ernst 2005) on the stopping problem + greedy entry | batch, model-free, structure-exploiting | Uses the exogenous dynamics to train from all counterfactual actions; no exploration needed |
| Contextual bandit (entry only, fixed 3 s exit) | one-step | Ablation: how much the learned sequential exit adds |

Deviations from amendment 5, recorded in `runs/v2-algos-lat450/deviations.json`:

- The bandit was registered with the validation-tuned convergence exit. It was
  implemented with a fixed 3 s exit, so that a model trained on training days does not
  depend on parameters tuned on validation days.
- FQI was described as "iterated to the 30 s horizon" but ran 15 iterations. The fitted
  values had stopped changing by about iteration 9.

All four use the same table, decision epochs, observation, fee sampling and validation
selection rule (per-fee rank sum). Code: `latency_arb/v2/train.py` (PPO),
`latency_arb/v2/train_dqn.py` (Double DQN), `latency_arb/v2/fqi.py` (FQI and bandit),
`latency_arb/v2/algos_experiment.py` (selection, test, paired comparisons).

## Results (frozen choices, unchanged simulator)

Test days 29 September-2 October (previously examined by attempt 1), 450 ms group
(primary), net USD for 0.001 BTC per trade, trades in brackets:

| Fee per side | PPO | Double DQN | FQI | Bandit | Best de-meaned rule | v1 raw-gap rule |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 bps | 76.91 (5,056) | 54.61 (4,218) | 55.37 (4,024) | 25.51 (5,081) | 76.77 (4,972) | 23.92 (2,058) |
| 1.0 bps | 37.91 (3,813) | 25.29 (2,186) | 20.33 (2,685) | 4.49 (1,193) | 34.80 (4,972) | 3.78 (158) |
| 2.0 bps | 2.21 (628) | 3.17 (240) | 0.80 (349) | 0.76 (66) | 1.94 (210) | 0.67 (15) |
| 3.5 bps | 0.05 (45) | 0.02 (12) | -0.06 (27) | 0.05 (5) | 0.05 (30) | 0.00 (0) |

Paired per-window comparisons (hour-block bootstrap 90% intervals; inconclusive below 100
trades or when the interval contains zero), from `runs/v2-leadlag-ppo-lat450/stats.txt` and
`runs/v2-algos-lat450/stats.txt`:

- PPO vs the best de-meaned rule: tie at 0.5 bps; PPO better at 1 bps (+3.12 USD,
  [+0.80, +5.34], 4/4 days); inconclusive at 2 and 3.5 bps.
- PPO vs Double DQN, FQI and the bandit: PPO better at 0.5 and 1 bps.
- Double DQN vs PPO at 2 bps: Double DQN better (+0.96 USD, [+0.1, +1.9]), with 240 trades
  against 628.
- 3.5 bps and 3 October: inconclusive.

Stress, HL-clock fills (fill at HL's own book as of t + 450 ms): at 0.5 bps PPO +27.05,
best rule +26.11, FQI +17.86, Double DQN +11.58, bandit -13.69 USD. At 1 bps and above
the learned policies lose money. The 150 ms group, an optimistic sensitivity, is in
`runs/v2-leadlag-ppo/stats.txt`.

Reading: the de-meaned signal is what makes trading possible at all. Learning the exit is
where most of the learned value comes from (the bandit ablation with a fixed exit earns
about an eighth of PPO at 1 bps). PPO is ahead of the other learners at low fees, but only
slightly ahead of a carefully tuned hand-written rule. All of this holds under the
simulator's received-book fill convention; under HL-clock fills the edge survives only
at about 0.5 bps per side.
