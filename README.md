# Reinforcement learning for cross-venue latency arbitrage

Offline reinforcement learning for BTC perpetual futures traded on Binance and
Hyperliquid (HL). Binance tends to move first and HL follows a few hundred
milliseconds later. The agent learns, from 30 days of recorded order books, when to
take a position on HL in the direction of that move and when to close it, with
fees, spread, slippage and execution delay charged in a replay simulator. Project
proposal: [docs/proposal.tex](docs/proposal.tex).

## Results

The version 2 agent was trained on 3-24 September 2026, selected on 25-28 September,
and evaluated on 29 September-2 October. All reported numbers come from the replay
simulator: orders fill 450 ms after the decision at the recorded HL depth, each trade
is 0.001 BTC (about USD 77), and the taker fee is charged on every fill.

PPO agent, four test days:

| Fee per side | Net P&L (USD) | Trades | Win rate | Profitable days | Max drawdown (USD) |
|---|---:|---:|---:|---:|---:|
| 0.5 bps | +76.91 | 5,056 | 72% | 4 / 4 | 0.17 |
| 1.0 bps | +37.91 | 3,813 | 60% | 4 / 4 | 0.34 |
| 2.0 bps | +2.21 | 628 | 44% | 3 / 4 | 0.53 |
| 3.5 bps | +0.05 | 45 | 44% | 2 / 4 | 0.27 |

Comparison at the same fees (net USD on the same test days):

| Fee per side | PPO | Double DQN | Fitted-Q iteration | Best tuned rule | Version 1 threshold rule |
|---|---:|---:|---:|---:|---:|
| 0.5 bps | **76.91** | 54.61 | 55.37 | 76.77 | 23.92 |
| 1.0 bps | **37.91** | 25.29 | 20.33 | 34.80 | 3.78 |
| 2.0 bps | 2.21 | **3.17** | 0.80 | 1.94 | 0.67 |

- At 0.5 and 1 bps per side the agent is profitable on every test day with
  sub-dollar drawdowns. In paired window-by-window comparisons, PPO beats Double DQN,
  fitted-Q iteration and the version 1 rule at both fees, and beats the best tuned
  rule at 1 bps (+3.12 USD, 90% interval +0.80 to +5.34).
- The same network adjusts to the fee it is given, trading about 5,000 times at
  0.5 bps and 45 times at 3.5 bps.
- With faster execution (150 ms) PPO earns +107.82 USD at 0.5 bps, +66.07 at 1 bps,
  +13.51 at 2 bps and +1.71 at 3.5 bps. At 2 bps it beats the best tuned rule by
  +4.45 USD (90% interval +2.33 to +6.92).
- What made this work: the raw price gap between the venues is dominated by a slow
  basis (a persistent offset) that never converges. Subtracting a 60-second moving
  average of the gap isolates the part that does, and learning when to exit carries
  most of the remaining value.

Scope: these are simulation results on recorded public books, not live trading. The
test days were also used by earlier experiments in this project, so a run on later
days (`latency_arb/v2/confirm.py`) is the next check. If fills are instead priced at
HL's own book clock, the 0.5 bps result remains positive (PPO +27.05 USD) and the
higher fee levels do not.

## Reports and documentation

- Final report, version 2 (formal model, algorithm comparison, results):
  [output/pdf/final_report_v2.pdf](output/pdf/final_report_v2.pdf)
- Final report, version 1 (the first two PPO designs):
  [output/pdf/final_report_v1.pdf](output/pdf/final_report_v1.pdf)
- Version 2 design, protocol history and commands: [docs/V2_LEADLAG.md](docs/V2_LEADLAG.md)
- Resuming runs and confirming on new days: [RESUME_V2.md](RESUME_V2.md)
- Pipeline implementation notes: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
- Data access guide: [docs/DATA_AND_TRAINING.md](docs/DATA_AND_TRAINING.md)

There is no live-order submission path.

## Install

Python 3.11 or later. Commands run from this repository's root in Linux/WSL.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[data,test,reports]'
pytest -q
~~~

The dependency versions used for the checked run are in
[requirements-lock.txt](requirements-lock.txt). Use that file with the PyTorch CPU
index to reproduce this environment.

## Verify the whole pipeline with synthetic data

Synthetic data checks the implementation; it is not evidence of a market edge.

~~~bash
python -m latency_arb.data.generate_episodes demo \
  --output data/demo --days 6 --steps-per-day 1000 --seed 7
python -m latency_arb.agent.train \
  --manifest data/demo/manifest.json --output runs/demo/ppo \
  --steps 30000 --seed 7 --decision-interval-ms 1000
python -m latency_arb.evaluate \
  --manifest data/demo/manifest.json --model runs/demo/ppo \
  --output runs/demo/validation --decision-interval-ms 1000 \
  --select-thresholds 1,2,4,7,10,20,100 --fee-grid 0.5,1,2,3.5
~~~

## Recorded data

Export and preparation are complete for **3 September–2 October 2026 UTC**:
**28,917,283 raw book rows**, reduced to **8,245 complete five-minute windows**
out of 8,640 (**95.43% coverage**). These contain 2,473,500 scheduled decision
snapshots and 7,420,500 replay rows. The fixed split is 22 training days
(5,984 windows), four validation days (1,121), and four test days (1,140).

The source has 20 Binance levels and **five** Hyperliquid levels. Funding
settlement rates are absent, so recorded experiments explicitly assume zero
funding. Rejecting entire windows with stale or missing quotes creates a
quality-filtered subset; results do not describe performance during excluded
periods or uninterrupted deployment.

The recorded-data module exports one venue and UTC day per query, caches Parquet
locally, and prepares causal replay archives. It accepts an explicitly authorized
ClickHouse client; importing it does not connect to anything. See
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for assumptions, execution grids,
quality exclusions, and the fixed 22/4/4-day split.

To prepare an already populated raw cache without opening a database connection:

~~~python
from latency_arb.data.recorded import prepare_recorded_dataset

manifest = prepare_recorded_dataset(
    None, "data/recorded-30d", cache_dir="data/recorded-cache",
    table="caerus.order_book_states",
)
~~~

For generic JSONL or Parquet snapshots, the streaming preparation command supports
the IPC serializer's parallel arrays and the sample files' price/size pairs:

~~~bash
python -m latency_arb.data.generate_episodes prepare INPUT1 INPUT2 \
  --output data/prepared --depth 5
~~~

Generic preparation requires at least three usable UTC days. Negative clock ages
are rejected by default; an explicit receive-age fallback is available for data
diagnostics. The recorded-month path instead excludes affected fixed windows and
never clips negative ages into a plausible measurement.

## Train, evaluate, and report

Run the complete protocol with three seeds, one million requested decisions per
seed, validation selection, and the frozen held-out comparison:

~~~bash
python -m latency_arb.experiment \
  --manifest data/recorded-30d/manifest.json \
  --output runs/recorded-ppo-2026-10-04 \
  --steps 1000000 --seeds 7,17,27 \
  --n-steps 4096 --batch-size 256 --epochs 5 \
  --decision-interval-ms 1000 --evaluate-test
python -m latency_arb.report \
  --run-dir runs/recorded-ppo-2026-10-04 \
  --manifest data/recorded-30d/manifest.json
~~~

Use **one writer per experiment output directory**. Re-running the same compatible
protocol reuses verified completed stages. An incomplete model/checkpoint
directory is preserved and cannot be resumed as a completed stage; use a new
output directory for a fresh run. Do not start a second writer while the current
experiment is running.

The experiment selects thresholds and PPO candidates on validation before opening
the final test comparison. Policies share execution assumptions: 150 ms delay,
3.5 bps taker fees per side, depth VWAP, and adverse slippage. The protocol also
compares frozen policies under the declared fee, delay, and spread stresses.

PPO saves weights, frozen normalization, provenance hashes, settings, and
per-episode trade/P&L logs. Model archives should only be loaded from trusted local
runs. Evaluation writes comparison metrics, every decision, fills, round trips,
equity paths, and cost sensitivity.

PDF generation uses optional ReportLab. Saved report data and charts remain
available when that package is absent. After the report data and PNG charts have
been generated, a separate Python runtime with ReportLab can render only the PDF:

~~~bash
python latency_arb/report.py \
  --run-dir runs/recorded-ppo-2026-10-04 --pdf-only
~~~

## Layout

| Location | Purpose |
|---|---|
| latency_arb/data/schema.py | Validated arrays, manifests, hashes, chronological boundaries |
| latency_arb/data/recorded.py | Bounded daily export and vectorized causal alignment |
| latency_arb/data/generate_episodes.py | Generic stream preparation and labeled synthetic data |
| latency_arb/env/latency_sim.py | Gymnasium execution, position accounting, and risk limits |
| latency_arb/baseline.py | Fixed-threshold and always-flat controls |
| latency_arb/agent/ | Lazy archive sampling, PPO training, frozen inference |
| latency_arb/evaluate.py | Full-episode comparisons and cost sensitivity |
| latency_arb/experiment.py | Compatible-stage resume, seed selection, and frozen test protocol |
| latency_arb/report.py | Saved result summaries, charts, and optional PDF |
| tests/ | Causality, arithmetic, risk, data integrity, and actual PPO save/load checks |

Generated data and run artifacts stay under ignored root-level data/ and runs/.
There is no live-order submission path.

## Project history

1. **Four-action PPO** (`runs/recorded-ppo-2026-10-04`): hold, long, short and exit
   every second at 3.5 bps per side. All seeds converged to holding flat, so the
   policy made no trades. A fixed 10 bps threshold rule in the same simulator earned
   +1.19 USD over 16 test trades.
2. **Opportunity-level PPO** (`runs/opportunity-ppo-2026-10-04`): skip or enter at
   screened gaps of at least 6 bps, with a supervised warm start and 4.5 bps fees.
   It also converged to near-zero trading on validation.
3. **Expected-value selector** (`runs/opportunity-value-2026-10-04`): a
   gradient-boosting model of net trade value. It made one trade on 3 October,
   the same trade as the threshold rule.
4. **Version 2** (`latency_arb/v2/`, `runs/v2-*`): analysing those results showed
   that the raw gap is mostly a slow basis that does not converge, so with that
   input the best policy really was to stay flat. Version 2 adds a causal basis
   estimate, formulates the problem as a semi-MDP with fee-conditioned entry and
   learned exits, and compares PPO, Double DQN, fitted-Q iteration and a
   contextual-bandit ablation against tuned rules (results above).

Each run directory keeps its configuration, models, logs and evaluation files. The
attempt 1 and 2 commands are below.

### Attempt 2 commands

~~~bash
.venv/bin/python -u -m latency_arb.opportunity_experiment \
  --manifest data/recorded-30d/manifest.json \
  --fresh-manifest data/recorded-fresh-2026-10-03/manifest.json \
  --output runs/opportunity-ppo-2026-10-04 \
  --steps 65536 --warm-epochs 50 --seeds 7,17,27
~~~

### Version 2 commands

See [docs/V2_LEADLAG.md](docs/V2_LEADLAG.md#commands).
