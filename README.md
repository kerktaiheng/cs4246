# Caerus offline reinforcement-learning research

Cost-aware replay and PPO for **Learning When Not to Trade**, the proposal in
[docs/proposal.tex](docs/proposal.tex). This branch replaces the original model
placeholders with a checked offline research pipeline.

Start with [the implementation and experiment handoff](docs/IMPLEMENTATION.md).
The pre-existing [data access guide](docs/DATA_AND_TRAINING.md) remains useful for
the collector connection and historical coverage; its placeholder-code status
predates this implementation.

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

At the 4 October 2026, 10:04 a.m. Singapore checkpoint, the baseline stage had
completed and PPO seed 7 was training. Completed real-data PPO and held-out
results were not yet available. See the run's status.json for its latest stage.

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

## Opportunity PPO follow-up

The original four-action run abstained on held-out data. The separate opportunity
study trains a binary entry selector: skip, or enter in the current gap direction.
An unchanged execution simulator handles the full trade, a 0.5 bps convergence
exit, the 30-second holding limit, costs, and delayed fills.

~~~bash
.venv/bin/python -u -m latency_arb.opportunity_experiment \
  --manifest data/recorded-30d/manifest.json \
  --fresh-manifest data/recorded-fresh-2026-10-03/manifest.json \
  --output runs/opportunity-ppo-2026-10-04 \
  --steps 65536 --warm-epochs 50 --seeds 7,17,27
~~~

Use a new, empty output directory. This runner preserves partial artifacts on
failure and does not restart or overwrite an existing study.

The study uses **4.5 bps per side** (9 bps round trip), 150 ms execution delay,
0.1 bps extra adverse slippage per fill, and 0.001 BTC position size. The 6 bps
candidate screen defines when the model can consider an entry; it is not a
claim that a 6 bps price gap pays these costs. Training labels and portfolio
rewards charge the complete actual modeled trade costs.

Training covers 3–24 September. Supervised actor initialization uses every causal
training candidate and its net fixed-exit outcome, including losses and rejected
future fills. Those trial outcomes can overlap and must never be summed as
portfolio returns. Genuine PPO then learns from continuous, nonoverlapping replay
with unchanged marked-equity rewards, undiscounted opportunity transitions, and
fixed positive reward scaling. Feature normalization is fitted on training data
only and stays frozen.

Both supervised-only and PPO-finetuned checkpoints are compared on 25–28
September, alongside fixed thresholds and cash. Validation chooses maximum net
dollars, with fewer trades breaking ties. A supervised-only winner is explicitly
identified as such; it does not demonstrate an improvement from PPO.

The previous 29 September–2 October test was already examined and is excluded
from this study. **3 October is a new, later test day**, prepared independently:
939,328 raw rows, 286/288 accepted fixed windows, and 85,800 one-second decisions.
The exact selected model is frozen before this test is opened. If no learned
validation candidate earns positive net dollars with actual trades, the fresh
test remains unused. One positive fresh day would be provisional evidence only.

Source is split into opportunity.py (causal features and binary replay),
agent/opportunity_train.py (training-only targets and genuine PPO),
opportunity_evaluate.py (full-episode economic comparison), and
opportunity_experiment.py (protocol, selection, and protected test access).
Generated outputs include protocol.json, event target provenance, separate
checkpoints, training trade logs, validation comparisons, selection.json,
summary.json, and status.json under the requested run directory.

The recorded replay still assumes zero funding because rates are unavailable.
Public sampled books do not model private fills, market impact, margin, or
movements between samples. Quality exclusions and fixed five-minute window
resets remain in force. Reused validation and multiple candidate selection can
overstate performance; the always-flat and threshold controls remain essential.


## Research outcome log: failed PPO attempts

**Attempt 1: FAILED to learn profitable trading.** The original four-action
PPO experiment completed operationally, but all three seeds and the higher-entropy
retry made zero completed validation trades. The selected policy also made zero
trades and earned $0 on the untouched 29 September–2 October test. This is
abstention, not a successful trading strategy. At 7 bps round-trip fees, the
separate 10 bps threshold control earned $1.188287 across 16 held-out trades.
That profit belongs to the rule-based control and must never be credited to PPO.

The original models, configuration, training logs, validation/test ledgers,
summary, and report remain preserved in runs/recorded-ppo-2026-10-04.
Its research_outcome.json explicitly records the research failure. The original
status.json still says completed because the computation completed; it does not
classify strategy quality.

**Attempt 2: FAILED to learn profitable trading.** The opportunity-level redesign
used causal entry features, binary skip/trade actions, fixed exits, supervised
initialization from net trade outcomes, and genuine PPO fine-tuning. Costs were
9 bps round trip plus spread and slippage. On 25–28 September validation:

| Candidate | Net USD | Completed trades |
|---|---:|---:|
| Seed 7, supervised initialization | 0.000000 | 0 |
| Seed 7, PPO fine-tuned | 0.000000 | 0 |
| Seed 17, supervised initialization | -0.107772 | 4 |
| Seed 17, PPO fine-tuned | 0.000000 | 0 |
| Seed 27, supervised initialization | -0.021333 | 4 |
| Seed 27, PPO fine-tuned | 0.000000 | 0 |

No learned candidate beat cash, so the protected 3 October test was not opened.
Artifacts and research_outcome.json remain in runs/opportunity-ppo-2026-10-04.
The saved report is runs/opportunity-ppo-2026-10-04/report/report.md.

The training audit found 317,232 candidate rows, with 99.57% concentrated on
18–23 September when Hyperliquid was persistently more expensive than Binance.
These correlated, overlapping examples are not 317,232 independent latency
opportunities. The persistent gap's cause is not proven by these fields.
Historical rows lack an explicit instrument symbol, and funding rates are
unavailable. Present quote age and price validity checks passed.

The next comparison directly estimates expected net trade dollars from the same
training-only outcomes. It is a supervised regression model, not another PPO
success claim. Its primary evaluation must use the user's stated 7 bps
round-trip fee, with 9 bps as a frozen-policy stress comparison. Every attempted
configuration and failure remains part of the research record. Software tests
do not establish profitability, and a positive validation or single fresh-day
result cannot establish a durable edge.
