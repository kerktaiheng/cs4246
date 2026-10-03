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

The verified clean range is **3 September–2 October 2026 UTC**, containing
**28,917,283 raw book rows**. The source has 20 Binance levels and **five**
Hyperliquid levels. Funding settlement rates are absent from the recorded table.

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

## Train and evaluate

~~~bash
python -m latency_arb.agent.train \
  --manifest data/recorded-30d/manifest.json --output runs/ppo-seed7 \
  --steps 1000000 --seed 7 --n-steps 4096 --batch-size 256 \
  --gamma 0.999 --decision-interval-ms 1000
python -m latency_arb.evaluate \
  --manifest data/recorded-30d/manifest.json --model runs/ppo-seed7 \
  --output runs/validation --decision-interval-ms 1000 \
  --select-thresholds 1,2,4,7,10,20,100
~~~

Evaluation defaults to validation. Use --split test only after fixing the
experiment protocol. Policies share the same execution assumptions, including
150 ms delay, 3.5 bps taker fees per side, depth VWAP, and adverse slippage.

PPO saves weights, frozen normalization, provenance hashes, settings, and
per-episode trade/P&L logs. Model archives should only be loaded from trusted local
runs. Evaluation writes comparison metrics, every decision, fills, round trips,
equity paths, and optional fee/delay/spread sensitivity.

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
| tests/ | Causality, arithmetic, risk, data integrity, and actual PPO save/load checks |

Generated data and run artifacts stay under ignored root-level data/ and runs/.
There is no live-order submission path.
