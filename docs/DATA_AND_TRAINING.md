# Data & Training Guide

How to pull the Caerus market-data set and train a PPO agent on it.
Audience: a teammate who has never touched this project.

Everything in the "Access" and "Pull" sections below was executed and verified on
2026-10-04. Numbers quoted under "Reference values" are reproducible checks — if your
pipeline disagrees with them, your pipeline is wrong.

---

## 0. Status: what exists, what you build

| Thing | State |
|---|---|
| Live capture (Binance + Hyperliquid → ClickHouse) | **Running**, 32 complete days banked |
| ClickHouse query access | **Works** (via SSH tunnel, see below) |
| `latency_arb/data/generate_episodes.py` | **Stub** — you build it |
| `latency_arb/env/latency_sim.py` | **Stub** — reward currently echoes the action back |
| `latency_arb/agent/train.py` | **Stub** — you build it |
| Baseline strategy | **Not written** |

Do not assume any Python in this repo works yet. The data is real; the model code is not.

---

## 1. Access

The ClickHouse ports are bound to **loopback only** on the capture box, so you cannot
connect directly. Open an SSH tunnel first.

```bash
ssh -fN -L 8123:127.0.0.1:8123 -i ~/.ssh/caerus-key.pem ubuntu@3.82.42.42
```

Credentials live on the box at `/data/caerus/deploy/.env` (user `caerus`). **Never commit
them.** Export into your shell:

```bash
export CH_USER=caerus
export CH_PASS=$(ssh -i ~/.ssh/caerus-key.pem ubuntu@3.82.42.42 \
  "grep CLICKHOUSE_PASSWORD /data/caerus/deploy/.env | cut -d= -f2")
```

Smoke test — should print two row counts:

```bash
curl -s "http://localhost:8123/?user=$CH_USER&password=$CH_PASS" \
  --data-binary "SELECT exchange, count() FROM caerus.order_book_states GROUP BY exchange FORMAT TSV"
```

Python deps (note: `stable-baselines3` is **missing** from `requirements.txt`, add it):

```bash
pip install clickhouse-connect polars numpy gymnasium torch stable-baselines3
```

---

## 2. The data

One table: `caerus.order_book_states`.

```
exchange        LowCardinality(String)   -- 'binance' | 'hyperliquid'
exchange_ts_ns  UInt64                   -- venue's own timestamp
local_ts_ns     UInt64                   -- OUR receipt time; use this for alignment
bid_prices      Array(Float64)           -- index 0 = best bid, descending
bid_sizes       Array(Float64)
ask_prices      Array(Float64)           -- index 0 = best ask, ascending
ask_sizes       Array(Float64)
```

```
ENGINE = MergeTree
PARTITION BY day(local_ts_ns)
ORDER BY (exchange, local_ts_ns)
```

Always filter on `exchange` and a `local_ts_ns` range — that hits both the partition and
the sort key. Anything else scans 2 GiB.

**The two venues are not symmetric.** This is the single most important fact about the
dataset:

| | depth per side | updates/sec | rows/day |
|---|---|---|---|
| binance | **20** (always) | 8.24 | ~820k |
| hyperliquid | **5** (always) | **1.63** | ~159k |

Binance is `btcusdt@depth20@100ms`. Hyperliquid is `l2Book` with `"fast":true`, which
returns exactly 5 levels — verified constant across 5.4M rows, it is not a bug.

Consequences you must respect:

- **Hyperliquid publishes ~1.6 times per second.** That is the ceiling on how fast the
  agent can react to the slow leg. A sub-second decision cadence is fiction. Use **1 Hz**.
- Any feature using levels 6–20 exists for Binance only. Don't build a symmetric
  20-level observation; it will be 75% padding on the HL side.

**Timestamps are UTC.** The box runs UTC. A "2026-10-03 16:19" row is 2026-10-04 00:19
Singapore time. Use `local_ts_ns` for cross-venue alignment, never `exchange_ts_ns` — the
venue clocks are not comparable to each other.

---

## 3. Data hygiene — read this before selecting a date range

Capture started **2026-08-27 02:31 UTC**. There is a **three-day hole** from an
out-of-memory incident:

| Date | Status |
|---|---|
| 2026-08-27 | Partial (capture began mid-day) |
| 2026-08-28 – 08-29 | Complete |
| 2026-08-30 | **Near-total outage** (7.8k rows, ~1%) |
| **2026-08-31** | **NO DATA AT ALL** |
| **2026-09-01** | **NO DATA AT ALL** |
| 2026-09-02 | **Partial** (~52%) |
| 2026-09-03 – 10-02 | **Complete — 30 consecutive days** |
| today | In progress, incomplete |

A naive `WHERE date BETWEEN '2026-08-27' AND '2026-10-02'` will silently hand you a
dataset with a 3-day discontinuity in the middle. If you then build episodes by row
index, you will create episodes that jump across a 72-hour price gap and the agent will
learn from fabricated moves.

**Default clean range: `2026-09-03` to `2026-10-02` (30 complete days).** Add 08-28 and
08-29 only if you explicitly handle the discontinuity.

Chronological split (never shuffle — that leaks future prices into training):

| Split | Dates | Days |
|---|---|---|
| train | 2026-09-03 → 09-24 | 22 |
| validation | 2026-09-25 → 09-28 | 4 |
| test (touch once, at the end) | 2026-09-29 → 10-02 | 4 |

The capture is still running, so the test window can be pushed later as data accrues.

### Server memory cap

ClickHouse on the box is capped at **1.5 GB**. Full-history high-cardinality aggregates
(e.g. `uniqExact` over all 33M rows) **will be killed** with `MEMORY_LIMIT_EXCEEDED`.
This is expected, not a fault. Query **one day at a time** and aggregate locally. All
queries below are day-scoped for this reason.

---

## 4. Pull recipe

Resample each venue to 1-second buckets (last observation in the bucket), then inner-join
on the second. This query is tested and runs within the memory cap:

```sql
WITH
  b AS (SELECT intDiv(local_ts_ns,1000000000) AS ts,
               argMax((bid_prices[1]+ask_prices[1])/2, local_ts_ns) AS mid,
               argMax(bid_prices[1], local_ts_ns) AS bid,
               argMax(ask_prices[1], local_ts_ns) AS ask
        FROM caerus.order_book_states
        WHERE exchange='binance'
          AND local_ts_ns >= toUnixTimestamp({day:String})*1000000000
          AND local_ts_ns <  toUnixTimestamp({day:String})*1000000000 + 86400000000000
        GROUP BY ts),
  h AS (SELECT intDiv(local_ts_ns,1000000000) AS ts,
               argMax((bid_prices[1]+ask_prices[1])/2, local_ts_ns) AS mid,
               argMax(bid_prices[1], local_ts_ns) AS bid,
               argMax(ask_prices[1], local_ts_ns) AS ask
        FROM caerus.order_book_states
        WHERE exchange='hyperliquid'
          AND local_ts_ns >= toUnixTimestamp({day:String})*1000000000
          AND local_ts_ns <  toUnixTimestamp({day:String})*1000000000 + 86400000000000
        GROUP BY ts)
SELECT ts, b.mid AS b_mid, b.bid AS b_bid, b.ask AS b_ask,
            h.mid AS h_mid, h.bid AS h_bid, h.ask AS h_ask,
       (b.mid - h.mid) / h.mid * 10000 AS gap_bps
FROM b INNER JOIN h USING (ts)
ORDER BY ts
```

Loop it per day and write Parquet — pull once, train many times. Never re-query
ClickHouse inside a training loop.

```python
import clickhouse_connect, polars as pl, os
from datetime import date, timedelta

client = clickhouse_connect.get_client(
    host="localhost", port=8123,
    username=os.environ["CH_USER"], password=os.environ["CH_PASS"])

SQL = open("docs/align_1s.sql").read()   # the query above

d, end = date(2026, 9, 3), date(2026, 10, 2)
while d <= end:
    df = pl.from_pandas(
        client.query_df(SQL, parameters={"day": d.isoformat()}))
    df.write_parquet(f"data/aligned/{d.isoformat()}.parquet")
    print(d, len(df))
    d += timedelta(days=1)
```

Expect ~86,200 rows per complete day (86,400 seconds minus brief dropouts).

### Reference values — verify your pull

Run on **2026-09-15**, 1-second alignment, `abs(gap_bps)`:

| Metric | Expected |
|---|---|
| aligned seconds | 86,192 |
| median gap | **0.956 bps** |
| p99 gap | 4.30 bps |
| max gap | 51.68 bps |
| % of seconds above 7 bps | **0.218%** |

That last row is the whole problem in one number. A round trip costs ~7 bps; roughly
**two seconds in a thousand** present a gap larger than that. The agent's job is to find
those and ignore everything else.

---

## 5. Environment conventions

Agree on these before anyone writes the env, or the baseline and the agent won't be
comparable.

- **Step rate: 1 Hz**, set by Hyperliquid's publish rate (§2).
- **Action delay: 150 ms.** An action chosen at `t` executes against the book at
  `t + 150ms`. Without this the agent trades on information it never had, and every
  result is worthless.
- **Actions:** `{hold flat, enter long, enter short, exit}` — discrete.
- **Fills: conservative.** Always cross the spread (buy at ask, sell at bid). Assume no
  queue priority and no maker fills. Optimistic fills are the most common way a backtest
  lies.
- **Fees:** parameterised, swept over `0.5 / 1 / 2 / 3.5` bps per side. Do not hardcode
  one tier — the headline result is the fee level at which the edge dies.
- **Reward:** realised PnL net of fees, spread and slippage. Position and loss limits
  enforced **by the environment**, not left to the policy.
- **Episodes:** contiguous slices (e.g. 1 hour = 3,600 steps). Never stitch across the
  08-30 → 09-02 hole.

---

## 6. Training PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = VecNormalize(DummyVecEnv([make_env("train")]),
                   norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", env,
            n_steps=4096,          # long rollouts: rewards are rare
            batch_size=256,
            gamma=0.999,           # positions are held for many steps
            ent_coef=0.01,         # keep exploring; the lazy optimum is "never trade"
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log="runs/")
model.learn(total_timesteps=5_000_000)
```

Why these settings, since you will be asked in the Q&A:

- `VecNormalize` — raw observations span basis points to dollar prices. Unnormalised
  inputs make the value function diverge.
- `gamma=0.999` — a trade spans many 1 Hz steps; at 0.99 the agent cannot see its own exit.
- `ent_coef` — **this is the main failure mode.** Doing nothing is a safe local optimum
  worth exactly zero, and it beats trading badly. Without entropy pressure PPO collapses
  to permanent abstention. Watch for it.
- `n_steps` high — with ~0.2% of steps actionable, short rollouts contain no signal at all.

### Diagnosing the collapse to inaction

Log **trade count per episode** alongside reward from the very first run. A flat reward
curve at exactly 0.0 with zero trades is not convergence, it is the degenerate policy.
Remedies, in the order to try them:

1. Raise `ent_coef` (0.01 → 0.05).
2. Curriculum: train first on the highest-dislocation days, then on everything.
3. Reward shaping — small positive credit for unrealised favourable movement while in a
   position, so the signal isn't only at exit.
4. Action masking: forbid `enter` when the gap is below a floor, shrinking the search space.

Only if all four fail should you fall back to DQN.

---

## 7. Evaluation

The baseline must run **inside the same environment** as the agent. A baseline evaluated
in a separate script with its own cost assumptions proves nothing.

Baseline: fixed threshold — enter when `abs(gap_bps) > X`, exit on convergence or timeout.
Sweep `X`, report the best-performing `X` so the comparison is honest.

Report for both, per fee level, on the untouched test window:

- net return after costs
- number of trades, and win rate
- max drawdown
- **selectivity**: trades taken ÷ opportunities above threshold

**Success = beating the tuned baseline under identical frictions.** Not absolute profit.
At retail fees the expected honest outcome is that neither is profitable, and the
deliverable is then the fee level at which the edge crosses zero. That is a legitimate
result; say so plainly rather than tuning until a number goes positive.

---

## 8. Gotchas, collected

1. Ports are loopback-only — the tunnel is mandatory.
2. Aug 31 and Sep 1 **do not exist**; Aug 30 and Sep 2 are partial.
3. ClickHouse dies above 1.5 GB — query per day.
4. Timestamps are UTC, not SGT.
5. Hyperliquid has 5 levels and updates ~1.6/s; Binance has 20 at ~8/s.
6. Use `local_ts_ns` for alignment, not `exchange_ts_ns`.
7. Never shuffle a chronological split.
8. PPO's failure mode here is learning to never trade. Log trade counts.
9. `stable-baselines3` is absent from `requirements.txt`.
10. Pull to Parquet once; don't query ClickHouse from a training loop.
