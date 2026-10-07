# Sub-second edge analysis (training days only)

Script: `subsecond_edge.py`; numbers: `subsecond_edge.json`; figures: `edge_vs_latency.png`,
`timing.png`. Six training days from the raw per-update cache: 4, 8, 12, 15, 19 and
23 September 2026 (UTC). Validation and test days were not read.

Definitions: a signal at time t with |gap - basis| >= X (basis = causal 60 s EMA of the
one-second gap) enters a Hyperliquid (HL) position in the signal direction, crossing the
spread, and exits 3 s later crossing it again. Net = gross - 2f - 0.2 bps, where f is the
fee per side and 0.1 bps slippage is charged per fill. Means are trade-weighted.

## Feed timing

Receive time minus exchange timestamp, so any clock offset is included:

- Binance: p10/p50/p90 = 73.7 / 75.8 / 78.9 ms, never below 72 ms.
- HL: 303.8 / 338.3 / 429.1 ms.
- HL publishes every 456 / 540 / 634 ms (p10/p50/p90; p99 878 ms), about 1.84 snapshots per second.
- Binance publishes about every 102 ms (8.6 to 9.5 snapshots per second). Its mid price changes on 0.7 to 6.1% of snapshots.
- The HL book held at a whole second is 624 ms old in HL's clock at the median (p90 847 ms).

## Edge against fill latency (one-second decision grid, received-book fills)

| X | Signals/day | L=0 | 150 | 300 | 500 | 1000 ms |
|---|---:|---:|---:|---:|---:|---:|
| 3 bps | 663 | 3.86 | 3.41 | 2.98 | 2.48 | 1.02 |
| 5 bps | 146 | 6.31 | 5.66 | 4.96 | 4.20 | 2.05 |
| 8 bps | 30 | 9.71 | 8.73 | 7.91 | 6.86 | 4.04 |

Values are mean gross bps per trade. At X=3 the edge is 0.37 bps at L=1500 ms, a loss of
about 0.28 bps per 100 ms of delay.

- X=3, f=1: net positive on 6/6 days up to L=300 ms, and on no day from L=800 ms.
- X=3, f=2: not positive on any day at any latency.
- X=5, f=2: positive on 6/6 days up to L=250 ms.

## Event-driven decisions against the one-second grid

L = 150 ms, with a 3 s debounce so trades do not overlap.

| X | Trades/day: grid / every Binance update | Gross bps: grid / event | Net at f=3.5: grid / event |
|---|---:|---:|---:|
| 3 | 502 / 743 | 3.39 / 3.23 | n/a |
| 5 | 110 / 180 | 5.42 / 5.11 | -1.78 / -2.09 |
| 8 | 21 / 42 | 8.59 / 8.30 | +1.39 / +1.10 |

- The grid never sees 28.5% (X=3), 36.0% (X=5) and 47.6% (X=8) of dislocation onsets within 10 s.
- A dislocation lasts 618 / 510 / 306 ms at the median.
- When the grid does catch an onset, it adds 452 ms of delay at the median, and it misses the next-second check on about 40% of onsets.

## The fill convention overstates the edge

The project simulator fills at the latest HL book received by t+L. That book reflects
HL's state about 340 ms earlier. If the order instead fills at HL's own book as of t+L
(HL clock, assuming our clock matches HL's), the X=3 grid edge falls from 3.39 to
2.04 bps. If it fills at the next HL snapshot (pessimistic), the edge is 0.63 bps.

With HL-clock fills, the X=3/5/8 grid gives 2.04 / 2.76 / 4.22 bps, against
2.64 / 3.94 / 5.77 bps for event-driven decisions. Comparing the same onsets, waiting
for the grid costs 1.0 to 1.9 bps per trade under HL-clock fills. Under received-book
fills the same cost appears as only 0.06 bps.

Daily net at f=1, X=3 with HL-clock fills, summed over trades:

- grid: -78 bps
- event-driven: +330 bps
- event-driven with a faster HL view: +538 bps

## What faster data could recover

At X=3 with HL-clock fills, the grid earns 2.04 bps per trade against an ideal of
3.36 bps, so 1.32 bps per trade is lost.

| Change | X=3 | X=5 (2.68 lost) | X=8 (4.33 lost) |
|---|---|---|---|
| Decide on every Binance update | +0.60 (45%) | +1.19 (44%) | +1.55 (36%) |
| Binance bookTicker, if it sees moves 50 ms sooner (at most +0.16, 12%, at X=3) | +0.08 (6%) | +0.16 (6%) | +0.24 (6%) |
| A faster HL view | +0.45 (34%); also removes the 19% of signals HL had already closed | +0.96 (36%) | +1.74 (40%) |

With pessimistic fills the three shares are 41%, 12 to 25% and 7%. Under received-book
fills there is no per-trade edge left to recover.

## What this data cannot show

- HL's true book when an order arrives, because HL snapshots are about 540 ms apart.
- Whether the capture host's clock (WSL2) matches the exchanges' clocks.
- What HL's exchange timestamp actually marks.
- Binance moves inside one 100 ms batch.
- How a higher-cadence HL feed would behave.
- Queue position, order size, market impact, or competition from other fast traders.

Event-driven decisions were evaluated only at Binance update times. The six days differ
about tenfold in signal count.
