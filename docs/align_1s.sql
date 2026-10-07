-- 1-second cross-venue alignment for one day.
-- Parameter: {day:String} e.g. '2026-09-15'
-- Resamples each venue to 1s buckets (last observation wins), then inner-joins on the second.
-- Day-scoped deliberately: the server is capped at 1.5 GB and full-history scans are killed.
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
