import clickhouse_connect

client = clickhouse_connect.get_client(
    host="localhost", port=8123,
    username="caerus", password="<from /data/caerus/deploy/.env on the box>",
    database="caerus",
)

# Pull one day, both venues, time-ordered — the replay stream for a backtest.
df = client.query_df("""
    SELECT exchange, exchange_ts_ns, local_ts_ns,
           bid_prices, bid_sizes, ask_prices, ask_sizes
    FROM order_book_states
    WHERE toDate(fromUnixTimestamp64Nano(toInt64(local_ts_ns))) = '2026-08-27'
    ORDER BY local_ts_ns
""")

# Arrays come back as numpy arrays per row; best bid/ask:
df["bid"] = df["bid_prices"].str[0]
df["ask"] = df["ask_prices"].str[0]
df["mid"] = (df["bid"] + df["ask"]) / 2

df.to_parquet("btc_20260827.parquet")  # export once, backtest from parquet