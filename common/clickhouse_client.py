"""@file clickhouse_client.py
@brief Explicit, optional export from ClickHouse into local replay input files.
@details Importing this module does not import the database driver, inspect
credentials, instantiate a client, or issue queries. The offline pipeline never
calls this exporter. A user must invoke export_day() or this module's CLI.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import re
from typing import Any


class ClickHouseClient:
    """@brief Compatibility wrapper whose construction is entirely offline.
    @details Calling fetch_arrow/query_df is explicit database authorization at
    the API level. The actual client and environment lookup are deferred until
    that first query; offline package imports never connect anywhere.
    """

    def __init__(self) -> None:
        """@brief Initialize an empty handle without reading credentials."""
        self._client: Any = None

    def _connection(self) -> Any:
        """@brief Open a configured connection on the first explicit query."""
        if self._client is None:
            self._client = client_from_environment()
        return self._client

    def fetch_arrow(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """@brief Run an explicitly supplied read query and return an Arrow table."""
        return self._connection().query_arrow(query, parameters=parameters)

    def query_df(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """@brief Run an explicitly supplied read query and return a DataFrame."""
        return self._connection().query_df(query, parameters=parameters)

    def close(self) -> None:
        """@brief Close an existing connection without opening a new one."""
        if self._client is not None:
            self._client.close()
            self._client = None


def client_from_environment() -> Any:
    """@brief Create a client only when explicitly requested, using environment values.
    @details Required: CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD.
    Optional: CLICKHOUSE_PORT (8123), CLICKHOUSE_DATABASE (caerus),
    CLICKHOUSE_SECURE (false). Credential values are never logged.
    @throws ValueError If required connection settings are absent or invalid.
    """
    required = ("CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Missing explicit ClickHouse configuration: " + ", ".join(missing))
    secure_value = os.environ.get("CLICKHOUSE_SECURE", "false").lower()
    if secure_value not in {"true", "false", "1", "0"}:
        raise ValueError("CLICKHOUSE_SECURE must be true, false, 1, or 0")
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.environ.get("CLICKHOUSE_DATABASE", "caerus"),
        secure=secure_value in {"true", "1"},
    )


def export_day(day: str, output: str | Path, *, table: str = "order_book_states") -> Path:
    """@brief Export exactly one UTC day to Parquet after explicit user invocation.
    @param day ISO calendar date interpreted in UTC.
    @param output A new Parquet output file; existing files are protected.
    @param table Simple database table name, validated before SQL interpolation.
    @details The time filter uses exact epoch nanoseconds with bound parameters.
    No conversion through floating-point seconds or server-local dates occurs.
    """
    selected_day = date.fromisoformat(day)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("table must be a simple SQL identifier")
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    epoch = date(1970, 1, 1)
    start_ns = (selected_day - epoch).days * 86_400_000_000_000
    end_ns = (selected_day + timedelta(days=1) - epoch).days * 86_400_000_000_000
    target.parent.mkdir(parents=True, exist_ok=True)
    client = client_from_environment()
    #! @details SQL identifiers are validated above; dates remain bound values.
    #! The result contains source book arrays and exact Int64 receive timestamps.
    query = (
        "SELECT exchange, toInt64(exchange_ts_ns) AS exchange_ts_ns, "
        "toInt64(local_ts_ns) AS local_ts_ns, bid_prices, bid_sizes, ask_prices, ask_sizes "
        f"FROM {table} WHERE local_ts_ns >= {{start:Int64}} "
        "AND local_ts_ns < {end:Int64} ORDER BY local_ts_ns"
    )
    try:
        import pyarrow.parquet as parquet
        writer = None
        try:
            with client.query_arrow_stream(
                query, parameters={"start": start_ns, "end": end_ns}, use_strings=True
            ) as batches:
                for batch in batches:
                    if writer is None:
                        writer = parquet.ParquetWriter(target, batch.schema)
                    writer.write_batch(batch)
            if writer is None:
                raise ValueError("requested UTC day contains no records")
        finally:
            if writer is not None:
                writer.close()
    finally:
        client.close()
    return target


def main(argv: list[str] | None = None) -> None:
    """@brief Expose an explicitly invoked exporter separate from offline preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--table", default="order_book_states")
    args = parser.parse_args(argv)
    print(export_day(args.day, args.output, table=args.table))


if __name__ == "__main__":
    main()
