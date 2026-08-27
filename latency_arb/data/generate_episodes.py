from __future__ import annotations

from pathlib import Path
from typing import Any


class EpisodeGenerator:
    """Simple placeholder for slicing ClickHouse data into episodes."""

    def __init__(self, client: Any, output_dir: str | Path | None = None) -> None:
        self.client = client
        self.output_dir = Path(output_dir or "./data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, query: str, limit: int = 1000) -> Path:
        table_path = self.output_dir / "episodes.parquet"
        data = self.client.fetch_arrow(query + f" LIMIT {limit}")
        data.to_pandas().to_parquet(table_path, index=False)
        return table_path
