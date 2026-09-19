"""Small DuckDB helpers shared by VirAssQ table scans."""

from pathlib import Path

import duckdb


def quote_sql_string(value: Path | str) -> str:
    """Quote a path or other literal for use in a DuckDB SQL expression."""

    return "'" + str(value).replace("'", "''") + "'"


def parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    """Return Parquet column names without scanning its rows."""

    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_sql_string(path)})"
    ).fetchall()
    return {str(row[0]) for row in rows}
