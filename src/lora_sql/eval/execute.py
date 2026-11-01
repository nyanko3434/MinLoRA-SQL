"""Run SQL against a SQLite database and compare result sets for execution accuracy.

Generated SQL is untrusted: databases are opened read-only (a generated DROP / DELETE raises
instead of corrupting the eval set) and every query runs under a wall-clock deadline (sqlite's own
`timeout` is only a lock wait, so a generated cross join could otherwise hang the eval forever).
"""

import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

DEFAULT_TIMEOUT_S = 30.0
_PROGRESS_EVERY = 10_000  # sqlite VM instructions between deadline checks
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


class ExecutionError(Exception):
    pass


def run_query(db_path: str | Path, query: str, timeout: float = DEFAULT_TIMEOUT_S) -> list[tuple]:
    """Executes `query` read-only against the SQLite db at `db_path`, returns fetched rows.

    Raises ExecutionError on any sqlite3 failure (syntax error, missing table, write attempt,
    deadline exceeded, missing db file, etc.) so callers can treat a broken generation as a
    scoring signal, not a crash.
    """
    deadline = time.monotonic() + timeout
    try:
        # mode=ro also means a missing file raises instead of silently creating an empty db.
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise ExecutionError(str(e)) from e
    try:
        conn.text_factory = lambda b: b.decode(errors="replace")
        # A truthy return aborts the statement with OperationalError("interrupted").
        conn.set_progress_handler(lambda: time.monotonic() > deadline, _PROGRESS_EVERY)
        return conn.execute(query).fetchall()
    except (sqlite3.Error, sqlite3.Warning, ValueError) as e:
        raise ExecutionError(str(e)) from e
    finally:
        conn.close()


def order_matters(gold_sql: str) -> bool:
    """Row order is only compared when the gold query sorts (test-suite-sql-eval's convention)."""
    return bool(_ORDER_BY.search(gold_sql))


def _normalize_row(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def results_match(predicted: list[tuple], gold: list[tuple], ordered: bool = False) -> bool:
    """Spider-style exec match. Unordered: multiset compare (duplicate-sensitive, and NULL-safe
    because rows are hashed, never sorted). Ordered: exact row sequence. Columns are compared
    positionally (no column-permutation search, unlike the official test-suite eval)."""
    pred_rows = [_normalize_row(r) for r in predicted]
    gold_rows = [_normalize_row(r) for r in gold]
    if ordered:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def execute_and_compare(
    db_path: str | Path,
    predicted_sql: str,
    gold_rows: list[tuple],
    ordered: bool,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[bool, str | None]:
    """(correct, execution error of the predicted query or None). Gold rows are precomputed by
    the caller. A predicted query that fails to execute scores as wrong rather than propagating."""
    try:
        predicted_rows = run_query(db_path, predicted_sql, timeout)
    except ExecutionError as e:
        return False, str(e)
    return results_match(predicted_rows, gold_rows, ordered), None
