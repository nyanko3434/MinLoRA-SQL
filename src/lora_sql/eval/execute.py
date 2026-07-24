"""Run SQL against a SQLite database and compare result sets for execution accuracy."""

import sqlite3
from pathlib import Path


class ExecutionError(Exception):
    pass


def run_query(db_path: str | Path, query: str, timeout: float = 5.0) -> list[tuple]:
    """Executes `query` against the SQLite db at `db_path`, returns fetched rows.

    Raises ExecutionError on any sqlite3 failure (syntax error, missing table, etc.)
    so callers can treat a broken generation as a scoring signal, not a crash.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="replace")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except sqlite3.Error as e:
        raise ExecutionError(str(e)) from e


def _normalize_row(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def results_match(predicted: list[tuple], gold: list[tuple]) -> bool:
    """Order-insensitive comparison of two result sets (Spider-style exec match)."""
    pred_set = sorted(_normalize_row(r) for r in predicted)
    gold_set = sorted(_normalize_row(r) for r in gold)
    return pred_set == gold_set


def execute_and_compare(db_path: str | Path, predicted_sql: str, gold_sql: str) -> bool:
    """Returns True iff predicted_sql executes and matches gold_sql's result set.

    Any execution failure on the predicted query counts as a mismatch rather
    than propagating, since a malformed generation should just score as wrong.
    """
    try:
        predicted_rows = run_query(db_path, predicted_sql)
    except ExecutionError:
        return False
    gold_rows = run_query(db_path, gold_sql)
    return results_match(predicted_rows, gold_rows)
