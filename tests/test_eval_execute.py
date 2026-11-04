"""Execution + comparison + held-out guard + gold preflight. CPU-only (tiny sqlite DBs in
tmp_path, no model), so these are safe to run while a training run holds the GPU."""

import sqlite3
import time
from pathlib import Path

import pytest

from lora_sql.data import Example
from lora_sql.eval import execute
from lora_sql.eval.execute import ExecutionError, execute_and_compare, order_matters, results_match, run_query
from lora_sql.eval.runner import check_held_out, gold_preflight, load_config

ROOT = Path(__file__).resolve().parents[1]


def _make_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT, score REAL);"
        "INSERT INTO t VALUES (1, 'a', 1.5), (2, NULL, NULL), (3, 'c', 2.0000001), (3, 'c', 2.0000001);"
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def db(tmp_path):
    return _make_db(tmp_path / "t.sqlite")


def _row_count(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT count(*) FROM t").fetchone()[0]
    finally:
        conn.close()


def test_null_rows_compare_without_crashing(db):
    rows = run_query(db, "SELECT name, score FROM t")
    assert (None, None) in rows
    # sorted() would raise TypeError comparing None with str/float; the multiset compare can't.
    assert results_match(rows, list(reversed(rows)))
    assert not results_match(rows, [r for r in rows if r != (None, None)])


def test_duplicates_matter():
    assert not results_match([(1,), (1,)], [(1,)])
    assert results_match([(1,), (2,), (1,)], [(1,), (1,), (2,)])


def test_row_order_only_matters_when_gold_sorts():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert order_matters("SELECT name FROM t ORDER  BY id DESC")
    assert order_matters("select name from t order by id")
    assert not order_matters("SELECT recorder FROM border_bytes")


def test_float_rounding():
    assert results_match([(2.0000001,)], [(2.0000002,)])
    assert not results_match([(2.0,)], [(2.1,)])


@pytest.mark.parametrize("sql", ["DELETE FROM t", "DROP TABLE t", "INSERT INTO t VALUES (9, 'z', 0.0)",
                                 "UPDATE t SET name = 'x'"])
def test_writes_are_rejected_and_db_untouched(db, sql):
    with pytest.raises(ExecutionError):
        run_query(db, sql)
    assert _row_count(db) == 4


def test_missing_db_raises_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope" / "nope.sqlite"
    with pytest.raises(ExecutionError):
        run_query(missing, "SELECT 1")
    assert not missing.exists()


def test_runaway_query_is_interrupted(db):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    t0 = time.monotonic()
    with pytest.raises(ExecutionError, match="interrupted"):
        run_query(db, endless, timeout=0.5)
    assert time.monotonic() - t0 < 5


def test_connection_closed_after_failed_query(db, monkeypatch):
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(execute.sqlite3, "connect", tracking_connect)
    with pytest.raises(ExecutionError):
        run_query(db, "SELECT nope FROM t")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_execute_and_compare_reports_exec_errors(db):
    gold = run_query(db, "SELECT id FROM t WHERE id = 1")
    assert execute_and_compare(db, "SELECT id FROM t WHERE id = 1", gold, ordered=False) == (True, None)
    assert execute_and_compare(db, "SELECT id FROM t WHERE id = 2", gold, ordered=False) == (False, None)
    correct, error = execute_and_compare(db, "SELEC id FROM t", gold, ordered=False)
    assert not correct and "syntax error" in error


def _cfg(eval_path="data/spider_data/dev.json", db_dir="data/spider_data/database"):
    return {"data": {"eval_path": eval_path, "db_dir": db_dir}}


def test_held_out_guard():
    check_held_out(_cfg())
    with pytest.raises(ValueError, match="held out"):
        check_held_out(_cfg(eval_path="data/spider_data/test.json"))
    with pytest.raises(ValueError, match="database"):
        check_held_out(_cfg(db_dir="data/spider_data/test_database"))


def _bird_cfg(eval_path="data/bird/dev_20240627/dev.json", db_dir="data/bird/dev_20240627/dev_databases"):
    return {"data": {"format": "bird", "eval_path": eval_path, "db_dir": db_dir}}


def test_held_out_guard_bird():
    check_held_out(_bird_cfg())
    with pytest.raises(ValueError, match="held out"):
        check_held_out(_bird_cfg(eval_path="data/bird/test.json"))
    # Each format runs only against its own database directory.
    with pytest.raises(ValueError, match="dev_databases"):
        check_held_out(_bird_cfg(db_dir="data/spider_data/database"))
    with pytest.raises(ValueError, match="database"):
        check_held_out({"data": {"format": "spider", "eval_path": "data/spider_data/dev.json",
                                 "db_dir": "data/bird/dev_20240627/dev_databases"}})
    with pytest.raises(ValueError, match="data.format"):
        check_held_out({"data": {"format": "wikisql", "eval_path": "dev.json", "db_dir": "database"}})


@pytest.mark.parametrize("path", ["configs/week1_parity.yaml", "configs/eval/coder7b_spider_dev.yaml",
                                  "configs/eval/coder7b_bird_dev.yaml"])
def test_eval_configs_pass_guard(path):
    cfg = load_config(ROOT / path)
    check_held_out(cfg)
    assert set(cfg["model"]) == {"name_or_path", "quant", "double_quant", "dtype"}


def test_gold_preflight(tmp_path):
    db_dir = tmp_path / "database"
    _make_db(db_dir / "toy" / "toy.sqlite")
    good = Example(db_id="toy", question="ids?", query="SELECT id FROM t", schema="")
    assert gold_preflight([good], db_dir) == [[(1,), (2,), (3,), (3,)]]

    broken = Example(db_id="toy", question="broken?", query="SELECT nope FROM t", schema="")
    with pytest.raises(RuntimeError, match=r"(?s)1 gold queries failed.*\[toy\] 'broken\?'"):
        gold_preflight([good, broken], db_dir)
