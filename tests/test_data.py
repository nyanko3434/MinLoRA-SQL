"""Tests for lora_sql.data: schema rendering, prompt formatting, and example loading.

render_schema is exercised directly against the real Spider tables.json shipped in
data/spider_data/ — the "perpetrator" db_id's rendered CREATE TABLE output has been
hand-verified as a golden oracle for the whole algorithm (PK/FK line placement, the
number/text/time/boolean/other type map, and identifier quoting for multi-word column
names like "Home Town"), so no synthetic fixture is needed for that test.
"""

import json
from pathlib import Path

from lora_sql.data import Example, format_prompt, load_examples, render_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_PATH = REPO_ROOT / "data" / "spider_data" / "tables.json"

GOLDEN_PERPETRATOR_SCHEMA = """CREATE TABLE perpetrator (
  Perpetrator_ID INTEGER,
  People_ID INTEGER,
  Date TEXT,
  Year INTEGER,
  Location TEXT,
  Country TEXT,
  Killed INTEGER,
  Injured INTEGER,
  PRIMARY KEY (Perpetrator_ID),
  FOREIGN KEY (People_ID) REFERENCES people(People_ID)
);

CREATE TABLE people (
  People_ID INTEGER,
  Name TEXT,
  Height INTEGER,
  Weight INTEGER,
  "Home Town" TEXT,
  PRIMARY KEY (People_ID)
);"""


def _load_table_entry(db_id: str) -> dict:
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return next(t for t in tables if t["db_id"] == db_id)


def test_render_schema_matches_golden_perpetrator_ddl():
    entry = _load_table_entry("perpetrator")
    assert render_schema(entry) == GOLDEN_PERPETRATOR_SCHEMA


def test_format_prompt_contains_schema_and_question_but_never_the_gold_query():
    example = Example(
        db_id="perpetrator",
        question="How many perpetrators are there?",
        query="SELECT count(*) FROM perpetrator",
        schema=GOLDEN_PERPETRATOR_SCHEMA,
    )
    prompt = format_prompt(example)

    assert GOLDEN_PERPETRATOR_SCHEMA in prompt
    assert example.question in prompt
    assert example.query not in prompt
    assert prompt.endswith("SQL: ")


def test_load_examples_maps_fields_and_shares_cached_schema_across_rows(tmp_path):
    # Real rows from data/spider_data/train_spider.json (db_id="perpetrator"), trimmed
    # to two, to exercise both field-mapping and the per-db_id schema cache (both rows
    # share db_id="perpetrator", so both must resolve to an identical, correctly
    # rendered schema without re-parsing tables.json per row).
    fixture_records = [
        {
            "db_id": "perpetrator",
            "question": "How many perpetrators are there?",
            "query": "SELECT count(*) FROM perpetrator",
        },
        {
            "db_id": "perpetrator",
            "question": "List the date of perpetrators in descending order of the number of people killed.",
            "query": "SELECT Date FROM perpetrator ORDER BY Killed DESC",
        },
    ]
    examples_path = tmp_path / "mini_eval.json"
    examples_path.write_text(json.dumps(fixture_records), encoding="utf-8")

    examples = load_examples(str(examples_path), tables_path=str(TABLES_PATH))

    assert len(examples) == 2
    for record, example in zip(fixture_records, examples):
        assert isinstance(example, Example)
        assert example.db_id == record["db_id"]
        assert example.question == record["question"]
        assert example.query == record["query"]
        assert example.schema == GOLDEN_PERPETRATOR_SCHEMA
    # same cached string object, not just equal content
    assert examples[0].schema is examples[1].schema


def test_load_examples_default_tables_path_resolves_sibling_tables_json(tmp_path):
    # Minimal single-table db, deliberately distinct from "perpetrator", to prove the
    # (path).parent / "tables.json" convention on its own — this is exactly what
    # eval/runner.py relies on by calling load_examples(cfg["data"]["eval_path"]) with
    # no second argument.
    tiny_tables = [
        {
            "db_id": "widgets",
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"],
            "table_names_original": ["widget"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
    ]
    tiny_records = [{"db_id": "widgets", "question": "How many widgets?", "query": "SELECT count(*) FROM widget"}]

    (tmp_path / "tables.json").write_text(json.dumps(tiny_tables), encoding="utf-8")
    examples_path = tmp_path / "eval.json"
    examples_path.write_text(json.dumps(tiny_records), encoding="utf-8")

    examples = load_examples(str(examples_path))  # no tables_path argument

    assert len(examples) == 1
    assert examples[0].db_id == "widgets"
    assert "CREATE TABLE widget (" in examples[0].schema
    assert "id INTEGER" in examples[0].schema
    assert "PRIMARY KEY (id)" in examples[0].schema
