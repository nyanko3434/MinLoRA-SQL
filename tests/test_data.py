"""Tests for lora_sql.data: schema rendering, prompt formatting, and example loading.

render_schema is exercised directly against the real Spider tables.json shipped in
data/spider_data/ — the "perpetrator" db_id's rendered CREATE TABLE output has been
hand-verified as a golden oracle for the whole algorithm (PK/FK line placement, the
number/text/time/boolean/other type map, and identifier quoting for multi-word column
names like "Home Town"), so no synthetic fixture is needed for that test.
"""

import json
from pathlib import Path

import pytest
from conftest import requires_spider
from transformers import AutoTokenizer

from lora_sql.data import (
    SYSTEM_PROMPT,
    Example,
    build_train_example,
    encode_prompt,
    format_prompt,
    load_bird_examples,
    load_examples,
    render_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_PATH = REPO_ROOT / "data" / "spider_data" / "tables.json"
CODER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
IM_END_ID = 151645

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


@requires_spider
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


@requires_spider
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


# A BIRD-style tables.json entry: BIRD type names, and a composite primary key given as a nested
# list (debit_card_specializing, formula_1, ... in the real BIRD dev).
BIRD_TABLES = [
    {
        "db_id": "shop",
        "column_names_original": [[-1, "*"], [0, "CustomerID"], [0, "Joined"], [1, "CustomerID"],
                                  [1, "Date"], [1, "Amount"], [1, "Paid At"]],
        "column_types": ["text", "integer", "date", "integer", "date", "real", "datetime"],
        "table_names_original": ["customers", "yearmonth"],
        "primary_keys": [1, [3, 4]],
        "foreign_keys": [[3, 1]],
    }
]
GOLDEN_BIRD_SHOP_SCHEMA = """CREATE TABLE customers (
  CustomerID INTEGER,
  Joined DATE,
  PRIMARY KEY (CustomerID)
);

CREATE TABLE yearmonth (
  CustomerID INTEGER,
  Date DATE,
  Amount REAL,
  "Paid At" DATETIME,
  PRIMARY KEY (CustomerID, Date),
  FOREIGN KEY (CustomerID) REFERENCES customers(CustomerID)
);"""


def test_render_schema_bird_types_and_composite_primary_key():
    assert render_schema(BIRD_TABLES[0]) == GOLDEN_BIRD_SHOP_SCHEMA


def test_load_bird_examples_evidence_and_sql_key(tmp_path):
    records = [
        {"db_id": "shop", "question": "How many customers?", "evidence": "customers refers to CustomerID",
         "SQL": "SELECT COUNT(CustomerID) FROM customers", "difficulty": "simple"},
        {"db_id": "shop", "question": "Total amount?", "evidence": "",
         "SQL": "SELECT SUM(Amount) FROM yearmonth", "difficulty": "moderate"},
    ]
    (tmp_path / "dev_tables.json").write_text(json.dumps(BIRD_TABLES), encoding="utf-8")
    (tmp_path / "dev.json").write_text(json.dumps(records), encoding="utf-8")

    with_hint = load_bird_examples(str(tmp_path / "dev.json"), str(tmp_path / "dev_tables.json"))
    assert [e.query for e in with_hint] == [r["SQL"] for r in records]
    assert with_hint[0].question == "How many customers?\nHint: customers refers to CustomerID"
    assert with_hint[1].question == "Total amount?"  # empty evidence adds no "Hint:" line
    assert with_hint[0].schema == GOLDEN_BIRD_SHOP_SCHEMA
    assert with_hint[0].schema is with_hint[1].schema

    no_hint = load_bird_examples(str(tmp_path / "dev.json"), str(tmp_path / "dev_tables.json"), use_evidence=False)
    assert [e.question for e in no_hint] == [r["question"] for r in records]
    # The hint reaches the prompt through the question; the template itself is unchanged.
    assert "Hint: customers refers to CustomerID" in format_prompt(with_hint[0])


BIRD_DEV = REPO_ROOT / "data" / "bird" / "dev_20240627"


@pytest.mark.skipif(not (BIRD_DEV / "dev.json").exists(), reason="BIRD dev not downloaded")
def test_real_bird_dev_loads():
    examples = load_bird_examples(str(BIRD_DEV / "dev.json"), str(BIRD_DEV / "dev_tables.json"))
    assert len(examples) == 1534
    assert len({e.db_id for e in examples}) == 11
    assert sum("\nHint: " in e.question for e in examples) == 1534 - 148  # 148 records have no evidence


@pytest.fixture(scope="module")
def coder_tokenizer():
    # local_files_only is the in-code equivalent of HF_HUB_OFFLINE=1 (which can't be set
    # reliably here: conftest has already imported transformers).
    try:
        return AutoTokenizer.from_pretrained(CODER_MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{CODER_MODEL} tokenizer is not in the local Hugging Face cache")


def _perpetrator_example() -> Example:
    return Example(
        db_id="perpetrator",
        question="How many perpetrators are there?",
        query="SELECT count(*) FROM perpetrator",
        schema=GOLDEN_PERPETRATOR_SCHEMA,
    )


def _department_management_example() -> Example:
    # Real train_spider row; its irregular double spaces exercise the byte-exact round-trip.
    return Example(
        db_id="department_management",
        question="How many heads of the departments are older than 56 ?",
        query="SELECT count(*) FROM head WHERE age  >  56",
        schema=render_schema(_load_table_entry("department_management")),
    )


def test_encode_prompt_renders_chatml_with_system_prompt(coder_tokenizer):
    example = _perpetrator_example()
    ids = encode_prompt(coder_tokenizer, example)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)

    text = coder_tokenizer.decode(ids)
    assert text.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    assert text.count(SYSTEM_PROMPT) == 1
    assert format_prompt(example) in text
    assert text.endswith("<|im_start|>assistant\n")
    assert "You are Qwen" not in text, "the template's default system turn leaked in"


@requires_spider
@pytest.mark.parametrize(
    "make_example",
    [_perpetrator_example, _department_management_example],
    ids=["perpetrator", "double-space"],
)
def test_build_train_example_layout(coder_tokenizer, make_example):
    example = make_example()
    ids, n_completion = build_train_example(coder_tokenizer, example)
    prompt_ids = encode_prompt(coder_tokenizer, example)
    sql_ids = coder_tokenizer(example.query, add_special_tokens=False).input_ids

    assert ids[: len(prompt_ids)] == prompt_ids
    assert ids[-1] == IM_END_ID
    assert n_completion == len(sql_ids) + 1
    assert len(ids) == len(prompt_ids) + n_completion
    assert coder_tokenizer.decode(ids[-n_completion:-1]) == example.query


def test_pad_and_eos_are_distinct(coder_tokenizer):
    assert coder_tokenizer.pad_token_id != coder_tokenizer.eos_token_id
