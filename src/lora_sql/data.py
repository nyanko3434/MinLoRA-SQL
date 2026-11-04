import json
import re
from dataclasses import dataclass
from pathlib import Path

# Spider types (number/text/time/boolean/others) plus the BIRD dev type names. Unknown -> TEXT.
_TYPE_MAP = {"number": "INTEGER", "text": "TEXT", "time": "TIME", "boolean": "BOOLEAN",
             "integer": "INTEGER", "real": "REAL", "date": "DATE", "datetime": "DATETIME"}
_UNSAFE_IDENTIFIER = re.compile(r"\W")

PROMPT_TEMPLATE = """You are a SQLite expert. Given the database schema below, write a single SQL query that answers the question.

{schema}

Question: {question}
SQL: """

# Must not be a substring of PROMPT_TEMPLATE, so the rendered prompt carries it exactly once.
SYSTEM_PROMPT = "You are a SQLite expert; respond with only the SQL query, with no explanation or formatting."


@dataclass
class Example:
    db_id: str
    question: str
    query: str
    schema: str


def _quote_identifier(name: str) -> str:
    return f'"{name}"' if _UNSAFE_IDENTIFIER.search(name) else name


def render_schema(table_entry: dict) -> str:
    """CREATE TABLE DDL text from one tables.json entry. Public (not underscored) so
    tests can call it directly against real tables.json data for a golden-string check."""
    column_names = table_entry["column_names_original"]
    column_types = table_entry["column_types"]
    table_names = table_entry["table_names_original"]
    # BIRD lists a composite key as a nested list ([[19, 20]]); flattened, its columns land in the
    # owning table's single PRIMARY KEY (...) line. Spider's keys are flat ints, so it is unchanged.
    primary_keys = {c for pk in table_entry["primary_keys"] for c in (pk if isinstance(pk, list) else [pk])}
    foreign_keys = table_entry["foreign_keys"]

    blocks = []
    for table_idx, table_name in enumerate(table_names):
        lines, pk_cols = [], []
        for i in range(1, len(column_names)):  # skip index 0, the [-1, "*"] sentinel
            owner_idx, col_name = column_names[i]
            if owner_idx != table_idx:
                continue
            sql_type = _TYPE_MAP.get(column_types[i], "TEXT")
            lines.append(f"  {_quote_identifier(col_name)} {sql_type}")
            if i in primary_keys:
                pk_cols.append(col_name)

        if pk_cols:
            lines.append(f"  PRIMARY KEY ({', '.join(_quote_identifier(c) for c in pk_cols)})")

        for from_idx, to_idx in foreign_keys:
            if column_names[from_idx][0] != table_idx:
                continue
            from_col = column_names[from_idx][1]
            to_table_idx, to_col = column_names[to_idx]
            to_table = table_names[to_table_idx]
            lines.append(
                f"  FOREIGN KEY ({_quote_identifier(from_col)}) "
                f"REFERENCES {_quote_identifier(to_table)}({_quote_identifier(to_col)})"
            )

        body = ",\n".join(lines)
        blocks.append(f"CREATE TABLE {_quote_identifier(table_name)} (\n{body}\n);")

    return "\n\n".join(blocks)


def format_prompt(example: Example) -> str:
    return PROMPT_TEMPLATE.format(schema=example.schema, question=example.question)


def encode_prompt(tokenizer, example: Example) -> list[int]:
    """ChatML prompt ids ending at the assistant header. The ONLY place prompt ids are built:
    training and eval both call this, so the model never sees two different prompt formats.

    The explicit system turn matters: without one, Qwen's template injects its own default
    ("You are Qwen, created by Alibaba Cloud...").
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(example)},
    ]
    # transformers v5 returns a BatchEncoding here, not a bare list of ids.
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True
    )
    return list(enc["input_ids"])


def build_train_example(tokenizer, example: Example) -> tuple[list[int], int]:
    """Training ids = prompt ids + SQL ids + <|im_end|>, and n_completion, the length of the
    trailing slice that carries loss (the SQL plus <|im_end|>)."""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id == tokenizer.eos_token_id, (
        f"<|im_end|> ({im_end_id}) is not the tokenizer's eos ({tokenizer.eos_token_id}); "
        "generation would not stop where training taught it to"
    )
    # Concatenate at the ID level; never re-tokenize a joined prompt+SQL string, which can
    # merge tokens across the boundary.
    completion_ids = tokenizer(example.query, add_special_tokens=False).input_ids + [im_end_id]
    return encode_prompt(tokenizer, example) + completion_ids, len(completion_ids)


def load_examples(path: str, tables_path: str | None = None) -> list[Example]:
    path = Path(path)
    resolved_tables_path = Path(tables_path) if tables_path is not None else path.parent / "tables.json"

    with open(resolved_tables_path, "r", encoding="utf-8") as f:
        tables_by_db_id = {t["db_id"]: t for t in json.load(f)}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    schema_cache: dict[str, str] = {}  # local to this call, keyed per db_id — many
    # examples share a db_id, and render_schema does non-trivial nested-loop work.
    examples = []
    for record in records:
        db_id = record["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = render_schema(tables_by_db_id[db_id])
        examples.append(Example(db_id=db_id, question=record["question"],
                                 query=record["query"], schema=schema_cache[db_id]))
    return examples


def bird_question(record: dict, use_evidence: bool = True) -> str:
    """BIRD's question text, with the expert evidence hint appended when present (the standard
    BIRD setting). It goes in Example.question, so the prompt template itself is unchanged."""
    evidence = (record.get("evidence") or "").strip()
    if use_evidence and evidence:
        return f"{record['question']}\nHint: {evidence}"
    return record["question"]


def load_bird_examples(path: str, tables_path: str, use_evidence: bool = True) -> list[Example]:
    """BIRD dev records (gold SQL under "SQL", schemas in dev_tables.json) as Examples, in file
    order. Zero-shot eval only: BIRD is never a training file."""
    with open(tables_path, "r", encoding="utf-8") as f:
        tables_by_db_id = {t["db_id"]: t for t in json.load(f)}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    schema_cache: dict[str, str] = {}
    examples = []
    for record in records:
        db_id = record["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = render_schema(tables_by_db_id[db_id])
        examples.append(Example(db_id=db_id, question=bird_question(record, use_evidence),
                                 query=record["SQL"], schema=schema_cache[db_id]))
    return examples
