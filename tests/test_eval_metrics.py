"""Spider difficulty (vendored eval_hardness) and its threading through the runner. CPU only: the
runner test stubs out the model, so it is safe while training holds the GPU."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from lora_sql.eval import runner
from lora_sql.eval.metrics import DIFFICULTIES, bootstrap_ci, eval_difficulties, holm, mcnemar_p, record_difficulty

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "spider_data" / "dev.json"


def test_dev_split_matches_official_counts():
    # The split the official Spider evaluation.py reports for dev.
    assert Counter(eval_difficulties(DEV)) == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}


@pytest.mark.parametrize("idx, expected", [
    (0, "easy"),     # SELECT count(*) FROM singer
    (8, "easy"),     # SELECT DISTINCT country FROM singer WHERE age > 20
    (2, "medium"),   # 3 select columns + ORDER BY
    (12, "hard"),    # WHERE age > (SELECT avg(age) ...): one nested query
    (26, "hard"),    # GROUP BY + ORDER BY count(*) + LIMIT: three components
    (24, "extra"),   # JOIN + WHERE + GROUP BY + ORDER BY + LIMIT
    (41, "extra"),   # JOIN + WHERE ... INTERSECT ...
])
def test_known_dev_examples(idx, expected):
    assert eval_difficulties(DEV)[idx] == expected


def test_record_without_sql_dict_is_unknown(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps([{"db_id": "x", "question": "q", "query": "SELECT 1"}]))
    assert eval_difficulties(path) == ["unknown"]


def test_run_eval_threads_difficulty(tmp_path, monkeypatch):
    """The runner attaches each example's difficulty by index: in results, the .jsonl and the
    summary's by_difficulty. The model is stubbed; generation returns the gold query."""
    cfg = runner.load_config(ROOT / "configs" / "week1_parity.yaml")
    for key in ("eval_path", "db_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    monkeypatch.setattr(runner, "load_model_for_eval", lambda cfg, ckpt=None: (SimpleNamespace(device="cpu"), None))
    monkeypatch.setattr(runner, "generate_sql", lambda model, tok, example, max_new, device: example.query)

    n = 30
    out = tmp_path / "eval.json"
    summary = runner.run_eval(cfg, out=str(out), limit=n)

    expected = eval_difficulties(DEV)[:n]
    assert set(expected) == set(DIFFICULTIES)  # the first 30 cover all four tiers
    report = json.loads(out.read_text())
    assert [r["difficulty"] for r in report["results"]] == expected
    rows = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    assert [r["difficulty"] for r in rows] == expected
    assert summary["by_difficulty"] == {
        d: {"accuracy": 1.0, "total": c, "correct": c} for d, c in Counter(expected).items()
    }


def test_bird_record_uses_its_own_difficulty_label():
    assert record_difficulty({"SQL": "SELECT 1", "difficulty": "challenging"}) == "challenging"


BIRD_CONFIG = ROOT / "configs" / "eval" / "coder7b_bird_dev.yaml"
BIRD_DEV = ROOT / "data" / "bird" / "dev_20240627" / "dev.json"


@pytest.mark.skipif(not BIRD_DEV.exists(), reason="BIRD dev not downloaded")
def test_run_eval_bird_format(tmp_path, monkeypatch):
    """The real BIRD config end to end with the model stubbed: the BIRD loader is used, the
    evidence hint reaches the prompt's question, golds run on dev_databases/, and BIRD's own
    difficulty labels are threaded through."""
    cfg = runner.load_config(BIRD_CONFIG)
    for key in ("eval_path", "tables_path", "db_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    seen = []
    monkeypatch.setattr(runner, "load_model_for_eval", lambda cfg, ckpt=None: (SimpleNamespace(device="cpu"), None))
    monkeypatch.setattr(runner, "generate_sql",
                        lambda model, tok, example, max_new, device: seen.append(example) or example.query)

    n = 20
    out = tmp_path / "bird.json"
    summary = runner.run_eval(cfg, out=str(out), limit=n)

    records = json.loads(BIRD_DEV.read_text())[:n]
    assert [e.query for e in seen] == [r["SQL"] for r in records]
    assert all(("\nHint: " in e.question) == bool(r["evidence"].strip()) for e, r in zip(seen, records))
    report = json.loads(out.read_text())
    assert [r["difficulty"] for r in report["results"]] == [r["difficulty"] for r in records]
    assert summary["overall_accuracy"] == 1.0
    assert report["meta"]["data"]["format"] == "bird"


def test_mcnemar_exact_values():
    assert mcnemar_p(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 9) == mcnemar_p(9, 3)  # two-sided: symmetric


def test_holm_adjusts_in_input_order_and_is_monotone():
    raw = [0.04, 0.01, 0.03]
    adj = holm(raw)
    assert adj == pytest.approx([0.06, 0.03, 0.06])  # 0.01*3, max(0.03*2, 0.03), max(0.04*1, 0.06)
    order = sorted(range(3), key=raw.__getitem__)
    assert [adj[i] for i in order] == sorted(adj)
    assert all(a >= r for a, r in zip(adj, raw))


def test_bootstrap_ci_brackets_observed_difference():
    import random
    rng = random.Random(0)
    a = [i % 4 != 0 for i in range(400)]   # 75% right
    b = [i % 2 == 0 for i in range(400)]   # 50% right
    resamples = [[rng.randrange(400) for _ in range(400)] for _ in range(1000)]
    lo, hi = bootstrap_ci(a, b, resamples)
    assert lo < 25.0 < hi and lo > 0
