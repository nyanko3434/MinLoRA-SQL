"""Aggregate execution accuracy, broken down by Spider difficulty (easy / medium / hard / extra)."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard", "extra")


# ---------------------------------------------------------------------------------------------
# Vendored from taoyds/spider, evaluation.py, at commit b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c
# https://github.com/taoyds/spider/blob/b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c/evaluation.py
# Copyright 2024 XLANG NLP Lab. Licensed under the Apache License, Version 2.0; a copy of the
# license is in licenses/spider-LICENSE at the repo root.
#
# Modifications: only the constants and functions eval_hardness needs are extracted, and
# eval_hardness is a module-level function instead of an Evaluator method (`self` dropped).
# Everything else is verbatim, including upstream quirks (count_agg over where[::2] reads not_op,
# and over `having` it sees the 'and'/'or' strings). Do not "fix" them: the official dev split
# (248 / 446 / 174 / 166) depends on them. Input is the parsed `sql` dict that ships with every
# Spider dev/train record, so process_sql.py is not needed.
# ---------------------------------------------------------------------------------------------

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"

# --------------------------------- end of vendored Spider code ---------------------------------


def record_difficulty(record: dict) -> str:
    """Spider: the official tier from the parsed `sql` dict. BIRD: its own `difficulty` label
    (simple / moderate / challenging). Anything else: "unknown"."""
    if "sql" in record:
        return eval_hardness(record["sql"])
    return record.get("difficulty") or "unknown"


def eval_difficulties(path) -> list[str]:
    """Difficulty for every record of an eval JSON, in file order (the same order the data
    loaders return)."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [record_difficulty(r) for r in records]


@dataclass
class EvalResult:
    db_id: str
    question: str
    predicted_sql: str
    gold_sql: str
    correct: bool
    difficulty: str = "unknown"


@dataclass
class MetricsSummary:
    overall_accuracy: float
    total: int
    correct: int
    by_difficulty: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_accuracy": self.overall_accuracy,
            "total": self.total,
            "correct": self.correct,
            "by_difficulty": self.by_difficulty,
        }


def compute_metrics(results: list[EvalResult]) -> MetricsSummary:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    overall = correct / total if total else 0.0

    buckets: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        buckets[r.difficulty].append(r)

    by_difficulty = {
        difficulty: {
            "accuracy": sum(1 for r in bucket if r.correct) / len(bucket),
            "total": len(bucket),
            "correct": sum(1 for r in bucket if r.correct),
        }
        for difficulty, bucket in buckets.items()
    }

    return MetricsSummary(
        overall_accuracy=overall,
        total=total,
        correct=correct,
        by_difficulty=by_difficulty,
    )


# ---- Paired significance over the same questions (used by scripts/summarize_*.py) ----

def mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (binomial, p = 0.5)."""
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=pvals.__getitem__)
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(a: list[bool], b: list[bool], resamples: list[list[int]]) -> tuple[float, float]:
    """95% paired bootstrap CI, in percentage points, of mean(a) - mean(b) over shared index resamples."""
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    stats = sorted(sum(diff[i] for i in s) / n for s in resamples)
    return 100 * stats[int(0.025 * len(stats))], 100 * stats[int(0.975 * len(stats)) - 1]
