"""Aggregate execution accuracy, optionally broken down by difficulty."""

from collections import defaultdict
from dataclasses import dataclass, field


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
