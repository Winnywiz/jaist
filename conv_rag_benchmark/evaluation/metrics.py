"""
Module 9 — evaluation metrics.

Aggregates per-turn diagnoses into the report numbers the project cares about:

  * **query-type accuracy** — accuracy broken down by the 8 query types,
  * **failure statistics** — % distribution over failure types and categories,
  * **difficulty curve** — accuracy by difficulty level (1..5),
  * **conversation-depth curve** — accuracy by turn number.

A turn is "successful" iff its :class:`Diagnosis` has ``success=True``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..config import get_logger
from .failure_taxonomy import category_of

logger = get_logger("evaluation.metrics")


class MetricsComputer:
    """Compute aggregate metrics from a flat list of evaluated turn records.

    Each record is expected to contain at least::

        {"query_type", "difficulty", "turn_id", "diagnosis": {...}}
    """

    def __init__(self, records: List[Dict]):
        self.records = records

    # -- public ------------------------------------------------------------- #
    def compute(self) -> Dict:
        total = len(self.records)
        success = sum(1 for r in self.records if r["diagnosis"]["success"])
        report = {
            "overall": {
                "n_turns": total,
                "accuracy": self._safe_div(success, total),
                "n_failures": total - success,
            },
            "query_type_accuracy": self.query_type_accuracy(),
            "failure_distribution": self.failure_distribution(),
            "failure_category_distribution": self.failure_category_distribution(),
            "difficulty_accuracy": self.difficulty_accuracy(),
            "turn_depth_accuracy": self.turn_depth_accuracy(),
        }
        return report

    def query_type_accuracy(self) -> Dict[str, Dict]:
        buckets: Dict[str, List[bool]] = defaultdict(list)
        for r in self.records:
            buckets[r.get("query_type", "Unknown")].append(r["diagnosis"]["success"])
        return {qt: {"accuracy": self._safe_div(sum(v), len(v)), "n": len(v)}
                for qt, v in sorted(buckets.items())}

    def failure_distribution(self) -> Dict[str, float]:
        """Percentage of *all turns* exhibiting each failure type (multi-label)."""
        counts: Dict[str, int] = defaultdict(int)
        total = len(self.records) or 1
        for r in self.records:
            for f in r["diagnosis"].get("failure_types", []):
                counts[f] += 1
        return {f: round(100.0 * c / total, 2)
                for f, c in sorted(counts.items(), key=lambda kv: -kv[1])}

    def failure_category_distribution(self) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        total = len(self.records) or 1
        for r in self.records:
            cats = {category_of(f) for f in r["diagnosis"].get("failure_types", [])}
            for c in cats:
                counts[c] += 1
        return {c: round(100.0 * n / total, 2)
                for c, n in sorted(counts.items(), key=lambda kv: -kv[1])}

    def difficulty_accuracy(self) -> Dict[int, Dict]:
        buckets: Dict[int, List[bool]] = defaultdict(list)
        for r in self.records:
            buckets[int(r.get("difficulty", 0))].append(r["diagnosis"]["success"])
        return {d: {"accuracy": self._safe_div(sum(v), len(v)), "n": len(v)}
                for d, v in sorted(buckets.items())}

    def turn_depth_accuracy(self) -> Dict[int, Dict]:
        buckets: Dict[int, List[bool]] = defaultdict(list)
        for r in self.records:
            buckets[int(r.get("turn_id", 0))].append(r["diagnosis"]["success"])
        return {t: {"accuracy": self._safe_div(sum(v), len(v)), "n": len(v)}
                for t, v in sorted(buckets.items())}

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _safe_div(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0
