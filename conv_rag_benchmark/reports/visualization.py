"""
Module 10 — matplotlib visualisations.

Renders the four charts the spec asks for and saves them under
``output/figures/``:

  1. failure distribution (bar, coloured by category),
  2. query-type performance (bar),
  3. difficulty performance curve (line),
  4. turn-depth performance curve (line).

Uses a non-interactive backend so it runs headless (CI / servers).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from ..config import get_logger
from ..evaluation.failure_taxonomy import category_of

logger = get_logger("reports.viz")

_CATEGORY_COLORS = {
    "Retrieval": "#4C72B0",
    "Conversation": "#DD8452",
    "Reasoning": "#55A868",
    "Generation": "#C44E52",
    "Unknown": "#999999",
}


class Visualizer:
    """Render benchmark metrics to PNG charts."""

    def __init__(self, figures_dir: str):
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)

    # -- public ------------------------------------------------------------- #
    def render_all(self, metrics: Dict) -> List[str]:
        """Render every chart; returns the list of written file paths."""
        paths = []
        try:
            paths.append(self.failure_distribution(metrics.get("failure_distribution", {})))
            paths.append(self.query_type_performance(metrics.get("query_type_accuracy", {})))
            paths.append(self.difficulty_curve(metrics.get("difficulty_accuracy", {})))
            paths.append(self.turn_depth_curve(metrics.get("turn_depth_accuracy", {})))
        except Exception as exc:  # pragma: no cover - never let plotting kill a run
            logger.warning("visualisation error: %s", exc)
        return [p for p in paths if p]

    def failure_distribution(self, dist: Dict[str, float]) -> Optional[str]:
        if not dist:
            return None
        labels = list(dist.keys())
        values = [dist[k] for k in labels]
        colors = [_CATEGORY_COLORS.get(category_of(k), "#999999") for k in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("% of turns exhibiting failure")
        ax.set_title("Failure Type Distribution")
        self._legend_by_category(ax)
        return self._save(fig, "failure_distribution.png")

    def query_type_performance(self, qt_acc: Dict[str, Dict]) -> Optional[str]:
        if not qt_acc:
            return None
        labels = list(qt_acc.keys())
        values = [qt_acc[k]["accuracy"] for k in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color="#4C72B0")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy by Query Type")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for b, v, n in zip(bars, values, [qt_acc[k]["n"] for k in labels]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8)
        return self._save(fig, "query_type_performance.png")

    def difficulty_curve(self, diff_acc: Dict) -> Optional[str]:
        if not diff_acc:
            return None
        xs = sorted(int(k) for k in diff_acc)
        ys = [diff_acc[x]["accuracy"] if x in diff_acc else diff_acc[str(x)]["accuracy"]
              for x in xs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o", color="#55A868", linewidth=2)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("difficulty (1=simple .. 5=multi-hop)")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy by Difficulty")
        ax.set_xticks(xs)
        return self._save(fig, "difficulty_performance.png")

    def turn_depth_curve(self, turn_acc: Dict) -> Optional[str]:
        if not turn_acc:
            return None
        xs = sorted(int(k) for k in turn_acc)
        ys = [turn_acc[x]["accuracy"] if x in turn_acc else turn_acc[str(x)]["accuracy"]
              for x in xs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="s", color="#DD8452", linewidth=2)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("turn number (conversation depth)")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy by Conversation Depth")
        ax.set_xticks(xs)
        return self._save(fig, "turn_depth_performance.png")

    # -- helpers ------------------------------------------------------------ #
    def _legend_by_category(self, ax) -> None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c)
                   for c in _CATEGORY_COLORS.values()]
        ax.legend(handles, list(_CATEGORY_COLORS.keys()), title="Category",
                  fontsize=8, loc="lower right")

    def _save(self, fig, name: str) -> str:
        path = os.path.join(self.figures_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("wrote %s", path)
        return path
