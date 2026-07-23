"""
End-to-end benchmark orchestrator.

Wires every module together::

    Dataset -> Graph -> Retriever -> Conversations (typed turns + grounded gold)
            -> Target RAG answers -> Failure classification -> Metrics -> Report/Charts

Produces three artefacts in ``output/``:
  * ``results.json``           — full per-turn records (conversation, question,
                                 query_type, gold, rag_answer, failure_types, ...),
  * ``summary.csv``            — one row per turn for spreadsheet analysis,
  * ``evaluation_report.json`` — aggregate metrics,
plus the four PNG charts under ``output/figures/``.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from .config import Config, get_logger
from .datasets.loader import DatasetLoader, Sample
from .embeddings import Embedder
from .evaluation.failure_classifier import FailureClassifier
from .evaluation.metrics import MetricsComputer
from .generation.conversation_generator import ConversationGenerator
from .graph.graph_builder import GraphBuilder
from .graph.retriever import GraphRetriever
from .interfaces.rag_interface import build_rag
from .llm import LLM
from .reports.visualization import Visualizer

logger = get_logger("pipeline")


class BenchmarkPipeline:
    """Runs the full conversational-RAG failure benchmark."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.config.ensure_dirs()
        # one shared generation LLM (questions/gold/mock-RAG); judge has its own.
        self.llm = LLM(model=self.config.gen_model, config=self.config)
        self.embedder = Embedder(config=self.config, llm=self.llm)
        logger.info("pipeline | openai=%s | gen=%s judge=%s | dataset=%s | rag=%s",
                    self.config.has_openai, self.config.gen_model,
                    self.config.judge_model, self.config.dataset, self.config.target_rag)

    # -- stages ------------------------------------------------------------- #
    def run(self) -> Dict:
        samples = self._load_data()
        kg = self._build_graph(samples)
        retriever = GraphRetriever(kg, config=self.config, llm=self.llm,
                                   embedder=self.embedder)
        conversations = self._generate_conversations(kg, retriever, samples)
        target = build_rag(self.config.target_rag, kg.chunks, self.config,
                           llm=self.llm, retriever=retriever, embedder=self.embedder)
        records = self._evaluate(conversations, target)
        report = MetricsComputer(records).compute()
        report["meta"] = {
            "dataset": self.config.dataset,
            "target_rag": self.config.target_rag,
            "n_conversations": len(conversations),
            "n_turns": len(records),
            "graph": kg.stats(),
            "openai": self.config.has_openai,
            "llm_cost": self.llm.cost_report(),
        }
        self._write_outputs(conversations, records, report)
        self._render_charts(report)
        logger.info("done | accuracy=%.3f | %s",
                    report["overall"]["accuracy"], self.llm.cost_report())
        return report

    def _load_data(self) -> List[Sample]:
        return DatasetLoader(self.config.dataset, path=self.config.dataset_path,
                             max_samples=self.config.max_samples).load()

    def _build_graph(self, samples: List[Sample]):
        chunks: List[str] = []
        for s in samples:
            chunks.extend(s.context)
        if not chunks:  # degenerate dataset: use the questions themselves
            chunks = [s.question for s in samples]
        gb = GraphBuilder(config=self.config, llm=self.llm, embedder=self.embedder)
        return gb.build(chunks)

    def _generate_conversations(self, kg, retriever, samples: List[Sample]):
        gen = ConversationGenerator(kg, config=self.config, llm=self.llm,
                                    retriever=retriever)
        seeds = samples[: self.config.num_conversations]
        convs = gen.generate_many(seeds)
        n_turns = sum(len(c.turns) for c in convs)
        logger.info("generated %d conversations (%d turns total)", len(convs), n_turns)
        return convs

    def _evaluate(self, conversations, target) -> List[Dict]:
        classifier = FailureClassifier(config=self.config)
        records: List[Dict] = []
        for conv in conversations:
            target.reset()
            for i, turn in enumerate(conv.turns):
                history = conv.history_upto(turn["turn_id"])
                resp = target.answer(turn["question"], history=history)
                gold = turn["gold"]
                diag = classifier.classify({
                    "question": turn["question"],
                    "query_type": turn["query_type"],
                    "expected_failure": turn.get("expected_failure"),
                    "gold_answer": gold["gold_answer"],
                    "gold_evidence": " ".join(turn["evidence"].get("chunks", []))[:1200],
                    "retrieved_evidence": resp.retrieved_context,
                    "rag_answer": resp.answer,
                    "is_unanswerable": turn.get("is_unanswerable", False),
                    "history": history,
                })
                records.append({
                    "conversation_id": conv.conversation_id,
                    "turn_id": turn["turn_id"],
                    "query_type": turn["query_type"],
                    "difficulty": turn["difficulty"],
                    "capability": turn["capability"],
                    "expected_failure": turn.get("expected_failure"),
                    "question": turn["question"],
                    "gold_answer": gold["gold_answer"],
                    "reasoning_chain": gold.get("reasoning_chain", []),
                    "supporting_nodes": gold.get("supporting_nodes", []),
                    "supporting_edges": gold.get("supporting_edges", []),
                    "rag_answer": resp.answer,
                    "retrieved_context": resp.retrieved_context,
                    "diagnosis": diag.to_dict(),
                })
        logger.info("evaluated %d turns against '%s' RAG", len(records), target.name)
        return records

    # -- outputs ------------------------------------------------------------ #
    def _write_outputs(self, conversations, records, report) -> None:
        out = self.config.output_dir
        results_path = os.path.join(out, "results.json")
        with open(results_path, "w", encoding="utf-8") as fw:
            json.dump({
                "conversations": [c.to_dict() for c in conversations],
                "turns": records,
            }, fw, ensure_ascii=False, indent=2)
        logger.info("wrote %s", results_path)

        report_path = os.path.join(out, "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as fw:
            json.dump(report, fw, ensure_ascii=False, indent=2)
        logger.info("wrote %s", report_path)

        csv_path = os.path.join(out, "summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fw:
            writer = csv.writer(fw)
            writer.writerow(["conversation_id", "turn_id", "query_type", "difficulty",
                             "success", "failure_types", "confidence",
                             "question", "gold_answer", "rag_answer"])
            for r in records:
                d = r["diagnosis"]
                writer.writerow([
                    r["conversation_id"], r["turn_id"], r["query_type"], r["difficulty"],
                    d["success"], "|".join(d["failure_types"]), d["confidence"],
                    r["question"], r["gold_answer"], r["rag_answer"],
                ])
        logger.info("wrote %s", csv_path)

    def _render_charts(self, report) -> None:
        Visualizer(self.config.figures_dir).render_all(report)
