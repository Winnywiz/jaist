"""
Central configuration for the benchmark.

Loads, in order of precedence:
    1. process environment variables,
    2. an optional ``config.yaml`` sitting next to this file,
    3. hard-coded defaults below.

It also locates the OpenAI key from the process env or the sibling
``RAG-DIVE/.env`` file (matching the convention used elsewhere in this repo), and
wires up paths to the JudgeAgent processed datasets so they can be reused.

Nothing here imports heavy dependencies, so importing :class:`Config` is cheap and
side-effect free apart from reading files.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Repo layout
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)  # D:\intershippu

#: JudgeAgent's processed datasets (corpus_chunks.jsonl, questions.json, graph.jsonl)
JUDGE_DATA_DIR = os.path.join(ROOT, "data")   # bundled local datasets (package)

#: RAG-DIVE/.env is the canonical secret store in this repo.
RAGDIVE_ENV = os.path.join(ROOT, ".env")   # package-root .env

#: Where the benchmark writes results / figures.
OUTPUT_DIR = os.path.join(ROOT, "result", "benchmark_quality")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

#: Optional YAML override file.
CONFIG_YAML = os.path.join(PACKAGE_DIR, "config.yaml")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent — safe to call repeatedly)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# .env / secret loading
# --------------------------------------------------------------------------- #
def _parse_env_file(path: str) -> Dict[str, str]:
    """Minimal ``.env`` parser (no python-dotenv dependency)."""
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val:
                env[key] = val
    return env


_ENV_FILE = _parse_env_file(RAGDIVE_ENV)


def openai_key() -> str:
    """OpenAI key from the process env first, then ``RAG-DIVE/.env``."""
    return os.environ.get("OPENAI_API_KEY", "") or _ENV_FILE.get("OPENAI_API_KEY", "")


# --------------------------------------------------------------------------- #
# Config object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Typed, immutable-ish runtime configuration for a benchmark run.

    Attributes:
        gen_model: OpenAI chat model used to *generate* questions / gold answers
            (and to power the mock RAG). Kept deliberately different from
            ``judge_model`` so the producer is not its own grader.
        judge_model: OpenAI chat model used for LLM-as-a-judge failure scoring.
        embed_model: OpenAI embedding model (used when sentence-transformers is
            unavailable).
        dataset: which dataset adapter to load (``multihoprag``, ``hotpotqa``,
            ``2wikimultihopqa``, ``musique`` or ``synthetic``).
        max_samples: cap on dataset rows loaded (None = all).
        num_conversations: how many conversations to generate.
        min_turns / max_turns: conversation length bounds (3–8 per the spec).
        seed: RNG seed for reproducibility.
        prefer_local_embeddings: try sentence-transformers before OpenAI.
        offline: force offline mode (no network) even if a key is present.
        extra: free-form dict for adapter-specific knobs.
    """

    gen_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-small"

    dataset: str = "synthetic"
    dataset_path: Optional[str] = None
    max_samples: Optional[int] = 20

    num_conversations: int = 5
    min_turns: int = 3
    max_turns: int = 6

    target_rag: str = "mock"  # one of: mock, vector, graph
    graph_mode: str = "typed"  # one of: typed, cooccur, none (graph ablation)
    seed: int = 42

    prefer_local_embeddings: bool = True
    offline: bool = False

    output_dir: str = OUTPUT_DIR
    figures_dir: str = FIGURES_DIR

    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, yaml_path: Optional[str] = None, **overrides: Any) -> "Config":
        """Build a Config from defaults <- config.yaml <- env <- kwargs."""
        data: Dict[str, Any] = {}

        # 1. YAML file (optional; only if PyYAML present)
        path = yaml_path or CONFIG_YAML
        if os.path.exists(path):
            try:
                import yaml  # type: ignore

                with open(path, "r", encoding="utf-8") as fr:
                    loaded = yaml.safe_load(fr) or {}
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception as exc:  # pragma: no cover - optional dep
                get_logger("config").warning("could not read %s: %s", path, exc)

        # 2. selected environment overrides
        env_map = {
            "gen_model": "GEN_MODEL",
            "judge_model": "JUDGE_MODEL",
            "embed_model": "EMBED_MODEL",
            "dataset": "BENCHMARK_DATASET",
            "seed": "BENCHMARK_SEED",
        }
        for field_name, env_name in env_map.items():
            if os.environ.get(env_name):
                data[field_name] = os.environ[env_name]

        # 3. explicit kwargs win
        data.update({k: v for k, v in overrides.items() if v is not None})

        # keep only known fields; stash the rest in ``extra``
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known}
        cfg = cls(**clean)
        cfg.extra.update(extra)
        return cfg

    # -- derived ------------------------------------------------------------ #
    @property
    def has_openai(self) -> bool:
        """True when an OpenAI key is available *and* we are not forced offline."""
        return (not self.offline) and bool(openai_key())

    def ensure_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
