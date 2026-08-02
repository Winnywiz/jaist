"""
Thin, offline-safe OpenAI chat/embedding wrapper shared across the benchmark.

Design goals (mirroring the repo's existing ``newmethod/llm.py``):
  * one client, one usage counter (so a run can print its cost),
  * never crash when there is no API key — callers branch on ``.available`` and
    fall back to deterministic heuristics,
  * a strict JSON helper, a plain-text helper, and an embedding helper.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

import numpy as np

from .config import Config, get_logger, openai_key

logger = get_logger("llm")

#: USD per 1M tokens (input, output) — best-effort, for the cost report only.
_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


class LLM:
    """Wrapper around the OpenAI chat + embedding APIs.

    Args:
        model: chat model id. Defaults to the config's ``gen_model``.
        config: the active :class:`Config` (used for the key + offline flag).
    """

    def __init__(self, model: Optional[str] = None, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.model = model or self.config.gen_model
        self.key = openai_key()
        self.available = self.config.has_openai
        self._client = None
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        if self.available:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.key)
            except Exception as exc:  # pragma: no cover - import/version issues
                logger.warning("OpenAI client unavailable (%s); running offline", exc)
                self.available = False

    # -- chat --------------------------------------------------------------- #
    def chat_json(self, system: str, user: str, temperature: float = 0.3,
                  retries: int = 2) -> Optional[dict]:
        """Return a parsed JSON object, or ``None`` when offline / on failure."""
        if not self.available:
            return None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                self._track(resp)
                return json.loads(resp.choices[0].message.content)
            except Exception as exc:
                logger.warning("chat_json attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
        return None

    def chat_text(self, system: str, user: str, temperature: float = 0.0) -> Optional[str]:
        """Return raw assistant text, or ``None`` when offline / on failure."""
        if not self.available:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            self._track(resp)
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning("chat_text failed: %s", exc)
            return None

    # -- embeddings --------------------------------------------------------- #
    def embed(self, texts: List[str], model: Optional[str] = None) -> Optional[np.ndarray]:
        """Return L2-normalised embeddings as a float32 array, or ``None``."""
        if not self.available:
            return None
        model = model or self.config.embed_model
        try:
            resp = self._client.embeddings.create(model=model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.clip(norms, 1e-8, None)
        except Exception as exc:
            logger.warning("embed failed: %s", exc)
            return None

    # -- bookkeeping -------------------------------------------------------- #
    def _track(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage:
            self.calls += 1
            self.in_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.out_tokens += getattr(usage, "completion_tokens", 0) or 0

    def cost_report(self) -> str:
        pin, pout = _PRICING.get(self.model, (None, None))
        if pin is None:
            return f"{self.calls} calls (no pricing for {self.model})"
        cost = self.in_tokens / 1e6 * pin + self.out_tokens / 1e6 * pout
        return (f"{self.calls} calls | in {self.in_tokens:,} out {self.out_tokens:,} "
                f"| ~${cost:.4f}")
