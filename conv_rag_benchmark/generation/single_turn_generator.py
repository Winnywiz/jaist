"""
The *single-turn* QA generator (the control method).

Where the Dynamic and Xie methods build a MULTI-turn conversation, this control asks
the seed question **and nothing else** — one grounded factual turn, no follow-ups.

It is deliberately not a separate algorithm: it is the adaptive generator
(:class:`AdaptiveConversationGenerator`) run for exactly ONE turn, i.e. turn 0, the
seed. Sharing the machinery is the point — everything except the *number of turns* is
held identical to the multi-turn methods, so a comparison isolates what the follow-up
strategy adds. When single-turn and Dynamic/Xie differ, the difference is the
follow-ups, not the seed authoring, retrieval, gold, or grading (all shared here).

Kept in its own file so the four methods map one-to-one onto four generator files:
    dynamic_generator.py      -> Dynamic (proposed)
    xie_generator.py          -> Xie sub-question decomposition
    single_turn_generator.py  -> Single-turn QA (this file)
    mtrag_generator.py        -> mtRAG human-conversation replay
"""
from __future__ import annotations

from .dynamic_generator import AdaptiveConversationGenerator, AdaptiveConversation


class SingleTurnQAGenerator(AdaptiveConversationGenerator):
    """Single-turn QA control: the grounded seed question only, no follow-ups.

    Inherits the full adaptive pipeline (seed grounding, retrieval, gold authoring,
    grading, the two document sets) and simply forces the conversation to one turn, so
    the loop produces only turn 0 (the seed) and stops. The static/dynamic content
    policy is irrelevant here — with no follow-up turns, no follow-up is ever
    conditioned on any history — but it is left untouched for construction symmetry.
    """

    def generate(self, seed, conversation_id: str,
                 n_turns: int = 1) -> AdaptiveConversation:
        # A single-turn conversation is the seed turn ONLY. Ignore any larger n_turns
        # the orchestrator passes, so this method can never accidentally grow follow-ups.
        return super().generate(seed, conversation_id, n_turns=1)
