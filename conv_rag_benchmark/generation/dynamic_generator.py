"""
The *adaptive* conversation generator (the proposed method).

Where a static generator pre-plans the whole question-type playlist up front and
treats its own gold answers as the conversation history, this generator **closes the
loop with the system under test**:

    ask a question -> read the RAG's REAL answer -> grade it ->
    let that result pick the NEXT question's type -> repeat.

This is the JudgeAgent idea applied to the conversational benchmark: the probe
chases the weaknesses the RAG actually reveals, instead of replaying a fixed script.

Design notes (so the benchmark stays valid):
  * We keep TWO histories.
      - ``truth_history``  : user question + *gold* answer.  Used to retrieve and to
        generate the next question + gold, so the benchmark stays grounded in the
        corpus, not in the RAG's (possibly wrong) replies.
      - ``rag_history``    : user question + the RAG's *real* answer.  This is what
        the RAG sees on later turns, and what the controller reacts to.
  * Only the question *type* is chosen adaptively (by :meth:`_next_type`); the
    question text and gold are still authored from the corpus truth.  So the questions
    stay well-formed — the adaptive part is only *which* questions get asked, and that
    they are scored against a live system.

The output of one conversation is a list of turns, each carrying both the gold and
the RAG's answer + the per-turn outcome, so you can report two things at once:
  1. benchmark quality (well_formed / gold_supported / gold_correct), and
  2. the RAG's failure profile (the diagnostic payoff of going adaptive).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import Config, get_logger
from ..graph.graph_builder import KnowledgeGraph
from ..graph.retriever import GraphRetriever
from ..interfaces.rag_interface import RAGInterface
from ..llm import LLM
from .gold_answer_generator import GoldAnswerGenerator, ABSTENTION
from .query_generator import QUERY_TYPES, QueryGenerator, QueryTurn


# ===================================================================================
#  QUALITY GUARDS
#
#  Each guard answers one yes/no question about a freshly generated turn: "is this
#  answer bad in a specific, known way?" If a guard says yes, the turn is thrown away
#  and the generator is asked to try again with an explanation of what went wrong.
#
#  Every guard is a plain function with no side effects, so each can be read, tested
#  and reasoned about on its own.
# ===================================================================================

#: Words too common to carry meaning. Removing them leaves only the words that
#: actually say something, which is what every guard below compares on.
COMMON_WORDS = set((
    "a an the of to in on at for and or but with without is are was were be been being "
    "this that these those it its their there here as by from into than then so if can "
    "could will would should may might must have has had do does did what which who whom "
    "whose how when where why not no yes about over under out up down you we they he she "
    "per each any some many much more most less"
).split())

#: A smaller subset of COMMON_WORDS: only articles, prepositions and forms of "to be" —
#: pure grammar, nothing that could ever be the point of a sentence. Guards that judge
#: SHORT answers use this instead, because COMMON_WORDS also drops words like "not",
#: "can", "did" and "more", which carry real meaning when an answer is only a few words
#: long ("They can use SMOTE [9]" is an answer; "was introduced in [7]" is not).
GRAMMAR_WORDS = set((
    "a an the of to in on at for and or but with is are was were be been "
    "this that these those it its their there here as by from into"
).split())


def meaningful_words(text, filler=COMMON_WORDS):
    """Split text into its meaningful words: lowercase, no punctuation, no filler.

    Words of 1-2 characters are dropped too, so "of a" disappears but "CO2" stays.
    Example: "What is the average degree?" -> ["average", "degree"]

    Pass a different `filler` set to be less aggressive about what counts as
    meaningless; see GRAMMAR_WORDS.
    """
    all_words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [word for word in all_words
            if word not in filler and len(word) > 2]


def overlap_ratio(words_a, words_b):
    """How much two sets of words overlap, from 0.0 (nothing shared) to 1.0 (identical).

    Shared words divided by total distinct words (the Jaccard ratio).
    Example: {cat, dog} vs {cat, bird} -> 1 shared / 3 total = 0.33
    """
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def gold_repeats_earlier_answer(gold, conversation_so_far, similarity_limit=0.8):
    """Is this answer just something we already said earlier in the conversation?

    Catches the stall where the benchmark asks slightly different questions that all
    land on the same fact, e.g. the answer "great wealth, power, and influence" coming
    back three turns in a row. Such a turn adds no new information.

    An answer counts as a repeat when it is identical to an earlier one, is fully
    contained inside an earlier one, or shares almost all of its words with one.
    Note the deliberate asymmetry: a LONGER answer that contains an earlier short one
    is fine, because it is adding detail rather than repeating.
    """
    new_answer = (gold or "").strip().lower()
    if not new_answer or new_answer.startswith(ABSTENTION[:20].lower()):
        return False
    new_words = set(re.findall(r"[a-z0-9]+", new_answer))
    if not new_words:
        return False

    for message in conversation_so_far:
        if message.get("role") != "assistant":
            continue
        earlier_answer = (message.get("content") or "").strip().lower()
        if not earlier_answer:
            continue
        if new_answer == earlier_answer or new_answer in earlier_answer:
            return True                                  # says nothing beyond a past answer
        earlier_words = set(re.findall(r"[a-z0-9]+", earlier_answer))
        if overlap_ratio(new_words, earlier_words) >= similarity_limit:
            return True                                  # different wording, same content
    return False


#: Matches a citation marker such as "[7]" or "[ 12 ]".
CITATION_MARKER = re.compile(r"\[\s*\d+\s*\]")


def gold_is_only_a_citation(gold):
    """Is this answer nothing but a pointer to a reference, with no real content?

    Some source papers contain sentences like "fmcn was acquainted in [7]". An answer
    copied from one of these tells you nothing you can check, because these corpora
    include the "[7]" marker but NOT the reference list it points to, so "[7]" can
    never be resolved into an actual paper.

    Only answers that BOTH contain a citation marker AND say almost nothing once that
    marker is removed are rejected. An answer that cites a source alongside a real
    fact is kept. Because the answers this guard sees are short, it strips only
    GRAMMAR_WORDS — the wider COMMON_WORDS list would eat words like "not" or "can"
    and push genuine short answers under the threshold.
    """
    answer = (gold or "").strip().lower()
    if not answer or not CITATION_MARKER.search(answer):
        return False                                     # no citation -> not our problem
    without_citation = CITATION_MARKER.sub("", answer)
    remaining = set(meaningful_words(without_citation, GRAMMAR_WORDS))
    return len(remaining) < 3                            # nothing of substance is left


#: Matches leftover LaTeX cross-references such as "TABREF1" or "SECREF27". Papers
#: converted from LaTeX keep these codes, but "Table TABREF1" points at nothing a
#: reader can look up, so any question or answer containing one is malformed.
LATEX_LABEL = re.compile(r"\b(?:TABREF|SECREF|FIGREF|BIBREF|EQREF)\d*\b", re.I)


def has_latex_label(text):
    """Does this text contain a leftover LaTeX reference code like "TABREF1"?"""
    return bool(LATEX_LABEL.search(text or ""))


def gold_only_restates_question(question, gold, required_new_words=2):
    """Does this answer just repeat the question back without answering it?

    A turn tests nothing if the answer contains only words the question already used,
    e.g. asking "What do secondary users aim to optimise?" and answering "secondary
    users aim to optimise ...". The answer has to introduce something new.

    Short answers are always accepted, because a real answer is often just a value:
    "10000", "Fubo" and "1.4 millions" are correct even though they add almost no new
    wording. Only longer answers, which should be explaining something, are checked.
    """
    answer_words = meaningful_words(gold)
    if len(answer_words) < 5:
        return False                                     # a short value answer is fine
    question_words = set(meaningful_words(question))
    added_words = {word for word in set(answer_words)
                   if word not in question_words and not word.isdigit()}
    return len(added_words) < required_new_words

logger = get_logger("generation.adaptive")

#: Outcome labels the per-turn grader emits for the RAG's answer.
OUTCOMES = ("correct", "wrong", "hallucinated", "abstained")

#: When the RAG keeps getting things right, escalate difficulty through this ramp.
_ESCALATION = ["Multi-Hop", "Comparative", "Ambiguous Reference"]

_GRADE_SYS = (
    "You grade a RAG system's ANSWER against the GOLD answer for one question. "
    "Choose exactly one label:\n"
    "  'correct'      - the answer matches the gold answer.\n"
    "  'abstained'    - the answer declines / says it does not know / cannot answer.\n"
    "  'hallucinated' - the gold says to abstain (unanswerable), but the answer "
    "confidently invents a specific reply.\n"
    "  'wrong'        - the answer is a confident but factually incorrect reply to "
    "an answerable question.\n"
    'Respond JSON: {"label": "<one of the four>", "correct": true/false}'
)


@dataclass
class AdaptiveTurn:
    """One probe turn: the asked question, the gold, the RAG's reply, the outcome."""

    turn_id: int
    query_type: str
    question: str
    gold: str
    evidence: str
    rag_answer: str
    outcome: str
    is_unanswerable: bool = False
    # GROUNDING GUARD gave up: all 3 attempts failed to produce a verified gold,
    # so this turn's answer key is untrustworthy — exclude it from per-type grading.
    guard_gave_up: bool = False
    # the retrieved text the QUESTION was authored from (the gold may be composed
    # against a different, question-specific re-retrieval stored in `evidence`)
    question_evidence: str = ""
    # the docs the RAG-under-test ACTUALLY retrieved to produce rag_answer (its own
    # retrieval, independent of what the question/gold were authored from)
    rag_retrieved_context: List[str] = field(default_factory=list)
    # COUNTERFACTUAL PROBE (Mode-1 interventional half): the SAME question re-answered
    # with the CLEAN gold history instead of the RAG's own (possibly poisoned) history.
    # Lets a failure be split into INHERITED (flips to correct here -> the conversation
    # caused it) vs INTRINSIC (still fails -> retrieval/generation would have failed
    # anyway). None = the probe was not run for this turn (turn 0, or --counterfactual off).
    cf_outcome: Optional[str] = None
    cf_rag_answer: str = ""
    # GENERATION PROVENANCE (thesis evidence): records what the follow-up generator was
    # allowed to condition on, so a run can PROVE the dynamic condition actually observed
    # the RAG's previous answer. Populated per non-seed turn. Keys: content_mode,
    # previous_gold_answer, previous_rag_answer, history_given_to_generator.
    provenance: Dict = field(default_factory=dict)
    # DIAGNOSTIC SLOTS (Comparative / Multi-Hop only): the expected answer components a
    # complete answer must cover, so the failure can be attributed by RULE (slot coverage)
    # instead of only by the LLM classifier. Empty for other types.
    expected_components: List = field(default_factory=list)
    # TWO SEPARATE DOCUMENT SETS (kept distinct on purpose — never merge them):
    #   question_generation_documents — the evidence the QUESTION GENERATOR retrieved and
    #     authored/verified the gold from (each: doc_id/rank/score/score_type/text).
    #   rag_retrieved_documents       — what the RAG-under-test retrieved to ANSWER this
    #     question (its own retrieval; same record shape).
    # Comparing the two lets the later failure stage separate a retrieval miss (gold
    # chunk available to the generator but absent from the RAG's retrieval) from a
    # generation miss (chunk in both, answer still wrong). See run_benchmark chain.
    question_generation_documents: List[Dict] = field(default_factory=list)
    rag_retrieved_documents: List[Dict] = field(default_factory=list)
    # PROVENANCE of the follow-up's CONTENT conditioning (the thesis IV, human-readable):
    #   "rag_history"   — dynamic: the generator saw the RAG's actual previous answer.
    #   "truth_history" — static: the generator saw only gold/truth history (no RAG answer).
    #   "seed"          — turn 0, taken from the dataset (not generated from either history).
    generation_source: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class AdaptiveConversation:
    conversation_id: str
    seed_question: str
    turns: List[AdaptiveTurn] = field(default_factory=list)
    type_sequence: List[str] = field(default_factory=list)
    outcome_sequence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "seed_question": self.seed_question,
            "type_sequence": self.type_sequence,
            "outcome_sequence": self.outcome_sequence,
            "turns": [t.to_dict() for t in self.turns],
        }


class AdaptiveConversationGenerator:
    """The proposed method: drive the conversation from the RAG-under-test's real answers."""

    def __init__(self, kg: KnowledgeGraph, target_rag: RAGInterface,
                 judge: LLM, config: Optional[Config] = None,
                 gen_llm: Optional[LLM] = None,
                 retriever: Optional[GraphRetriever] = None,
                 type_policy: str = "controller",
                 content_policy: str = "static",
                 forced_types: Optional[List[str]] = None,
                 quality_gate: bool = False,
                 strict_gold: bool = False,
                 counterfactual: bool = False,
                 local_extractor=None,
                 skip_seed_grounding: bool = False,
                 inject_unanswerable_at: Optional[int] = None):
        # skip_seed_grounding (opt-in): use seed.question verbatim as the seed — do not
        # widen/rewrite it. Set when the orchestrator pre-grounds ONE shared seed and
        # feeds the SAME seed to every method (so the only difference is the follow-up
        # strategy, not the starting question). Default False = original behaviour.
        self.skip_seed_grounding = skip_seed_grounding
        # inject_unanswerable_at (opt-in): force ONE follow-up turn (this turn_id) to be an
        # Unanswerable probe, while every other turn stays adaptive. Lets the proposed
        # method deliberately test the knowledge-boundary / hallucination axis without a
        # prior hallucination (the controller only reaches Unanswerable reactively). None =
        # off (original behaviour).
        self.inject_unanswerable_at = inject_unanswerable_at
        #: 'textpairs' arm: a GraphBuilder used ONLY to extract entities/relations from
        #: the retrieved chunks at generation time (no persistent graph). It lets the
        #: no-graph arm author Comparative questions, so both arms can be compared on
        #: the SAME question types. None = untouched behaviour.
        self.local_extractor = local_extractor
        self._local_cache: Dict[str, tuple] = {}
        self.kg = kg
        self.target_rag = target_rag
        self.quality_gate = quality_gate
        self.strict_gold = strict_gold
        # counterfactual: after the real answer, re-ask the SAME turn with the clean gold
        # history to separate inherited (multi-turn) failures from intrinsic ones. Opt-in
        # because it costs one extra RAG call per non-seed turn.
        self.counterfactual = counterfactual
        self.judge = judge
        self.config = config or Config.load()
        self.gen_llm = gen_llm or LLM(config=self.config)
        self.retriever = retriever or GraphRetriever(
            kg, config=self.config, llm=self.gen_llm)
        self.query_gen = QueryGenerator(config=self.config, llm=self.gen_llm)
        # strict_gold: use the strict composer (gold uses ONLY evidence facts) + verify
        # each gold is grounded AND correct before accepting (trustworthy answer key).
        self.gold_gen = GoldAnswerGenerator(config=self.config, llm=self.gen_llm,
                                            strict=strict_gold)
        # "controller" = outcome-driven _next_type (the proposed method).
        # "random"     = the ablation: still a live loop, but the next type is
        #                chosen at random and the RAG's outcome is ignored. This
        #                isolates how much the controller adds over plain adaptivity.
        self.type_policy = type_policy
        # content_policy = the THESIS independent variable for question CONTENT:
        #   "static"  — the follow-up is generated WITHOUT access to the RAG's previous
        #               answer (the generator sees truth_history / gold). It could have
        #               been written before the RAG responded.
        #   "dynamic" — the follow-up is generated AFTER observing the RAG's actual
        #               previous answer (the generator sees rag_history). It conditions
        #               its content on what the RAG really said.
        # NOTE: gold-answer generation ALWAYS uses truth_history regardless of this flag,
        # so the answer key can never inherit a RAG error. (See generate().)
        self.content_policy = content_policy
        # forced_types: a fixed per-turn type sequence for follow-ups (turns 1..n-1),
        # cycled. When set, it OVERRIDES the type policy so static and dynamic runs get
        # EXACTLY the same question type at every turn — the paired-comparison control.
        # Turn 0 is always the factual seed regardless. Also disables the repeat-driven
        # type switch (see generate()) so the forced sequence is truly identical across
        # conditions.
        self.forced_types = list(forced_types) if forced_types else None
        self._rng = random.Random(self.config.seed)

    # -- the adaptive loop -------------------------------------------------- #
    def generate(self, seed, conversation_id: str, n_turns: int) -> AdaptiveConversation:
        # corpus-only datasets (e.g. MedQA) ship NO seed question -> generate a factual
        # one from the seed's evidence, else turn 0 is an empty question (which tanked
        # MedQA's "Follow-Up"/seed grounding).
        seed_q = (seed.question or "").strip()
        seed_q_source = "(question taken from the dataset, not authored from retrieval)"
        if not seed_q:
            ctx = (seed.context[0][:400] if getattr(seed, "context", None) else "")
            seed_ev = self.retriever.retrieve(ctx, k=6) if ctx else None
            ev_text = seed_ev.evidence_text(1500) if seed_ev else ctx
            out = self.gen_llm.chat_json(
                'Write ONE clear, specific factual question answerable from the TEXT. '
                'Do not mention "text" or "passage". Respond JSON: {"question": "..."}',
                f"TEXT: {ev_text}") or {}
            seed_q = (str(out.get("question") or "").strip()
                      or (ctx.split(".")[0][:120] if ctx else "What is described here?"))
            seed_q_source = ev_text

        conv = AdaptiveConversation(conversation_id=conversation_id, seed_question=seed_q)
        truth_history: List[Dict] = []   # user + GOLD  (keeps questions grounded)
        rag_history: List[Dict] = []     # user + RAG   (what the RAG sees / we react to)
        qtype = "Follow-Up"              # turn 0 is always the factual seed
        n_correct = 0

        for turn_id in range(n_turns):
            # 1. retrieve evidence around the running (truth) conversation
            focus = truth_history[-2]["content"] if len(truth_history) >= 2 \
                else seed_q
            evidence = self.retriever.retrieve(
                focus if turn_id else seed_q, k=8,
                conversation_history=truth_history)

            # 'textpairs' control: no persistent graph, but extract entities/relations
            # from the retrieved chunks so Comparative is authorable. Without this the
            # no-graph arm yields ZERO Comparative turns (the guard below reroutes every
            # one to Multi-Hop), which confounds arm with question-type mix.
            # Use ALL retrieved chunks: with only 4 no relation has two distinct sources,
            # so no shared-attribute pair exists and every Comparative would reroute.
            if self.local_extractor is not None and not evidence.nodes:
                evidence.nodes, evidence.edges = self.local_extractor.extract_local(
                    evidence.chunks, cache=self._local_cache)

            # 2. author the question + gold from corpus truth
            gave_up = False        # guard verdict for THIS turn
            q_ev = seed_q_source if turn_id == 0 \
                else " ".join(evidence.chunks[:5])[:1500]
            turn_provenance: Dict = {}     # generation provenance (follow-up turns only)
            if turn_id == 0:
                turn = self._seed_turn(seed)
                # honest label: the seed is a plain factual anchor, NOT a coreference
                # probe — typing it "Follow-Up" polluted that type's per-type stats.
                turn.query_type = "Seed"
                turn.question = seed_q          # use the (possibly generated) seed question
                gold_evidence = evidence
                gold = self.gold_gen.generate(turn, gold_evidence, truth_history)
                # SEED GUARD: the seed anchors the whole conversation, so an
                # ungrounded seed gold poisons every later turn. Attempts: widen
                # retrieval (dataset seed questions — e.g. MultiHopRAG's cross-
                # article source questions — are often ungroundable against the
                # chunk corpus), then REWRITE the seed question from corpus text.
                for _attempt in range(4):
                    grounded = not gold.gold_answer.startswith(ABSTENTION[:20])
                    if grounded and self.strict_gold:
                        grounded = self._gold_ok(turn.question, gold.gold_answer,
                                                 gold_evidence)
                    # skip_seed_grounding: the caller supplied an ALREADY-grounded seed
                    # (shared across methods), so never rewrite it — use it verbatim.
                    if grounded or _attempt == 3 or self.skip_seed_grounding:
                        break
                    if _attempt < 2:
                        gold_evidence = self.retriever.retrieve(
                            seed_q, k=16 if _attempt == 0 else 24)
                    else:
                        # last resort: replace the ungroundable seed with a
                        # question authored FROM the corpus evidence itself.
                        # Strip LaTeX cross-ref labels (TABREF1, ...) from the text so the
                        # model can't copy them into the seed question (the seed bypasses
                        # the in-loop placeholder guard).
                        ev_text = gold_evidence.evidence_text(1500) \
                            if gold_evidence else ""
                        ev_text = LATEX_LABEL.sub(" ", ev_text)
                        out = self.gen_llm.chat_json(
                            'Write ONE clear, specific factual question answerable '
                            'from the TEXT. Do not mention "text" or "passage". Do NOT '
                            'reference tables, sections, figures, or citations by label '
                            '(e.g. "Table TABREF1", "Section SECREF2", "[BIBREF3]"). '
                            'Respond JSON: {"question": "..."}',
                            f"TEXT: {ev_text}") or {}
                        newq = str(out.get("question") or "").strip()
                        if has_latex_label(newq):     # belt-and-suspenders
                            newq = LATEX_LABEL.sub("", newq).strip()
                        if newq:
                            seed_q = newq
                            turn.question = newq
                            conv.seed_question = newq
                            q_ev = ev_text      # the rewrite authored FROM this text
                            gold_evidence = self.retriever.retrieve(seed_q, k=8)
                    gold = self.gold_gen.generate(turn, gold_evidence, truth_history)
                gave_up = not grounded
            else:
                # OPT-IN UNANSWERABLE INJECTION: force this one follow-up to be an
                # Unanswerable probe (knowledge-boundary / hallucination test), leaving
                # every other turn adaptive. Overrides the controller's type for this turn.
                if self.inject_unanswerable_at == turn_id:
                    qtype = "Unanswerable"
                # GROUNDING GUARD: try up to 3 times to author a question of this type
                # whose gold is actually SUPPORTED by retrieved evidence. Each attempt
                # re-retrieves for the (possibly drifted) question; Comparative/Multi-Hop
                # widen k and merge an entity-focused retrieval (both entities covered).
                # We reject ungroundable instances (gold == abstention) and retry, so the
                # answer key stays grounded without abandoning E's hard question types.
                # STRUCTURAL FALLBACK: a grounded Comparative needs a pair with a
                # shared stated attribute. When the retrieved evidence offers none,
                # switch to Multi-Hop (same difficulty tier) instead of letting the
                # generator freelance an uncomposable comparison (~0.48 give-up).
                if qtype == "Comparative" and "shared attribute" not in \
                        self.query_gen._comparable_pairs(evidence.nodes, evidence.edges):
                    qtype = "Multi-Hop"
                turn = gold = gold_evidence = None
                feedback = None      # tell the next attempt WHY the last one failed
                # COVERAGE-GUIDED GENERATION: the facts already answered this conversation,
                # so the generator targets a NEW fact instead of recycling one (cuts both
                # repetition and the give-ups that repetition causes).
                covered = [m.get("content", "") for m in truth_history
                           if m.get("role") == "assistant"]
                gen_type = qtype     # may be switched below if repeats persist
                # THE INDEPENDENT VARIABLE: which history the question generator may
                # condition on. dynamic -> the RAG's ACTUAL prior answers (rag_history);
                # static -> gold (truth_history), i.e. no access to the RAG's answer.
                # Everything downstream (retrieval grounding, gold, coverage, structural
                # checks) stays on truth_history so ONLY question content is affected.
                gen_history = (rag_history if self.content_policy == "dynamic"
                               else truth_history)
                # PROVENANCE: record what the generator was allowed to see, so the run
                # can prove the dynamic condition actually observed the RAG's answer.
                # truth_history / rag_history hold turns 0..turn_id-1 (not yet advanced).
                _prev_gold = next((m["content"] for m in reversed(truth_history)
                                   if m.get("role") == "assistant"), "")
                _prev_rag = next((m["content"] for m in reversed(rag_history)
                                  if m.get("role") == "assistant"), "")
                turn_provenance = {
                    "content_mode": self.content_policy,
                    "generator_saw_rag_answer": self.content_policy == "dynamic",
                    "previous_gold_answer": _prev_gold,
                    "previous_rag_answer": _prev_rag,
                    "history_given_to_generator": [dict(m) for m in gen_history[-6:]],
                }
                for _attempt in range(3):
                    cand = self.query_gen.generate(gen_type, turn_id, evidence,
                                                   gen_history, feedback=feedback,
                                                   covered=covered)
                    # TRUTHFULNESS guard: drop any gold fact leaked as an unstated
                    # presupposition BEFORE gold/retrieval, so the gold stays aligned.
                    # ONLY Follow-Up: it uses a pronoun and is the type that smuggles
                    # descriptive clauses from the gold. Clarification/Comparative/etc.
                    # legitimately NAME entities, so stripping them makes them vague.
                    if cand.query_type == "Follow-Up":
                        cand.question = self._strip_leak(cand.question, truth_history)
                    # PRE-SEND QUALITY GATE: score well-formedness, rewrite if low.
                    if self.quality_gate:
                        cand.question = self._refine_quality(cand.question, truth_history)
                    # STRUCTURAL TYPE CHECK (prompts request, code enforces):
                    # pronoun-based types must actually contain a pronoun and not
                    # name their target — Ambiguous-Reference probes without this
                    # were plain factual questions that never caught anything.
                    err = self._type_structure_error(cand, truth_history)
                    if err and _attempt < 2:
                        feedback = err
                        continue
                    # PLACEHOLDER guard (question): reject a question that references a
                    # table/section/figure/citation by its raw LaTeX label ("Table TABREF1")
                    # — the label is meaningless standalone. Ask about the content instead.
                    if has_latex_label(cand.question) and _attempt < 2:
                        feedback = ('the question references a raw LaTeX placeholder label '
                                    '(e.g. "Table TABREF1", "Section SECREF27"); ask about '
                                    'the actual content, not the table/section/figure label.')
                        continue
                    multi = cand.query_type in ("Comparative", "Multi-Hop")
                    # Type-dependent retrieval: Follow-Up uses a PRONOUN ("it"/"they")
                    # so it NEEDS the history to resolve the referent; named-entity types
                    # (Comparative/Correction/...) retrieve better WITHOUT the history,
                    # which otherwise anchors to the seed topic and returns stale evidence.
                    # CONVERSATIONAL QUERY REWRITING (ZeQR / CQR): rewrite the (often
                    # pronoun-based) question into a SELF-CONTAINED query using the
                    # conversation, then retrieve on THAT -> resolves "it"/"they"/"that"
                    # so the evidence matches the gold (fixes Follow-Up grounding).
                    retr_q = self._self_contained(cand.question, truth_history)
                    ev = self.retriever.retrieve(retr_q, k=16 if multi else 8,
                                                 conversation_history=None)
                    _te = cand.target_entity
                    if isinstance(_te, (list, tuple)):
                        _te = " ".join(str(x) for x in _te)
                    if _te and str(_te).strip():
                        ev = self._merge_evidence(
                            ev, self.retriever.retrieve(str(_te), k=8))
                    if cand.query_type == "Comparative":
                        ev = self._merge_evidence(ev, self.retriever.retrieve(retr_q, k=10))
                    g = self.gold_gen.generate(cand, ev, truth_history)
                    turn, gold, gold_evidence = cand, g, ev
                    # accept once grounded (real answer, not the abstention string),
                    # or if the type is intentionally Unanswerable
                    grounded = cand.is_unanswerable or \
                        not g.gold_answer.startswith(ABSTENTION[:20])
                    # strict_gold: additionally VERIFY the gold is supported+correct
                    # (a trustworthy answer key — so a correct RAG isn't marked wrong).
                    if grounded and self.strict_gold and not cand.is_unanswerable:
                        grounded = self._gold_ok(cand.question, g.gold_answer, ev)
                    # PLACEHOLDER guard (gold): a gold that quotes a LaTeX label
                    # ("Table TABREF25 shows that...") references something meaningless
                    # standalone. Reject, ask for the fact stated without the label.
                    if grounded and not cand.is_unanswerable \
                            and has_latex_label(g.gold_answer):
                        grounded = False
                        if _attempt < 2:
                            feedback = ('the answer references a raw LaTeX placeholder label '
                                        '(e.g. "Table TABREF25 shows..."); state the actual '
                                        'fact without the table/section/figure label.')
                            continue
                    # CITATION-ONLY guard: reject a gold whose entire content is a bare
                    # reference marker ('...was introduced in [7]') — untestable, since the
                    # corpus has no reference list to resolve [7]. Ask for a real fact.
                    if grounded and not cand.is_unanswerable \
                            and gold_is_only_a_citation(g.gold_answer):
                        grounded = False
                        if _attempt < 2:
                            feedback = ('the answer is essentially just a citation marker '
                                        '(e.g. "...in [7]") with no substantive content; ask '
                                        'about a concrete, testable fact from the EVIDENCE.')
                            continue
                    # ADDS-NEW-INFO guard: reject a long gold that merely restates the
                    # question and introduces no new content (a near-empty answer that tests
                    # nothing and tanks well_formed). Short value answers are exempt.
                    if grounded and not cand.is_unanswerable \
                            and gold_only_restates_question(cand.question, g.gold_answer):
                        grounded = False
                        if _attempt < 2:
                            feedback = ('the answer just restates the question and adds no new '
                                        'information; ask about a fact whose ANSWER introduces '
                                        'new, specific content not already in the question.')
                            continue
                    # NO-REPEAT guard: reject a follow-up whose gold merely restates a
                    # fact already established earlier (the conversation-stall that
                    # tanks well_formed). Retry, telling the generator to move on.
                    if grounded and not cand.is_unanswerable \
                            and gold_repeats_earlier_answer(g.gold_answer, truth_history):
                        grounded = False
                        if _attempt < 2:
                            # a pronoun/back-referring type that keeps circling the same
                            # fact won't escape by retrying the SAME type — switch to a
                            # fact-forcing type (Multi-Hop) that must surface a new fact.
                            if gen_type in ("Ambiguous Reference", "Follow-Up",
                                            "Clarification", "Correction") \
                                    and not self.forced_types:
                                gen_type = "Multi-Hop"
                            feedback = ('the answer merely repeats a fact already '
                                        'established earlier in the conversation; ask '
                                        'about a DIFFERENT, previously-uncovered fact.')
                            continue
                    if grounded:
                        break
                    feedback = (
                        f'the question "{cand.question[:120]}" asked for a fact the '
                        "retrieved evidence does not state, so no gold answer could "
                        "be composed."
                        if g.gold_answer.startswith(ABSTENTION[:20]) else
                        f'the drafted gold for "{cand.question[:120]}" failed '
                        "verification against the evidence (unsupported or off-target).")
                gave_up = not grounded   # last attempt's verdict survives the loop
            gold_text = gold.gold_answer
            ev_text = " ".join(gold_evidence.chunks[:5])[:1500]

            # 3. ASK THE REAL RAG (it sees its own prior answers, not the gold)
            rag_resp = self.target_rag.answer(turn.question, rag_history)
            rag_answer = rag_resp.answer or ""

            # 4. grade the RAG's answer -> outcome
            outcome = self._grade(turn.question, gold_text, rag_answer,
                                  turn.is_unanswerable)

            # 4b. COUNTERFACTUAL PROBE (Mode 1, opt-in): re-ask the SAME question with the
            # CLEAN gold history instead of the RAG's own (possibly poisoned) one. At this
            # point `truth_history` has NOT yet been advanced for the current turn, so it
            # holds exactly turns 0..turn_id-1 of gold -- the clean history for this
            # question. Comparing the two outcomes isolates the multi-turn effect: a
            # failure that flips to "correct" here was INHERITED from the conversation; one
            # that persists is INTRINSIC. Skipped on turn 0 (both histories are empty).
            cf_outcome: Optional[str] = None
            cf_answer = ""
            if self.counterfactual and turn_id > 0:
                cf_resp = self.target_rag.answer(turn.question, truth_history)
                cf_answer = cf_resp.answer or ""
                cf_outcome = self._grade(turn.question, gold_text, cf_answer,
                                         turn.is_unanswerable)

            # TWO DOCUMENT SETS, kept separate: what the generator saw vs what the RAG
            # retrieved. Both carry real doc_id/rank/score (see RetrievalResult /
            # RAGResponse); doc_ids share the kg.chunks index space when the RAG is built
            # over the same corpus, so they are directly comparable.
            qg_docs = gold_evidence.scored_documents() if gold_evidence is not None \
                and hasattr(gold_evidence, "scored_documents") else []
            gen_source = "seed" if turn_id == 0 else (
                "rag_history" if self.content_policy == "dynamic" else "truth_history")
            conv.turns.append(AdaptiveTurn(
                turn_id=turn_id, query_type=turn.query_type, question=turn.question,
                gold=gold_text, evidence=ev_text, rag_answer=rag_answer,
                rag_retrieved_context=list(rag_resp.retrieved_context or []),
                outcome=outcome, is_unanswerable=turn.is_unanswerable,
                guard_gave_up=gave_up, question_evidence=q_ev,
                cf_outcome=cf_outcome, cf_rag_answer=cf_answer,
                provenance=turn_provenance,
                expected_components=list((getattr(turn, "meta", None) or {}).get("components", [])),
                question_generation_documents=qg_docs,
                rag_retrieved_documents=list(rag_resp.retrieved_docs or []),
                generation_source=gen_source))
            conv.type_sequence.append(turn.query_type)
            conv.outcome_sequence.append(outcome)

            # 5. advance both histories
            truth_history += [{"role": "user", "content": turn.question},
                              {"role": "assistant", "content": gold_text}]
            rag_history += [{"role": "user", "content": turn.question},
                            {"role": "assistant", "content": rag_answer}]

            # 6. THE ADAPTIVE STEP: the RAG's outcome chooses the next type
            if outcome == "correct":
                n_correct += 1
            qtype = self._choose_type(outcome, turn.query_type, n_correct, turn_id)

        logger.info("conv %s | types=%s | outcomes=%s", conversation_id,
                    conv.type_sequence, conv.outcome_sequence)
        return conv

    @staticmethod
    def _type_structure_error(cand, history=None) -> Optional[str]:
        """Feedback text when a type is structurally wrong (pronoun types: no pronoun
        or names its target; Clarification: references nothing from the conversation).
        Returns None when the structure is fine."""
        import re
        q = cand.question or ""
        if cand.query_type in ("Follow-Up", "Ambiguous Reference"):
            # PERSONAL pronouns only: this/that/these/those match as conjunctions
            # ("argues that ...") and let non-coreference questions slip through.
            if not re.search(r"\b(he|she|it|they|him|her|them|his|hers|its|"
                             r"their|theirs)\b", q, re.I):
                return (f'the question "{q[:100]}" contains no pronoun, but a '
                        f"{cand.query_type} question must refer to its target entity "
                        "ONLY by a pronoun (he/she/it/they).")
            te = cand.target_entity
            if isinstance(te, (list, tuple)):
                te = " ".join(str(x) for x in te)
            te = (str(te).strip() if te else "")
            if te and te.lower() in q.lower():
                return (f'the question explicitly names "{te}", but a '
                        f"{cand.query_type} question must refer to it ONLY by a pronoun.")
        elif cand.query_type == "Clarification" and history:
            hist_toks = set(re.findall(r"[a-z]{4,}", " ".join(
                str(t.get("content") or "") for t in history).lower()))
            q_toks = set(re.findall(r"[a-z]{4,}", q.lower())) - {
                "what", "which", "would", "could", "should", "please", "about",
                "clarify", "explain", "exactly", "mean", "more", "detail", "details"}
            if q_toks and hist_toks and not (q_toks & hist_toks):
                return (f'the question "{q[:100]}" mentions nothing from the '
                        "conversation — a Clarification must ask about a detail a "
                        "previous answer actually contained.")
        return None

    def _self_contained(self, question, history):
        """Conversational query rewriting (ZeQR/CQR): turn a pronoun/elliptical question
        into a standalone search query by resolving references against the conversation,
        so retrieval finds the right entity's chunks. Falls back to the original."""
        if not history:
            return question
        hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        out = self.gen_llm.chat_json(
            "Rewrite the QUESTION as a fully self-contained search query: replace every "
            "pronoun/reference (it, they, that, this, the former) with the actual entity "
            "named in the CONVERSATION. Keep it concise. Respond JSON: {\"query\": \"...\"}",
            f"CONVERSATION:\n{hist}\n\nQUESTION: {question}") or {}
        return (str(out.get("query") or "").strip() or question)

    def _strip_leak(self, question, history):
        """TRUTHFULNESS guard (Gricean Maxim of Quality): rewrite the question to drop any
        presupposition (descriptive clause / assumed fact) that was NOT already stated in
        the conversation, while keeping the core ask and any pronoun. Prevents the gold
        evidence leaking into the question as an unstated assumption. Falls back safely."""
        if not history:
            return question
        hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        out = self.gen_llm.chat_json(
            "Check the QUESTION for presuppositions NOT established in the CONVERSATION. "
            "If it assumes any fact/descriptive clause that was never stated (e.g. 'when "
            "he was X', 'the famous Y'), rewrite it to remove ONLY that unstated "
            "assumption, keeping the core question and any pronoun. If it assumes nothing "
            "unstated, return it unchanged. Respond JSON: {\"question\": \"...\"}",
            f"CONVERSATION:\n{hist}\n\nQUESTION: {question}") or {}
        return (str(out.get("question") or "").strip() or question)

    def _gold_ok(self, question, gold, evidence):
        """Verify the gold is BOTH grounded in the evidence AND a correct answer to the
        question — so the answer key is trustworthy and won't mark a correct RAG wrong."""
        ev = " ".join(evidence.chunks[:5])[:1800]
        out = self.gen_llm.chat_json(
            "Judge the GOLD answer. Reply true ONLY if it is (a) fully supported by the "
            "EVIDENCE (no facts beyond it) AND (b) a correct answer to the QUESTION. "
            'JSON: {"ok": true/false}',
            f"EVIDENCE: {ev}\nQUESTION: {question}\nGOLD: {gold}") or {}
        return bool(out.get("ok"))

    def _refine_quality(self, question, history):
        """PRE-SEND QUALITY GATE (self-refine / generate-then-critique): score the
        question's well-formedness GIVEN the conversation; if it is below 4/5, rewrite it
        to be clearer WITHOUT changing what it asks and WITHOUT naming an entity the
        question intentionally refers to by pronoun (preserves the coreference test).
        Runs only when quality_gate=True. One LLM call (score + rewrite together)."""
        hist = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:]) or "(start)"
        out = self.gen_llm.chat_json(
            "Rate the QUESTION's well-formedness 1-5 (clear, answerable, self-contained "
            "GIVEN THE CONVERSATION; a pronoun is fine if the referent is clear). If it is "
            "below 4, rewrite it to be clearer WITHOUT changing what it asks and WITHOUT "
            "naming an entity it intentionally refers to by pronoun. "
            'JSON: {"score": 1-5, "improved": "<question, rewritten only if score<4>"}',
            f"CONVERSATION:\n{hist}\nQUESTION: {question}") or {}
        imp = str(out.get("improved") or "").strip()
        try:
            score = int(out.get("score") or 5)
        except (TypeError, ValueError):
            score = 5
        return imp if (score < 4 and imp) else question

    @staticmethod
    def _merge_evidence(a, b):
        """Union two RetrievalResults (dedup nodes by id, chunks by text, edges by triple).

        Chunks are merged TOGETHER WITH their aligned ``chunk_ids``/``scores`` so the
        per-turn ``question_generation_documents`` metadata (doc_id/rank/score) stays
        valid after a merge; ``score_type`` is inherited when ``a`` has none."""
        seen_n = {n.get("id") for n in a.nodes}
        a.nodes = a.nodes + [n for n in b.nodes if n.get("id") not in seen_n]
        # pad a's aligned lists defensively so appends stay index-aligned with chunks
        while len(a.chunk_ids) < len(a.chunks):
            a.chunk_ids.append(None)
        while len(a.scores) < len(a.chunks):
            a.scores.append(None)
        seen_c = set(a.chunks)
        b_ids = b.chunk_ids or [None] * len(b.chunks)
        b_scores = b.scores or [None] * len(b.chunks)
        for i, c in enumerate(b.chunks):
            if c not in seen_c:
                a.chunks.append(c)
                a.chunk_ids.append(b_ids[i] if i < len(b_ids) else None)
                a.scores.append(b_scores[i] if i < len(b_scores) else None)
                seen_c.add(c)
        if b.score_type and not a.score_type:
            a.score_type = b.score_type
        seen_e = {(e.get("source"), e.get("target"), e.get("relation")) for e in a.edges}
        a.edges = a.edges + [e for e in b.edges
                             if (e.get("source"), e.get("target"), e.get("relation")) not in seen_e]
        return a

    def generate_many(self, seeds, n_turns: int) -> List[AdaptiveConversation]:
        out = []
        for i, seed in enumerate(seeds[: self.config.num_conversations]):
            out.append(self.generate(seed, f"conv-{i:03d}", n_turns))
        return out

    # -- type selection ----------------------------------------------------- #
    def _choose_type(self, outcome: str, last_type: str, n_correct: int,
                     turn_id: int) -> str:
        """Dispatch to the controller (the proposed method) or the random ablation."""
        if self.forced_types:
            # Fixed sequence: _choose_type is called at the end of `turn_id` and returns
            # the type for turn_id+1, so follow-up turn t (>=1) -> forced_types[(t-1) % L].
            # With turn_id being the current turn, that index is turn_id % L.
            return self.forced_types[turn_id % len(self.forced_types)]
        if self.type_policy == "random":
            # still a live loop, but the outcome is IGNORED: pick any other type.
            return self._rng.choice([t for t in QUERY_TYPES if t != last_type])
        return self._next_type(outcome, last_type, n_correct, turn_id)

    # -- the controller (this is what replaces D's _plan_sequence) ---------- #
    def _next_type(self, outcome: str, last_type: str, n_correct: int,
                   turn_id: int = 0) -> str:
        """Pick the next question type from how the RAG just did.

        Outcome-driven, but extended so every type in the taxonomy is reachable
        (the earlier controller could never emit Topic Shift, and only reached
        Clarification by accident):

        wrong        -> Correction        (can it recover from its own mistake?)
        hallucinated -> Unanswerable      (will it ever admit it doesn't know?)
        abstained    -> Clarification / Follow-Up  (alternate: ask it to elaborate,
                        or re-engage with a concrete follow-up)
        correct      -> climb the difficulty ramp (Multi-Hop / Comparative /
                        Ambiguous Reference); on every 4th straight success inject a
                        Topic Shift to test whether sustained success left it
                        clinging to stale context.
        """
        if outcome == "wrong":
            nxt = "Correction"
        elif outcome == "hallucinated":
            nxt = "Unanswerable"
        elif outcome == "abstained":
            # re-engage: alternate a clarification ask with a concrete follow-up
            nxt = "Clarification" if turn_id % 2 == 0 else "Follow-Up"
        elif n_correct % 4 == 0:  # sustained success -> change subject (memory reset)
            nxt = "Topic Shift"
        else:                     # climb the difficulty ramp
            nxt = _ESCALATION[n_correct % len(_ESCALATION)]
        if nxt == last_type:  # never ask the same type twice in a row
            # (was ->Follow-Up after Clarification, which breaks the pronoun referent)
            nxt = "Clarification" if last_type != "Clarification" else "Comparative"
        # A pronoun Follow-Up needs a prior turn that ESTABLISHED a single clear entity.
        # It breaks after (a) MULTI-ENTITY turns (Comparative/Multi-Hop/Ambiguous -> "it"
        # is ambiguous) and (b) a CLARIFICATION (a meta-question that introduces NO new
        # entity, so "it" has no referent: "What amount did it pay?" after "Can you
        # clarify which model...?"). In those cases route Follow-Up to a self-contained
        # type instead of a pronoun question.
        if nxt == "Follow-Up" and last_type in {"Comparative", "Multi-Hop",
                                                "Ambiguous Reference", "Clarification"}:
            nxt = "Comparative" if last_type == "Clarification" else "Clarification"
        return nxt

    # -- per-turn grader ---------------------------------------------------- #
    def _grade(self, question: str, gold: str, rag_answer: str,
               is_unanswerable: bool) -> str:
        if not rag_answer.strip():
            return "abstained"
        if self.judge.available:
            out = self.judge.chat_json(
                _GRADE_SYS,
                f"QUESTION: {question}\nGOLD: {gold}\n"
                f"UNANSWERABLE: {is_unanswerable}\nRAG ANSWER: {rag_answer}")
            if out and out.get("label") in OUTCOMES:
                return out["label"]
        # offline / failure fallback: cheap lexical heuristic
        low = rag_answer.lower()
        if any(p in low for p in ("i don't know", "cannot answer", "not answerable",
                                  "no information", "unable to")):
            return "abstained"
        if is_unanswerable:
            return "hallucinated"
        return "correct" if gold.lower()[:30] in low or low[:30] in gold.lower() \
            else "wrong"

    # -- seed turn ---------------------------------------------------------- #
    def _seed_turn(self, seed) -> QueryTurn:
        spec = QUERY_TYPES["Follow-Up"]
        return QueryTurn(
            question=seed.question, query_type="Follow-Up", turn_id=0, difficulty=1,
            capability="Factual Retrieval (seed)",
            expected_failure=spec.get("expected_failure", "Missing Retrieval"),
            is_unanswerable=False, meta={"seed_id": seed.id})
