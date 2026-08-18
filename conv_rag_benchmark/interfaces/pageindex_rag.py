"""
PageIndex target RAG — connect your friend's PageIndex system to the benchmark.

WHAT TO DO: implement the ONE method marked ``TODO`` below (``_pageindex_retrieve``) so
it calls your PageIndex (the hosted API or the local package) and returns the retrieved
passages. Everything else — generation, logging the two document sets, scoring, wiring
into every conversation method — is already handled by the base class and the orchestrator.

WHY IT'S SHAPED THIS WAY:
  * It subclasses ``_LexicalCorpusRAG`` so GENERATION reuses the SAME OpenAI backend as
    every other RAG (vector/selfrag/crag/…). That keeps the comparison controlled: only
    the RETRIEVAL differs (PageIndex vs cosine), not the answer model. If instead you want
    PageIndex's OWN end-to-end answer, override ``answer`` and set the answer string
    yourself — but then note in the write-up that PageIndex uses a different generator.
  * ``_pageindex_retrieve`` must return REAL records: ``{doc_id, rank, score, score_type,
    text}``. Put PageIndex's actual relevance score in ``score`` and name it in
    ``score_type`` (e.g. "pageindex"). NEVER fabricate a score — the whole
    retrieval-vs-generation attribution rests on these being real.

HOW IT GETS SELECTED: importing this module registers the name "pageindex", so
``--rag pageindex`` works. See "ENABLING IT" at the bottom.

    python -m compare.experiment --method proposed --rag pageindex --dataset qasper \
        --convos 15 --turns 8 --seed 42 --label main
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .rag_interface import _LexicalCorpusRAG, RAGInterface, RAGResponse


class PageIndexRAG(_LexicalCorpusRAG):
    """PageIndex retrieval + the shared LLM generator.

    The corpus arrives as ``chunks`` (a list of passage strings) — the same corpus every
    other RAG is built over. Build (or connect to) your PageIndex index from these in
    ``__init__`` if you need to, then answer each query in ``_pageindex_retrieve``.
    """

    name = "pageindex"

    def __init__(self, chunks: List[str], config=None, llm=None, embedder=None,
                 k: int = 5, use_history: bool = True, **kwargs):
        # Base sets up self.chunks, self.llm (the shared generator), self.k, etc.
        # We keep use_embeddings=False because PageIndex does its own indexing/retrieval —
        # we don't need the base's cosine index. (Flip to True only if you want a cosine
        # fallback available.)
        super().__init__(chunks, config=config, llm=llm, embedder=embedder, k=k,
                         use_embeddings=False, use_history=use_history)
        # Map passage text -> its index in self.chunks, so a PageIndex hit that returns one
        # of OUR corpus passages can be given a doc_id that lines up with
        # question_generation_documents (same corpus => comparable doc_ids). Hits that are
        # NOT one of our chunks (e.g. a PageIndex tree-summary node) fall back to the
        # backend's own id — still fine, just not index-aligned.
        self._text_to_id = {c: i for i, c in enumerate(self.chunks)}

        # ------------------------------------------------------------------ #
        # TODO(friend): build / connect your PageIndex index here if needed.
        #   e.g.  self._pi = pageindex.build(self.chunks)          # local package
        #    or   self._pi_doc_id = pageindex_api.submit(self.chunks)  # hosted API
        # Leave as-is if PageIndex is queried statelessly per request.
        # ------------------------------------------------------------------ #

    # ====================================================================== #
    #  THE ONE METHOD TO IMPLEMENT
    # ====================================================================== #
    def _pageindex_retrieve(self, query: str) -> List[Dict]:
        """Return PageIndex's top passages for ``query`` as ranked records.

        REQUIRED shape — one dict per retrieved passage, best first:
            {
              "doc_id": <int index into self.chunks, or PageIndex's own id string>,
              "rank":   <1-based int>,          # position in PageIndex's ranking
              "score":  <float>,                # PageIndex's REAL relevance score
              "score_type": "pageindex",        # names the metric (do not fake)
              "text":   <the passage string>,
            }

        Use ``self._resolve_doc_id(text, fallback_id)`` to map a returned passage back to
        our corpus index when it is one of our chunks.
        """
        # ------------------------------------------------------------------ #
        # TODO(friend): replace this body with a real PageIndex call.
        #
        # Example skeleton (pseudo — adapt to the real PageIndex client):
        #
        #   hits = self._pi.query(query, top_k=self.k)          # your PageIndex call
        #   docs = []
        #   for rank, h in enumerate(hits, start=1):
        #       docs.append({
        #           "doc_id": self._resolve_doc_id(h.text, h.id),
        #           "rank": rank,
        #           "score": float(h.score),        # PageIndex's real score
        #           "score_type": "pageindex",
        #           "text": h.text,
        #       })
        #   return docs
        # ------------------------------------------------------------------ #
        raise NotImplementedError(
            "PageIndexRAG._pageindex_retrieve is a stub — implement it to call PageIndex. "
            "See the TODO in conv_rag_benchmark/interfaces/pageindex_rag.py.")

    # -- helpers (already done) -------------------------------------------- #
    def _resolve_doc_id(self, text: str, fallback_id):
        """Give a PageIndex hit a doc_id: our corpus index if the text is one of our
        chunks (so it lines up with question_generation_documents), else PageIndex's own
        id. Never invents an index."""
        idx = self._text_to_id.get(text)
        return idx if idx is not None else fallback_id

    def answer(self, question: str, history: Optional[List[Dict]] = None) -> RAGResponse:
        history = history or []
        # Same history-conditioned query the other corpus RAGs use, so retrieval is
        # driven identically — only the retriever backend differs.
        query = question
        if self.use_history and history:
            query = " ".join(t["content"] for t in history[-2:]) + " " + question

        docs = self._pageindex_retrieve(query)          # <-- PageIndex retrieval
        docs = self._reindexed(docs)                    # normalise rank to final order
        ctx = [d["text"] for d in docs]
        ans = self._generate(question, ctx, history)    # <-- shared LLM generation
        return RAGResponse(answer=ans, retrieved_context=ctx, retrieved_docs=docs,
                           meta={"rag": self.name})


# ========================================================================== #
#  ENABLING IT
#  Importing this module self-registers the name "pageindex" with the benchmark,
#  so `--rag pageindex` resolves. Trigger the import in ONE of these ways:
#
#   (A) add this single line near the bottom of conv_rag_benchmark/connectors.py:
#           from .interfaces import pageindex_rag  # noqa: F401  (registers "pageindex")
#
#   (B) or run through a tiny launcher that imports it first:
#           python -c "import conv_rag_benchmark.interfaces.pageindex_rag; \
#                      from compare.experiment import main; main()" \
#               --method proposed --rag pageindex --dataset qasper --convos 15 --turns 8
# ========================================================================== #
def _register() -> None:
    from ..connectors import register_rag
    register_rag("pageindex",
                 lambda chunks, config, **kw: PageIndexRAG(chunks, config=config, **kw))


_register()
