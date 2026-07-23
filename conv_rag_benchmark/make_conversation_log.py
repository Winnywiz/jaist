"""
Export benchmark conversations as a READABLE log for human evaluation — whole
conversations in order (unlike make_human_eval's shuffled per-question CSV), so a
rater can judge dialogue-level properties: coherence, natural flow, whether each
typed probe does what its label claims.

Run:  python -m conv_rag_benchmark.make_conversation_log <build.json> [--out file.md]
"""
import argparse
import json
import os

HEADER = """# Conversation log for human evaluation
Source: {src}  ({n} conversations)

## Instructions for the evaluator
For every TURN answer (write y/n next to the [ ] boxes):
  Q1 natural   — does the question sound like something a real user would ask next?
  Q2 fair      — is the gold answer actually supported by the shown EVIDENCE?
For every CONVERSATION:
  C1 coherent  — do the turns follow each other sensibly (no jarring jumps except
                 where the type is 'Topic Shift')?
Please do NOT skip turns. "RAG" lines show the tested system's answer — you are NOT
rating the RAG, only the benchmark's questions and answer key.

---
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-convos", type=int, default=None)
    args = ap.parse_args(argv)

    d = json.load(open(args.build, encoding="utf-8"))
    convs = d.get("conversations", [])
    if args.max_convos:
        convs = convs[: args.max_convos]
    out_path = args.out or os.path.splitext(args.build)[0] + "_humanlog.md"

    lines = [HEADER.format(src=os.path.basename(args.build), n=len(convs))]
    for ci, c in enumerate(convs):
        lines.append(f"\n## Conversation {ci + 1}  (id: {c.get('conversation_id', ci)})")
        for t in c.get("turns", []):
            lines.append(f"\n**Turn {t.get('turn_id')} — type: {t.get('query_type')}**")
            lines.append(f"- QUESTION: {t.get('question', '')}")
            lines.append(f"- GOLD ANSWER: {t.get('gold', '')}")
            import re

            def _bullets(text):
                seen, out = set(), []
                for sent in re.split(r"(?<=[.!?])\s+", str(text or "").replace("\n", " ")):
                    sent = sent.strip()
                    if sent and sent not in seen:  # retrieval often repeats chunks
                        seen.add(sent)
                        out.append(f"    - {sent}")
                return out

            # Two evidence blocks: what authored the QUESTION, and what the GOLD
            # was composed against (a question-specific re-retrieval). One sentence
            # per bullet so the rater can scan for the supporting sentence.
            q_ev = t.get("question_evidence") or ""
            gold_ev = t.get("evidence") or ""
            if q_ev and q_ev.strip() != gold_ev.strip():
                lines.append("- EVIDENCE the QUESTION was authored from:")
                lines += _bullets(q_ev)
                lines.append("- EVIDENCE the GOLD ANSWER was composed against:")
            elif q_ev:
                lines.append("- EVIDENCE (authored the question AND grounds the gold):")
            else:
                lines.append("- EVIDENCE the GOLD ANSWER was composed against "
                             "(question-authoring retrieval not recorded in this build):")
            lines += _bullets(gold_ev)
            lines.append(f"- RAG: {t.get('rag_answer', '')}")
            lines.append("- [ ] Q1 natural   [ ] Q2 fair")
        lines.append("\n**[ ] C1 coherent (whole conversation)**")
        lines.append("\n---")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    n_turns = sum(len(c.get("turns", [])) for c in convs)
    print(f"wrote {len(convs)} conversations / {n_turns} turns -> {out_path}")


if __name__ == "__main__":
    main()
