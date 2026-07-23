"""
Probe-difficulty control: what fraction of turns are 'verbatim softballs' — the
gold is contained in ONE evidence sentence, so the RAG can answer by copying a
sentence rather than understanding? A cleaned benchmark with a low failure rate
is only meaningful if the probes still have teeth.

A turn is VERBATIM-ANSWERABLE when a single evidence sentence contains >= 80% of
the gold's content tokens. Offline, no LLM calls.

Run:  python -m conv_rag_benchmark.probe_difficulty [build.json ...]
"""
import argparse
import json
import re
from collections import defaultdict

ABST = "Not answerable"


def _content_tokens(s):
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


def verbatim_answerable(gold: str, evidence: str) -> bool:
    g = _content_tokens(gold)
    if not g:
        return False
    for sent in re.split(r"(?<=[.!?])\s+", evidence or ""):
        s = _content_tokens(sent)
        if len(g & s) / len(g) >= 0.8:
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    for f in args.files:
        d = json.load(open(f, encoding="utf-8"))
        by_type = defaultdict(lambda: [0, 0])
        for c in d.get("conversations", []):
            for t in c.get("turns", []):
                gold = t.get("gold") or ""
                if not gold.strip() or gold.startswith(ABST):
                    continue
                soft = verbatim_answerable(gold, t.get("evidence") or "")
                by_type[t.get("query_type", "?")][1] += 1
                by_type[t.get("query_type", "?")][0] += int(soft)
        tot = sum(v[1] for v in by_type.values())
        softtot = sum(v[0] for v in by_type.values())
        print(f"\n=== {f} ===")
        print(f"{'type':<20}{'verbatim':>9}{'total':>7}{'rate':>7}")
        for k, v in sorted(by_type.items()):
            print(f"{k:<20}{v[0]:>9}{v[1]:>7}{v[0]/v[1] if v[1] else 0:>7.2f}")
        print(f"{'ALL':<20}{softtot:>9}{tot:>7}{softtot/tot if tot else 0:>7.2f}")


if __name__ == "__main__":
    main()
