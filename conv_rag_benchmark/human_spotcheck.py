"""
Human spot-check of the LLM judge on OUR OWN generated runs.

The MTRAG external validation (mtrag_validation/) measured the judge against
human annotators on *their* data. This closes the remaining loop: the thesis
author personally labels a stratified sample of turns from our own builds, and
we measure judge-vs-author agreement on the exact 4-way outcome scheme the
judge uses (correct / wrong / hallucinated / abstained).

Anchoring guard: the labeling file NEVER contains the judge's outcome — that
lives in a separate key file joined only at scoring time. Label blind.

Workflow (from the repo root):
    python -m conv_rag_benchmark.human_spotcheck sample --n 40   # draw items
    python -m conv_rag_benchmark.human_spotcheck label           # label them (resumable)
    python -m conv_rag_benchmark.human_spotcheck score           # agreement + kappa
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("conv_rag_benchmark/output")
ITEMS = OUT / "spotcheck_items.json"     # what the human sees (no judge labels)
KEY = OUT / "spotcheck_key.json"         # judge outcome per item id (do not open!)
LABELS = OUT / "spotcheck_labels.json"   # the human's answers, saved as you go

OUTCOMES = ("correct", "wrong", "hallucinated", "abstained")
SHORTCUTS = {"c": "correct", "w": "wrong", "h": "hallucinated", "a": "abstained"}


def _turns(pattern="conv_rag_benchmark/output/eval_*.json"):
    """Yield (build_name, turn) for every graded turn in every eval build."""
    for f in sorted(glob.glob(pattern)):
        name = Path(f).stem.replace("eval_", "")
        try:
            build = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for c in build.get("conversations", []):
            for t in c.get("turns", []):
                if t.get("outcome") in OUTCOMES and not t.get("guard_gave_up"):
                    yield name, t


def sample(n: int, seed: int = 42, exclude_key: str = ""):
    """Stratified draw: spread the sample across the four judge outcomes so the
    human sees failures too, not 90% 'correct' (round-robin over outcome pools).

    ``exclude_key``: path to a prior spotcheck_key.json whose (build, turn_id,
    query_type) turns are skipped — so a re-run draws a FRESH, non-overlapping
    sample (Round 2 must not reuse turns the annotator already saw, or the
    validation is contaminated)."""
    seen = set()
    if exclude_key and Path(exclude_key).exists():
        for v in json.loads(Path(exclude_key).read_text(encoding="utf-8")).values():
            seen.add((v.get("build"), v.get("turn_id"), v.get("query_type")))
    pools = defaultdict(list)
    for name, t in _turns():
        if (name, t.get("turn_id"), t.get("query_type")) in seen:
            continue
        pools[t["outcome"]].append((name, t))
    rng = random.Random(seed)
    for p in pools.values():
        rng.shuffle(p)
    picked, i = [], 0
    while len(picked) < n and any(pools.values()):
        for oc in OUTCOMES:                      # round-robin across outcomes
            if pools[oc] and len(picked) < n:
                picked.append(pools[oc].pop())
        i += 1
    rng.shuffle(picked)

    items, key = [], {}
    for idx, (name, t) in enumerate(picked):
        iid = f"item-{idx:03d}"
        items.append({
            "id": iid,
            "question": t["question"],
            "gold": t["gold"],
            "rag_answer": t["rag_answer"],
            "is_unanswerable": bool(t.get("is_unanswerable")),
        })
        key[iid] = {"judge_outcome": t["outcome"], "build": name,
                    "query_type": t.get("query_type"), "turn_id": t.get("turn_id")}
    ITEMS.write_text(json.dumps(items, indent=1), encoding="utf-8")
    KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"sampled {len(items)} turns -> {ITEMS}")
    print(f"judge answer key (do NOT open before labeling) -> {KEY}")
    print("next: python -m conv_rag_benchmark.human_spotcheck label")


def label():
    """Interactive, resumable labeling loop."""
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    todo = [it for it in items if it["id"] not in labels]
    print(f"{len(labels)}/{len(items)} already labeled; {len(todo)} to go.\n"
          "Label WHAT THE RAG DID (its action), not whether that action was good.\n"
          "Decide in this order:\n"
          "  Did the RAG DECLINE (say 'I don't know' / 'not in the context' / refuse)?\n"
          "     -> a = abstained   (ALWAYS use this for a decline, EVEN IF declining\n"
          "                         was the right thing to do on an unanswerable Q.\n"
          "                         Do NOT call a decline 'correct'.)\n"
          "  Otherwise it gave a real answer -- is it right?\n"
          "     -> c = correct       (matches the gold's facts; extra true detail is OK)\n"
          "     -> w = wrong         (confidently states something the gold doesn't)\n"
          "     -> h = hallucinated  (gold is unanswerable, but it invented specifics)\n"
          "  Tie-breakers:\n"
          "     partial answer (some gold facts missing, none wrong) -> lean c\n"
          "     names a DIFFERENT set of things than the gold        -> w\n"
          "  s = skip, q = quit (progress is saved)\n")
    for it in todo:
        print("=" * 72)
        print(f'[{it["id"]}]  unanswerable-by-design: {it["is_unanswerable"]}')
        print(f'\nQUESTION: {it["question"]}')
        print(f'\nGOLD:     {it["gold"]}')
        print(f'\nRAG SAID: {it["rag_answer"]}\n')
        while True:
            ans = input("your label [c/w/h/a/s/q]: ").strip().lower()
            if ans in ("q", "s") or ans in SHORTCUTS:
                break
            print("  please type one of: c w h a s q")
        if ans == "q":
            break
        if ans == "s":
            continue
        labels[it["id"]] = SHORTCUTS[ans]
        LABELS.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"\nsaved {len(labels)}/{len(items)} labels -> {LABELS}")
    if len(labels) == len(items):
        print("all done! next: python -m conv_rag_benchmark.human_spotcheck score")


def _kappa(pairs):
    """Cohen's kappa, multiclass (judge label vs human label)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for j, h in pairs if j == h) / n
    jm, hm = Counter(j for j, _ in pairs), Counter(h for _, h in pairs)
    pe = sum(jm[c] * hm[c] for c in set(jm) | set(hm)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def score():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    pairs = [(key[i]["judge_outcome"], h) for i, h in labels.items() if i in key]
    n = len(pairs)
    agree = sum(1 for j, h in pairs if j == h)

    # binary collapse: did the RAG behave well? (correct, or abstained when the
    # turn was unanswerable-by-design is handled per-item below)
    items = {it["id"]: it for it in json.loads(ITEMS.read_text(encoding="utf-8"))}
    def good(label, iid):
        return label == ("abstained" if items[iid]["is_unanswerable"] else "correct")
    bpairs = [(good(key[i]["judge_outcome"], i), good(h, i))
              for i, h in labels.items() if i in key]
    bagree = sum(1 for j, h in bpairs if j == h)

    conf = Counter(pairs)
    report = {
        "n_labeled": n,
        "fourway_agreement": round(agree / n, 3) if n else None,
        "fourway_kappa": _kappa(pairs),
        "binary_agreement": round(bagree / n, 3) if n else None,
        "binary_kappa": _kappa(bpairs),
        "confusion_judge_vs_human": {f"{j}|{h}": c for (j, h), c in conf.most_common()},
        "disagreements": [
            {"id": i, "judge": key[i]["judge_outcome"], "human": h,
             "build": key[i]["build"], "query_type": key[i]["query_type"]}
            for i, h in labels.items()
            if i in key and key[i]["judge_outcome"] != h],
    }
    out = OUT / "spotcheck_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\nsaved -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["sample", "label", "score"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default="",
                    help="path to a prior key.json to draw a fresh, non-overlapping sample")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "sample":
        sample(args.n, args.seed, args.exclude)
    elif args.mode == "label":
        label()
    else:
        score()


if __name__ == "__main__":
    main()
