"""
compare/annotate_score.py — score the classifier against human labels (the thesis DV).

Reads the review file produced by ``compare/annotate_sample.py`` AFTER you have filled in
``true_failure_type`` on the cases, and reports, PER METHOD (and overall):

  * attribution accuracy  — how often the classifier's predicted failure_type equals your
    true label (the headline dependent variable);
  * layer accuracy        — the coarser retrieval-vs-generation attribution (each type maps
    to a layer via failure/taxonomy.py);
  * Cohen's kappa         — classifier-vs-human agreement corrected for chance, so the
    number is honest even when one type dominates.

Only cases with a non-empty ``true_failure_type`` are scored; the rest are reported as
"pending". ``not_a_failure`` (the classifier over-fired) is a valid true label and counts
as a miss against whatever type the classifier predicted — which is the correct penalty.

This reads only the review file + the taxonomy; it changes nothing.

Usage:
    python -m compare.annotate_score
    python -m compare.annotate_score --file compare/result/main/annotation/review_todo.json
    python -m compare.annotate_score --show-disagreements
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import failure.taxonomy as tax

VALID_TYPES = set(tax.FAILURE_TYPES_BY_KEY) | {"not_a_failure"}
#: failure_type -> layer (retrieval / generation), authoritative from the taxonomy.
TYPE_LAYER = {k: getattr(v.layer, "value", str(v.layer))
              for k, v in tax.FAILURE_TYPES_BY_KEY.items()}
TYPE_LAYER["not_a_failure"] = "none"


def _layer_of(failure_type: Optional[str]) -> Optional[str]:
    if not failure_type:
        return None
    return TYPE_LAYER.get(failure_type)


def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    """Cohen's kappa between two raters over the same items. `pairs` = [(rater_a, rater_b)].
    Returns None when it is undefined (fewer than 2 items or a single constant category)."""
    n = len(pairs)
    if n < 2:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    if len(labels) < 2:
        return 1.0 if all(a == b for a, b in pairs) else 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def score(cases: List[Dict]) -> Dict:
    """Group by method, compute type/layer accuracy + kappa. Returns a report dict."""
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    invalid: List[Tuple[str, str]] = []
    pending = 0
    for c in cases:
        true_t = (c.get("true_failure_type") or "").strip()
        if not true_t:
            pending += 1
            continue
        if true_t not in VALID_TYPES:
            invalid.append((c.get("id", "?"), true_t))
            continue
        by_method[c["method"]].append(c)

    report = {"pending": pending, "invalid": invalid, "methods": {}, "overall": {}}
    all_type_pairs: List[Tuple[str, str]] = []
    all_layer_pairs: List[Tuple[str, str]] = []

    for method in sorted(by_method):
        rows = by_method[method]
        type_pairs, layer_pairs, disagreements = [], [], []
        for c in rows:
            pred_t = c["classifier_predicted"].get("failure_type") or "(none)"
            true_t = c["true_failure_type"].strip()
            pred_l = c["classifier_predicted"].get("layer") or _layer_of(pred_t) or "(none)"
            true_l = _layer_of(true_t) or "none"
            type_pairs.append((pred_t, true_t))
            layer_pairs.append((pred_l, true_l))
            if pred_t != true_t:
                disagreements.append({"id": c["id"], "predicted": pred_t,
                                      "true": true_t, "query_type": c.get("query_type")})
        n = len(rows)
        type_acc = sum(1 for p, t in type_pairs if p == t) / n
        layer_acc = sum(1 for p, t in layer_pairs if p == t) / n
        report["methods"][method] = {
            "n": n,
            "type_accuracy": round(type_acc, 3),
            "layer_accuracy": round(layer_acc, 3),
            "type_kappa": (round(k, 3) if (k := cohen_kappa(type_pairs)) is not None else None),
            "layer_kappa": (round(k, 3) if (k := cohen_kappa(layer_pairs)) is not None else None),
            "disagreements": disagreements,
        }
        all_type_pairs += type_pairs
        all_layer_pairs += layer_pairs

    if all_type_pairs:
        n = len(all_type_pairs)
        report["overall"] = {
            "n": n,
            "type_accuracy": round(sum(1 for p, t in all_type_pairs if p == t) / n, 3),
            "layer_accuracy": round(sum(1 for p, t in all_layer_pairs if p == t) / n, 3),
            "type_kappa": (round(k, 3) if (k := cohen_kappa(all_type_pairs)) is not None else None),
            "layer_kappa": (round(k, 3) if (k := cohen_kappa(all_layer_pairs)) is not None else None),
        }
    return report


def _print(report: Dict, show_disagreements: bool):
    inv = report["invalid"]
    if inv:
        print(f"!! {len(inv)} case(s) have an INVALID true_failure_type "
              f"(not one of {sorted(VALID_TYPES)}):")
        for cid, val in inv[:20]:
            print(f"     {cid}: {val!r}")
    print(f"\n# Attribution accuracy (classifier vs human)   pending={report['pending']}")
    print("| Method | n | type acc | layer acc | type kappa | layer kappa |")
    print("|---|--:|--:|--:|--:|--:|")
    for method, m in report["methods"].items():
        tk = "n/a" if m["type_kappa"] is None else f"{m['type_kappa']:.3f}"
        lk = "n/a" if m["layer_kappa"] is None else f"{m['layer_kappa']:.3f}"
        print(f"| {method} | {m['n']} | {m['type_accuracy']:.3f} | "
              f"{m['layer_accuracy']:.3f} | {tk} | {lk} |")
    ov = report.get("overall")
    if ov:
        tk = "n/a" if ov["type_kappa"] is None else f"{ov['type_kappa']:.3f}"
        lk = "n/a" if ov["layer_kappa"] is None else f"{ov['layer_kappa']:.3f}"
        print(f"| **overall** | {ov['n']} | {ov['type_accuracy']:.3f} | "
              f"{ov['layer_accuracy']:.3f} | {tk} | {lk} |")
    print("\n  type acc  = predicted failure_type == your true label (the DV)")
    print("  layer acc = retrieval-vs-generation attribution only")
    print("  kappa     = Cohen's kappa, agreement beyond chance (1=perfect, 0=chance)")
    if show_disagreements:
        print("\n# Disagreements (classifier predicted → your true label)")
        for method, m in report["methods"].items():
            if not m["disagreements"]:
                continue
            print(f"\n  {method}:")
            for d in m["disagreements"]:
                print(f"    {d['predicted']:18} -> {d['true']:18}  "
                      f"[{d.get('query_type')}]  {d['id']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score classifier vs human failure labels.")
    ap.add_argument("--file", default=os.path.join("compare", "result", "main",
                                                    "annotation", "review_todo.json"),
                    help="the review file you filled in")
    ap.add_argument("--show-disagreements", action="store_true",
                    help="list every case where the classifier and your label differ")
    ap.add_argument("--out", default=None,
                    help="also write the report JSON here (e.g. .../annotation/scores.json)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"Review file not found: {args.file}\n"
              "Run `python -m compare.annotate_sample` first.")
        return
    data = json.load(open(args.file, encoding="utf-8"))
    cases = data.get("cases", [])
    report = score(cases)
    _print(report, args.show_disagreements)

    if not report["methods"]:
        print("\n(No labelled cases yet — fill 'true_failure_type' in the review file.)")
    out_path = args.out or os.path.join(os.path.dirname(args.file), "scores.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(report, fw, ensure_ascii=False, indent=2)
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
