"""
Multi-turn failure localisation.

Answers "did this failure come from the multi-turn setting, or would the turn have
failed on its own?". The first three analyses are *observational* (they read only the
result JSONs already on disk, no API calls); the fourth is *interventional* and needs
runs produced with ``run_benchmark --counterfactual``.

Four analyses, all reported per (dataset, rag) and pooled:

  1. POSITION EFFECT   accuracy as a function of turn depth. If turn 0 is healthy
                       and accuracy decays with depth, the conversation itself is
                       the stressor, not retrieval.
  2. TYPE EFFECT       accuracy on history-DEPENDENT query types (the turn cannot
                       be understood without the conversation) versus SELF-CONTAINED
                       ones. Failure concentrated in the former is multi-turn
                       evidence; failure spread evenly points at retrieval/generation.
  3. PROPAGATION       P(correct at t | previous turn correct) versus
                       P(correct at t | previous turn failed). A large gap means
                       errors cascade through the history, i.e. a turn is being
                       blamed for a failure it inherited.
  4. COUNTERFACTUAL    (interventional) re-answer each turn with the CLEAN gold history
                       instead of the RAG's own. Only the history changes, so a failure
                       that flips to correct was CAUSED by the conversation (INHERITED),
                       and one that persists is INTRINSIC to the turn. Confirms analysis
                       3 causally and yields a McNemar's test. Needs --counterfactual runs.

CAVEAT baked into the output: the generator is *adaptive* — the outcome of turn t
chooses the query type of turn t+1 (adaptive_generator._choose_type). So depth and
query type are confounded, and conditioning on depth partly conditions on past
success. Analysis 1 is therefore also reported WITHIN a single query type, which
removes the type-mix confound (the selection effect remains — see README note).

Usage:
    python -m conv_rag_benchmark.analyze_multiturn
    python -m conv_rag_benchmark.analyze_multiturn --root result/benchmark_quality \
        --out result/benchmark_quality/multiturn_localisation.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

#: Query types whose question is only interpretable given the conversation.
#: These probe the *conversational* capability (taxonomy category "Conversation").
HISTORY_DEPENDENT = {
    "Ambiguous Reference",
    "Follow-Up",
    "Clarification",
    "Correction",
    "Topic Shift",
}

#: Query types that are answerable as a standalone question — they probe retrieval
#: and reasoning, not the conversation.
SELF_CONTAINED = {
    "Seed",
    "Multi-Hop",
    "Comparative",
    "Temporal",
}

#: Graded on abstention, not on content — kept out of both groups so a correct
#: refusal is never counted as a conversational failure.
EXCLUDED_TYPES = {"Unanswerable"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_runs(root: Path) -> List[Dict]:
    """Every result JSON under *root* that carries per-turn conversations."""
    runs = []
    for path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or "conversations" not in data:
            continue  # graph_ragtest.json / raptor_tree.json / *_docsim.json
        runs.append({
            "path": path,
            "dataset": path.parent.name,
            "rag": data.get("rag", "?"),
            "conversations": data["conversations"],
        })
    return runs


def iter_turns(runs: Iterable[Dict]) -> Iterable[Tuple[Dict, Dict, Dict]]:
    """Yield (run, conversation, turn) for every turn in every run."""
    for run in runs:
        for conv in run["conversations"]:
            for turn in conv.get("turns", []):
                yield run, conv, turn


def is_correct(turn: Dict) -> bool:
    return turn.get("outcome") == "correct"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval — honest at the small n these runs have."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Pooled two-proportion z-test -> (z, two-sided p). p via a normal-CDF approx."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # two-sided p from the error function
    return (z, math.erfc(abs(z) / math.sqrt(2)))


def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """McNemar's test for paired binary outcomes, continuity-corrected.

    ``b`` and ``c`` are the DISCORDANT counts (pairs that agree carry no signal). Returns
    (chi-square, two-sided p) with 1 degree of freedom; the survival function of a
    chi-square_1 is ``erfc(sqrt(chi2/2))``.
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / n
    return (chi2, math.erfc(math.sqrt(chi2 / 2)))


def rate(k: int, n: int) -> Dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n,
            "rate": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


# --------------------------------------------------------------------------- #
# 1. position effect
# --------------------------------------------------------------------------- #
def position_effect(turns: List[Tuple[Dict, Dict, Dict]],
                    only_type: str | None = None) -> Dict:
    """Accuracy per turn depth, optionally restricted to one query type."""
    by_depth: Dict[int, List[bool]] = defaultdict(list)
    for _, _, turn in turns:
        if only_type and turn.get("query_type") != only_type:
            continue
        if turn.get("query_type") in EXCLUDED_TYPES:
            continue
        by_depth[turn.get("turn_id", -1)].append(is_correct(turn))
    depths = sorted(d for d in by_depth if d >= 0)
    curve = {d: rate(sum(by_depth[d]), len(by_depth[d])) for d in depths}

    # first third vs last third of the conversation, as a single testable contrast
    if depths:
        cut = max(1, len(depths) // 3)
        early = [ok for d in depths[:cut] for ok in by_depth[d]]
        late = [ok for d in depths[-cut:] for ok in by_depth[d]]
        z, p = two_proportion_z(sum(early), len(early), sum(late), len(late))
        contrast = {
            "early": rate(sum(early), len(early)),
            "late": rate(sum(late), len(late)),
            "delta": round(
                (sum(early) / len(early) if early else 0)
                - (sum(late) / len(late) if late else 0), 4),
            "z": round(z, 3), "p": round(p, 5),
        }
    else:
        contrast = {}
    return {"curve": curve, "early_vs_late": contrast,
            "restricted_to": only_type or "all types"}


# --------------------------------------------------------------------------- #
# 2. type effect
# --------------------------------------------------------------------------- #
def type_effect(turns: List[Tuple[Dict, Dict, Dict]]) -> Dict:
    per_type: Dict[str, List[bool]] = defaultdict(list)
    outcomes: Dict[str, Counter] = defaultdict(Counter)
    for _, _, turn in turns:
        qt = turn.get("query_type", "?")
        per_type[qt].append(is_correct(turn))
        outcomes[qt][turn.get("outcome", "?")] += 1

    dep = [ok for qt, v in per_type.items() if qt in HISTORY_DEPENDENT for ok in v]
    ind = [ok for qt, v in per_type.items() if qt in SELF_CONTAINED for ok in v]
    z, p = two_proportion_z(sum(dep), len(dep), sum(ind), len(ind))

    return {
        "per_type": {qt: {**rate(sum(v), len(v)), "outcomes": dict(outcomes[qt])}
                     for qt, v in sorted(per_type.items())},
        "history_dependent": rate(sum(dep), len(dep)),
        "self_contained": rate(sum(ind), len(ind)),
        "delta": round((sum(ind) / len(ind) if ind else 0)
                       - (sum(dep) / len(dep) if dep else 0), 4),
        "z": round(z, 3), "p": round(p, 5),
    }


def type_effect_clean_history(turns_by_conv: Dict[Tuple, List[Dict]]) -> Dict:
    """Analysis 2, restricted to turns whose PREVIOUS turn was answered correctly.

    Necessary because the adaptive policy ties query type to the previous outcome
    almost deterministically (Correction/Follow-Up only ever follow a failure,
    Ambiguous Reference/Topic Shift only ever follow a success). Comparing the raw
    groups therefore mixes "this capability is hard" with "this turn inherited a
    poisoned history". Conditioning on a clean history removes the second effect,
    so the remaining gap is attributable to the conversational capability itself.
    """
    dep: List[bool] = []
    ind: List[bool] = []
    per_type: Dict[str, List[bool]] = defaultdict(list)
    for turns in turns_by_conv.values():
        ordered = sorted(turns, key=lambda t: t.get("turn_id", 0))
        for prev, cur in zip(ordered, ordered[1:]):
            if not is_correct(prev) or cur.get("query_type") in EXCLUDED_TYPES:
                continue
            qt = cur.get("query_type", "?")
            per_type[qt].append(is_correct(cur))
            if qt in HISTORY_DEPENDENT:
                dep.append(is_correct(cur))
            elif qt in SELF_CONTAINED:
                ind.append(is_correct(cur))
    z, p = two_proportion_z(sum(dep), len(dep), sum(ind), len(ind))
    return {
        "per_type": {qt: rate(sum(v), len(v)) for qt, v in sorted(per_type.items())},
        "history_dependent": rate(sum(dep), len(dep)),
        "self_contained": rate(sum(ind), len(ind)),
        "delta": round((sum(ind) / len(ind) if ind else 0)
                       - (sum(dep) / len(dep) if dep else 0), 4),
        "z": round(z, 3), "p": round(p, 5),
    }


# --------------------------------------------------------------------------- #
# 3. propagation
# --------------------------------------------------------------------------- #
def propagation(turns_by_conv: Dict[Tuple, List[Dict]]) -> Dict:
    """P(correct at t | prev correct) vs P(correct at t | prev failed).

    Reported twice. The raw contrast is confounded: ``_choose_type`` picks the next
    query type FROM the outcome, so a post-failure turn is also a different (harder)
    type. The stratified figure conditions on the current turn's type and pools the
    per-type differences weighted by stratum size (Cochran–Mantel–Haenszel style),
    which removes that confound. Conversation-level difficulty is NOT controlled —
    both remain upper bounds until the truth_history re-run.
    """
    after_ok: List[bool] = []
    after_bad: List[bool] = []
    strata: Dict[str, Tuple[List[bool], List[bool]]] = defaultdict(lambda: ([], []))
    for turns in turns_by_conv.values():
        ordered = sorted(turns, key=lambda t: t.get("turn_id", 0))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.get("query_type") in EXCLUDED_TYPES:
                continue
            ok, bad = strata[cur.get("query_type", "?")]
            (after_ok if is_correct(prev) else after_bad).append(is_correct(cur))
            (ok if is_correct(prev) else bad).append(is_correct(cur))

    z, p = two_proportion_z(sum(after_ok), len(after_ok),
                            sum(after_bad), len(after_bad))

    # size-weighted mean of the within-type differences
    per_type, num, den = {}, 0.0, 0.0
    for qt, (ok, bad) in sorted(strata.items()):
        if not ok or not bad:
            continue
        d = sum(ok) / len(ok) - sum(bad) / len(bad)
        w = len(ok) + len(bad)
        per_type[qt] = {"after_correct": rate(sum(ok), len(ok)),
                        "after_failure": rate(sum(bad), len(bad)),
                        "delta": round(d, 4)}
        num += d * w
        den += w

    return {
        "after_correct": rate(sum(after_ok), len(after_ok)),
        "after_failure": rate(sum(after_bad), len(after_bad)),
        "delta": round((sum(after_ok) / len(after_ok) if after_ok else 0)
                       - (sum(after_bad) / len(after_bad) if after_bad else 0), 4),
        "z": round(z, 3), "p": round(p, 5),
        "delta_stratified_by_type": round(num / den, 4) if den else None,
        "per_type": per_type,
    }


# --------------------------------------------------------------------------- #
# 4. counterfactual (the interventional half)
# --------------------------------------------------------------------------- #
def counterfactual_effect(turns: List[Tuple[Dict, Dict, Dict]]) -> Dict | None:
    """Paired real-vs-counterfactual outcomes from the Mode-1 probe.

    Each non-seed turn carries ``cf_outcome``: the SAME question re-answered with the
    clean gold history instead of the RAG's own (possibly poisoned) one. Comparing it to
    the real ``outcome`` turns analysis 3 from correlational into CAUSAL — the only thing
    that changed is the history, so a failure that flips to correct was *caused* by the
    conversation, not the retriever or the generator.

    Returns None when no turn carries a counterfactual (runs made without
    ``--counterfactual``), so old result files are simply skipped.

    2x2 paired table (real x counterfactual):
        a = real correct, cf correct       b = real correct, cf FAILED  (clean history hurt)
        c = real FAILED, cf correct        d = real FAILED, cf failed
    Of the real failures (c + d): ``c`` are INHERITED (the clean history fixes them) and
    ``d`` are INTRINSIC. McNemar's test runs on the discordant pairs (b, c).
    """
    a = b = c = d = 0
    for _, _, turn in turns:
        cf = turn.get("cf_outcome")
        if cf is None:
            continue
        if turn.get("query_type") in EXCLUDED_TYPES:
            continue
        real_ok = is_correct(turn)
        cf_ok = cf == "correct"
        if real_ok and cf_ok:
            a += 1
        elif real_ok and not cf_ok:
            b += 1
        elif not real_ok and cf_ok:
            c += 1
        else:
            d += 1
    n_paired = a + b + c + d
    if n_paired == 0:
        return None
    chi2, p = mcnemar(b, c)
    real_failures = c + d
    return {
        "n_paired": n_paired,
        "table": {"real_ok_cf_ok": a, "real_ok_cf_fail": b,
                  "real_fail_cf_ok": c, "real_fail_cf_fail": d},
        "n_real_failures": real_failures,
        "inherited": rate(c, real_failures),   # clean history flips the failure to correct
        "intrinsic": rate(d, real_failures),   # failure survives the clean history
        "clean_history_regressions": b,        # clean history broke a real success
        "mcnemar_chi2": round(chi2, 3),
        "mcnemar_p": round(p, 5),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def analyse(turns: List[Tuple[Dict, Dict, Dict]]) -> Dict:
    by_conv: Dict[Tuple, List[Dict]] = defaultdict(list)
    for run, conv, turn in turns:
        by_conv[(str(run["path"]), conv.get("conversation_id"))].append(turn)

    # the most frequent type is the one worth a within-type depth curve
    common = Counter(t.get("query_type") for _, _, t in turns
                     if t.get("query_type") not in EXCLUDED_TYPES)
    top_type = common.most_common(1)[0][0] if common else None

    return {
        "n_turns": len(turns),
        "outcomes": dict(Counter(t.get("outcome") for _, _, t in turns)),
        "position_effect": position_effect(turns),
        "position_effect_within_type": position_effect(turns, only_type=top_type),
        "type_effect": type_effect(turns),
        "type_effect_clean_history": type_effect_clean_history(by_conv),
        "propagation": propagation(by_conv),
        "counterfactual": counterfactual_effect(turns),
    }


def _fmt(r: Dict) -> str:
    if not r or r.get("n", 0) == 0:
        return "     -    "
    return f"{r['rate']:.3f} (n={r['n']})"


def print_report(report: Dict) -> None:
    p = report["pooled"]
    print("=" * 78)
    print(f"MULTI-TURN FAILURE LOCALISATION — {p['n_turns']} turns pooled")
    print(f"outcomes: {p['outcomes']}")
    print("=" * 78)

    print("\n[1] POSITION EFFECT — accuracy by turn depth (all types)")
    for d, r in p["position_effect"]["curve"].items():
        bar = "#" * int(round((r["rate"] or 0) * 40))
        print(f"  turn {d:>2}  {_fmt(r):<16} {bar}")
    c = p["position_effect"]["early_vs_late"]
    if c:
        print(f"  early {_fmt(c['early'])} vs late {_fmt(c['late'])}  "
              f"delta={c['delta']:+.3f}  z={c['z']}  p={c['p']}")

    w = p["position_effect_within_type"]
    print(f"\n[1b] POSITION EFFECT within a single type ({w['restricted_to']}) "
          f"— removes the adaptive type-mix confound")
    for d, r in w["curve"].items():
        print(f"  turn {d:>2}  {_fmt(r)}")
    c = w["early_vs_late"]
    if c:
        print(f"  early {_fmt(c['early'])} vs late {_fmt(c['late'])}  "
              f"delta={c['delta']:+.3f}  z={c['z']}  p={c['p']}")

    t = p["type_effect"]
    print("\n[2] TYPE EFFECT — accuracy per query type")
    for qt, r in t["per_type"].items():
        group = ("history-dependent" if qt in HISTORY_DEPENDENT
                 else "self-contained" if qt in SELF_CONTAINED else "excluded")
        print(f"  {qt:<22} {_fmt(r):<16} [{group}]  {r['outcomes']}")
    print(f"  history-dependent {_fmt(t['history_dependent'])} vs "
          f"self-contained {_fmt(t['self_contained'])}")
    print(f"  delta={t['delta']:+.3f} (self-contained minus history-dependent)  "
          f"z={t['z']}  p={t['p']}")

    ch = p["type_effect_clean_history"]
    print("\n[2b] TYPE EFFECT on a CLEAN history (previous turn was correct)")
    print("     — the adaptive policy ties type to the previous outcome, so only this")
    print("       contrast separates capability difficulty from inherited failure")
    for qt, r in ch["per_type"].items():
        group = ("history-dependent" if qt in HISTORY_DEPENDENT
                 else "self-contained" if qt in SELF_CONTAINED else "-")
        print(f"  {qt:<22} {_fmt(r):<16} [{group}]")
    print(f"  history-dependent {_fmt(ch['history_dependent'])} vs "
          f"self-contained {_fmt(ch['self_contained'])}")
    print(f"  delta={ch['delta']:+.3f}  z={ch['z']}  p={ch['p']}")

    g = p["propagation"]
    print("\n[3] PROPAGATION — does a failed turn poison the next one?")
    print(f"  P(correct | previous turn correct) = {_fmt(g['after_correct'])}")
    print(f"  P(correct | previous turn failed)  = {_fmt(g['after_failure'])}")
    print(f"  delta={g['delta']:+.3f}  z={g['z']}  p={g['p']}")
    if g.get("delta_stratified_by_type") is not None:
        print(f"  delta stratified by query type = {g['delta_stratified_by_type']:+.3f}"
              "   <- controls for the adaptive type switch after a failure")
        for qt, s in g["per_type"].items():
            print(f"     {qt:<22} after-ok {_fmt(s['after_correct']):<16} "
                  f"after-fail {_fmt(s['after_failure']):<16} {s['delta']:+.3f}")

    cf = p.get("counterfactual")
    if cf:
        print("\n[4] COUNTERFACTUAL — re-answer each turn with the CLEAN gold history")
        print("     — the INTERVENTIONAL half: only the history changes, so a failure that")
        print("       flips to correct was CAUSED by the conversation, not retrieval/gen")
        t = cf["table"]
        print(f"  paired turns={cf['n_paired']}   real failures={cf['n_real_failures']}")
        print(f"  2x2  real-ok/cf-ok={t['real_ok_cf_ok']:<5} "
              f"real-ok/cf-FAIL={t['real_ok_cf_fail']:<5} "
              f"real-FAIL/cf-ok={t['real_fail_cf_ok']:<5} "
              f"real-FAIL/cf-fail={t['real_fail_cf_fail']}")
        print(f"  INHERITED (clean history fixes it)   {_fmt(cf['inherited'])}")
        print(f"  INTRINSIC (fails even on clean hist)  {_fmt(cf['intrinsic'])}")
        print(f"  McNemar chi2={cf['mcnemar_chi2']}  p={cf['mcnemar_p']}  "
              f"(clean-history regressions={cf['clean_history_regressions']})")

    print("\n" + "-" * 78)
    print("PER (dataset, rag) — history-dependent / self-contained / propagation delta")
    print("-" * 78)
    print(f"  {'dataset':<14} {'rag':<10} {'hist-dep':<16} {'self-cont':<16} {'prop':>7}")
    for key, sub in sorted(report["by_run"].items()):
        ds, rag = key.split("|")
        te, pr = sub["type_effect"], sub["propagation"]
        print(f"  {ds:<14} {rag:<10} {_fmt(te['history_dependent']):<16} "
              f"{_fmt(te['self_contained']):<16} {pr['delta']:>+7.3f}")

    print("\nHOW TO READ THIS")
    print("  [2] large positive delta -> failures concentrate in turns that need the")
    print("      conversation: the multi-turn stage, not retrieval or the generator.")
    print("  [2] delta near zero      -> the conversation is not the discriminator;")
    print("      attribute with the retrieval/generation split instead (2x2 on")
    print("      gold_retrieved x outcome).")
    print("  [3] large positive delta -> errors cascade; a failing turn is partly")
    print("      inheriting a poisoned history. Confirm causally with [4].")
    print("  [4] high INHERITED share + significant McNemar -> the multi-turn setting")
    print("      itself is a failure source: the clean-history rerun repairs failures")
    print("      that the poisoned conversation caused. High INTRINSIC -> retrieval/")
    print("      generation is the culprit and the conversation is not to blame.")
    print("  [1] decay with depth that SURVIVES [1b] -> genuine depth degradation.")
    print("      Decay only in [1] -> an artefact of the adaptive type mix.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="result/benchmark_quality", type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="write the full report as JSON")
    args = ap.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"no result JSONs with conversations under {args.root}")
    all_turns = list(iter_turns(runs))

    by_run: Dict[str, Dict] = {}
    grouped: Dict[str, List] = defaultdict(list)
    for run, conv, turn in all_turns:
        grouped[f"{run['dataset']}|{run['rag']}"].append((run, conv, turn))
    for key, turns in grouped.items():
        by_run[key] = analyse(turns)

    report = {"root": str(args.root), "n_runs": len(runs),
              "pooled": analyse(all_turns), "by_run": by_run}
    print_report(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
