"""
parallel_multiseed.py — run the attribution comparison over MANY seeds, in parallel.

`multiseed.py` runs seeds sequentially in-process. At ~14 min/seed measured, 40 seeds x
3 datasets = 120 runs = ~28h sequential. This spawns each (dataset, seed) as its own
subprocess across a worker pool, and RESUMES: any seed whose
`result/attribution_{ds}_seed{s}.json` already exists with a matching config is skipped,
so a crash at hour 4 does not restart from zero.

Aggregates into the SAME `result/multiseed_{dataset}.json` schema `multiseed.py` emits,
so `fair_macro.py` reads the output unchanged.

Why more seeds: the Conversation category fires only ~8-12 cases/seed, so with 3 seeds
its estimate swung [0.0, 0.25, 0.111] on qasper — uninterpretable. Seeds are the unit of
independence here (each shuffles WHICH documents get injected).

Run from repo root:
    python -m DYNAMICQA.parallel_multiseed --datasets qasper multihoprag hfdocqa \
        --seeds 0 39 --n 20 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, pstdev
from threading import Lock
from typing import Dict, List, Optional

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "result", "attribution")
CATS = ["Retrieval", "Generation", "Conversation"]

_print_lock = Lock()
_done = 0
_total = 0
_t0 = 0.0


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _path(ds: str, seed: int) -> str:
    return os.path.join(RESULT_DIR, f"attribution_{ds}_seed{seed}.json")


def already_done(ds: str, seed: int, n: int) -> bool:
    """True if this seed ran with a matching config and a real (non-offline) LLM."""
    p = _path(ds, seed)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            c = json.load(f).get("config", {})
        return (c.get("dataset") == ds and c.get("n_seeds") == n
                and c.get("llm") is True)
    except Exception:
        return False


def run_one(ds: str, seed: int, n: int) -> Optional[str]:
    """Run one (dataset, seed) as a subprocess. Returns an error string, or None on success."""
    global _done
    t = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "DYNAMICQA.shared.harness",
         "--dataset", ds, "--n", str(n), "--seed", str(seed)],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    el = time.time() - t
    with _print_lock:
        _done += 1
        pct = 100.0 * _done / _total if _total else 0
        elapsed = (time.time() - _t0) / 60.0
        eta = (elapsed / _done) * (_total - _done) if _done else 0
        status = "ok " if r.returncode == 0 else "FAIL"
        print(f"[{_done:>3}/{_total}] {pct:5.1f}%  {status}  {ds:<12} seed={seed:<3} "
              f"{el/60:5.1f}m  | elapsed {elapsed:5.1f}m  ETA {eta:5.1f}m", flush=True)
    if r.returncode != 0:
        return f"{ds} seed={seed}: exit {r.returncode}\n{r.stderr[-400:]}"
    if not os.path.exists(_path(ds, seed)):
        return f"{ds} seed={seed}: no result file written (no failures fired?)"
    return None


def aggregate(ds: str, seeds: List[int], n: int) -> Optional[Dict]:
    """Rebuild multiseed_{ds}.json from the per-seed attribution files."""
    acc: Dict[str, Dict[str, List[float]]] = {}
    fired_per_seed = []
    used = []
    for s in seeds:
        p = _path(ds, s)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            report = json.load(f)["report"]
        methods = report["methods"]
        if not methods:
            continue
        used.append(s)
        first = next(iter(methods.values()))
        fired_per_seed.append({"seed": s, "fired": {
            c: first["by_category"].get(c, {}).get("n", 0) for c in CATS}})
        for m, r in methods.items():
            d = acc.setdefault(m, {k: [] for k in ["macro", "micro"] + CATS})
            bycat = r["by_category"]
            vals = [bycat[c]["acc"] for c in CATS if c in bycat]
            d["macro"].append(round(mean(vals), 3) if vals else 0.0)
            d["micro"].append(r["accuracy"])
            for c in CATS:
                if c in bycat:
                    d[c].append(bycat[c]["acc"])
    if not acc:
        return None
    out = {"dataset": ds, "n": n, "seeds": used, "fired_per_seed": fired_per_seed,
           "per_method": {m: {k: {"mean": round(mean(v), 3) if v else None,
                                 "sd": round(pstdev(v), 3) if len(v) > 1 else 0.0,
                                 "values": v}
                             for k, v in d.items()}
                          for m, d in acc.items()}}
    path = os.path.join(RESULT_DIR, f"multiseed_{ds}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log(f"  aggregated {ds}: {len(used)} seeds -> {path}")
    return out


def main(argv=None) -> None:
    global _total, _t0
    ap = argparse.ArgumentParser(description="parallel multi-seed attribution runner")
    ap.add_argument("--datasets", nargs="+", default=["qasper", "multihoprag", "hfdocqa"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 39],
                    metavar=("FIRST", "LAST"), help="inclusive seed range")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run seeds even if a matching result already exists")
    args = ap.parse_args(argv)

    os.makedirs(RESULT_DIR, exist_ok=True)
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))

    jobs = []
    skipped = 0
    for ds in args.datasets:
        for s in seeds:
            if not args.no_resume and already_done(ds, s, args.n):
                skipped += 1
                continue
            jobs.append((ds, s))

    _total = len(jobs)
    _t0 = time.time()
    print(f"# datasets={args.datasets} seeds={seeds[0]}..{seeds[-1]} n={args.n} "
          f"workers={args.workers}")
    print(f"# {len(jobs)} runs to do, {skipped} already done (resumed)")
    print(f"# at ~14 min/seed this is ~{len(jobs) * 14.1 / 60 / args.workers:.1f}h "
          f"wall-clock at {args.workers} workers\n")

    errors: List[str] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, ds, s, args.n): (ds, s) for ds, s in jobs}
            for fu in as_completed(futs):
                err = fu.result()
                if err:
                    errors.append(err)

    print(f"\n# all runs finished in {(time.time() - _t0)/60:.1f} min")
    if errors:
        print(f"# !! {len(errors)} runs FAILED:")
        for e in errors[:10]:
            print("   -", e.replace("\n", "\n     "))

    print("\n# aggregating ...")
    for ds in args.datasets:
        aggregate(ds, seeds, args.n)

    print("\n# done. Next:  python -m DYNAMICQA.fair_macro")


if __name__ == "__main__":
    main()
