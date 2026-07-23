"""
Cross-RAG matrix: run methods D, E, F against several target RAGs and collect
what each METHOD found about each RAG (the failure profile). This answers
"test the RAG with D/E/F" — the RAG is the system under test; the method is the
tester. Question quality is a method property and is reported separately.

Run:  python -m conv_rag_benchmark.run_rag_matrix
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

PY = sys.executable
_ap = argparse.ArgumentParser()
_ap.add_argument("--convos", default="6")
_ap.add_argument("--turns", default="3")
_ap.add_argument("--rags", default="vector,selfrag,raptor,graph")
_ap.add_argument("--dch", default="300")
_ap.add_argument("--gates", action="store_true",
                 help="run with --quality-gate + --strict-gold on all methods")
_args, _ = _ap.parse_known_args()

RAGS = [r.strip() for r in _args.rags.split(",") if r.strip()]
CONVOS, TURNS, DCH, DATASET = _args.convos, _args.turns, _args.dch, "multihoprag"
OUT = os.path.join("conv_rag_benchmark", "output", "MultiHopRAG")
COMBINED = os.path.join(OUT, "rag_matrix_DEF_gated.json" if _args.gates
                        else "rag_matrix_DEF.json")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(mod, extra):
    subprocess.run([PY, "-u", "-m", mod, "--dataset", DATASET, *extra],
                   check=True, env=ENV)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def profile(d):
    """pull (failure_rate, outcomes) from a saved result json."""
    rf = d.get("rag_failure", {})
    return {"failure_rate": rf.get("failure_rate"),
            "outcomes": rf.get("outcomes", {}),
            "failures_by_probe": rf.get("failures_by_probe", {})}


def main():
    results = {}
    for rag in RAGS:
        print(f"\n########## RAG = {rag} ##########", flush=True)
        cell = {}
        gate_args = ["--quality-gate", "--strict-gold"] if _args.gates else []
        gate_sfx = "_strictgold_qgate" if _args.gates else ""
        # E (adaptive controller)
        run("conv_rag_benchmark.build_e_adaptive",
            ["--rag", rag, "--convos", CONVOS, "--turns", TURNS, "--d-chunks", DCH,
             *gate_args])
        e = load(os.path.join(OUT, f"quality_e{gate_sfx}.json"))
        cell["E"] = {"quality": e["quality"]["E"], **profile(e)}
        # D (random type)
        run("conv_rag_benchmark.build_e_adaptive",
            ["--rag", rag, "--convos", CONVOS, "--turns", TURNS, "--d-chunks", DCH,
             "--type-policy", "random", *gate_args])
        dd = load(os.path.join(OUT, f"quality_e_randomtype{gate_sfx}.json"))
        cell["D"] = {"quality": dd["quality"]["E"], **profile(dd)}
        # F (all-types)
        run("conv_rag_benchmark.build_alltypes",
            ["--rag", rag, "--convos", CONVOS, *gate_args])
        ff = load(os.path.join(OUT, "quality_alltypes.json"))
        cell["F"] = {"quality": ff["quality"], **profile(ff)}

        results[rag] = cell
        with open(COMBINED, "w", encoding="utf-8") as fw:      # incremental save
            json.dump(results, fw, ensure_ascii=False, indent=2)
        print(f"  saved partial -> {COMBINED}", flush=True)

    # ---- print the matrices ----
    print(f"\n\n=== RAG FAILURE RATE (convos/cell={CONVOS}) — each method's view ===")
    print(f"{'RAG':<10}{'D':>10}{'E':>10}{'F':>10}")
    for rag in RAGS:
        c = results[rag]
        print(f"{rag:<10}{str(c['D']['failure_rate']):>10}"
              f"{str(c['E']['failure_rate']):>10}{str(c['F']['failure_rate']):>10}")

    for metric in ("well_formed", "gold_supported", "gold_correct"):
        print(f"\n=== QUESTION QUALITY: {metric} (a method property) ===")
        print(f"{'RAG':<10}{'D':>10}{'E':>10}{'F':>10}")
        for rag in RAGS:
            c = results[rag]
            print(f"{rag:<10}{str(c['D']['quality'].get(metric)):>10}"
                  f"{str(c['E']['quality'].get(metric)):>10}"
                  f"{str(c['F']['quality'].get(metric)):>10}")
    print(f"\nSaved combined -> {COMBINED}")


if __name__ == "__main__":
    main()
