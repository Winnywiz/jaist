"""
compare entry point.

Runs the PROPOSED dynamic method against the STATIC baselines on failure-attribution
accuracy, and writes the result to compare/result/.

Run from the repo root (D:\\intershippu):

    python -m compare.run --dataset multihoprag --n 20        # real run (needs API key)
    python -m compare.run --dataset multihoprag --n 8 --offline   # plumbing smoke test
"""
from compare.shared.harness import main

if __name__ == "__main__":
    main()
