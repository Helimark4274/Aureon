"""Usage:
  python examples/run_eval.py --csv examples/questions_template.csv --backend mock          # test pipeline, no API
  python examples/run_eval.py --csv my_questions.csv --backend anthropic --yes               # real model (costs money)
"""
import argparse, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aureon.evalkit import load_csv, run, report

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
p.add_argument("--model", default="claude-sonnet-4-6")
p.add_argument("--n", type=int, default=5, help="candidates per question")
p.add_argument("--threshold", type=float, default=0.7)
p.add_argument("--train-frac", type=float, default=0.5)
p.add_argument("--exact", action="store_true", help="require exact match instead of 'truth contained in answer'")
p.add_argument("--out", default="RESULTS_REAL.md")
p.add_argument("--yes", action="store_true", help="confirm API spend")
a = p.parse_args()

rows = load_csv(a.csv)
if len(rows) < 10:
    sys.exit(f"Only {len(rows)} usable rows in {a.csv}; need at least 10 (aim for 200+).")
calls = len(rows) * a.n
if a.backend == "anthropic":
    print(f"{len(rows)} questions x {a.n} candidates = {calls} API calls (sequential, so this will take a while).")
    if not a.yes:
        sys.exit("Re-run with --yes to proceed.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    from aureon.llm_anthropic import anthropic_generator
    gen, pass_truth = anthropic_generator(model=a.model), False
else:
    from aureon.sim import make_generator
    print("MOCK backend: synthetic answers. Numbers say nothing about a real model.")
    gen, pass_truth = make_generator(0, defaultdict(lambda: 0.7)), True

res = run(rows, gen, a.train_frac, 0, a.threshold, a.n, a.exact, pass_truth)
md = report(*res)
open(a.out, "w", encoding="utf-8").write(md)
print(md)
print(f"\nSaved to {a.out}")
