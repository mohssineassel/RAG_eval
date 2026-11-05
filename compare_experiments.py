"""
Compare every experiment saved in experiments/<name>/summary.json.

    uv run python compare_experiments.py              # all experiments
    uv run python compare_experiments.py k1 k2        # only names containing "k1" or "k2"
"""
import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"

filters = sys.argv[1:]
summaries = []
for path in sorted(EXPERIMENTS_DIR.glob("*/summary.json")):
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if not filters or any(word in summary["experiment"] for word in filters):
        summaries.append(summary)

if not summaries:
    sys.exit(f"No experiments found in {EXPERIMENTS_DIR}")

names = [s["experiment"] for s in summaries]
width = max(12, *(len(n) + 2 for n in names))

print("\n =========== EXPERIMENT COMPARISON SUMMARY =========")
print(f"{'Metric':<22}" + "".join(f"{n:<{width}}" for n in names))
print("-" * (22 + width * len(names)))

for metric in summaries[0]["metrics"]:
    row = f"{metric.replace('_', ' ').title():<22}"
    for s in summaries:
        score = s["metrics"].get(metric)
        row += f"{'-' if score is None else f'{score:.2f}':<{width}}"
    print(row)

print(f"{'Test cases':<22}" + "".join(f"{s['number_of_test_cases']:<{width}}" for s in summaries))
