
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
print(f"{'Metric':<26}" + "".join(f"{n:<{width}}" for n in names))
print("-" * (26 + width * len(names)))

all_metrics = list(dict.fromkeys(m for s in summaries for m in s["metrics"]))

for metric in all_metrics:
    row = f"{metric.replace('_', ' ').title():<26}"
    for s in summaries:
        score = s["metrics"].get(metric)
        row += f"{'-' if score is None else f'{score:.2f}':<{width}}"
    print(row)

print(f"{'Test cases':<26}" + "".join(f"{s['number_of_test_cases']:<{width}}" for s in summaries))
print(f"{'Trick questions':<26}" + "".join(f"{s.get('number_of_trick_questions', 0):<{width}}" for s in summaries))
