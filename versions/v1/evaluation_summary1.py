import json
from statistics import mean

with open("evaluated_results1_fastOne.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

metrics = ["answer_relevancy_score", "faithfulness_score", "contextual_relevancy_score", "correctness_score"]

print("\n =========== RAG EVAL SUMMARY =========")

threshold = 0.7

for metric in metrics:

    scores = [result[metric] for result in results ]
    average_score = mean(scores) if scores else None
    passed = sum(score >= threshold for score in scores)
    failed = sum(score < threshold for score in scores)

    pass_rate = (passed / len(scores)) * 100 

    print(metric.replace("_", " ").title())
    print(f"Average Score: {average_score:.2f}")
    print(f"Passed: {passed}/{len(scores)}")
    print(f"Failed: {failed}/{len(scores)}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    print()

