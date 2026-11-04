import json
from statistics import mean


with open("evaluated_results3_fastOne.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

experiment = {
    "experiment": "experiment_v3_k2",
    "retreiver_k" : 2,
    "number_of_test_cases": len(results),

    "metrics": {
        "answer_relevancy": mean([result["answer_relevancy_score"] for result in results]),
        "faithfulness": mean([result["faithfulness_score"] for result in results]),
        "contextual_relevancy": mean([result["contextual_relevancy_score"] for result in results]),
        "correctness": mean([result["correctness_score"] for result in results])
    }

}

with open("experiment_v3_k2.jsonl", "w", encoding="utf-8") as f:
    json.dump(experiment, f, indent=4, ensure_ascii=False)

print("Experiment summary saved to experiment_v3_k2.jsonl.")

