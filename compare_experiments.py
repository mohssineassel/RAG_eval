import json

with open("experiment_v3_k2.jsonl", "r", encoding="utf-8") as f:
    k2 = json.load(f)

with open("experiment_v2_k1.jsonl", "r", encoding="utf-8") as f:
    k1 = json.load(f)

with open("experiment_v1_k3.jsonl", "r", encoding="utf-8") as f:
    k3 = json.load(f)


print("\n =========== EXPERIMENT COMPARISON SUMMARY =========")

print(
    f"{'Metric':<22}"
    f"{'k=1':<10}"
    f"{'k=2':<10}"
    f"{'k=3':<10}"    
)

print("-"* 50)

for metric in k3["metrics"]:
    score_k3 = k3["metrics"][metric]
    score_k2 = k2["metrics"][metric]
    score_k1 = k1["metrics"][metric]

    print(
        f"{metric.replace('_', ' ').title():<22} "
        f"{score_k3:<10.2f}"
        f"{score_k2:<10.2f}"
        f"{score_k1:<10.2f}"

    )