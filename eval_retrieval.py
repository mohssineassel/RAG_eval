"""
For every question with "evidence" in the dataset (a phrase from the PDF that contains
the answer), check where the chunk holding that phrase ranks in the search results:

"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_PATH = ROOT / "eval_dataset.jsonl"
RESULTS_DIR = ROOT / "experiments" / "retrieval"


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def gold_rank(retrieved: list[str], evidence: list[str]) -> int | None:
    phrases = [normalize(e) for e in evidence]
    for rank, chunk in enumerate(retrieved, start=1):
        if any(p in normalize(chunk) for p in phrases):
            return rank
    return None


def hit_and_mrr(ranks: list[int | None], k: int) -> tuple[float, float]:
    hits = [r is not None and r <= k for r in ranks]
    reciprocal = [1 / r if hit else 0.0 for r, hit in zip(ranks, hits)]
    return mean(hits), mean(reciprocal)


def evaluate_embeddings(provider: str, dataset: list[dict], ks: list[int]) -> dict:
    from RAG1.vectorstore import build_vector_store

    vector_store = build_vector_store(provider)
    rows = []
    for row in dataset:
        docs = vector_store.similarity_search(row["question"], k=max(ks))
        rows.append({"question": row["question"], "rank": gold_rank([d.page_content for d in docs], row["evidence"])})

    ranks = [r["rank"] for r in rows]
    scores = {}
    for k in ks:
        hit_rate, mrr = hit_and_mrr(ranks, k)
        scores[f"k{k}"] = {"hit_rate": hit_rate, "mrr": mrr}
    return {"embeddings": provider, "number_of_questions": len(rows), "scores": scores, "per_question": rows}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval only (hit rate and MRR, no LLM).")
    parser.add_argument("--embeddings", nargs="+", default=[os.getenv("EMBEDDING_PROVIDER", "openai")],
                        choices=["openai", "gemini", "ollama"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    ks = sorted(args.k)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [r for r in (json.loads(line) for line in f if line.strip()) if r.get("evidence")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for provider in args.embeddings:
        result = evaluate_embeddings(provider, dataset, ks)
        with open(RESULTS_DIR / f"{provider}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        results.append(result)

    print(f"\n =========== RETRIEVAL ({len(dataset)} questions with evidence) =========")
    header = f"{'Embeddings':<12}" + "".join(f"{'hit@' + str(k):<9}{'mrr@' + str(k):<9}" for k in ks)
    print(header)
    print("-" * len(header))
    for result in results:
        print(f"{result['embeddings']:<12}" + "".join(
            f"{result['scores'][f'k{k}']['hit_rate']:<9.2f}{result['scores'][f'k{k}']['mrr']:<9.2f}" for k in ks))

    # which questions the retriever gets wrong
    for result in results:
        misses = [r for r in result["per_question"] if r["rank"] != 1]
        if misses:
            print(f"\n{result['embeddings']}: right chunk NOT ranked first for")
            for r in misses:
                where = f"rank {r['rank']}" if r["rank"] else f"not in top {max(ks)}"
                print(f"  - {r['question']} ({where})")


if __name__ == "__main__":
    main()
