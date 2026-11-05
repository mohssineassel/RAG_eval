"""
Run a full RAG experiment: generate answers -> evaluate them -> save a summary.

Examples:
    uv run python run_experiment.py --k 2 --provider groq --embeddings gemini --name my_test

Each run is saved in experiments/<name>/ (answers.jsonl, evaluated.jsonl, summary.json).
"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import RAG1.tools as tl
from RAG1.agent import ask

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_PATH = ROOT / "eval_dataset.jsonl"
EXPERIMENTS_DIR = ROOT / "experiments"
THRESHOLD = 0.7
METRICS = ["answer_relevancy", "faithfulness", "contextual_relevancy", "correctness"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_answers(dataset: list[dict], k: int) -> list[dict]:
    """Step 1: ask the RAG agent every question in the dataset."""
    

    tl.k = k  

    results = []
    for i, test_case in enumerate(dataset, start=1):
        question = test_case["question"]
        print(f"[{i}/{len(dataset)}] Question: {question}")
        rag_result = ask(question)

        if not rag_result["tool_called"]:
            print("  ⚠️ Agent answered without retrieving anything")
        print(f"  RAG Answer: {rag_result['answer']}")

        results.append({
            "question": question,
            "expected_answer": test_case["expected_answer"],
            "rag_answer": rag_result["answer"],
            "rag_context": rag_result["context"],
            "tool_called": rag_result["tool_called"],
        })
    return results


def evaluate_answers(results: list[dict], provider: str) -> list[dict]:
    """Step 2: score every answer with DeepEval."""
    from deepeval.evaluate import AsyncConfig, CacheConfig, evaluate
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    from RAG1.judge import get_judge

    judge = get_judge(provider)

    test_cases = []
    for result in results:
        context = result["rag_context"]
        test_cases.append(LLMTestCase(
            input=result["question"],
            actual_output=result["rag_answer"],
            expected_output=result["expected_answer"],
            retrieval_context=[context] if isinstance(context, str) else context,
        ))

    metrics = [
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge),
        GEval(
            name="Correctness",
            criteria="Determine whether the model's response is factually correct and aligns with the expected answer.",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            threshold=THRESHOLD,
            model=judge,
        ),
    ]

    evaluation = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        cache_config=CacheConfig(use_cache=False, write_cache=False),
        # gemini/groq and local ollama can't handle parallel metric calls
        async_config=AsyncConfig(run_async=provider == "openai"),
    )

    
    scores_by_question = {}
    for test_result in evaluation.test_results:
        scores = {f"{m}_score": None for m in METRICS}
        for metric in test_result.metrics_data or []:
            name = metric.name.strip().lower()
            if name.startswith("correctness"):
                name = "correctness"
            scores[f"{name.replace(' ', '_')}_score"] = metric.score
        scores_by_question[test_result.input] = scores

    return [{**result, **scores_by_question.get(result["question"], {})} for result in results]


def summarize(evaluated: list[dict], config: dict) -> dict:
    """Step 3: average score and pass rate for each metric."""
    metrics, pass_rates = {}, {}
    for metric in METRICS:
        scores = [r[f"{metric}_score"] for r in evaluated if r.get(f"{metric}_score") is not None]
        metrics[metric] = mean(scores) if scores else None
        pass_rates[metric] = sum(s >= THRESHOLD for s in scores) / len(scores) if scores else None

    return {
        "experiment": config["name"],
        "config": config,
        "number_of_test_cases": len(evaluated),
        "metrics": metrics,
        "pass_rates": pass_rates,
    }


def print_summary(summary: dict):
    print(f"\n =========== {summary['experiment']} ({summary['number_of_test_cases']} test cases) =========")
    for metric in METRICS:
        score, rate = summary["metrics"][metric], summary["pass_rates"][metric]
        if score is None:
            print(f"{metric.replace('_', ' ').title():<22} no scores")
        else:
            print(f"{metric.replace('_', ' ').title():<22} avg {score:.2f}   pass rate {rate:.0%}")


def run(name: str, k: int, provider: str, embeddings: str, limit: int | None):
    out_dir = EXPERIMENTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_jsonl(DATASET_PATH)
    dataset = dataset[:limit] if limit else dataset

    results = generate_answers(dataset, k)
    write_jsonl(out_dir / "answers.jsonl", results)

    evaluated = evaluate_answers(results, provider)
    write_jsonl(out_dir / "evaluated.jsonl", evaluated)

    config = {"name": name, "retriever_k": k, "provider": provider, "embeddings": embeddings, "limit": limit}
    summary = summarize(evaluated, config)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments.")
    parser.add_argument("--k", type=int, nargs="+", default=[2], help="retriever k (several values = several experiments)")
    parser.add_argument("--provider", default=os.getenv("PROVIDER", "openai"), choices=["openai", "groq", "gemini", "ollama"])
    parser.add_argument("--embeddings", default=os.getenv("EMBEDDING_PROVIDER", "openai"), choices=["openai", "gemini", "ollama"])
    parser.add_argument("--limit", type=int, default=None, help="only use the first N questions")
    parser.add_argument("--name", default=None, help="experiment name (default: <provider>_<embeddings>_k<k>)")
    args = parser.parse_args()

    # RAG1 reads these when it is imported, so set them before the first import
    os.environ["PROVIDER"] = args.provider
    os.environ["EMBEDDING_PROVIDER"] = args.embeddings

    for k in args.k:
        name = args.name or f"{args.provider}_{args.embeddings}_k{k}"
        if args.name and len(args.k) > 1:
            name = f"{args.name}_k{k}"
        run(name, k, args.provider, args.embeddings, args.limit)


if __name__ == "__main__":
    main()
