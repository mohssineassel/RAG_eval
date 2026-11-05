"""
Run a full RAG experiment: generate answers -> evaluate them -> save a summary.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_PATH = ROOT / "eval_dataset.jsonl"
EXPERIMENTS_DIR = ROOT / "experiments"
THRESHOLD = 0.7
METRICS = ["answer_relevancy", "faithfulness", "contextual_relevancy", "correctness", "hallucination_resistance"]
TRICK_TYPES = {"unanswerable", "false_premise"}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_trick(row: dict) -> bool:
    return row.get("type", "answerable") in TRICK_TYPES


def metric_applies(metric: str, row: dict) -> bool:
    if metric == "faithfulness":
        return True
    return is_trick(row) == (metric == "hallucination_resistance")


def is_fully_scored(row: dict) -> bool:
    return all(row.get(f"{m}_score") is not None for m in METRICS if metric_applies(m, row))


def load_previous(path: Path, resume: bool) -> dict[str, dict]:
    if not resume or not path.exists():
        return {}
    return {row["question"]: row for row in read_jsonl(path)}


def generate_answers(dataset: list[dict], k: int, path: Path, resume: bool) -> list[dict]:
    """Step 1: ask the RAG agent every question in the dataset (saved after each answer)."""

    import RAG1.tools as tl
    from RAG1.agent import ask

    tl.k = k  

    previous = load_previous(path, resume)
    if previous:
        print(f"Resuming: {len(previous)} answers already saved")

    results = []
    for i, test_case in enumerate(dataset, start=1):
        question = test_case["question"]
        if question in previous:
            results.append(previous[question])
            continue

        print(f"[{i}/{len(dataset)}] Question: {question}")
        rag_result = ask(question)

        if not rag_result["tool_called"]:
            print("  ⚠️ Agent answered without retrieving anything")
        print(f"  RAG Answer: {rag_result['answer']}")

        results.append({
            "question": question,
            "type": test_case.get("type", "answerable"),
            "expected_answer": test_case["expected_answer"],
            "rag_answer": rag_result["answer"],
            "rag_context": rag_result["context"],
            "tool_called": rag_result["tool_called"],
        })
        write_jsonl(path, results)
    return results


def evaluate_answers(results: list[dict], provider: str, path: Path, resume: bool) -> list[dict]:
    """Step 2: score every answer with DeepEval (saved after each test case)."""
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, evaluate
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    from RAG1.judge import get_judge

    judge = get_judge(provider)

    def to_test_case(result: dict) -> LLMTestCase:
        context = result["rag_context"]
        return LLMTestCase(
            input=result["question"],
            actual_output=result["rag_answer"],
            expected_output=result["expected_answer"],
    
            retrieval_context=([context] if isinstance(context, str) else context) or ["(nothing retrieved)"],
        )

    normal_metrics = [
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

    trick_metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge),
        GEval(
            name="Hallucination Resistance",
            evaluation_steps=[
                "The question is a trick question: either the retrieved context does not contain the answer, or the question contains a false premise.",
                "The expected output describes the ideal behavior: saying the information is not available, or rejecting and correcting the false premise.",
                "Give a high score if the actual output declines to answer or clearly rejects the false premise, without inventing facts that are not in the retrieval context.",
                "Give a low score if the actual output accepts the false premise or states facts (numbers, names, explanations) that are not supported by the retrieval context.",
            ],
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            threshold=THRESHOLD,
            model=judge,
        ),
    ]

    previous = load_previous(path, resume)
    already_done = {q: row for q, row in previous.items() if is_fully_scored(row)}
    if already_done:
        print(f"Resuming: {len(already_done)} test cases already scored")

    evaluated = []
    for i, result in enumerate(results, start=1):
        if result["question"] in already_done:
            evaluated.append(already_done[result["question"]])
            continue

        evaluation = evaluate(
            test_cases=[to_test_case(result)],
            metrics=trick_metrics if is_trick(result) else normal_metrics,
            cache_config=CacheConfig(use_cache=False, write_cache=False),
            # gemini/groq and local ollama can't handle parallel metric calls
            async_config=AsyncConfig(run_async=provider == "openai"),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        )

        scores = {f"{m}_score": None for m in METRICS}
        for test_result in evaluation.test_results:
            for metric in test_result.metrics_data or []:
                name = metric.name.replace("[GEval]", "").strip().lower()
                scores[f"{name.replace(' ', '_')}_score"] = metric.score
        row = {**result, **scores}
        evaluated.append(row)

        shown = ", ".join(f"{m} {row[f'{m}_score']:.2f}" for m in METRICS if row.get(f"{m}_score") is not None)
        missing = "" if is_fully_scored(row) else "   ⚠️ not fully scored, rerun with --resume"
        print(f"[{i}/{len(results)}] {result['question']} -> {shown}{missing}")

        # keep rows from the previous run that we haven't reached yet
        later = [previous[r["question"]] for r in results[i:] if r["question"] in previous]
        write_jsonl(path, evaluated + later)

    return evaluated


def summarize(evaluated: list[dict], config: dict) -> dict:
    """Step 3: average score and pass rate for each metric."""
    metrics, pass_rates, scored = {}, {}, {}
    for metric in METRICS:
        scores = [r[f"{metric}_score"] for r in evaluated if r.get(f"{metric}_score") is not None]
        metrics[metric] = mean(scores) if scores else None
        pass_rates[metric] = sum(s >= THRESHOLD for s in scores) / len(scores) if scores else None
        
        expected = sum(metric_applies(metric, r) for r in evaluated)
        scored[metric] = f"{len(scores)}/{expected}"

    
    ranks = [r["retrieval_rank"] for r in evaluated if "retrieval_rank" in r]
    if ranks:
        metrics["retrieval_hit_rate"] = mean(rank is not None for rank in ranks)
        metrics["retrieval_mrr"] = mean(1 / rank if rank else 0.0 for rank in ranks)

    return {
        "experiment": config["name"],
        "config": config,
        "number_of_test_cases": len(evaluated),
        "number_of_trick_questions": sum(is_trick(r) for r in evaluated),
        "metrics": metrics,
        "pass_rates": pass_rates,
        "scored": scored,
    }


def print_summary(summary: dict):
    print(f"\n =========== {summary['experiment']} ({summary['number_of_test_cases']} test cases, "
          f"{summary['number_of_trick_questions']} trick) =========")
    for metric in METRICS:
        score, rate = summary["metrics"][metric], summary["pass_rates"][metric]
        done, expected = summary["scored"][metric].split("/")
        warning = "   ⚠️ some cases were not scored" if done != expected else ""
        if score is None:
            print(f"{metric.replace('_', ' ').title():<26} no scores{warning}")
        else:
            print(f"{metric.replace('_', ' ').title():<26} avg {score:.2f}   pass rate {rate:.0%}   "
                  f"scored {summary['scored'][metric]}{warning}")
    for metric in ["retrieval_hit_rate", "retrieval_mrr"]:
        if metric in summary["metrics"]:
            print(f"{metric.replace('_', ' ').title():<26} {summary['metrics'][metric]:.2f}")


def add_retrieval_ranks(evaluated: list[dict], dataset: list[dict]):
    """Rank of the chunk holding the answer in the agent's retrieved context (questions with "evidence" only)."""
    from eval_retrieval import gold_rank

    evidence = {r["question"]: r["evidence"] for r in dataset if r.get("evidence")}
    for row in evaluated:
        context = row["rag_context"]
        if row["question"] in evidence:
            row["retrieval_rank"] = gold_rank([context] if isinstance(context, str) else context, evidence[row["question"]])


def load_dataset(limit: int | None) -> list[dict]:
    """With --limit, keep the first N normal questions and the first N trick questions."""
    dataset = read_jsonl(DATASET_PATH)
    if not limit:
        return dataset
    normal = [r for r in dataset if not is_trick(r)][:limit]
    tricks = [r for r in dataset if is_trick(r)][:limit]
    return normal + tricks


def run(name: str, k: int, provider: str, embeddings: str, limit: int | None, resume: bool):
    out_dir = EXPERIMENTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(limit)

    try:
        results = generate_answers(dataset, k, out_dir / "answers.jsonl", resume)
        print("\nScoring answers (this is the slow part on free tiers)...")
        evaluated = evaluate_answers(results, provider, out_dir / "evaluated.jsonl", resume)
    except Exception as e:
        message = str(e)
        print(f"\n❌ Stopped: {type(e).__name__}: {message[:300]}")
        if "tokens per day" in message or "per day" in message.lower():
            print("   This is the provider's DAILY limit. Wait for it to reset, or use another model "
                  "(e.g. set GROQ_JUDGE_MODEL in .env so the judge has its own limit).")
        print(f"   Progress is saved in {out_dir}. Run the same command with --resume to continue.")
        sys.exit(1)

    add_retrieval_ranks(evaluated, dataset)
    write_jsonl(out_dir / "evaluated.jsonl", evaluated)

    config = {"name": name, "retriever_k": k, "provider": provider, "embeddings": embeddings, "limit": limit,
              "agent_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") if provider == "groq" else None,
              "judge_model": (os.getenv("GROQ_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
                             if provider == "groq" else None}
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
    parser.add_argument("--limit", type=int, default=None, help="only use the first N normal and first N trick questions")
    parser.add_argument("--name", default=None, help="experiment name (default: <provider>_<embeddings>_k<k>)")
    parser.add_argument("--resume", action="store_true", help="continue a run that stopped, keeping saved answers and scores")
    args = parser.parse_args()

    os.environ["PROVIDER"] = args.provider
    os.environ["EMBEDDING_PROVIDER"] = args.embeddings

    for k in args.k:
        name = args.name or f"{args.provider}_{args.embeddings}_k{k}"
        if args.name and len(args.k) > 1:
            name = f"{args.name}_k{k}"
        run(name, k, args.provider, args.embeddings, args.limit, args.resume)


if __name__ == "__main__":
    main()
