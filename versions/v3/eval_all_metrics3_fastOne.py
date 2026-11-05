from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 
import json
from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.evaluate import CacheConfig
from deepeval.evaluate import evaluate,CacheConfig, AsyncConfig
import os
from deepeval.models import OllamaModel
from RAG1.judge import LangChainJudge
provider = os.getenv("PROVIDER", "openai")


if provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    name = os.getenv("GEMINI_MODEL")
    judge = LangChainJudge(ChatGoogleGenerativeAI(model=name, temperature=0), name)
elif provider == "groq":
    from langchain_groq import ChatGroq
    name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    judge = LangChainJudge(ChatGroq(model=name, temperature=0, max_retries=8), name, structured_method="json_schema")
elif provider == "ollama":
    from deepeval.models import OllamaModel
    judge = OllamaModel(model="qwen2.5:3b")
else:
    judge = None 

with open("result3.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

questions_tests = []

for i,result in enumerate(results, start=1):

    print(f"Evaluating test case {i}/{len(results)}")

    context = result["rag_context"]

    if isinstance(context, str):
        retrieval_context = [context]
    else:
        retrieval_context = context

    test_case = LLMTestCase(
        input = result["question"],
        actual_output = result["rag_answer"],
        expected_output = result["expected_answer"],
        retrieval_context = retrieval_context
    )

    questions_tests.append(test_case)

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)
contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=judge)
correctness = GEval(
        name="Correctness",
        criteria = (
            "Determine whether the model's response is factually correct and aligns with the expected answer."
        ),
        evaluation_params = [
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge
)


evaluation_results = evaluate(
    test_cases=questions_tests,
    metrics=[answer_relevancy, faithfulness, contextual_relevancy, correctness],
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False
    ),
    async_config=AsyncConfig(run_async=provider == "openai"),
)

evaluated_results = []

for i, (original_result, test_result) in enumerate(zip(results, evaluation_results.test_results), start=1):

    answer_relevancy_score = None
    faithfulness_score = None
    contextual_relevancy_score = None
    correctness_score = None

    print("\nMetrics :")

    for metric in test_result.metrics_data or []:

        metric_name = metric.name.strip().lower()

        if metric_name == "answer relevancy":
            answer_relevancy_score = metric.score

        elif metric_name == "faithfulness":
            faithfulness_score = metric.score

        elif metric_name == "contextual relevancy":
            contextual_relevancy_score = metric.score

        elif metric_name.startswith("correctness"):
            correctness_score = metric.score

    evaluated_result = {
            **original_result,
            "answer_relevancy_score": answer_relevancy_score,
            "faithfulness_score": faithfulness_score,
            "contextual_relevancy_score": contextual_relevancy_score,
            "correctness_score": correctness_score,
        }  

    evaluated_results.append(evaluated_result)

with open("evaluated_results3_fastOne.jsonl", "w", encoding="utf-8") as f:
    for result in evaluated_results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("Evaluation completed. Results saved to evaluated_results3_fastOne.jsonl.")