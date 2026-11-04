import json
from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


with open("result1.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

evaluated_results = []

for i,result in enumerate(results, start=1):

    print(f"Evaluating test case {i}/{len(results)}")

    test_case = LLMTestCase(
        input = result["question"],
        actual_output = result["rag_answer"],
        expected_output = result["expected_answer"],
        retrieval_context = [result["rag_context"]]
    )

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
    correctness_metric = GEval(
            name="Correctness",
            criteria = (
                "Determine whether the model's response is factually correct and aligns with the expected answer."
            ),
            evaluation_params = [
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ]
    )

    answer_relevancy_metric.measure(test_case)
    faithfulness_metric.measure(test_case)
    contextual_relevancy_metric.measure(test_case)
    correctness_metric.measure(test_case)

    evaluated_result = {
        **result,
        "answer_relevancy_score": answer_relevancy_metric.score,
        "faithfulness_score": faithfulness_metric.score,
        "contextual_relevancy_score": contextual_relevancy_metric.score,
        "correctness_score": correctness_metric.score,
    }

    evaluated_results.append(evaluated_result)

    print("answer_relevancy_score:", answer_relevancy_metric.score)
    print("faithfulness_score:", faithfulness_metric.score)
    print("contextual_relevancy_score:", contextual_relevancy_metric.score)
    print("correctness_score:", correctness_metric.score)   


with open("evaluated_results1.jsonl", "w", encoding="utf-8") as f:
    for evaluated_result in evaluated_results:
        f.write(json.dumps(evaluated_result) + "\n")

print("Evaluation completed. Results saved to evaluated_results1.jsonl.")





    