
from RAG1.agent import ask
import json

with open("eval_dataset.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

    results = []
    for test_case in dataset:
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]

        print(f"Question: {question}")  
        rag_result = ask(question)

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "rag_answer": rag_result["answer"],
            "rag_context": rag_result["context"],
            "tool_called": rag_result["tool_called"],
        }
        if not rag_result["tool_called"]:
            print("  ⚠️ Agent answered without retrieving anything")

        results.append(result)
        print(f"RAG Answer: {rag_result['answer']}")


with open("result1.jsonl", "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")


print("Evaluation completed. Results saved to result1.jsonl.")  