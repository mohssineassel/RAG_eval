from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
import json

test_case = LLMTestCase(
    input = "Where do penguins live?",
    actual_output = "Penguins live in a wide range of environments in the Southern Hemisphere, including the icy coasts of Antarctica, the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator.",
    expected_output = "Penguins live in the Southern Hemisphere, including Antarctica, South America, southern Africa, and the Galapagos Islands.",
)

metric = GEval(
    name="Correctness",
    criteria = (
        "Determine whether the model's response is factually correct and aligns with the expected answer."
    ),
    evaluation_params = [
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]
)


metric.measure(test_case = test_case)

print("Score:" , metric.score)
print("Reason:" , metric.reason)
print("Passed:" , metric.is_successful())
