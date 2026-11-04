from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input = "what do pengiuns eat?",
    actual_output = ("Penguins primarily eat fish, mohssine is a nice person"),
)

metric = AnswerRelevancyMetric(threshold=0.7)

metric.measure(test_case)

print("Score:" , metric.score)
print("Reason:" , metric.reason)
print("Passed:" , metric.is_successful())