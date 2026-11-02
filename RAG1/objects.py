from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="The answer to the question based on the provided context.")
    context: str = Field(description="The context from which the answer was derived.")