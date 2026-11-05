import os
from deepeval.models import DeepEvalBaseLLM


class LangChainJudge(DeepEvalBaseLLM):
    """Wraps any LangChain chat model so DeepEval can use it as a judge."""

    def __init__(self, llm, name: str, structured_method: str | None = None):
        self.model = llm
        self.name = name
        # "json_schema" for models whose function calling is unreliable (gpt-oss on Groq)
        self.structured_kwargs = {"method": structured_method} if structured_method else {}

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.model.with_structured_output(schema, **self.structured_kwargs).invoke(prompt)
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self.model.with_structured_output(schema, **self.structured_kwargs).ainvoke(prompt)
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return self.name


def get_judge(provider: str):
    """Return the DeepEval judge for a provider (None = DeepEval's default OpenAI model)."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        name = os.getenv("GEMINI_MODEL")
        return LangChainJudge(ChatGoogleGenerativeAI(model=name, temperature=0), name)
    if provider == "groq":
        from langchain_groq import ChatGroq
        name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return LangChainJudge(ChatGroq(model=name, temperature=0, max_retries=8), name, structured_method="json_schema")
    if provider == "ollama":
        from deepeval.models import OllamaModel
        return OllamaModel(model="qwen2.5:3b")
    return None