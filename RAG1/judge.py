import os
import re
from deepeval.models import DeepEvalBaseLLM


def _is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "ratelimit" in text or "rate limit" in text or "resource_exhausted" in text or "429" in text


def _parse_json(text: str, schema):
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return schema.model_validate_json(match.group(0) if match else text)


class LangChainJudge(DeepEvalBaseLLM):
    

    def __init__(self, llm, name: str, structured_method: str | None = None):
        self.model = llm
        self.name = name
        self.structured_kwargs = {"method": structured_method} if structured_method else {}

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self.model.invoke(prompt).content
        try:
            return self.model.with_structured_output(schema, **self.structured_kwargs).invoke(prompt)
        except Exception as e:
            if _is_rate_limit(e):
                raise
            return _parse_json(self.model.invoke(prompt).content, schema)

    async def a_generate(self, prompt: str, schema=None):
        if schema is None:
            return (await self.model.ainvoke(prompt)).content
        try:
            return await self.model.with_structured_output(schema, **self.structured_kwargs).ainvoke(prompt)
        except Exception as e:
            if _is_rate_limit(e):
                raise
            return _parse_json((await self.model.ainvoke(prompt)).content, schema)

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
        # a separate judge model gets its own Groq daily token limit
        name = os.getenv("GROQ_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return LangChainJudge(ChatGroq(model=name, temperature=0, max_retries=8), name, structured_method="json_schema")
    if provider == "ollama":
        from deepeval.models import OllamaModel
        return OllamaModel(model="qwen2.5:3b")
    return None
