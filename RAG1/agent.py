import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import RAG1.tools as tl
from .prompt import system_agent_prompt
from .objects import Answer
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy

provider = os.getenv("PROVIDER", "openai")

if provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0)
elif provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
elif provider == "ollama":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    model=llm,
    tools=[tl.collecting_info],
    system_prompt=system_agent_prompt,
    # gpt-oss on Groq often emits malformed JSON for the structured-output tool,
    # so Groq answers in plain text and ask() falls back to the last message.
    response_format=None if provider == "groq" else ToolStrategy(Answer),
)


def ask(question: str) -> dict:
    """
    Ask a question to the agent and return the answer plus the chunks actually retrieved.
    """
    tl.retrieved_chunks.clear()  # reset before each question

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    structured = result.get("structured_response")

    if structured is not None:
        answer = structured.answer
    else:
        answer = result["messages"][-1].content  # fallback: the model's plain text reply
        if provider != "groq":
            print("  ⚠️ Structured output failed, using raw text")

    return {
        "answer": answer,
        "context": list(tl.retrieved_chunks),  # real retriever output
        "tool_called": len(tl.retrieved_chunks) > 0,
    }


if __name__ == "__main__":
    print(ask("Why can't penguins fly?"))