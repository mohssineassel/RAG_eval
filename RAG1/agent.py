import RAG1.tools as tl
from .prompt import system_agent_prompt
from langchain_openai import ChatOpenAI
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from .objects import Answer


llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Initialize the language model with GPT-4o and temperature 0

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer)

def ask(question: str) -> dict:
    """
    Ask a question to the agent and return the answer.
    """
    tl.last_retrieved_context = []  # Reset the last retrieved context before each query

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )  # Invoke the agent with the question
    
    structured_response = result["structured_response"]  # Extract the structured response from the result

    return {
        "answer": structured_response.answer,  # Return the answer from the structured response
        "context": structured_response.context,  # Return the context from the structured response
    }


print(ask("What is the main topic of the paper?"))  # Example usage of the ask function