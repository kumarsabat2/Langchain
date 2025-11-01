from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from agent_step1 import calculator  # import our tool


# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# Simple function: decide if we should call the calculator

def agent(query: str):
    # Ask LLM whether to calculate or answer directly
    prompt = f"""
You are a smart AI assistant.
If the user asks a math expression, respond exactly as:
"CALCULATE: <expression>"
Otherwise, just answer normally.

User: {query}
"""
    response = llm.invoke(prompt)
    message = response.content.strip()

    if message.startswith("CALCULATE:"):
        expr = message.replace("CALCULATE:", "").strip()
        result = calculator(expr)
        return f"Result = {result}"
    else:
        return message


if __name__ == "__main__":
    print("Agent test")
    print(agent("What is 2 + 5 * 3?"))
    print(agent("Who invented Python?"))