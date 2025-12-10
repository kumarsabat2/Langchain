from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import os
# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize AI Model
llm = ChatOpenAI(temperature=0.2)

def explain_fraud(transaction):
    """
    Uses AI to explain fraud risk in business language.
    """
    prompt = f"""
You are a retail fraud detection expert.

Transaction Data:
{transaction['data']}

Rule Flags:
{transaction['flags']}

Respond with:
1. Risk level (Low / Medium / High)
2. Explanation
3. Action recommendation
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
