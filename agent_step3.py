from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Step 1. Define a simple tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [
    Tool(
        name="Multiplier",
        func=lambda x: multiply(*map(int, x.split())),
        description="Multiply two numbers. Input should be 'a b'."
    )
]

# Step 2. Initialize an LLM
llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-3.5-turbo"

# Step 3. Create the ReAct agent
agent = create_react_agent(llm=llm, tools=tools)

# Step 4. Create an executor to run the agent
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 5. Ask the agent a question that requires tool use
result = executor.invoke({"input": "Use the multiplier to multiply 7 and 9"})
print("Final Output:", result["output"])
