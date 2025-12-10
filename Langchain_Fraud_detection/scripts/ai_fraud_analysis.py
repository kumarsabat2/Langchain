import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load dataset
path = r"F:\Agentic AI\Langchain\Langchain_Fraud_detection\Dataset\transactions.csv"
df = pd.read_csv(path)

# Rule logic
def apply_rules(row):
    flags = []
    if row["quantity"] > 20:
        flags.append("Unusually high quantity purchased")
    if row["price"] > 50 and row["payment_type"] == "Cash":
        flags.append("High-value cash transaction")
    if row["refund_flag"] == "Yes":
        flags.append("Refund flagged transaction")
    if row["discount_used"] == "Yes" and row["price"] > 10:
        flags.append("Discount applied on expensive item")
    return flags

suspicious = []
for _, row in df.iterrows():
    flags = apply_rules(row)
    if flags:
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "data": row.to_dict(),
            "flags": flags
        })

from dotenv import load_dotenv
import os
# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.2)

print("\n🧠 AI FRAUD REPORT\n")

for txn in suspicious:
    prompt = f"""
You are an expert retail fraud analyst.

Transaction Data:
{txn['data']}

Rule Flags:
{txn['flags']}

Provide:
1. Risk level (Low/Medium/High)
2. Explanation
3. Recommended action
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\nTransaction: {txn['transaction_id']}")
    print(response.content)
