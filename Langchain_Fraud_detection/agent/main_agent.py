import pandas as pd
from rule_tool import apply_fraud_rules
from explanation_tool import explain_fraud

# Load transaction data
path = r"F:\Agentic AI\Langchain\Langchain_Fraud_detection\Dataset\transactions.csv"
df = pd.read_csv(path)

# Run rule engine
suspicious = apply_fraud_rules(df)

print("\n🤖 FRAUD DETECTION AGENT REPORT\n")

# If nothing suspicious found
if not suspicious:
    print("No fraud detected.")
else:
    # Run AI reasoning for every fraud case
    for txn in suspicious:
        print(f"\nTransaction ID: {txn['transaction_id']}")
        explanation = explain_fraud(txn)
        print(explanation)
