import pandas as pd

file_path = r"F:\Agentic AI\Langchain\Langchain_Fraud_detection\Dataset\transactions.csv"
df = pd.read_csv(file_path)

fraud_cases = []

for index, row in df.iterrows():
    flags = []

    # Rule 1: High quantity low price (possible bulk fraud)
    if row["quantity"] > 20:
        flags.append("Unusually high quantity purchased")

    # Rule 2: High value cash transaction
    if row["price"] > 50 and row["payment_type"] == "Cash":
        flags.append("High-value cash transaction")

    # Rule 3: Refund present
    if row["refund_flag"] == "Yes":
        flags.append("Refund flagged transaction")

    # Rule 4: Discount on expensive item
    if row["discount_used"] == "Yes" and row["price"] > 10:
        flags.append("Discount applied on expensive item")

    if flags:
        fraud_cases.append({
            "transaction_id": row["transaction_id"],
            "item": row["item_name"],
            "price": row["price"],
            "flags": flags
        })

# Display results
print("\n🚨 Suspicious Transactions Found:\n")
for case in fraud_cases:
    print(case)
