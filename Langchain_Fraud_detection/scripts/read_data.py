import pandas as pd

file_path = r"F:\Agentic AI\Langchain\Langchain_Fraud_detection\Dataset\transactions.csv"

df = pd.read_csv(file_path)

print("\n✅ Dataset Loaded Successfully!\n")
print(df.head())
print("\nTotal Records:", len(df))
