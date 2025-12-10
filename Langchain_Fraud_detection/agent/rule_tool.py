import pandas as pd

def apply_fraud_rules(df: pd.DataFrame):
    """
    Applies rule-based fraud detection.
    Returns only suspicious transactions with flags.
    """
    suspicious = []

    for _, row in df.iterrows():
        flags = []

        if row["quantity"] > 20:
            flags.append("Unusually high quantity purchased")

        if row["price"] > 50 and row["payment_type"] == "Cash":
            flags.append("High-value cash transaction")

        if row["refund_flag"] == "Yes":
            flags.append("Refund flagged transaction")

        if row["discount_used"] == "Yes" and row["price"] > 10:
            flags.append("Discount applied on expensive item")

        if flags:
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "data": row.to_dict(),
                "flags": flags
            })

    return suspicious
