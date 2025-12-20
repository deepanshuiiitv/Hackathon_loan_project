# Feature engineering (DTI, income) — OpenML compatible

import pandas as pd

# Final feature list used by the model
FEATURES = [
    "AGE",
    "monthly_income",
    "DTI",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "BILL_AMT1",
    "BILL_AMT2",
]

# for training purpose only
def engineer_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for TRAINING data (OpenML schema).
    Maps x-columns to business-meaningful features.
    """

    # -----------------------------
    # OpenML column mapping
    # -----------------------------
    df = df.rename(columns={
        "x1": "LIMIT_BAL",     # credit limit
        "x5": "AGE",
        "x6": "PAY_0",
        "x7": "PAY_2",
        "x8": "PAY_3",
        "x12": "BILL_AMT1",
        "x13": "BILL_AMT2",
        "x18": "PAY_AMT1",
        "x19": "PAY_AMT2",
        "x20": "PAY_AMT3",
    })

    # -----------------------------
    # Income proxy (monthly)
    # -----------------------------
    df["monthly_income"] = df["LIMIT_BAL"] / 10

    # -----------------------------
    # Estimated EMI
    # -----------------------------
    df["estimated_emi"] = df[
        ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]
    ].mean(axis=1)

    # -----------------------------
    # Debt-to-Income Ratio
    # -----------------------------
    df["DTI"] = (df["estimated_emi"] / df["monthly_income"]).clip(0, 1)

    return df[FEATURES]

# for api
def engineer_applicant_features(applicant):
    """
    Feature engineering for API input (single applicant).
    Must EXACTLY match training feature names.
    """

    dti = (
        applicant.existing_monthly_debt + applicant.new_loan_emi
    ) / applicant.monthly_income

    input_df = pd.DataFrame([{
        "AGE": applicant.age,
        "monthly_income": applicant.monthly_income,
        "DTI": dti,
        "PAY_0": applicant.PAY_0,
        "PAY_2": applicant.PAY_2,
        "PAY_3": applicant.PAY_3,
        "BILL_AMT1": applicant.BILL_AMT1,
        "BILL_AMT2": applicant.BILL_AMT2,
    }])

    return input_df, dti
