
from fastapi import Depends, FastAPI
from contextlib import asynccontextmanager
from database.database import SessionLocal

from requests import Session

# from explain import llm_explanation
from database.models import PredictionLog
from perform_operation.explain import llm_explanation
from perform_operation.schemas import LoanApplicant
from perform_operation.features import engineer_applicant_features
from perform_operation.model import train_model, predict_pd, explain
from perform_operation.decision import decision_policy
from database.database import Base, engine, SessionLocal

# from app.explain import llm_explanation  # optional
FEATURE_TEXT = {
    "AGE": "the applicant’s age",
    "monthly_income": "the applicant’s monthly income",
    "DTI": "the proportion of income used for existing debt payments",
    "PAY_0": "most recent payment status",
    "PAY_2": "payment status from two months ago",
    "PAY_3": "payment status from three months ago",
    "BILL_AMT1": "most recent outstanding credit card balance",
    "BILL_AMT2": "outstanding credit card balance from the previous month",
}

def humanize_feature(f):
    return FEATURE_TEXT.get(f, f.replace("_", " ").lower())

Base.metadata.create_all(bind=engine)

# -----------------------------
# Lifespan Event Handler
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    train_model()
    print("Model loaded successfully.")
    yield
    # Shutdown (optional cleanup)
    print("Application shutdown.")


app = FastAPI(
    title="Credit Risk API",
    version="1.0",
    lifespan=lifespan
)


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(applicant: LoanApplicant):

    input_df, dti = engineer_applicant_features(applicant)
    pd_value = float(round(predict_pd(input_df), 2))
    decision, interest = decision_policy(pd_value)

    explanation = explain(input_df)

    top_factors = [
        (humanize_feature(str(f)), float(v))
        for f, v in explanation[:3]
    ]

    explanation_text = llm_explanation(
        decision=decision,
        interest=interest,
        factors=top_factors
    )
    
    db = SessionLocal()
    
    # 🔹 SAVE EVERYTHING
    log = PredictionLog(
        age=applicant.age,
        monthly_income=applicant.monthly_income,
        existing_monthly_debt=applicant.existing_monthly_debt,
        new_loan_emi=applicant.new_loan_emi,
        PAY_0=applicant.PAY_0,
        PAY_2=applicant.PAY_2,
        PAY_3=applicant.PAY_3,
        BILL_AMT1=applicant.BILL_AMT1,
        BILL_AMT2=applicant.BILL_AMT2,

        dti_percent=float(round(dti * 100, 2)),
        decision=decision,
        interest_type=interest,
        explanation=explanation_text,
        top_risk_factors=[
            {"feature": f, "impact": v} for f, v in top_factors
        ]
    )

    db.add(log)
    db.commit()

    return {
        "dti_percent": float(round(dti * 100, 2)),
        "decision": decision,
        "interest_type": interest,
        "top_risk_factors": [
            {"feature": f, "impact": v} for f, v in top_factors
        ],
        "explanation": explanation_text
    }
