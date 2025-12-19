
from fastapi import Depends, FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from typing import List
from requests import Session

# --- Project Imports ---
from database.database import SessionLocal, Base, engine
from database.models import PredictionLog
from perform_operation.schemas import LoanApplicant
from perform_operation.features import engineer_applicant_features
from perform_operation.model import train_model, predict_pd, explain
from perform_operation.decision import decision_policy
from perform_operation.explain import llm_explanation

# --- New Import for AI Extraction ---
# Note: Ensure ai_engine.py is inside a folder named 'extract_fields' 
# or change this import to match your folder structure (e.g., 'perform_operation.ai_engine')
try:
    from extract_fields.ai_engine import extract_text, extract_fields
except ImportError:
    # Fallback if you placed it in perform_operation
    from perform_operation.ai_engine import extract_text, extract_fields

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

# Create Database Tables
Base.metadata.create_all(bind=engine)

# -----------------------------
# Lifespan Event Handler
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load/Train Model
    print("Initializing model...")
    train_model()
    print("Model loaded successfully.")
    yield
    # Shutdown
    print("Application shutdown.")

app = FastAPI(
    title="Credit Risk API",
    version="1.0",
    lifespan=lifespan
)

# -----------------------------
# 1. Manual Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(applicant: LoanApplicant):
    # 1. Feature Engineering
    input_df, dti = engineer_applicant_features(applicant)
    
    # 2. Prediction & Decision
    pd_value = float(round(predict_pd(input_df), 2))
    decision, interest = decision_policy(pd_value)

    # 3. Explanation
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
    
    # 4. Save to Database
    db = SessionLocal()
    try:
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
            top_risk_factors=[{"feature": f, "impact": v} for f, v in top_factors]
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        db.close()

    return {
        "dti_percent": float(round(dti * 100, 2)),
        "decision": decision,
        "interest_type": interest,
        "top_risk_factors": [{"feature": f, "impact": v} for f, v in top_factors],
        "explanation": explanation_text
    }


# -----------------------------
# 2. File-Based Prediction Endpoint (NEW)
# -----------------------------
@app.post("/predict_from_files")
async def predict_from_files(files: List[UploadFile] = File(...)):
    """
    Accepts PDF/Image files, extracts applicant data, 
    and runs the credit risk prediction model automatically.
    """
    
    combined_text = ""

    # A. Read & Extract Text from All Files
    for file in files:
        content = await file.read()
        # Pass content bytes and filename to your AI engine
        text = extract_text(content, filename=file.filename)
        combined_text += " ; " + text  # Merge text with delimiter

    # B. Extract Structured Fields
    data = extract_fields(combined_text)
    
    # C. Create Applicant Object (Mapping Dictionary -> Pydantic Model)
    # We use .get() with defaults to ensure safety if extraction misses a field
    applicant = LoanApplicant(
        age=int(data.get("age", 30)),
        monthly_income=float(data.get("monthly_income", 0)),
        existing_monthly_debt=float(data.get("existing_monthly_debt", 0)),
        new_loan_emi=float(data.get("new_loan_emi", 0)),
        PAY_0=int(data.get("PAY_0", 0)),
        PAY_2=int(data.get("PAY_2", 0)),
        PAY_3=int(data.get("PAY_3", 0)),
        BILL_AMT1=float(data.get("BILL_AMT1", 0)),
        BILL_AMT2=float(data.get("BILL_AMT2", 0)),
    )

    # D. Run Prediction Logic (Same as /predict)
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

    # E. Save to Database
    db = SessionLocal()
    try:
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
            top_risk_factors=[{"feature": f, "impact": v} for f, v in top_factors]
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        db.close()

    # F. Return Response (Including extracted data for verification)
    return {
        "status": "success",
        "extracted_data": data, # Helpful for frontend to show what was read
        "prediction": {
            "dti_percent": float(round(dti * 100, 2)),
            "decision": decision,
            "interest_type": interest,
            "top_risk_factors": [{"feature": f, "impact": v} for f, v in top_factors],
            "explanation": explanation_text
        }
    }