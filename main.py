from fastapi import Depends, FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

from contextlib import asynccontextmanager
from typing import List
from requests import Session

# -----------------------------
# Project Imports (UNCHANGED)
# -----------------------------
from database.database import SessionLocal, Base, engine
from database.models import PredictionLog
from ai_services.perform_operation.schemas import LoanApplicant
from ai_services.perform_operation.features import engineer_applicant_features
from ai_services.perform_operation.model import train_model, predict_pd, explain
from ai_services.perform_operation.decision import decision_policy
from ai_services.perform_operation.explain import llm_explanation

# -----------------------------
# AI Extraction
# -----------------------------
try:
    from ai_services.extract_fields.ai_engine import extract_text, extract_fields
except ImportError:
    from ai_services.extract_fields.ai_engine import extract_text, extract_fields


# -----------------------------
# Feature Humanization
# -----------------------------
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


# -----------------------------
# Database Init
# -----------------------------
Base.metadata.create_all(bind=engine)


# -----------------------------
# Lifespan (Model Load)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing model...")
    train_model()
    print("Model ready.")
    yield
    print("Application shutdown.")


# -----------------------------
# App Init
# -----------------------------
app = FastAPI(
    title="Credit Risk API",
    version="1.0",
    lifespan=lifespan
)

# -----------------------------
# UI Setup
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -----------------------------
# UI ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/manual", response_class=HTMLResponse)
def manual_ui(request: Request):
    return templates.TemplateResponse("manual.html", {"request": request})


@app.get("/files", response_class=HTMLResponse)
def file_ui(request: Request):
    return templates.TemplateResponse("file.html", {"request": request})
from sqlalchemy.orm import Session

@app.get("/lender-dashboard", response_class=HTMLResponse)
def lender_dashboard(request: Request):
    db: Session = SessionLocal()
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .all()
    )
    db.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "logs": logs
        }
    )



# -----------------------------
# 1. Manual Prediction API
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
            dti_percent=round(dti * 100, 2),
            decision=decision,
            interest_type=interest,
            explanation=explanation_text,
            top_risk_factors=[{"feature": f, "impact": v} for f, v in top_factors]
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return {
        "dti_percent": round(dti * 100, 2),
        "decision": decision,
        "interest_type": interest,
        "top_risk_factors": [{"feature": f, "impact": v} for f, v in top_factors],
        "explanation": explanation_text
    }


# -----------------------------
# 2. File-Based Prediction API (EXACTLY 2 FILES)
# -----------------------------
@app.post("/predict_from_files")
async def predict_from_files(files: List[UploadFile] = File(...)):

    if len(files) != 2:
        return {
            "status": "error",
            "message": "Exactly 2 documents required (Salary Slip + Bank Statement)."
        }

    combined_text = ""

    for file in files:
        content = await file.read()
        text = extract_text(content, filename=file.filename)
        combined_text += " ; " + text

    data = extract_fields(combined_text)

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
            dti_percent=round(dti * 100, 2),
            decision=decision,
            interest_type=interest,
            explanation=explanation_text,
            top_risk_factors=[{"feature": f, "impact": v} for f, v in top_factors]
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return {
        "status": "success",
        "extracted_data": data,
        "prediction": {
            "dti_percent": round(dti * 100, 2),
            "decision": decision,
            "interest_type": interest,
            "top_risk_factors": [{"feature": f, "impact": v} for f, v in top_factors],
            "explanation": explanation_text
        }
    }
