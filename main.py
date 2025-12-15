from fastapi import FastAPI, UploadFile, File
import shutil
from database import Base, engine, SessionLocal
from models import LoanApplication
from ai_engine import extract_text, extract_fields, calculate_metrics
from schemas import LoanResponse

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Loan Underwriting System")


@app.get("/")
def home():
    return {"message": "AI Loan Underwriting API is running"}


@app.post("/apply-loan", response_model=LoanResponse)
def apply_loan(name: str, pdf: UploadFile = File(...)):
    file_path = f"temp_{pdf.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(pdf.file, buffer)

    text = extract_text(file_path)
    fields = extract_fields(text)
    dti, risk_score, decision = calculate_metrics(fields)

    db = SessionLocal()
    loan = LoanApplication(
        name=name,
        age=fields["age"],
        income=fields["income"],
        credit_score=fields["credit_score"],
        dti=dti,
        risk_score=risk_score,
        decision=decision
    )
    db.add(loan)
    db.commit()

    return LoanResponse(
        name=name,
        income=fields["income"],
        credit_score=fields["credit_score"],
        dti=dti,
        risk_score=risk_score,
        decision=decision
    )

@app.get("/lender-dashboard")
def dashboard():
    db = SessionLocal()
    return db.query(LoanApplication).all()
