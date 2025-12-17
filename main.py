from fastapi import FastAPI, UploadFile, File,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil

from database import Base, engine, SessionLocal
from models import LoanApplication
from ai_engine import extract_text, extract_fields, calculate_metrics

Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/apply-loan-ui", response_class=HTMLResponse)
def apply_ui(request: Request):
    return templates.TemplateResponse("apply_loan.html", {"request": request})

@app.get("/dashboard-ui", response_class=HTMLResponse)
def dashboard_ui(request: Request):
    db = SessionLocal()
    loans = db.query(LoanApplication).all()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "loans": loans}
    )

@app.post("/apply-loan")
@app.post("/apply-loan")
def apply_loan(
    name: str = Form(...),
    pdf: UploadFile = File(...)
):
    path = f"temp_{pdf.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    text = extract_text(path)
    fields = extract_fields(text)

    # DECIDE NAME: Prefer PDF extraction, fallback to Form Input
    final_name = fields.get("name") if fields.get("name") else name

    dti, risk, decision = calculate_metrics(fields)

    db = SessionLocal()
    loan = LoanApplication(
        name=final_name,
        age=fields["age"],
        income=fields["income"],
        credit_score=fields["credit_score"],
        dti=dti,
        risk_score=risk,
        decision=decision
    )
    db.add(loan)
    db.commit()

    return {
        "name": final_name,
        "income": fields["income"],
        "credit_score": fields["credit_score"],
        "risk_score": risk,
        "decision": decision
    }

@app.get("/lender-dashboard")
def lender_dashboard():
    db = SessionLocal()
    return db.query(LoanApplication).all()
