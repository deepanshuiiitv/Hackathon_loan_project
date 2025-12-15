import pdfplumber
import re
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_fields(text):
    doc = nlp(text)

    income = re.search(r"(\d{4,7})", text)
    age = re.search(r"Age[:\s]+(\d{2})", text)
    credit = re.search(r"Credit Score[:\s]+(\d{3})", text)

    return {
        "income": float(income.group(1)) if income else 0,
        "age": int(age.group(1)) if age else 30,
        "credit_score": int(credit.group(1)) if credit else 650,
    }

def calculate_metrics(data):
    income = data["income"]
    credit = data["credit_score"]

    dti = 0.3 if income > 30000 else 0.6

    risk_score = (
        (750 - credit) * 0.5 +
        dti * 100 +
        (40 - data["age"]) * 0.2
    )

    if risk_score < 50:
        decision = "APPROVE (LOW INTEREST)"
    elif risk_score < 100:
        decision = "APPROVE (HIGHER INTEREST)"
    else:
        decision = "REJECT"

    return dti, risk_score, decision
