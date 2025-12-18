from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
from database.database import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    monthly_income = Column(Float)
    existing_monthly_debt = Column(Float)
    new_loan_emi = Column(Float)
    PAY_0 = Column(Integer)
    PAY_2 = Column(Integer)
    PAY_3 = Column(Integer)
    BILL_AMT1 = Column(Float)
    BILL_AMT2 = Column(Float)
    dti_percent = Column(Float)
    decision = Column(String(20))
    interest_type = Column(String(20))
    explanation = Column(String(1000))
    top_risk_factors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
