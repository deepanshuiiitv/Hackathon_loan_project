from sqlalchemy import Column, Integer, String, Float
from database import Base

class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    credit_score = Column(Integer, nullable=False)
    dti = Column(Float, nullable=False)
    risk_score = Column(Float)
    decision = Column(String(50))
    
