from pydantic import BaseModel

class LoanResponse(BaseModel):
    name: str
    income: float
    credit_score: int
    dti: float
    risk_score: float
    decision: str
