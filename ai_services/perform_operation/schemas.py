 # Request / response models


from pydantic import BaseModel

class LoanApplicant(BaseModel):
    age: int
    monthly_income: float
    existing_monthly_debt: float
    new_loan_emi: float
    PAY_0: int
    PAY_2: int
    PAY_3: int
    BILL_AMT1: float
    BILL_AMT2: float
