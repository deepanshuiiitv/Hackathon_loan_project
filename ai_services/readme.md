
<-- Change .env for database configutation -->

"run on http://127.0.0.1:8000/docs"

----use this for test --------------------------------------------------->

{
  "age": 32,
  "monthly_income": 80000,
  "existing_monthly_debt": 15000,
  "new_loan_emi": 18000,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 1,
  "BILL_AMT1": 120000,
  "BILL_AMT2": 110000
}

------------Used dataset fields-------------------------------------------->

ID
LIMIT_BAL
SEX
EDUCATION
MARRIAGE
AGE
Repayment status history (6)-->
PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6
Bill statement amounts (6)-->
BILL_AMT1
BILL_AMT2
BILL_AMT3
BILL_AMT4
BILL_AMT5
BILL_AMT6
Past payment amounts (6)-->
PAY_AMT1
PAY_AMT2
PAY_AMT3
PAY_AMT4
PAY_AMT5
PAY_AMT6
Target variable (1)-->
default.payment.next.month

----------------------------------------------------------