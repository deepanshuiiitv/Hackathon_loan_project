# 🏦 LoanLens AI

**Automated Loan Risk Assessment & Decision System**

LoanLens AI automates **loan application processing** by extracting key financial and applicant data from **PDFs and scanned images** using **OCR (Tesseract), rule-based parsing, and lightweight NLP**.
It converts **unstructured documents into structured insights** to produce **transparent, explainable loan decisions** for **KYC and credit evaluation**.

---

## 🚀 Key Capabilities

* 📄 PDF & scanned image ingestion
* 🔍 OCR using **Tesseract**
* 🧠 Rule-based, deterministic field extraction
* ⚖️ Explainable credit risk assessment
* 🏦 Lender dashboard with audit trail
* 🗄️ Database-backed prediction logs

---

## 🧠 High-Level Workflow

1. Upload documents **or** enter details manually
2. OCR extracts raw text
3. Rule-based parsing → structured fields
4. Feature engineering + credit model
5. Decision policy → **Approved / High Interest / Rejected**
6. Results stored for lender review

---

## 📊 Dataset Reference (UCI Credit Card Default)

### Applicant Attributes

* `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`, `LIMIT_BAL`

### Repayment Status (6 months)

* `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`

### Bill Amounts (6 months)

* `BILL_AMT1` → `BILL_AMT6`

### Payment Amounts (6 months)

* `PAY_AMT1` → `PAY_AMT6`

### Target

* `default.payment.next.month`

---

## 🧪 API Test Payload (`/predict`)

```json
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
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create & Activate Virtual Environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (CMD / PowerShell)

```bat
python -m venv venv
venv\Scripts\activate
```

---

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔎 OCR Setup (Tesseract – REQUIRED)

### 🐧 Linux

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Set path in code:

```python
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
```

---

### 🪟 Windows

1. Download installer:
   [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install (default path recommended)

Set path in code:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify:

```bash
tesseract --version
```

---

## 🗄️ Database Configuration

### Option 1: `.env` (Recommended)

```
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/databasename
```

### Option 2: Direct in `database.py`

```python
DATABASE_URL = "mysql+pymysql://root:password@localhost:3306/databasename"
```

---

## ▶️ Run the Application

```bash
uvicorn main:app --reload
```

---

## 🌐 Access

* 📘 **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* 🧪 Manual Prediction API
* 📄 Document Upload (OCR)
* 🏦 Lender Dashboard

---

## 🏁 Why LoanLens AI

* ✅ Deterministic & explainable (no black-box hallucinations)
* ✅ Realistic fintech workflow
* ✅ OCR + rules + ML (enterprise-friendly)
* ✅ Production-ready UI & database

---

