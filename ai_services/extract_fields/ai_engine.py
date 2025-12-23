import pdfplumber
import re
import cv2  # OpenCV for advanced image processing
import numpy as np
from transformers import pipeline
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# --- 1. CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

FIELD_CONFIG = {
    # --- PERSONAL INFO ---
    "age": {
        "question": "What is the age of the applicant?",
        "patterns": [r"Age[:\-\s]+(\d{2})", r"(\d{2})\s*years\s*old"],
        "type": "int",
        "default": 30
    },

    # --- INCOME & DEBT ---
    "monthly_income": {
        "question": "What is the monthly income?",
        "patterns": [
            r"(?:Income|Salary|Earnings).*?([\d,]+(?:k|000)?)",
            r"(?:Net|Approx).*?([\d,]+)"
        ],
        "type": "currency"
    },
    "existing_monthly_debt": {
        "question": "What is the total existing monthly debt or liability?",
        "patterns": [
            r"(?:liabilities|repayment|commitments|obligations)[^;]*?([\d,]+)",
            r"amount to\s*([\d,]+)"
        ],
        "type": "currency"
    },
    "new_loan_emi": {
        "question": "What is the expected or proposed EMI for the new loan?",
        "patterns": [
            # The [\s₹Rs\.\-:]+ part consumes any garbage symbols before the number
            r"(?:Expected|Estimated|Proposed)[^;]*?(?:Installment|EMI)[^;]*?[\s₹Rs\.\-:]+([\d,]+)",
            r"(?:Requested|New)[^;]*?Loan[^;]*?[\s₹Rs\.\-:]+([\d,]+)"
        ],
        "type": "currency"
    },

    # --- PAYMENT HISTORY ---
    "PAY_0": {
        "question": "What is the payment status for the latest month?",
        "patterns": [r"(?:Latest month|Current billing month|Most recent cycle)[^;]*?[:\-\s]+([^;]+)"],
        "type": "payment_status"
    },
    "PAY_2": {
        "question": "What was the payment status two months ago?",
        "patterns": [r"(?:Two months ago|2 months ago)[^;]*?[:\-\s]+([^;]+)"],
        "type": "payment_status"
    },
    "PAY_3": {
        "question": "What was the payment status three months ago?",
        "patterns": [r"(?:Three months ago|3 months ago)[^;]*?[:\-\s]+([^;]+)"],
        "type": "payment_status"
    },

    # --- CREDIT CARD BILLS ---
    "BILL_AMT1": {
        "question": "What is the current outstanding credit card balance?",
        "patterns": [
            r"(?:Current|Present|Latest)[^;]*?(?:balance|outstanding|dues)[^;]*?([\d,\. ]{3,})",
            r"(?:balance|outstanding|dues)[^;]*?(?:Current|Present|Latest)[^;]*?([\d,\. ]{3,})"
        ],
        "type": "currency"
    },
    "BILL_AMT2": {
        "question": "What was the previous outstanding credit card balance?",
        "patterns": [
            r"(?:Previous|Prior|Last)[^;]*?(?:billing|balance|outstanding|cycle)[^;]*?([\d,\. ]{3,})",
            r"(?:billing|balance|outstanding|cycle)[^;]*?(?:Previous|Prior|Last)[^;]*?([\d,\. ]{3,})"
        ],
        "type": "currency"
    }
}

# --- 2. INITIALIZATION ---
print("Loading Transformers Pipeline...")
try:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
except Exception as e:
    print(f"⚠️ Pipeline loading failed: {e}")
    qa_pipeline = None


# --- 3. HELPER FUNCTIONS ---

def preprocess_text(text):
    if not text: return ""
    text = text.replace('\n', ' ; ')
    text = text.replace('"', '').replace("'", "")
    # Remove obvious currency symbols (if OCR caught them)
    text = text.replace('₹', '').replace('Rs.', '') 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_currency(text_val):
    if not text_val: return 0.0
    clean = re.sub(r"[^\d\.kK]", "", str(text_val).lower())
    try:
        if "k" in clean:
            return float(clean.replace("k", "")) * 1000
        return float(clean)
    except ValueError:
        return 0.0

def parse_payment_status(text_val):
    text = str(text_val).lower()
    
    # Check for specific delay numbers
    delay_match = re.search(r"delayed? by (\d+)", text)
    if delay_match: return int(delay_match.group(1))

    # Risk Keywords
    if any(x in text for x in ["minimum", "partial", "late", "delay", "not cleared", "overdue", "missed"]):
        if any(x in text for x in ["overdue", "missed", "not cleared"]): return 2
        return 1

    # Safe Keywords
    if any(x in text for x in ["time", "timely", "full", "clear", "paid"]):
        return 0
        
    return 0

def clean_image_for_ocr(image):
    """
    Advanced Image Pre-processing to fix 'Rupee Artifacts'.
    1. Upscales image 3x to separate symbol from digits.
    2. Applies thresholding to make text crisp black/white.
    3. (Optional) Morphological opening to disconnect touching characters.
    """
    try:
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. 3x UPSCALING (Critical for separating '₹' from '9')
        img_cv = cv2.resize(img_cv, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 3. Binary Thresholding (Standard Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Morphological Opening (Optional - remove small noise)
        # kernel = np.ones((1, 1), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return Image.fromarray(binary)
    except Exception:
        return image

def clean_artifact_logic(value, income, field_name):
    """
    Sanitizes values based on financial logic.
    """
    if income == 0 or value == 0: return value
    
    # LOGIC:
    # EMI shouldn't typically be > 40% of Income. 
    # Total Debt can be higher (up to 300% is possible for mortgages etc).
    
    threshold_ratio = 0.40 if "emi" in field_name else 3.0
    
    # If Value is suspiciously high compared to income
    if value > (income * threshold_ratio):
        val_str = str(int(value))
        
        # Check for common OCR artifacts: 2, 3, 4, 8, 7
        if val_str[0] in ['2', '3', '4', '8', '7']:
            try:
                stripped_val = float(val_str[1:])
                # Verify sanity: Is the stripped value > 500?
                if stripped_val > 500:
                    print(f"🔧 Logic Fix ({field_name}): {value} -> {stripped_val}")
                    return stripped_val
            except:
                pass
    return value

def sanitize_financial_data(data):
    """
    Post-processing to fix 'Impossible' financial numbers.
    """
    # 1. Sanitize Income First
    inc = data.get("monthly_income", 0)
    # If income is suspiciously large (> 200k), check for artifacts
    if inc > 200000:
        val_str = str(int(inc))
        if val_str[0] in ['2', '3', '4', '8']: 
            try:
                new_inc = float(val_str[1:])
                if new_inc > 5000: 
                    data["monthly_income"] = new_inc
                    inc = new_inc
                    print(f"🔧 Logic Fix (Income): {val_str} -> {new_inc}")
            except: pass

    # 2. Sanitize EMI & Debts based on (Corrected) Income
    if inc > 0:
        # Check EMI
        emi = data.get("new_loan_emi", 0)
        data["new_loan_emi"] = clean_artifact_logic(emi, inc, "new_loan_emi")

        # Check other debts
        for key in ["existing_monthly_debt", "BILL_AMT1", "BILL_AMT2"]:
            val = data.get(key, 0)
            data[key] = clean_artifact_logic(val, inc, key)
            
    return data

# --- 4. EXTRACT TEXT ---

def extract_text(file_content, filename=""):
    text = ""
    filename = filename.lower()
    file_stream = io.BytesIO(file_content)

    if filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception: pass
        
        # OCR Fallback
        if len(text.strip()) < 50:
            try:
                file_stream.seek(0)
                doc = fitz.open(stream=file_stream.read(), filetype="pdf")
                ocr_text = ""
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    
                    # Clean & Upscale Image
                    img = clean_image_for_ocr(img)
                    
                    # Run Tesseract with Block Config (--psm 6)
                    # This helps reading lines of data accurately
                    ocr_text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
                text = ocr_text
            except Exception: pass

    elif filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            image = Image.open(file_stream)
            image = clean_image_for_ocr(image)
            # Run Tesseract with Block Config
            text = pytesseract.image_to_string(image, config='--psm 6')
        except Exception: pass

    return text

# --- 5. EXTRACT FIELDS ---

def extract_fields(text):
    clean_text = preprocess_text(text)
    extracted_data = {}
    short_context = clean_text[:3000]

    for field_key, config in FIELD_CONFIG.items():
        val = None
        
        # A. Regex
        if "patterns" in config:
            for pat in config["patterns"]:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    break
        
        # B. AI Fallback
        if not val and qa_pipeline:
            try:
                res = qa_pipeline(question=config["question"], context=short_context)
                if res['score'] > 0.1: val = res['answer']
            except: pass

        # C. Parse
        final_val = config.get("default", 0)
        if val:
            try:
                target_type = config["type"]
                if target_type == "int":
                    nums = re.findall(r"\d+", str(val))
                    if nums: final_val = int(nums[0])
                elif target_type in ["float", "currency"]:
                    final_val = parse_currency(val)
                elif target_type == "payment_status":
                    final_val = parse_payment_status(val)
            except: pass
        
        extracted_data[field_key] = final_val

    # --- FINAL LOGIC SANITIZATION ---
    extracted_data = sanitize_financial_data(extracted_data)
            
    return extracted_data