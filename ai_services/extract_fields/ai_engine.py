import pdfplumber
import re
import spacy
from transformers import pipeline
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# --- CONFIGURATION ---
# Adjust path if needed for your server environment
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# FIELD CONFIGURATION
FIELD_CONFIG = {
    # --- PERSONAL INFO ---
    "age": {
        "question": "What is the age of the applicant?",
        "patterns": [
            r"Age[:\-\s]+(\d{2})", 
            r"(\d{2})\s*years\s*old"
        ],
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
            # "Total recurring liabilities stand at..."
            # We use [^;]+ to ensure we don't jump to the next line looking for a number
            r"(?:liabilities|repayment|commitments|obligations)[^;]*?([\d,]+)",
            r"amount to\s*([\d,]+)"
        ],
        "type": "currency"
    },
    "new_loan_emi": {
        "question": "What is the expected or proposed EMI for the new loan?",
        "patterns": [
            r"(?:Expected|Estimated|Proposed)[^;]*?(?:Installment|EMI)[^;]*?([\d,]+)",
            r"(?:Requested|New)[^;]*?Loan[^;]*?([\d,]+)"
        ],
        "type": "currency"
    },

    # --- PAYMENT HISTORY (Stricter Boundaries) ---
    "PAY_0": {
        "question": "What is the payment status for the latest month?",
        "patterns": [
            r"(?:Latest month|Current billing month|Most recent cycle)[^;]*?[:\-\s]+([^;]+)"
        ],
        "type": "payment_status"
    },
    "PAY_2": {
        "question": "What was the payment status two months ago?",
        "patterns": [
            r"(?:Two months ago|2 months ago)[^;]*?[:\-\s]+([^;]+)"
        ],
        "type": "payment_status"
    },
    "PAY_3": {
        "question": "What was the payment status three months ago?",
        "patterns": [
            r"(?:Three months ago|3 months ago)[^;]*?[:\-\s]+([^;]+)"
        ],
        "type": "payment_status"
    },

    # --- CREDIT CARD BILLS (Distinct Keywords) ---
    "BILL_AMT1": {
        "question": "What is the current outstanding credit card balance?",
        "patterns": [
            # 1. "Current Balance/Outstanding: 15000"
            # REMOVED 'amount' from the list to avoid "repayment amount"
            # ADDED 'outstanding' to catch "outstanding amount"
            r"(?:Current|Present|Latest)[^;]*?(?:billing|balance|outstanding|dues)[^;]*?([\d,]{3,})",
            
            # 2. "Balance/Outstanding (Current): 15000" (Swapped order support)
            r"(?:billing|balance|outstanding|dues)[^;]*?(?:Current|Present|Latest)[^;]*?([\d,]{3,})"
        ],
        "type": "currency"
    },
    "BILL_AMT2": {
        "question": "What was the previous outstanding credit card balance?",
        "patterns": [
            # 1. "Previous Balance: 12000"
            r"(?:Previous|Prior|Last)[^;]*?(?:billing|balance|outstanding|cycle)[^;]*?([\d,]{3,})",
            
            # 2. "Balance (Previous): 12000"
            r"(?:billing|balance|outstanding|cycle)[^;]*?(?:Previous|Prior|Last)[^;]*?([\d,]{3,})"
        ],
        "type": "currency"
    }
}

# --- INITIALIZATION ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ SpaCy model not found. Continuing without it.")
    nlp = None

print("Loading Transformers Pipeline...")
# Ensure you have internet connection on first run to download the model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# --- HELPER FUNCTIONS ---

def preprocess_text(text):
    if not text: return ""
    # 1. Replace Newlines with a explicit delimiter ' ; '
    text = text.replace('\n', ' ; ')
    
    # 2. Clean up quotes
    text = text.replace('"', '').replace("'", "")
    
    # 3. Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_currency(text_val):
    if not text_val: return 0.0
    clean = re.sub(r"[^\d\.kK]", "", str(text_val)).lower()
    try:
        if "k" in clean:
            return float(clean.replace("k", "")) * 1000
        return float(clean)
    except ValueError:
        return 0.0

def parse_payment_status(text_val):
    """
    Intelligent mapping of payment status descriptions to integers.
    """
    text = str(text_val).lower()
    
    # 1. Explicit delay numbers (e.g. "delayed by 2 months")
    delay_match = re.search(r"delayed? by (\d+)", text)
    if delay_match:
        return int(delay_match.group(1))

    # 2. Heuristic Mappings
    if any(x in text for x in ["time", "timely", "full", "clear"]):
        return 0  # Paid on time
    
    if any(x in text for x in ["minimum due", "partial", "late", "delay"]):
        return 1  # Moderate risk (Late/Partial)
        
    if any(x in text for x in ["no payment", "overdue", "missed"]):
        return 2  # High risk

    return 0  # Default to Safe

# --- UPDATED EXTRACTOR FUNCTION ---

def extract_text(file_content, filename=""):
    """
    Extracts text from file bytes (UploadFile content).
    Args:
        file_content (bytes): The binary content of the file.
        filename (str): The name of the file (to determine type).
    """
    text = ""
    filename = filename.lower()
    
    # Convert bytes to a file-like stream
    file_stream = io.BytesIO(file_content)

    # A. PDF HANDLING
    if filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception as e:
            print(f"⚠️ PDF Text Extract failed: {e}")

        # OCR Fallback for scanned PDFs
        if len(text.strip()) < 50:
            print(f"⚠️ Scanned PDF detected ({filename}). Switching to OCR...")
            try:
                # Reset stream position
                file_stream.seek(0)
                # Open with PyMuPDF (fitz)
                doc = fitz.open(stream=file_stream.read(), filetype="pdf")
                ocr_text = ""
                for page in doc:
                    pix = page.get_pixmap(dpi=200) # Render page to image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    ocr_text += pytesseract.image_to_string(image) + "\n"
                text = ocr_text
            except Exception as e:
                print(f"❌ PDF OCR failed: {e}")

    # B. IMAGE HANDLING
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            image = Image.open(file_stream)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"❌ Image OCR failed: {e}")

    return text


# --- MAIN EXTRACTOR FUNCTION ---

def extract_fields(text):
    clean_text = preprocess_text(text)
    extracted_data = {}
    
    short_context = clean_text[:2000]

    for field_key, config in FIELD_CONFIG.items():
        val = None
        
        # A. Try Regex First (Strict Mode)
        if "patterns" in config:
            for pat in config["patterns"]:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    break
        
        # B. AI Fallback (Transformers)
        if not val:
            try:
                result = qa_pipeline(question=config["question"], context=short_context)
                answer = result['answer']
                if len(answer) < 60:
                    val = answer
            except Exception:
                pass

        # C. Parse Data
        final_val = config.get("default", 0)

        if val:
            try:
                target_type = config["type"]
                
                if target_type == "int":
                    nums = re.findall(r"\d+", str(val))
                    if nums: final_val = int(nums[0])
                    
                elif target_type == "float" or target_type == "currency":
                    parsed_amt = parse_currency(val)
                    # --- SAFETY CHECK ---
                    # Ensure Debt isn't exactly the same as Income (common parsing error)
                    if field_key == "existing_monthly_debt" and parsed_amt == extracted_data.get("monthly_income", -1):
                        pass # Ignore this bad match, keep default
                    else:
                        final_val = parsed_amt
                    
                elif target_type == "payment_status":
                    final_val = parse_payment_status(val)
                    
                else:
                    final_val = str(val).strip()
                    
            except Exception:
                pass 

        extracted_data[field_key] = final_val

    return extracted_data