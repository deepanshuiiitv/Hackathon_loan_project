import pdfplumber
import re
import spacy
from transformers import pipeline
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# --- MODEL INITIALIZATION ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

print("Loading Transformers Pipeline...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text(file_path):
    """
    Hybrid Extractor: 
    1. Handles Images (.jpg, .png) using OCR.
    2. Handles PDFs using Digital Extraction first.
    3. Fallback: Uses PyMuPDF + OCR for scanned PDFs.
    """
    text = ""
    ext = os.path.splitext(file_path)[1].lower()

    # --- CASE 1: IMAGE FILES ---
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        print(f"🖼️ Processing Image: {file_path}")
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"❌ Image OCR failed: {e}")

    # --- CASE 2: PDF FILES ---
    elif ext == ".pdf":
        print(f"📄 Processing PDF: {file_path}")
        
        # Attempt A: Digital Extraction (Fast)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            print(f"Digital extraction error: {e}")

        # Attempt B: OCR Fallback (PyMuPDF + Tesseract)
        if len(text.strip()) < 50:
            print("⚠️ Scanned PDF detected. Switching to OCR (via PyMuPDF)...")
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(file_path)
                ocr_text = ""
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Render page to an image (pixmap)
                    pix = page.get_pixmap(dpi=300)
                    
                    # Convert raw bytes to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Run OCR on the image
                    ocr_text += pytesseract.image_to_string(image) + "\n"
                
                text = ocr_text
            except Exception as e:
                print(f"❌ OCR Failed: {e}. Ensure Tesseract is installed.")

    return text

def preprocess_text(text):
    if not text: return ""
    text = text.replace('"', '').replace("'", "")
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_income_value(text_val):
    if not text_val: return 0.0
    clean = re.sub(r"[^\d\.kK]", "", text_val).lower()
    try:
        if "k" in clean:
            return float(clean.replace("k", "")) * 1000
        return float(clean)
    except ValueError:
        return 0.0

def extract_fields(text):
    clean_text = preprocess_text(text)
    
    data = {
        "name": None,
        "income": 0.0,
        "age": 30,
        "credit_score": 650
    }

    # --- 1. NAME EXTRACTION ---
    try:
        short_context = clean_text[:500] 
        result = qa_pipeline(question="What is the name of the applicant?", context=short_context)
        candidate_name = result['answer']
        if len(candidate_name) > 3 and not any(char.isdigit() for char in candidate_name):
            data["name"] = re.sub(r"^(Mr\.|Ms\.|Mrs\.|Dr\.)\s*", "", candidate_name).strip()
    except Exception:
        pass

    if not data["name"]:
        name_patterns = [
            r"Name[:\-\s]+([A-Za-z\s\.]+)", 
            r"Applicant[:\-\s]+([A-Za-z\s\.]+)"
        ]
        for pat in name_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip()
                data["name"] = " ".join(raw_name.split()[:3]) 
                break

    # --- 2. INCOME EXTRACTION ---
    income_patterns = [
        r"(?:Income|Salary|Earnings).*?([\d,]+(?:k|000)?)", 
        r"Rs\.?\s*([\d,]+)", 
        r"INR\s*([\d,]+)"
    ]
    for pat in income_patterns:
        match = re.search(pat, clean_text, re.IGNORECASE)
        if match:
            val = parse_income_value(match.group(1))
            if 1000 < val < 10000000:
                data["income"] = val
                break

    # --- 3. AGE EXTRACTION ---
    try:
        short_context = clean_text[:1000]
        result = qa_pipeline(question="What is the age of the applicant?", context=short_context)
        age_match = re.search(r"(\d{2})", result['answer'])
        if age_match:
            data["age"] = int(age_match.group(1))
    except Exception:
        pass

    # --- 4. CREDIT SCORE EXTRACTION ---
    score_found = False
    
    # Attempt 1: Transformers
    try:
        result = qa_pipeline(question="What is the credit score?", context=clean_text[:1000])
        score_match = re.search(r"\b(\d{3})\b", result['answer'])
        if score_match:
            score = int(score_match.group(1))
            if 300 <= score <= 900:
                data["credit_score"] = score
                score_found = True
    except Exception:
        pass

    # Attempt 2: Relaxed Regex
    if not score_found:
        relaxed_patterns = [
            r"Credit Score.{0,50}?(\d{3})", 
            r"CIBIL.{0,50}?(\d{3})",
            r"Score.{0,50}?(\d{3})"
        ]
        for pat in relaxed_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    data["credit_score"] = score
                    score_found = True
                    break

    # Attempt 3: SpaCy Matcher
    if not score_found and nlp:
        doc = nlp(clean_text)
        matcher = spacy.matcher.Matcher(nlp.vocab)
        p1 = [{"LOWER": "credit"}, {"LOWER": "score", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LIKE_NUM": True}]
        p2 = [{"LOWER": "cibil"}, {"LOWER": "score", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LIKE_NUM": True}]
        matcher.add("CREDIT_SCORE", [p1, p2])
        matches = matcher(doc)
        for _, start, end in matches:
            try:
                val_text = re.sub(r"\D", "", doc[start:end][-1].text)
                score = int(val_text)
                if 300 <= score <= 900:
                    data["credit_score"] = score
                    break
            except ValueError:
                continue

    return data

def calculate_metrics(data):
    income = data["income"]
    credit = data["credit_score"]
    age = data["age"]

    dti = 0.3 if income > 30000 else 0.6
    
    risk_score = (
        (750 - credit) * 0.5 +
        dti * 100 +
        (40 - age) * 0.2
    )

    if risk_score < 50:
        decision = "APPROVE (LOW INTEREST)"
    elif risk_score < 100:
        decision = "APPROVE (HIGHER INTEREST)"
    else:
        decision = "REJECT"

    return dti, risk_score, decision