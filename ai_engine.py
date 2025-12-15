import pdfplumber
import re
import spacy
from transformers import pipeline

# --- MODEL INITIALIZATION ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model isn't downloaded
    print("⚠️ SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Load Transformers (using a faster model for CPU efficiency)
print("Loading Transformers Pipeline...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text(pdf_path):
    """
    Reads text from a digital PDF.
    Since your files are digital (from Word), we do NOT need OCR.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract_text() is perfect for digital PDFs
            text += page.extract_text() or ""
    return text

def preprocess_text(text):
    """
    CRITICAL STEP: Cleans up the messy output from different PDF layouts.
    """
    # 1. Remove quotes (fixes the CSV-style issue in Sample 8)
    text = text.replace('"', '').replace("'", "")
    
    # 2. Replace newlines with spaces (fixes broken sentences)
    text = text.replace('\n', ' ')
    
    # 3. Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_income_value(text_val):
    """Converts '$50,000', '50k', 'Rs 50000' to float."""
    if not text_val: return 0.0
    # Remove currency symbols and commas
    clean = re.sub(r"[^\d\.kK]", "", text_val).lower()
    try:
        if "k" in clean:
            return float(clean.replace("k", "")) * 1000
        return float(clean)
    except ValueError:
        return 0.0

def extract_fields(text):
    # --- STEP 1: CLEAN THE INPUT ---
    # This solves the issue where "Income" and "90000" are on different lines
    clean_text = preprocess_text(text)
    
    data = {
        "name": None,
        "income": 0.0,
        "age": 30,
        "credit_score": 650
    }

    # --- 1. NAME EXTRACTION (Transformers + Regex) ---
    # Attempt 1: Transformers (Best for narrative texts like "submitted by Ms. Priya...")
    try:
        short_context = clean_text[:500] # Name is usually at the top
        result = qa_pipeline(question="What is the name of the applicant?", context=short_context)
        candidate_name = result['answer']
        
        # Validation: A valid name usually has 2+ words and no digits
        if len(candidate_name) > 3 and not any(char.isdigit() for char in candidate_name):
            # Clean up titles like "Mr.", "Ms."
            data["name"] = re.sub(r"^(Mr\.|Ms\.|Mrs\.|Dr\.)\s*", "", candidate_name).strip()
    except Exception as e:
        print(f"Name QA failed: {e}")

    # Attempt 2: Regex Fallback (Best for forms like "Name: John Doe")
    if not data["name"]:
        name_patterns = [
            r"Name[:\-\s]+([A-Za-z\s\.]+)", 
            r"Applicant[:\-\s]+([A-Za-z\s\.]+)"
        ]
        for pat in name_patterns:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                # Take first 3 words max to avoid capturing garbage text
                raw_name = match.group(1).strip()
                data["name"] = " ".join(raw_name.split()[:3]) 
                break

    # --- 2. INCOME EXTRACTION (Regex + SpaCy) ---
    # Strategy: Regex is often safer for Income than NLP for simple forms
    # We look for "Income" followed by ANY characters (.*?) until a number appear
    income_patterns = [
        r"(?:Income|Salary|Earnings).*?([\d,]+(?:k|000)?)", # Matches: "Income... 90000"
        r"Rs\.?\s*([\d,]+)",                                 # Matches: "Rs. 90,000"
        r"INR\s*([\d,]+)"                                    # Matches: "INR 90,000"
    ]
    
    for pat in income_patterns:
        match = re.search(pat, clean_text, re.IGNORECASE)
        if match:
            val = parse_income_value(match.group(1))
            # Filter: Income is usually > 1000 and < 10,000,000
            if 1000 < val < 10000000:
                data["income"] = val
                break

    # --- 3. AGE EXTRACTION (Transformers) ---
    # Transformers work best on "Natural Language" (sentences)
    # We pass the cleaned text so it understands the context better
    try:
        # We assume the age is in the first 1000 characters to save speed
        short_context = clean_text[:1000]
        result = qa_pipeline(question="What is the age of the applicant?", context=short_context)
        
        # Extract digits from the answer (e.g., "28 years" -> 28)
        age_str = result['answer']
        age_match = re.search(r"(\d{2})", age_str)
        if age_match:
            data["age"] = int(age_match.group(1))
    except Exception as e:
        print(f"Transformers QA error: {e}")

    # --- 4. CREDIT SCORE EXTRACTION (SpaCy) ---
    # Strategy: Try SpaCy first (Context aware), then Regex (Pattern strict)
    score_found = False
    # Attempt 1: Transformers (BEST for Narrative sentences like "score is 760")
    try:
        # We ask specifically for the score
        result = qa_pipeline(question="What is the credit score?", context=clean_text[:1000])
        answer = result['answer']
        # Look for a 3-digit number in the answer
        score_match = re.search(r"\b(\d{3})\b", answer)
        if score_match:
            score = int(score_match.group(1))
            if 300 <= score <= 900:
                data["credit_score"] = score
                score_found = True
    except Exception as e:
        print(f"Credit Score QA failed: {e}")

    # Attempt 2: Relaxed Regex (Allows words in between)
    if not score_found:
        # Pattern explanation:
        # "Credit Score" OR "CIBIL"
        # followed by ANY text (.{0,50}?) up to 50 chars (non-greedy)
        # followed by a 3-digit number (\d{3})
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

    # Attempt 3: Strict SpaCy Matcher (Fallback for structured forms)
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

    # Metric Logic
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