from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import random


# Feature mapping from your code
FEATURES = {
    "AGE": "the applicant's age",
    "monthly_income": "the applicant's monthly income",
    "DTI": "the proportion of income used for existing debt payments",
    "PAY_0": "most recent payment status",
    "PAY_2": "payment status from two months ago",
    "PAY_3": "payment status from three months ago",
    "BILL_AMT1": "most recent outstanding credit card balance",
    "BILL_AMT2": "outstanding credit card balance from the previous month",
}


# # Initialize the LLM pipeline
# hf_pipeline = pipeline(
#     task="text2text-generation",
#     model="google/flan-t5-base",
#     max_new_tokens=150,
#     do_sample=True,
#     temperature=0.7,
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

def map_feature_to_text(feature_key):
    """Convert feature key to human-readable text"""
    feature_key = str(feature_key).strip()
    
    if feature_key in FEATURES:
        return FEATURES[feature_key]
    
    for key, value in FEATURES.items():
        if key.lower() == feature_key.lower():
            return value
    
    return feature_key

def llm_explanation(decision, interest, factors):
    """
    Generate polite loan decision explanations
    
    Parameters:
    - decision: "APPROVED" or "DISAPPROVED" (any case)
    - interest: Interest type string (e.g., "Low Interest")
    - factors: List of tuples like [("PAY_0", -0.5089), ("BILL_AMT1", -0.2395)]
    """
    # Extract and map feature names to human-readable text
    FEATURESs = []
    for factor in factors:
        if isinstance(factor, (list, tuple)) and len(factor) >= 1:
            feature_key = factor[0]
            FEATURES = map_feature_to_text(feature_key)
            FEATURESs.append(FEATURES)
    
    # Create the reasons string
    if len(FEATURESs) == 0:
        reasons = "various application factors"
    elif len(FEATURESs) == 1:
        reasons = FEATURESs[0]
    elif len(FEATURESs) == 2:
        reasons = f"{FEATURESs[0]} and {FEATURESs[1]}"
    else:
        reasons = ", ".join(FEATURESs[:-1]) + f", and {FEATURESs[-1]}"
    
    # Normalize inputs
    decision_lower = str(decision).lower().strip()
    interest_lower = str(interest).lower().strip()
    
    # Template-based approach
    if decision_lower == "approved":
        templates = [
            f"We're pleased to approve your loan application. Our decision considered factors including {reasons}. You qualify for {interest_lower} interest.",
            f"Congratulations! Your loan application has been approved based on our review of {reasons}. Your approved interest rate is {interest_lower}.",
            f"Good news - your loan application is approved. We assessed factors such as {reasons}, and you qualify for {interest_lower} interest.",
            f"Your loan application has been approved. After reviewing {reasons}, we're happy to offer you {interest_lower} interest.",
            f"We are pleased to inform you that your loan has been approved. This decision was based on {reasons}, and your interest rate will be {interest_lower}.",
        ]
    else:  # disapproved or denied
        templates = [
            f"We regret that we cannot approve your loan application at this time. We understand this is disappointing. Our decision considered {reasons}.",
            f"After careful review, we're unable to approve your loan application. We recognize this may not be the news you hoped for. Factors considered include {reasons}.",
            f"Unfortunately, we cannot approve your loan request. We appreciate your application and understand this may be difficult. Our assessment reviewed {reasons}.",
            f"Your loan application has not been approved. We understand this may be disappointing. The decision was based on factors like {reasons}.",
            f"After evaluating your application, we are unable to approve your loan at this time. We considered {reasons} and understand this may be disappointing news.",
        ]
    
    explanation = random.choice(templates)
    return explanation