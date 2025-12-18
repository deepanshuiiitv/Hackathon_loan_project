  # Decision policy
def decision_policy(pd: float):
    if pd < 5:
        return "APPROVED", "Low Interest"
    elif pd <= 15:
        return "APPROVED", "High Interest"
    else:
        return "REJECTED", None
