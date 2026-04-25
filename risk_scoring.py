def get_risk_level(prob):
    score = round(prob * 100, 2)

    if score < 30:
        return score, "SAFE", "Image is likely authentic."
    elif score < 80:
        return score, "SUSPICIOUS", "Image may be AI-generated. Verify source."
    else:
        return score, "HIGH", "High probability of AI-generated content."