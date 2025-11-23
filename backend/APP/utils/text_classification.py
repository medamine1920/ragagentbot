def guess_domain_from_text(text: str) -> str:
    text = text.lower()

    if any(word in text for word in ["invoice", "balance sheet", "expense", "profit", "financial"]):
        return "finance"
    elif any(word in text for word in ["contract", "court", "legal", "agreement", "law"]):
        return "legal"
    elif any(word in text for word in ["AI", "python", "server", "algorithm", "machine learning", "API"]):
        return "tech"
    elif any(word in text for word in ["hospital", "patient", "medical", "symptom", "diagnosis"]):
        return "health"
    else:
        return "general"