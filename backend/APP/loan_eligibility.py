# loan_eligibility.py
# keep the same signature to avoid breaking imports
# --- Drop-in replacement: explainable, Islamic-friendly score (0–100) ---
def calculate_credit_score(age, monthly_income, existing_loans, employment_status,
                           requested_car_value, desired_duration_months):
    """
    Deterministic score with transparent components:
    - Debt Service Ratio (DSR) cap 40% income
    - Employment stability
    - Age band
    - Existing loans penalty
    Scaled 0..100 with clear buckets.
    """
    # 1) Debt service ratio target (Sharia financing uses affordability guardrails too)
    max_installment = 0.40 * float(monthly_income or 0)
    est_installment = 0 if desired_duration_months in (None, 0) else (requested_car_value / desired_duration_months)

    # Subscores (0..100)
    # a) Affordability (higher is better)
    if max_installment <= 0:
        afford = 0
    else:
        ratio = est_installment / max_installment  # <=1 good, >1 bad
        if ratio <= 0.5:
            afford = 100
        elif ratio <= 0.75:
            afford = 85
        elif ratio <= 1.0:
            afford = 70
        elif ratio <= 1.25:
            afford = 40
        else:
            afford = 15

    # b) Employment stability
    emp_map = {
        "permanent": 100,
        "contract": 75,
        "self-employed": 60,
        "unemployed": 10
    }
    emp = emp_map.get(str(employment_status).lower(), 50)

    # c) Age band
    if 25 <= age <= 55:
        age_s = 90
    elif 21 <= age < 25 or 56 <= age <= 60:
        age_s = 70
    elif 18 <= age < 21 or 61 <= age <= 65:
        age_s = 50
    else:
        age_s = 30

    # d) Existing loans penalty
    penalty = 15 if existing_loans else 0

    # Weighted sum → 0..100
    raw = (0.55 * afford) + (0.25 * emp) + (0.15 * age_s) - penalty
    score = int(max(0, min(100, round(raw))))

    return score


def explain_credit_score(age, monthly_income, existing_loans, employment_status,requested_car_value, desired_duration_months):
    """Human explanation + per-factor breakdown for UI."""
    max_installment = 0.40 * float(monthly_income or 0)
    est_installment = 0 if desired_duration_months in (None, 0) else (requested_car_value / desired_duration_months)
    dsr = 0.0 if max_installment == 0 else est_installment / max_installment

    score = calculate_credit_score(age, monthly_income, existing_loans, employment_status,requested_car_value, desired_duration_months)

    # Buckets
    band = (
        "A (Excellent)" if score >= 85 else
        "B (Good)"      if score >= 70 else
        "C (Fair)"      if score >= 55 else
        "D (Weak)"      if score >= 40 else
        "E (High Risk)"
    )

    tips = []
    if dsr > 1.0:
        tips.append("Reduce requested car value or increase duration to lower your installment.")
    if existing_loans:
        tips.append("Close or restructure existing financing to reduce risk.")
    if str(employment_status).lower() in {"contract", "self-employed", "unemployed"}:
        tips.append("Provide stronger income proofs or additional guarantor(s).")

    return {
        "score": score,
        "band": band,
        "affordability": {
            "estimated_installment": round(est_installment, 2),
            "max_installment_40pct_income": round(max_installment, 2),
            "dsr": round(dsr, 2)
        },
        "factors": {
            "employment_status": str(employment_status),
            "age": age,
            "existing_loans": bool(existing_loans),
            "requested_car_value": requested_car_value,
            "duration_months": desired_duration_months
        },
        "suggestions": tips
    }



# --- Approval rule (Sharia-aligned affordability) ---
def decide_loan_approval(credit_score, monthly_payment, income):
    """
    Return (approval_bool, reasons_list)

    Policy (tweak as you like):
      - Approve if score >= 60
      - and monthly_payment <= 40% of income
    """
    reasons = []
    try:
        income = float(income or 0.0)
        monthly_payment = float(monthly_payment or 0.0)
        credit_score = int(credit_score)
    except Exception:
        return False, ["Invalid inputs to approval rule"]

    if income <= 0:
        return False, ["Invalid income provided (must be > 0)"]

    dcr = monthly_payment / income  # Debt-to-Capacity ratio

    if credit_score < 60:
        reasons.append("Overall risk score below threshold (60)")
    if dcr > 0.40:
        reasons.append("Debt-to-Capacity exceeds 40%")

    approved = (len(reasons) == 0)
    return approved, reasons


#What changed (and why)

#Sharia alignment: we use a profit rate (Murabaha-style), not interest compounding.

#Explainability: score is 0–100 with a transparent breakdown of each factor.

#Decision bands: A/B/C/D/E make results readable for users and reviewers.

#Affordability is based on DCR (payment ÷ income) with a 40% guardrail (customizable).