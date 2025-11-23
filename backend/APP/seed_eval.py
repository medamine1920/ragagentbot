# seed_eval.py
import uuid
from datetime import datetime
import random
from cassandra.cluster import Cluster
import os

from loan_eligibility import calculate_credit_score, calculate_monthly_plan

CASSANDRA_HOST = os.getenv("CASSANDRA_HOSTS", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
KEYSPACE       = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")

def connect():
    cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT)
    return cluster.connect(KEYSPACE)

def is_eligible(row):
    """Simple rule-based eligibility check"""
    reasons = []
    if row.credit_score < 600:
        reasons.append("Low credit score")
    if row.monthly_income < 800:
        reasons.append("Income below threshold")
    if row.existing_loans:
        reasons.append("Has existing loans")
    if row.employment_status in ["unemployed", "temporary"]:
        reasons.append("Unstable employment")

    eligible = len(reasons) == 0
    return eligible, "; ".join(reasons) if reasons else "Eligible"

def _norm_job(job: str) -> str:
    if not isinstance(job, str):
        job = str(job or "")
    j = job.strip().lower()
    if j in {"permanent", "full time", "full-time"}:
        return "full-time"
    if j in {"contractor", "contract"}:
        return "contract"
    if j in {"self employed", "self-employed", "freelance"}:
        return "self-employed"
    if j in {"temp", "temporary"}:
        return "temporary"
    if j in {"unemployed", "jobless"}:
        return "unemployed"
    return j

def make_case(fn, nid, age, job, income, has_loans, car_value, months, law_note):
    job_clean = _norm_job(job)

    score = calculate_credit_score(
        age=age,
        monthly_income=income,
        existing_loans=has_loans,
        employment_status=job_clean,
        car_value=car_value
    )

    class Row: pass
    r = Row()
    r.credit_score = score
    r.monthly_income = float(income)
    r.existing_loans = bool(has_loans)
    r.employment_status = job_clean

    eligible, _ = is_eligible(r)
    y_true = "Eligible" if eligible else "NotEligible"

    return {
        "case_id": uuid.uuid4(),
        "full_name": fn,
        "national_id": nid,
        "age": int(age),
        "employment_status": job_clean,
        "monthly_income": float(income),
        "existing_loans": bool(has_loans),
        "requested_car_value": float(car_value),
        "desired_duration_months": int(months),
        "y_true": y_true,
        "law_refs": law_note,
        "domain": "leasing",
        "created_at": datetime.utcnow()
    }

def seed(n=64):
    names  = ["Aymen", "Maya", "Salma", "Karim", "Nour", "Yassine", "Leila", "Rami"]
    jobs   = ["permanent", "full-time", "contract", "self-employed", "unemployed", "temporary"]
    ages   = [23, 27, 35, 46, 52]
    incomes= [600, 900, 1200, 2100, 3000, 4200]
    loans  = [True, False]
    cars   = [25000, 38000, 45000, 82000]
    durs   = [24, 36, 48, 60]

    NOTES = {
        "income":   "Meets/violates affordability threshold per policy.",
        "loans":    "Existing obligations beyond acceptable risk.",
        "emp":      "Employment stability requirement not met.",
        "general":  "Approved per aggregate policy checks."
    }

    random.seed(7)
    cases = []
    while len(cases) < n:
        fn = random.choice(names)
        nid = str(random.randint(10_000_000, 99_999_999))
        age = random.choice(ages)
        job = random.choice(jobs)
        inc = random.choice(incomes)
        has = random.choice(loans)
        car = random.choice(cars)
        dur = random.choice(durs)

        if inc < 800:
            note = NOTES["income"]
        elif has:
            note = NOTES["loans"]
        elif _norm_job(job) in ["unemployed", "temporary"]:
            note = NOTES["emp"]
        else:
            note = NOTES["general"]

        cases.append(make_case(fn, nid, age, job, inc, has, car, dur, note))

    return cases

def save_cases(session, cases):
    q = """
    INSERT INTO eval_cases (
      case_id, full_name, national_id, age, employment_status, monthly_income,
      existing_loans, requested_car_value, desired_duration_months, y_true,
      law_refs, domain, created_at
    ) VALUES (%(case_id)s, %(full_name)s, %(national_id)s, %(age)s, %(employment_status)s,
      %(monthly_income)s, %(existing_loans)s, %(requested_car_value)s,
      %(desired_duration_months)s, %(y_true)s, %(law_refs)s, %(domain)s, %(created_at)s)
    """
    for c in cases:
        session.execute(q, c)

if __name__ == "__main__":
    s = connect()
    cases = seed(n=64)
    save_cases(s, cases)
    print(f"Seeded {len(cases)} eval cases ✔")
