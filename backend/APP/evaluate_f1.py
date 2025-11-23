# evaluate_f1.py
import uuid
from datetime import datetime
from cassandra.cluster import Cluster
import os
import matplotlib.pyplot as plt
from loan_eligibility import calculate_credit_score
from sklearn.metrics import classification_report

CASSANDRA_HOST = os.getenv("CASSANDRA_HOSTS", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
KEYSPACE       = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")

LABELS = ["Eligible", "NotEligible"]
ARTIFACTS_DIR = "/app/eval_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
def connect():
    cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT)
    return cluster.connect(KEYSPACE)

def fetch_cases(session):
    rows = session.execute("SELECT * FROM eval_cases")
    return list(rows)

def is_eligible(row):
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

def predict(row):
    score = calculate_credit_score(
        age=row.age,
        monthly_income=row.monthly_income,
        existing_loans=row.existing_loans,
        employment_status=row.employment_status,
        car_value=row.requested_car_value
    )
    class R: pass
    r = R()
    r.credit_score = score
    r.monthly_income = row.monthly_income
    r.existing_loans = row.existing_loans
    r.employment_status = row.employment_status

    eligible, reason = is_eligible(r)
    y_pred = "Eligible" if eligible else "NotEligible"
    return y_pred, reason

def prf1(counts):
    precisions, recalls, f1s = [], [], []
    tp_sum = fp_sum = fn_sum = 0

    for lbl in LABELS:
        tp = counts[lbl]["tp"]
        fp = counts[lbl]["fp"]
        fn = counts[lbl]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        tp_sum += tp; fp_sum += fp; fn_sum += fn

    macro = {
        "precision": sum(precisions)/len(LABELS),
        "recall":    sum(recalls)/len(LABELS),
        "f1":        sum(f1s)/len(LABELS),
    }
    micro_prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    micro_rec  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    micro_f1   = (2*micro_prec*micro_rec)/(micro_prec+micro_rec) if (micro_prec+micro_rec) else 0.0

    micro = {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1}
    return macro, micro

def save_case_result(session, run_id, case_id, y_true, y_pred, reason):
    session.execute("""
    INSERT INTO eval_results (run_id, case_id, y_true, y_pred, match, reason)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (run_id, case_id, y_true, y_pred, (y_true==y_pred), reason))

def save_metrics(session, run_id, n, macro, micro, notes):
    session.execute("""
    INSERT INTO eval_metrics (run_id, timestamp, support, precision_macro, recall_macro, f1_macro,
                              precision_micro, recall_micro, f1_micro, notes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (run_id, datetime.utcnow(), n,
          macro["precision"], macro["recall"], macro["f1"],
          micro["precision"], micro["recall"], micro["f1"],
          notes))
    
plt.savefig(os.path.join(ARTIFACTS_DIR, "metrics_bar.png"))
plt.close()

plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))
plt.close()


def main():
    session = connect()
    cases = fetch_cases(session)
    if not cases:
        print("No eval cases found. Run seed_eval.py first.")
        return

    run_id = uuid.uuid4()
    counts = {lbl: {"tp":0,"fp":0,"fn":0} for lbl in LABELS}
    rows_print = []

    for row in cases:
        y_true = row.y_true
        y_pred, reason = predict(row)
        save_case_result(session, run_id, row.case_id, y_true, y_pred, reason)
        rows_print.append((str(row.case_id)[:8], y_true, y_pred, "✓" if y_true==y_pred else "✗", reason))

        for lbl in LABELS:
            if y_true == lbl and y_pred == lbl:
                counts[lbl]["tp"] += 1
            elif y_true != lbl and y_pred == lbl:
                counts[lbl]["fp"] += 1
            elif y_true == lbl and y_pred != lbl:
                counts[lbl]["fn"] += 1

    macro, micro = prf1(counts)
    save_metrics(session, run_id, len(cases), macro, micro,
                 notes="Logic-based evaluation of leasing eligibility.")

    print("\n=== Per-Case Results (sample) ===")
    print("{:<10} {:<13} {:<13} {:<3} {}".format("case_id","y_true","y_pred","ok","reason"))
    for r in rows_print[:15]:
        print("{:<10} {:<13} {:<13} {:<3} {}".format(*r))

    print("\n=== Aggregated Metrics ===")
    print(f"Support: {len(cases)}")
    print(f"Macro  P/R/F1: {macro['precision']:.3f} / {macro['recall']:.3f} / {macro['f1']:.3f}")
    print(f"Micro  P/R/F1: {micro['precision']:.3f} / {micro['recall']:.3f} / {micro['f1']:.3f}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
