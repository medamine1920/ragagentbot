# evaluate_model.py
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from credit_model import make_pipeline
from sklearn.model_selection import train_test_split

def evaluate(df: pd.DataFrame):
    df = df.dropna(subset=["age","monthly_income","requested_car_value","desired_duration_months",
                           "employment_status","existing_loans","defaulted"])
    X = df[["age","monthly_income","requested_car_value","desired_duration_months",
            "employment_status","existing_loans"]]
    y = df["defaulted"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    pipe = make_pipeline()
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    yproba = pipe.predict_proba(Xte)[:,1]

    print("Confusion matrix:\n", confusion_matrix(yte, yhat))
    print("\nClassification report:\n", classification_report(yte, yhat, digits=3))
    print("ROC AUC:", roc_auc_score(yte, yproba))
