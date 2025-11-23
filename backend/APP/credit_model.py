# credit_model.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("/app/data/models/credit_model.joblib")

FEATURES_NUM = ["age", "monthly_income", "requested_car_value", "desired_duration_months"]
FEATURES_CAT = ["employment_status", "existing_loans"]

def make_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])

def train_and_save(df: pd.DataFrame):
    """
    df must contain FEATURES + 'defaulted' (0/1).
    """
    df = df.dropna(subset=FEATURES_NUM + FEATURES_CAT + ["defaulted"])
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df["defaulted"].astype(int)

    pipe = make_pipeline()
    pipe.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict_proba_single(model, row: dict) -> float:
    X = pd.DataFrame([{
        "age": row["age"],
        "monthly_income": row["monthly_income"],
        "requested_car_value": row["requested_car_value"],
        "desired_duration_months": row["desired_duration_months"],
        "employment_status": row["employment_status"],
        "existing_loans": bool(row["existing_loans"]),
    }])
    p = model.predict_proba(X)[0,1]
    return float(p)
