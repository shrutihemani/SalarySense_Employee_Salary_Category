"""Train the SalarySense employee salary-category classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

def make_dataset(n=1200, seed=42):
    """Create a reproducible demonstration dataset; not real employee/payroll data."""
    rng = np.random.default_rng(seed)
    years = rng.integers(0, 26, n)
    education = rng.choice(["Diploma", "Bachelor", "Master", "PhD"], n, p=[.15, .48, .29, .08])
    level = rng.choice(["Entry", "Associate", "Mid", "Senior", "Lead"], n, p=[.22, .2, .25, .22, .11])
    department = rng.choice(["IT", "Finance", "HR", "Sales", "Operations", "Marketing"], n)
    edu_points = pd.Series(education).map({"Diploma": 0, "Bachelor": 1, "Master": 2, "PhD": 3}).to_numpy()
    level_points = pd.Series(level).map({"Entry": 0, "Associate": 1, "Mid": 2, "Senior": 3, "Lead": 4}).to_numpy()
    dept_points = pd.Series(department).map({"IT": 1.1, "Finance": .8, "HR": .3, "Sales": .7, "Operations": .5, "Marketing": .6}).to_numpy()
    score = years * .42 + edu_points * .8 + level_points * 1.45 + dept_points + rng.normal(0, 1.1, n)
    # Three categories are created from a transparent synthetic score.
    category = np.where(score < 4.0, "Entry Salary", np.where(score < 8.0, "Mid Salary", "High Salary"))
    return pd.DataFrame({
        "years_experience": years,
        "education_level": education,
        "job_level": level,
        "department": department,
        "salary_category": category
    })

def train():
    data = make_dataset()
    X = data[["years_experience", "education_level", "job_level", "department"]]
    y = data["salary_category"]
    numeric = ["years_experience"]
    categorical = ["education_level", "job_level", "department"]
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced"))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"Evaluation accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to: {MODEL_PATH}")
    print("Note: this is a synthetic educational model, not a real salary estimator.")

if __name__ == "__main__":
    train()
