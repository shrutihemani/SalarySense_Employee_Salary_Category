from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

EDUCATION_LEVELS = ["Diploma", "Bachelor", "Master", "PhD"]
JOB_LEVELS = ["Entry", "Associate", "Mid", "Senior", "Lead"]
DEPARTMENTS = ["IT", "Finance", "HR", "Sales", "Operations", "Marketing"]

def validate_payload(payload):
    errors = []
    try:
        years = int(payload.get("years_experience", ""))
        if not 0 <= years <= 50:
            errors.append("Years of experience must be between 0 and 50.")
    except (TypeError, ValueError):
        years = None
        errors.append("Years of experience must be a whole number.")
    education = payload.get("education_level", "")
    level = payload.get("job_level", "")
    department = payload.get("department", "")
    if education not in EDUCATION_LEVELS:
        errors.append("Select a valid education level.")
    if level not in JOB_LEVELS:
        errors.append("Select a valid job level.")
    if department not in DEPARTMENTS:
        errors.append("Select a valid department.")
    if errors:
        return None, errors
    return {
        "years_experience": years,
        "education_level": education,
        "job_level": level,
        "department": department,
    }, []

@app.get("/")
def index():
    return render_template(
        "index.html",
        education_levels=EDUCATION_LEVELS,
        job_levels=JOB_LEVELS,
        departments=DEPARTMENTS,
        prediction=None,
        error=None,
        form_data={}
    )

@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None), 200 if model is not None else 503

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return jsonify(
            endpoint="/prediction",
            method="POST",
            required_fields=["years_experience", "education_level", "job_level", "department"]
        )
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    payload = payload or {}
    values, errors = validate_payload(payload)
    if errors:
        if request.is_json:
            return jsonify(errors=errors), 400
        return render_template("index.html", education_levels=EDUCATION_LEVELS,
            job_levels=JOB_LEVELS, departments=DEPARTMENTS, prediction=None,
            error=" ".join(errors), form_data=payload), 400
    if model is None:
        return jsonify(error="Model is not available. Run python train_model.py first."), 503
    frame = pd.DataFrame([values])
    predicted = str(model.predict(frame)[0])
    if request.is_json:
        return jsonify(predicted_salary_category=predicted, input=values)
    return render_template("index.html", education_levels=EDUCATION_LEVELS,
        job_levels=JOB_LEVELS, departments=DEPARTMENTS, prediction=predicted,
        error=None, form_data=values)

@app.errorhandler(404)
def not_found(_error):
    return jsonify(error="Route not found"), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
