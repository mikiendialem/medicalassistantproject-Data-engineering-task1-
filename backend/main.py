"""
Symptom-to-possible-conditions triage API.

This is a DECISION-SUPPORT / EDUCATIONAL tool, not a diagnostic device.
It ranks possible conditions based on reported symptoms using a classical
ML model trained on a small public dataset. It does not replace clinical
evaluation. Every response includes a disclaimer and confidence scores.
"""
import json
import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")


def resolve_model_dir() -> Path:
    candidates = [
        BASE_DIR.parent / "model",
        BASE_DIR / "model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_DIR = resolve_model_dir()
FRONTEND_DIR = BASE_DIR.parent / "frontend"
CONFIG_PATH = FRONTEND_DIR / "config.json"

app = FastAPI(title="Symptom Triage Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model artifacts once at startup ---
classifier_path = MODEL_DIR / "disease_classifier.joblib"
symptom_list_path = MODEL_DIR / "symptom_list.json"
disease_list_path = MODEL_DIR / "disease_list.json"

if not classifier_path.exists() or not symptom_list_path.exists() or not disease_list_path.exists():
    missing = [
        str(path)
        for path in (classifier_path, symptom_list_path, disease_list_path)
        if not path.exists()
    ]
    raise RuntimeError(
        "Missing model artifacts. Run train.py first or place the files in one of the expected model directories: "
        + ", ".join(missing)
    )

model = joblib.load(classifier_path)
with open(symptom_list_path) as f:
    SYMPTOM_LIST: List[str] = json.load(f)
with open(disease_list_path) as f:
    DISEASE_LIST: List[str] = json.load(f)

DISCLAIMER = (
    "This tool provides ranked possibilities based on a machine learning model "
    "and is NOT a medical diagnosis. It is trained on a small demo dataset and "
    "should not be relied on for real health decisions. Always consult a "
    "licensed healthcare professional for any health concern. If you are "
    "experiencing a medical emergency, contact emergency services immediately."
)


class SymptomRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptom keys the user is experiencing")
    top_k: int = Field(5, ge=1, le=10)


class PredictionItem(BaseModel):
    condition: str
    confidence: float


class PredictionResponse(BaseModel):
    results: List[PredictionItem]
    disclaimer: str
    unrecognized_symptoms: List[str]


class InfermedicaEvidenceItem(BaseModel):
    id: str
    choice_id: str
    source: str | None = None


class InfermedicaDiagnosisRequest(BaseModel):
    sex: str = Field(..., description="Assigned sex at birth, for the Infermedica API")
    age: int = Field(..., ge=0, le=130)
    evidence: List[InfermedicaEvidenceItem] = Field(..., min_length=1)


class InfermedicaDiagnosisResponse(BaseModel):
    raw_response: dict
    disclaimer: str


INFERMEDICA_DIAGNOSIS_URL = os.environ.get(
    "INFERMEDICA_DIAGNOSIS_URL", "https://api.infermedica.com/v3/diagnosis"
)
INFERMEDICA_APP_ID = os.environ.get("INFERMEDICA_APP_ID", "")
INFERMEDICA_APP_KEY = os.environ.get("INFERMEDICA_APP_KEY", "")


@app.get("/symptoms")
def get_symptoms():
    """Return the full list of symptom keys the model understands."""
    return {"symptoms": sorted(SYMPTOM_LIST)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_frontend_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/frontend/index.html", include_in_schema=False)
def serve_legacy_frontend_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/config.json", include_in_schema=False)
def serve_config_json():
    return FileResponse(CONFIG_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: SymptomRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Provide at least one symptom.")

    recognized = [s for s in req.symptoms if s in SYMPTOM_LIST]
    unrecognized = [s for s in req.symptoms if s not in SYMPTOM_LIST]

    if not recognized:
        raise HTTPException(
            status_code=400,
            detail="None of the provided symptoms were recognized. Use GET /symptoms for valid options.",
        )

    # Build the one-hot feature vector the model expects
    input_vector = pd.DataFrame(
        [[1 if s in recognized else 0 for s in SYMPTOM_LIST]], columns=SYMPTOM_LIST
    )

    proba = model.predict_proba(input_vector)[0]
    classes = model.classes_

    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[: req.top_k]

    results = [
        PredictionItem(condition=cond, confidence=round(float(prob), 4))
        for cond, prob in ranked
    ]

    return PredictionResponse(
        results=results,
        disclaimer=DISCLAIMER,
        unrecognized_symptoms=unrecognized,
    )


@app.post("/github/infermedica/diagnosis", response_model=InfermedicaDiagnosisResponse)
def infermedica_diagnosis(req: InfermedicaDiagnosisRequest):
    if not INFERMEDICA_APP_ID or not INFERMEDICA_APP_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Infermedica credentials are not configured. Set INFERMEDICA_APP_ID and "
                "INFERMEDICA_APP_KEY to use the GitHub-sourced API integration."
            ),
        )

    payload = {
        "sex": req.sex,
        "age": {"value": req.age},
        "evidence": [item.model_dump(exclude_none=True) for item in req.evidence],
    }

    try:
        response = requests.post(
            INFERMEDICA_DIAGNOSIS_URL,
            json=payload,
            headers={
                "App-Id": INFERMEDICA_APP_ID,
                "App-Key": INFERMEDICA_APP_KEY,
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Infermedica API request failed: {exc}") from exc

    try:
        response_data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Infermedica API returned invalid JSON.") from exc

    return InfermedicaDiagnosisResponse(raw_response=response_data, disclaimer=DISCLAIMER)
