from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from src.preprocessing import preprocess_text


MODEL_PATH = Path("models/spam_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")
METADATA_PATH = Path("models/metadata.json")


@lru_cache(maxsize=1)
def load_model_bundle() -> tuple[Any, Any, Dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Vectorizer file not found. Train the model first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Model metadata not found.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, vectorizer, metadata


def _safe_probability(model: Any, features: Any) -> Optional[float]:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)
            if probabilities.size == 0:
                return None
            positive_index = 1 if model.classes_.tolist() == [0, 1] else np.argmax(model.classes_)
            return float(probabilities[0][positive_index])
        except Exception:
            return None
    return None


def predict_message(message: str) -> Dict[str, Any]:
    if message is None or not str(message).strip():
        raise ValueError("Please enter a message before analyzing.")

    if len(str(message).strip()) > 5000:
        raise ValueError("Message is too long. Please keep it under 5000 characters.")

    model, vectorizer, metadata = load_model_bundle()
    cleaned_message = preprocess_text(message)
    if not cleaned_message:
        raise ValueError("Message is empty after preprocessing.")

    features = vectorizer.transform([cleaned_message])
    prediction = model.predict(features)[0]

    result_label = "SPAM" if int(prediction) == 1 else "NOT SPAM"
    probability = _safe_probability(model, features)
    confidence = round(float(probability) * 100, 2) if probability is not None else None

    explanation = explain_prediction(model, vectorizer, cleaned_message)
    return {
        "label": result_label,
        "raw_label": int(prediction),
        "confidence": confidence,
        "message": message,
        "metadata": metadata,
        "explanation": explanation,
    }


def explain_prediction(model: Any, vectorizer: Any, cleaned_message: str) -> List[str]:
    feature_names = vectorizer.get_feature_names_out()
    tokens = cleaned_message.split()
    if not tokens:
        return []

    token_set = set(tokens)
    weights = None

    if hasattr(model, "coef_"):
        if model.coef_.ndim > 1:
            weights = np.abs(model.coef_[0])
        else:
            weights = np.abs(model.coef_)
    elif hasattr(model, "feature_log_prob_"):
        weights = np.abs(model.feature_log_prob_[1] - model.feature_log_prob_[0])

    if weights is not None:
        top_indices = np.argsort(weights)[::-1][:25]
        candidates = [feature_names[idx] for idx in top_indices if feature_names[idx] in token_set]
        return candidates[:8]

    return [token for token in tokens if len(token) > 3][:8]
