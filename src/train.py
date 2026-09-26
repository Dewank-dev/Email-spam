from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.evaluate import build_model_comparison_table, compute_confusion_matrix, compute_metrics
from src.preprocessing import encode_labels, load_dataset, transform_texts

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


MODEL_CONFIG = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.7),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

if XGBClassifier is not None:
    MODEL_CONFIG["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)


VECTORIZERS = {
    "CountVectorizer": CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english"),
    "TF-IDF": TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english"),
}


def prepare_training_data(data_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = load_dataset(data_path)
    df["text_preprocessed"] = transform_texts(df["text"])

    X = df["text_preprocessed"]
    y = encode_labels(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return df, X_train, X_test, y_train, y_test


def evaluate_vectorizer_and_model(
    vectorizer_name: str,
    vectorizer,
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, object]:
    fitted_vectorizer = vectorizer
    X_train_vec = fitted_vectorizer.fit(X_train)
    X_train_transformed = X_train_vec.transform(X_train)
    X_test_transformed = X_train_vec.transform(X_test)

    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    metrics = compute_metrics(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)

    return {
        "vectorizer_name": vectorizer_name,
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": cm,
        "model": model,
        "vectorizer": fitted_vectorizer,
    }


def make_final_model_bundle(best_result: Dict[str, object], metadata: Dict[str, object]) -> None:
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(best_result["model"], model_dir / "spam_model.pkl")
    joblib.dump(best_result["vectorizer"], model_dir / "vectorizer.pkl")
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_and_save_model(data_path: str | Path = "data/spam.csv") -> Dict[str, object]:
    df, X_train, X_test, y_train, y_test = prepare_training_data(data_path)

    all_results: List[Dict[str, object]] = []

    for vectorizer_name, base_vectorizer in VECTORIZERS.items():
        for model_name, base_model in MODEL_CONFIG.items():
            vectorizer_instance = base_vectorizer.__class__(**base_vectorizer.get_params())
            model_instance = base_model.__class__(**base_model.get_params())
            result = evaluate_vectorizer_and_model(
                vectorizer_name,
                vectorizer_instance,
                model_name,
                model_instance,
                X_train,
                X_test,
                y_train,
                y_test,
            )
            all_results.append(result)

    comparison_rows = build_model_comparison_table(all_results)
    best_result = max(all_results, key=lambda entry: (entry["f1"], entry["accuracy"]))

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    metadata = {
        "model_name": best_result["model_name"],
        "vectorizer_name": best_result["vectorizer_name"],
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset_size": int(len(df)),
        "spam_messages": int((df["label"] == "spam").sum()),
        "ham_messages": int((df["label"] == "ham").sum()),
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1_score": best_result["f1"],
        "number_of_features": int(best_result["vectorizer"].get_feature_names_out().shape[0]),
        "confusion_matrix": best_result["confusion_matrix"].tolist(),
        "preprocessing": {
            "text_lowercase": True,
            "remove_urls": True,
            "remove_punctuation": True,
            "tokenize": True,
            "remove_stopwords": True,
            "stemming": True,
            "extra_spaces": True,
        },
    }

    make_final_model_bundle(best_result, metadata)
    return {
        "best_result": best_result,
        "comparison": comparison_rows,
        "metadata": metadata,
    }


if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Selected model: {result['metadata']['model_name']}")
    print(f"Selected vectorizer: {result['metadata']['vectorizer_name']}")
    print(f"F1-score: {result['metadata']['f1_score']:.4f}")
