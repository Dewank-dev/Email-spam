from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NLTK_DATA_DIR = Path.home() / ".nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.append(str(NLTK_DATA_DIR))


def ensure_nltk_data() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
    ]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab/english")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)


ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def normalize_label(value: object) -> str:
    if value is None:
        return "ham"
    label = str(value).strip().lower()
    if label in {"spam", "1", "true", "yes", "y"}:
        return "spam"
    return "ham"


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    text_column_candidates = ["text", "message", "sms", "content", "email", "body", "msg"]
    label_column_candidates = ["label", "target", "spam", "is_spam", "class"]

    found_text_col = next((col for col in text_column_candidates if col in df.columns), None)
    found_label_col = next((col for col in label_column_candidates if col in df.columns), None)

    if found_text_col is None or found_label_col is None:
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["text", "label"]
            found_text_col, found_label_col = "text", "label"
        else:
            raise ValueError(
                "Dataset must contain at least one text/message column and a label column. "
                "Expected columns like: message, text, label, spam."
            )

    df = df[[found_text_col, found_label_col]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].fillna("ham")
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["text", "label"]).copy()
    df["label"] = df["label"].apply(normalize_label)

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both spam and ham labels for training.")

    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\w+@\w+\.\w+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    tokens = word_tokenize(cleaned)
    filtered_tokens = [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(filtered_tokens)


def transform_texts(texts: Iterable[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]


def encode_labels(labels: Iterable[str]) -> pd.Series:
    label_map = {"spam": 1, "ham": 0}
    return pd.Series(labels, dtype=str).map(label_map)
