import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label definitions with metadata
LABELS = [
    {"key": "toxic",         "name": "Toxic",          "color": "#ff4757", "description": "General toxic content"},
    {"key": "severe_toxic",  "name": "Severely Toxic",  "color": "#ff2f3b", "description": "Highly toxic / extreme content"},
    {"key": "obscene",       "name": "Obscene",         "color": "#ff6b35", "description": "Obscene or vulgar language"},
    {"key": "threat",        "name": "Threat",          "color": "#ff9f43", "description": "Direct or implicit threats"},
    {"key": "insult",        "name": "Insult",          "color": "#ffd32a", "description": "Personal insults or attacks"},
    {"key": "identity_hate", "name": "Identity Hate",   "color": "#a29bfe", "description": "Hate based on identity/group"},
]

LABEL_KEYS = [l["key"] for l in LABELS]
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model_cache.joblib")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "train.csv")
SAMPLE_SIZE = 20000   # rows used for fast training

_pipeline = None        # cached pipeline
_model_meta = {}        # metadata exposed via /api/stats


def _train_and_save():
    """Train TF-IDF + multi-label Logistic Regression and cache to disk."""
    logger.info("Training model from data sample (%d rows)…", SAMPLE_SIZE)
    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE + 5000)
    df = df[["comment_text"] + LABEL_KEYS].dropna()
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)

    X = df["comment_text"].astype(str).tolist()
    Y = df[LABEL_KEYS].values

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
            min_df=2,
        )),
        ("clf", MultiOutputClassifier(
            LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
            ),
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, Y_train)

    # Compute per-label accuracy on validation set
    Y_pred = pipeline.predict(X_val)
    per_label_acc = {}
    for i, lbl in enumerate(LABEL_KEYS):
        correct = (Y_pred[:, i] == Y_val[:, i]).mean()
        per_label_acc[lbl] = round(float(correct) * 100, 1)

    meta = {
        "sample_size": len(X_train),
        "val_size":    len(X_val),
        "per_label_accuracy": per_label_acc,
        "model_type":  "TF-IDF + Logistic Regression (Multi-Label)",
        "features":    50000,
        "ngram_range": "1-2",
    }
    joblib.dump({"pipeline": pipeline, "meta": meta}, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return pipeline, meta


def load_model():
    """Load cached model or train it if not found."""
    global _pipeline, _model_meta
    if _pipeline is not None:
        return _pipeline

    if os.path.exists(MODEL_PATH):
        logger.info("Loading cached model from %s", MODEL_PATH)
        obj = joblib.load(MODEL_PATH)
        _pipeline  = obj["pipeline"]
        _model_meta = obj["meta"]
    else:
        _pipeline, _model_meta = _train_and_save()

    return _pipeline


def get_model_stats():
    """Return metadata about the loaded model."""
    load_model()
    return _model_meta


def predict(text: str) -> dict:
    """
    Run inference on a single text string.

    Returns
    -------
    {
        "is_harmful": bool,
        "overall_confidence": float,   # 0-100
        "labels": [
            {"key": ..., "name": ..., "flagged": bool, "confidence": float, "color": ...},
            ...
        ],
        "summary": str
    }
    """
    pipe = load_model()

    # Binary predictions
    pred_labels = pipe.predict([text])[0]                # shape (6,)

    # Per-class probabilities from each sub-estimator
    proba_list = pipe.predict_proba([text])              # list of (1,2) arrays
    probs = np.array([p[0][1] for p in proba_list])     # probability of class=1

    label_results = []
    for i, lbl in enumerate(LABELS):
        label_results.append({
            "key":        lbl["key"],
            "name":       lbl["name"],
            "color":      lbl["color"],
            "description": lbl["description"],
            "flagged":    bool(pred_labels[i] == 1),
            "confidence": round(float(probs[i]) * 100, 1),
        })

    flagged_labels = [l for l in label_results if l["flagged"]]
    overall_conf   = round(float(max(probs)) * 100, 1)
    is_harmful     = len(flagged_labels) > 0

    if not is_harmful:
        summary = "✅ This text appears clean and does not contain threatening or harmful content."
    elif any(l["key"] == "threat" and l["flagged"] for l in label_results):
        summary = "🚨 CRITICAL: Direct threat detected in this text. Immediate review recommended."
    elif any(l["key"] == "severe_toxic" and l["flagged"] for l in label_results):
        summary = "⚠️ Severely toxic content detected. Content moderation required."
    else:
        cats = ", ".join(l["name"] for l in flagged_labels)
        summary = f"⚠️ Potentially harmful content detected: {cats}."

    return {
        "is_harmful":          is_harmful,
        "overall_confidence":  overall_conf,
        "labels":              label_results,
        "summary":             summary,
        "flagged_count":       len(flagged_labels),
    }
