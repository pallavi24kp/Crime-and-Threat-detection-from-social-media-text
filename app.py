"""
SENTINEL Flask API
Wraps the NLP crime detection pipeline and exposes REST endpoints
for the frontend to consume.
"""

import os
import threading
import time
import pickle

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Import ML pipeline functions ──────────────────────────────────────────────
from crime_detection import (
    load_data,
    preprocess_dataframe,
    preprocess_text,
    get_models,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="/frontend")
CORS(app)  # Allow requests from file:// and localhost frontends

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# ── Shared model state ────────────────────────────────────────────────────────
MODEL_STATE = {
    "ready": False,
    "error": None,
    "model": None,
    "label_encoder": None,
    "best_name": None,
    "best_acc": None,
    "text_col": None,
    "classes": [],
}

MODEL_PATH = "trained_model.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Label → UI metadata mapping
# ─────────────────────────────────────────────────────────────────────────────

def label_to_ui(label_str: str):
    """
    Map a raw label string to a risk level and UI category.
    Adapt the keyword lists below to match your dataset's actual class names.
    """
    lbl = str(label_str).lower()

    threat_keywords  = ["threat", "violence", "violent", "extremis", "weapon",
                        "kill", "bomb", "terror", "harm", "attack", "murder"]
    abuse_keywords   = ["abuse", "abusive", "harass", "hate", "offensive",
                        "toxic", "vulgar", "insult", "bully"]
    safe_keywords    = ["safe", "neutral", "normal", "clean", "benign",
                        "not", "no", "0", "false", "negative"]

    if any(k in lbl for k in threat_keywords):
        return "Critical Threat", "High", "threat"
    elif any(k in lbl for k in abuse_keywords):
        return "Abuse Signal", "Medium", "abuse"
    elif any(k in lbl for k in safe_keywords):
        return "Safe Content", "Low", "safe"
    else:
        # Fallback: numeric labels — 0 = safe, anything > 0 = escalating risk
        try:
            val = int(float(lbl))
            if val == 0:
                return "Safe Content", "Low", "safe"
            elif val == 1:
                return "Suspicious Activity", "Medium", "abuse"
            else:
                return "Critical Threat", "High", "threat"
        except ValueError:
            return lbl.title(), "Unknown", "safe"


# ─────────────────────────────────────────────────────────────────────────────
# Background model training
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """Load data, train models, pick the best, store in MODEL_STATE."""
    global MODEL_STATE
    try:
        print("[SENTINEL] Loading data…")
        train_df, _ = load_data()

        # Auto-detect columns
        cols      = train_df.columns.tolist()
        label_col = cols[-1]
        text_keys = ("tweet", "text", "message", "post", "content", "comment")
        text_col  = next((c for c in cols if c.lower() in text_keys), cols[1])

        MODEL_STATE["text_col"] = text_col
        print(f"[SENTINEL] Using text_col='{text_col}'  label_col='{label_col}'")

        # Preprocess
        train_df = preprocess_dataframe(train_df, text_col)

        # Encode labels
        le = LabelEncoder()
        y  = le.fit_transform(train_df[label_col])
        MODEL_STATE["classes"] = le.classes_.tolist()
        print(f"[SENTINEL] Classes: {le.classes_}")

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[text_col], y,
            test_size=0.2, random_state=42, stratify=y,
        )

        # Train all models, pick best
        best_model = None
        best_acc   = 0.0
        best_name  = ""

        for name, pipeline in get_models().items():
            print(f"[SENTINEL] Training {name}…")
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            acc   = accuracy_score(y_val, preds)
            print(f"[SENTINEL]   {name}: val_acc = {acc:.4f}")
            if acc > best_acc:
                best_acc   = acc
                best_model = pipeline
                best_name  = name

        print(f"[SENTINEL] Best model: {best_name} (acc={best_acc:.4f})")
        print(f"[SENTINEL] Retraining '{best_name}' on full training data…")
        best_model.fit(train_df[text_col], y)

        # Persist to disk for fast future restarts
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": best_model,
                "le": le,
                "best_name": best_name,
                "best_acc": best_acc,
                "text_col": text_col,
            }, f)
        print(f"[SENTINEL] Model saved to {MODEL_PATH}")

        MODEL_STATE.update({
            "ready": True,
            "model": best_model,
            "label_encoder": le,
            "best_name": best_name,
            "best_acc": float(best_acc),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        MODEL_STATE["error"] = str(exc)


def load_or_train():
    """Load a pre-trained model if available, otherwise train from scratch."""
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[SENTINEL] Loading cached model from {MODEL_PATH}…")
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            MODEL_STATE.update({
                "ready": True,
                "model": saved["model"],
                "label_encoder": saved["le"],
                "best_name": saved["best_name"],
                "best_acc": float(saved["best_acc"]),
                "text_col": saved["text_col"],
                "classes": saved["le"].classes_.tolist(),
            })
            print(f"[SENTINEL] Model loaded: {saved['best_name']} (acc={saved['best_acc']:.4f})")
            return
        except Exception as exc:
            print(f"[SENTINEL] Cache load failed ({exc}), retraining…")

    train_model()


# ─────────────────────────────────────────────────────────────────────────────
# Frontend Route
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the SENTINEL frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/frontend/<path:filename>", methods=["GET"])
def frontend_static(filename):
    """Serve any additional static assets from the frontend directory."""
    return send_from_directory(FRONTEND_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    """Frontend polls this to know when the model is ready."""
    return jsonify({
        "ready":      MODEL_STATE["ready"],
        "error":      MODEL_STATE["error"],
        "model":      MODEL_STATE["best_name"],
        "accuracy":   MODEL_STATE["best_acc"],
        "classes":    MODEL_STATE["classes"],
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze a piece of text for crime/threat signals.

    Request body: { "text": "<user text>" }
    Response:     { "label", "display_title", "risk_level", "category",
                    "confidence", "model_used", "preprocessed_text" }
    """
    if not MODEL_STATE["ready"]:
        err = MODEL_STATE.get("error")
        return jsonify({
            "error": err or "Model is still initializing. Please wait and retry."
        }), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request must include a 'text' field."}), 400

    raw_text = str(data["text"]).strip()
    if not raw_text:
        return jsonify({"error": "Text cannot be empty."}), 400

    model = MODEL_STATE["model"]
    le    = MODEL_STATE["label_encoder"]

    # Preprocess
    cleaned = preprocess_text(raw_text)

    # Predict
    pred_idx  = model.predict([cleaned])[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    # Confidence (probability) — LinearSVC uses decision function
    confidence = None
    try:
        proba     = model.predict_proba([cleaned])[0]
        confidence = float(max(proba))
    except AttributeError:
        try:
            dec    = model.decision_function([cleaned])[0]
            # Normalise with a sigmoid-like approach for multi-class
            if hasattr(dec, "__len__"):
                exp_dec = [pow(2.718281828, d) for d in dec]
                s       = sum(exp_dec)
                confidence = float(max(exp_dec) / s)
            else:
                # Binary SVM: scale decision value to [0,1]
                confidence = float(1 / (1 + pow(2.718281828, -abs(dec))))
        except Exception:
            confidence = None

    display_title, risk_level, category = label_to_ui(str(pred_label))

    return jsonify({
        "label":             str(pred_label),
        "display_title":     display_title,
        "risk_level":        risk_level,
        "category":          category,          # "threat" | "abuse" | "safe"
        "confidence":        round(confidence * 100, 1) if confidence else None,
        "model_used":        MODEL_STATE["best_name"],
        "preprocessed_text": cleaned,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[SENTINEL] Starting background model training…")
    threading.Thread(target=load_or_train, daemon=True).start()
    print("[SENTINEL] API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
