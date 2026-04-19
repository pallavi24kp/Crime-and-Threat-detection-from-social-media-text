"""
app.py — Flask REST API for Crime & Threat Detection
"""
import os
import time
import logging
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model_utils import load_model, predict, get_model_stats, LABELS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pre-load model at startup
logger.info("Pre-loading model…")
load_model()
logger.info("Model ready.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the frontend SPA."""
    if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "Crime & Threat Detection API",
        "version": "1.0.0",
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return model metadata and label definitions."""
    meta = get_model_stats()
    return jsonify({
        "model": meta,
        "labels": LABELS,
        "data_info": {
            "dataset": "Jigsaw Toxic Comment Classification",
            "source":  "Kaggle / Jigsaw",
            "classes": len(LABELS),
            "task":    "Multi-label text classification",
        },
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze text for crime/threat indicators.

    Request body:
        { "text": "<input text>" }

    Response:
        {
            "text": str,
            "is_harmful": bool,
            "overall_confidence": float,
            "labels": [...],
            "summary": str,
            "flagged_count": int,
            "processing_time_ms": float
        }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text too long. Maximum 10,000 characters."}), 400

    t0 = time.perf_counter()
    try:
        result = predict(text)
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    result["text"]               = text
    result["processing_time_ms"] = elapsed_ms
    logger.info("Analyzed %d chars → harmful=%s (%.1f ms)", len(text), result["is_harmful"], elapsed_ms)
    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def batch_analyze():
    """
    Analyze multiple texts in one request.

    Request body:
        { "texts": ["text1", "text2", ...] }

    Maximum 50 texts per request.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field in request body."}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list."}), 400
    if len(texts) > 50:
        return jsonify({"error": "Maximum 50 texts per batch request."}), 400

    results = []
    for text in texts:
        text = str(text).strip()
        if not text:
            results.append({"error": "Empty text skipped."})
            continue
        try:
            result = predict(text)
            result["text"] = text
            results.append(result)
        except Exception as exc:
            results.append({"text": text, "error": str(exc)})

    harmful_count = sum(1 for r in results if r.get("is_harmful", False))
    return jsonify({
        "total":         len(results),
        "harmful_count": harmful_count,
        "clean_count":   len(results) - harmful_count,
        "results":       results,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
