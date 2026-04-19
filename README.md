# 🛡️ CrimeShield — Crime & Threat Detection from Social Media Text

> **Real-time multi-label NLP classification** of toxic, threatening, and hate-speech content using TF-IDF + Logistic Regression (fast inference) with BERT fine-tuning support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [BERT Fine-Tuning](#bert-fine-tuning)
- [Label Definitions](#label-definitions)

---

## Overview

CrimeShield detects crime and threat-related content in social media text. It performs **multi-label classification** across 6 categories simultaneously — a single piece of text can be flagged for multiple threat types (e.g., both "Toxic" and "Threatening").

The system ships as a **full-stack web application**:

| Component | Technology |
|-----------|-----------|
| ML Backend | Python · Flask · scikit-learn |
| Model | TF-IDF (50K features, 1-2 ngrams) + Multi-label Logistic Regression |
| Heavy Model | BERT (`bert-base-uncased`) via HuggingFace Transformers |
| Frontend | Vanilla HTML · CSS · JavaScript |
| Dataset | Jigsaw Toxic Comment Classification (Kaggle) |

---

## Features

- ✅ **Single-text analysis** — analyze any text for 6 threat categories with confidence scores
- ✅ **Batch analysis** — process up to 50 texts in one API call
- ✅ **Real-time API** — Flask REST API with CORS support
- ✅ **Premium dark UI** — glassmorphism, animated confidence bars, analysis history
- ✅ **Analysis history** — persisted in browser localStorage
- ✅ **Model statistics** — per-label accuracy, training details via `/api/stats`
- ✅ **BERT ready** — `crime_detection.py` for full BERT fine-tuning

---

## Architecture

```
Browser (HTML/CSS/JS)
        │
        │  HTTP POST /api/analyze
        ▼
Flask REST API (app.py)
        │
        │  predict(text)
        ▼
model_utils.py
   ├── TF-IDF Vectorizer (50K features, 1-2 ngrams)
   └── Multi-label Logistic Regression (6 outputs)
        │
        ▼
[toxic, severe_toxic, obscene, threat, insult, identity_hate]
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Web Framework | Flask 2.x |
| CORS | flask-cors |
| ML | scikit-learn (TF-IDF, LogisticRegression, MultiOutputClassifier) |
| Deep Learning | PyTorch + HuggingFace Transformers (BERT) |
| Data | pandas, numpy |
| Model Caching | joblib |
| Frontend | HTML5, CSS3 (custom), Vanilla JS |
| Fonts | Inter, JetBrains Mono (Google Fonts) |

---

## Dataset

**Jigsaw Toxic Comment Classification Challenge** (Wikipedia talk-page comments)

| File | Size | Description |
|------|------|-------------|
| `data/train.csv` | ~68 MB | 159,571 labeled comments |
| `data/test.csv` | ~60 MB | 153,164 test comments |
| `data/test_labels.csv` | ~5 MB | Ground-truth labels for test set |

**Label columns:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## Project Structure

```
NLP-crime/
├── app.py                  # Flask REST API
├── crime_detection.py      # BERT fine-tuning script
├── model_utils.py          # Model loading, training, inference
├── requirements.txt        # Python dependencies
├── model_cache.joblib      # Cached sklearn model (auto-generated)
├── results/                # BERT training checkpoints
├── logs/                   # Training logs
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
└── frontend/
    ├── index.html          # Single-page application
    ├── style.css           # Premium dark theme styles
    └── app.js              # Frontend logic & API integration
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Start the Flask API

```bash
python app.py
```

On first run, the model will **auto-train** on a 20,000-row sample of `data/train.csv` and cache to `model_cache.joblib`. Subsequent starts load instantly.

Expected output:
```
INFO  Pre-loading model…
INFO  Training model from data sample (20000 rows)…
INFO  Model saved to model_cache.joblib
INFO  Model ready.
 * Running on http://0.0.0.0:5000
```

### 3. Open the Frontend

Open `frontend/index.html` in your browser — no build step required:

```
NLP-crime/frontend/index.html   ← open directly in browser
```

Or serve it with Python:
```bash
cd frontend
python -m http.server 8080
# Open http://localhost:8080
```

---

## API Reference

All endpoints are at `http://localhost:5000`.

### `GET /api/health`

Health check.

```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00Z",
  "service": "Crime & Threat Detection API",
  "version": "1.0.0"
}
```

---

### `GET /api/stats`

Model metadata and label definitions.

```json
{
  "model": {
    "sample_size": 17000,
    "val_size": 3000,
    "per_label_accuracy": {
      "toxic": 92.1,
      "severe_toxic": 98.4,
      ...
    },
    "model_type": "TF-IDF + Logistic Regression (Multi-Label)"
  },
  "labels": [...],
  "data_info": { "dataset": "Jigsaw Toxic Comment Classification", ... }
}
```

---

### `POST /api/analyze`

Analyze a single text.

**Request:**
```json
{ "text": "I will find you and make you pay!" }
```

**Response:**
```json
{
  "text": "I will find you and make you pay!",
  "is_harmful": true,
  "overall_confidence": 87.3,
  "flagged_count": 2,
  "summary": "🚨 CRITICAL: Direct threat detected...",
  "processing_time_ms": 12.4,
  "labels": [
    { "key": "toxic",   "name": "Toxic",  "flagged": true,  "confidence": 87.3, "color": "#ff4757" },
    { "key": "threat",  "name": "Threat", "flagged": true,  "confidence": 74.1, "color": "#ff9f43" },
    { "key": "obscene", "name": "Obscene","flagged": false, "confidence": 12.0, "color": "#ff6b35" },
    ...
  ]
}
```

---

### `POST /api/batch`

Analyze up to 50 texts.

**Request:**
```json
{ "texts": ["text one", "text two", "text three"] }
```

**Response:**
```json
{
  "total": 3,
  "harmful_count": 1,
  "clean_count": 2,
  "results": [...]
}
```

---

## Frontend

The UI (`frontend/index.html`) features:

| Section | Description |
|---------|-------------|
| **Hero** | Live model stats (accuracy, training samples, categories) |
| **Text Analyzer** | Input + animated results with category confidence bars |
| **Batch Analyzer** | Multi-line input, summary stats per result |
| **History** | Past analyses persisted in localStorage |
| **About & API Docs** | Architecture info and curl examples |

**Tips:**
- Press `Ctrl+Enter` to submit text quickly
- Click example buttons (Threat / Toxic / Clean) to load sample texts
- Click any history entry to re-analyze it

---

## BERT Fine-Tuning

For state-of-the-art performance, use `crime_detection.py` to fine-tune BERT:

```bash
# Ensure data/train.csv is present
python crime_detection.py
```

Training details:
- Model: `bert-base-uncased`
- Epochs: 2
- Batch size: 16
- Max sequence length: 128
- Learning rate: 2e-5
- Checkpoints saved in `./results/`

> ⚠️ Requires a CUDA GPU for practical training time (~3–4 hours on a modern GPU).

---

## Label Definitions

| Label | Color | Description |
|-------|-------|-------------|
| `toxic` | 🔴 Red | Generally rude or disrespectful content |
| `severe_toxic` | 🔴 Dark Red | Highly aggressive / extreme toxic content |
| `obscene` | 🟠 Orange | Vulgar or sexually explicit language |
| `threat` | 🟡 Amber | Direct or implicit threats against someone |
| `insult` | 🟡 Yellow | Personal attacks or degrading remarks |
| `identity_hate` | 🟣 Purple | Hate speech targeting identity groups |

---

## Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| CUDA (optional) | ≥ 11.6 (for BERT training) |

```
flask, flask-cors, scikit-learn, pandas, numpy, joblib
transformers, torch, datasets  (for BERT)
```
