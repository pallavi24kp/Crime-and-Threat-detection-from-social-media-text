# SENTINEL — NLP Crime & Threat Detection System

> AI-powered text analysis for detecting threats, abuse, and suspicious activity in social media and communication logs.

---

## Features

- **Real-time text classification** — paste any text and get an instant risk assessment
- **Multi-model pipeline** — trains Logistic Regression, Naive Bayes, and Linear SVM; selects the best automatically
- **Confidence scoring** — probability-calibrated confidence displayed per prediction
- **Risk levels** — categorises results into `Critical Threat`, `Abuse Signal`, or `Safe Content`
- **Instant startup** — pre-trained model loads from cache; no retraining needed on every restart
- **Unified server** — Flask serves both the REST API and the frontend from a single port

---

## Project Structure

```
NLP-crime/
├── app.py                  # Flask API + frontend server
├── crime_detection.py      # ML pipeline (preprocessing, models, training)
├── requirements.txt        # Python dependencies
├── start.ps1               # Windows startup script
├── data/                   # Training & test datasets (not committed)
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
├── frontend/
│   └── index.html          # SENTINEL web UI (single-file, no build step)
└── trained_model.pkl       # Cached trained model (not committed)
```

---

## Setup

### Prerequisites
- Python 3.9+
- Windows (PowerShell) or any OS with a bash-compatible shell

### Install

```powershell
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

### Data

Place the dataset files in the `data/` directory:

```
data/train.csv
data/test.csv
data/test_labels.csv
```

The following columns are expected:
- `id` — unique identifier
- `tweet` / `text` / `message` — the text to classify
- Label column (last column) — the target class

---

## Running

### Windows (recommended)

```powershell
.\start.ps1
```

This activates the virtual environment, starts the Flask server, and auto-opens the browser.

### Manual

```powershell
venv\Scripts\python.exe app.py
```

Then open **http://localhost:5000** in your browser.

---

## API Reference

### `GET /api/status`
Returns the current model state.

```json
{
  "ready": true,
  "model": "Naive Bayes",
  "accuracy": 0.9895,
  "classes": ["0", "1"]
}
```

### `POST /api/analyze`
Classify a text sample.

**Request:**
```json
{ "text": "Your text here" }
```

**Response:**
```json
{
  "label": "1",
  "display_title": "Critical Threat",
  "risk_level": "High",
  "category": "threat",
  "confidence": 94.3,
  "model_used": "Naive Bayes",
  "preprocessed_text": "your text here"
}
```

---

## How It Works

1. **Preprocessing** — lowercasing, URL/mention removal, non-alpha stripping
2. **Feature extraction** — TF-IDF with up to 50,000 unigram + bigram features
3. **Training** — three classifiers trained in parallel; best validation accuracy wins
4. **Caching** — best model serialized to `trained_model.pkl` for instant future loads
5. **Serving** — Flask exposes REST endpoints; the same server serves the frontend HTML

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn (LR, NB, LinearSVC), TF-IDF |
| Frontend | HTML, Vanilla JS, Tailwind CSS (CDN) |
| Fonts | Google Fonts (Manrope, Inter) |

---

## License

For academic and research use only.
