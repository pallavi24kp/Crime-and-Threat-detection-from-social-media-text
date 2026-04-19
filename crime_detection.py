"""NLP-based Crime & Threat Detection from Social Media Text.

Detects threats, abusive/violent intent, and suspicious criminal activity.
Pipeline: preprocessing → TF-IDF feature extraction → classification.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data(
    train_path=None,
    test_path=None,
    test_labels_path=None,
):
    """Load train/test CSVs from the data/ directory."""
    train_path       = train_path       or os.path.join(DATA_DIR, "train.csv")
    test_path        = test_path        or os.path.join(DATA_DIR, "test.csv")
    test_labels_path = test_labels_path or os.path.join(DATA_DIR, "test_labels.csv")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Merge test labels if available
    if os.path.exists(test_labels_path):
        test_labels = pd.read_csv(test_labels_path)
        test_df = test_df.merge(test_labels, on="id", how="left")

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nTrain columns:", train_df.columns.tolist())
    print("Train label distribution:\n", train_df.iloc[:, -1].value_counts())
    return train_df, test_df


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"@\w+", " ", text)                     # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # strip hashtag symbol
    text = re.sub(r"[^a-z\s]", " ", text)                 # remove non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(preprocess_text)
    return df


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION  (TF-IDF)
# ─────────────────────────────────────────────

def build_tfidf(max_features=50_000, ngram_range=(1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
    )


# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", build_tfidf()),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                        class_weight="balanced")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", build_tfidf()),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", build_tfidf()),
            ("clf",   LinearSVC(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
    }


# ─────────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_val, y_train, y_val):
    best_model = None
    best_acc   = 0.0
    best_name  = ""

    for name, pipeline in get_models().items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        print(f"\n{'='*50}")
        print(f"Model: {name}  |  Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, preds))
        if acc > best_acc:
            best_acc   = acc
            best_model = pipeline
            best_name  = name

    print(f"\n Best model: {best_name}  (accuracy = {best_acc:.4f})")
    return best_model, best_name


# ─────────────────────────────────────────────
# 6. GENERATE SUBMISSION
# ─────────────────────────────────────────────

def generate_submission(model, test_df: pd.DataFrame, text_col: str,
                        id_col: str = "id", out_path: str = "submission.csv"):
    preds = model.predict(test_df[text_col])
    sub   = pd.DataFrame({id_col: test_df[id_col], "label": preds})
    sub.to_csv(out_path, index=False)
    print(f"\n Submission saved to: {out_path}")
    return sub


# ─────────────────────────────────────────────
# 7. OPTIONAL – LSTM (deep learning)
# ─────────────────────────────────────────────

def build_lstm_model(vocab_size, embedding_dim=128, max_len=100, num_classes=2):
    """
    Requires: pip install tensorflow
    Call this function only if you want the LSTM variant.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                             Dropout, Bidirectional, GlobalMaxPooling1D)

        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        return model
    except ImportError:
        print("TensorFlow not installed. Skipping LSTM model.")
        return None


def prepare_sequences(texts, max_len=100, max_words=20_000):
    """Tokenise texts and pad sequences for LSTM."""
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tok.fit_on_texts(texts)
        seqs = tok.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
        return padded, tok
    except ImportError:
        return None, None


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────
    train_df, test_df = load_data()

    # Auto-detect text & label columns
    # Assumes last column = label, first non-id text column = tweet/text
    cols       = train_df.columns.tolist()
    label_col  = cols[-1]
    text_col   = [c for c in cols if c.lower() in ("tweet", "text", "message", "post", "content")]
    text_col   = text_col[0] if text_col else cols[1]   # fallback: second column
    id_col     = cols[0]

    print(f"\nUsing  text_col='{text_col}'  label_col='{label_col}'  id_col='{id_col}'")

    # ── Preprocess ────────────────────────────
    train_df = preprocess_dataframe(train_df, text_col)
    test_df  = preprocess_dataframe(test_df,  text_col)

    # ── Encode labels ─────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(train_df[label_col])
    print("\nLabel classes:", le.classes_)

    # ── Split ─────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[text_col], y,
        test_size=0.2, random_state=42, stratify=y
    )

    # ── Train & evaluate ──────────────────────
    best_model, best_name = train_and_evaluate(X_train, X_val, y_train, y_val)

    # ── Retrain best model on full training set ──
    print(f"\n Retraining '{best_name}' on full training data…")
    best_model.fit(train_df[text_col], y)

    # ── Submission ────────────────────────────
    if id_col in test_df.columns:
        generate_submission(best_model, test_df, text_col, id_col)
    else:
        print("No id column found in test set – skipping submission file.")

    # ── Optional: LSTM ────────────────────────
    use_lstm = False   # set True to run deep-learning variant
    if use_lstm:
        MAX_LEN   = 100
        padded, tok = prepare_sequences(train_df[text_col].tolist(), max_len=MAX_LEN)
        if padded is not None:
            lstm = build_lstm_model(
                vocab_size  = len(tok.word_index) + 1,
                max_len     = MAX_LEN,
                num_classes = len(le.classes_),
            )
            if lstm:
                X_tr, X_vl, y_tr, y_vl = train_test_split(
                    padded, y, test_size=0.2, random_state=42, stratify=y
                )
                lstm.fit(X_tr, y_tr, epochs=5, batch_size=64,
                         validation_data=(X_vl, y_vl))


if __name__ == "__main__":
    main()
