import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load dataset
df = pd.read_csv("train.csv")  # change path if needed

# Example: For Jigsaw dataset (toxic classification)
# Combine labels into one (binary classification)
df['label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)

df = df[['comment_text', 'label']]
df = df.dropna()

# 2. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment_text'], df['label'], test_size=0.2, random_state=42
)

# 3. Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

# 4. Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(train_labels)
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': list(val_labels)
})

# 5. Load BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# 7. Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
    }

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 9. Train
trainer.train()

# 10. Evaluate
results = trainer.evaluate()
print("Evaluation Results:", results)

# 11. Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return torch.argmax(probs).item()

