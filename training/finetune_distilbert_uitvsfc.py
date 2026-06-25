"""
finetune_distilbert_uitvsfc.py  —  YOUR FIRST FINE-TUNING RUN
=============================================================
Goal: fine-tune `distilbert-base-multilingual-cased` on UIT-VSFC (Vietnamese
student-feedback sentiment, 3 classes) and read off an F1 score. This is a
"training wheels" pipeline: a clean, solved task so you can confirm the
mechanics work before pointing them at the harder ViRDC task.

HOW TO RUN (Google Colab):
  1. Open colab.research.google.com -> New notebook.
  2. Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4 is free & enough).
  3. Paste each numbered SECTION below into its own cell, run top to bottom.
     (You CAN run the whole file at once, but going cell-by-cell is the point —
      you watch each stage and learn where things can break.)

Expect ~3-8 minutes total on a T4. A healthy macro-F1 here is roughly 0.85-0.92.
"""

# =============================================================================
# SECTION 1 — Install libraries
# transformers: the model + Trainer.  datasets: easy dataset loading.
# evaluate: metric functions (F1).  accelerate: lets Trainer use the GPU.
# (In Colab, prefix with ! to run as a shell command: !pip install ...)
# =============================================================================
# !pip install -q transformers datasets evaluate accelerate

# =============================================================================
# SECTION 2 — Confirm the GPU is actually on
# If this prints "cuda" you're set. If it prints "cpu", go fix Runtime type;
# training on CPU will be painfully slow.
# =============================================================================
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch sees device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =============================================================================
# SECTION 3 — Load the dataset
# UIT-VSFC on the Hugging Face Hub. Fields: 'sentence' (text),
# 'sentiment' (0=negative, 1=neutral, 2=positive), 'topic' (ignore for now).
# load_dataset downloads + caches it and gives you train/validation/test splits.
# If this exact id ever fails to load, search the Hub for "vietnamese students
# feedback" and swap the id — the rest of the script is dataset-agnostic.
# =============================================================================
from datasets import load_dataset

raw = load_dataset("uitnlp/vietnamese_students_feedback")
print(raw)                              # shows the splits and their sizes
print(raw["train"][0])                  # peek at one example

TEXT_COL  = "sentence"
LABEL_COL = "sentiment"
NUM_LABELS = 3

# =============================================================================
# SECTION 4 — Tokenize
# A model can't read text; it reads integer token IDs. The tokenizer that
# SHIPS WITH the model knows exactly how this model was trained to split text,
# so we always load the matching one by name.
#   - truncation=True: cut comments longer than max_length (BERT-family models
#     have a hard length limit; long inputs must be trimmed).
#   - we map the tokenizer over every split in one shot; batched=True is just
#     a speed optimization (process many rows per call).
# =============================================================================
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, max_length=128)

tokenized = raw.map(tokenize, batched=True)

# The Trainer expects the label column to be literally named "labels".
# Rename it if needed so we don't have to think about it again.
if LABEL_COL != "labels":
    tokenized = tokenized.rename_column(LABEL_COL, "labels")

# A "data collator" pads every batch to the length of its longest member at
# training time (dynamic padding) — more efficient than padding everything to 128.
from transformers import DataCollatorWithPadding
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =============================================================================
# SECTION 5 — Load the model (pretrained body + fresh classification head)
# AutoModelForSequenceClassification = the pretrained DistilBERT PLUS a small
# untrained head sized to num_labels. The body knows language; the head starts
# random and is what fine-tuning mainly teaches. The warning about
# "newly initialized weights" is EXPECTED — that's the head.
# =============================================================================
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# =============================================================================
# SECTION 6 — Define how we score the model
# Accuracy alone lies on imbalanced data (a model can ignore a rare class and
# still look "accurate"). macro-F1 averages the F1 of each class equally, so a
# rare class can't be swept under the rug. We report both.
# `compute_metrics` receives raw model outputs (logits) + true labels; argmax
# turns the 3 per-class scores into the single predicted class.
# =============================================================================
import numpy as np
import evaluate

acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# =============================================================================
# SECTION 7 — Training configuration
# TrainingArguments is the knob panel. The ones that matter for you:
#   - num_train_epochs: how many full passes over the data (3 is a fine start).
#   - learning_rate 2e-5: the standard starting point for fine-tuning BERT-family.
#   - per_device_train_batch_size: how many examples per step (16 fits a T4).
#   - eval_strategy="epoch": run the validation metrics after every epoch so you
#     can watch F1 climb. (Older transformers call this `evaluation_strategy`.)
#   - load_best_model_at_end: keep the epoch with the best validation F1, not
#     just the last one — guards against over-training.
# =============================================================================
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="distilbert-uitvsfc",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",          # don't try to log to W&B etc.
)

# =============================================================================
# SECTION 8 — The Trainer ties it all together and runs the training loop
# Under the hood each step: feed a batch -> model predicts -> compare to truth
# via a loss function -> backpropagate -> nudge the weights. The Trainer hides
# that loop (you'll write it by hand later, on purpose, to understand it).
# =============================================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()                  # <-- this is the actual fine-tuning

# =============================================================================
# SECTION 9 — Final evaluation on the TEST split
# validation was used during training (to pick the best epoch), so the honest
# final number comes from test, which the model never trained or tuned on.
# =============================================================================
test_metrics = trainer.evaluate(tokenized["test"])
print("\nTEST RESULTS:", test_metrics)
print(f"\n>>> Your first fine-tuned macro-F1: {test_metrics['eval_macro_f1']:.4f}")

# =============================================================================
# SECTION 10 — Sanity-check it on your own sentence
# Proves end-to-end inference works: text -> tokens -> model -> predicted class.
# =============================================================================
labels_text = {0: "negative", 1: "neutral", 2: "positive"}
sample = "Giảng viên dạy rất hay và nhiệt tình"   # "The lecturer teaches very well and enthusiastically"
inputs = tokenizer(sample, return_tensors="pt", truncation=True).to(model.device)
with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()
print(f"\nSample: {sample}\nPredicted sentiment: {labels_text[pred]}")
