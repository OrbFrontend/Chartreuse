"""Fine-tune the configured ettin encoder as a binary style classifier.

Trains on the active axis's train/validation splits, selects the best checkpoint
by validation F1, and writes it to the active classifier directory. Uses GPU if
available.

  python -m classifier.train                 # default 3e-5, 4 epochs
  python -m classifier.train --lr 5e-5 --epochs 5
  python -m classifier.train --demo          # no-network self-check
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from core.paths import SPLITS, CLASSIFIER, ETTIN_MODEL

# Base encoder is env-swappable (ETTIN_MODEL); the CLASSIFIER dir slug tracks it, so a
# different encoder lands in its own models/<slug>-<axis> dir. Single source: core.paths.
MODEL = ETTIN_MODEL


def class_weights(path):
    c = collections.Counter(json.loads(l)["label"] for l in open(path) if l.strip())
    n, k = sum(c.values()), len(c)
    # inverse-frequency weights, normalized so an even split gives [1, 1]
    return torch.tensor([n / (k * c[i]) for i in range(k)], dtype=torch.float)


class WeightedTrainer(Trainer):
    """CrossEntropy with class weights to counter the ~3:1 negative skew."""
    def __init__(self, *a, weights=None, **kw):
        super().__init__(*a, **kw)
        self._weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels, weight=self._weights.to(out.logits.device),
            label_smoothing=0.1)  # softens softmax saturation -> usable probabilities
        return (loss, out) if return_outputs else loss


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0)
    return {"f1": f1, "precision": p, "recall": r,
            "accuracy": accuracy_score(labels, preds)}


def main(lr, epochs, batch, out):
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, attn_implementation="sdpa")

    ds = load_dataset("json", data_files={
        "train": str(SPLITS / "train.jsonl"),
        "validation": str(SPLITS / "val.jsonl")})

    def prep(b):
        enc = tok(b["text"], truncation=True, max_length=128)
        enc["labels"] = [int(x) for x in b["label"]]
        return enc

    ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

    args = TrainingArguments(
        output_dir=out, learning_rate=lr, num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch",
        metric_for_best_model="f1", greater_is_better=True,
        load_best_model_at_end=True, weight_decay=1e-5,
        fp16=torch.cuda.is_available(), report_to="none",
        logging_steps=20, save_total_limit=2)

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        compute_metrics=metrics,
        data_collator=DataCollatorWithPadding(tok),  # pad per-batch (variable lengths)
        weights=class_weights(str(SPLITS / "train.jsonl")),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    trainer.train()
    print("best val:", trainer.evaluate())
    trainer.save_model(out)
    tok.save_pretrained(out)
    print("saved ->", out)


def demo():
    # logic check: weighted CE down-weights the majority class, and inverse-freq
    # weights normalize to [1, 1] on a balanced split.
    import tempfile, os
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.jsonl")
    with open(p, "w") as f:
        for lbl in [0, 0, 0, 1]:  # 3:1
            f.write(json.dumps({"label": lbl}) + "\n")
    w = class_weights(p)
    assert abs(w[0] - 4 / 6) < 1e-6 and abs(w[1] - 4 / 2) < 1e-6, w
    assert w[1] > w[0]  # minority weighted higher

    bal = os.path.join(d, "b.jsonl")
    with open(bal, "w") as f:
        f.write(json.dumps({"label": 0}) + "\n" + json.dumps({"label": 1}) + "\n")
    wb = class_weights(bal)
    assert torch.allclose(wb, torch.ones(2)), wb

    m = metrics((np.array([[2., 0.], [0., 2.], [2., 0.], [0., 2.]]),
                 np.array([0, 1, 1, 1])))
    assert m["accuracy"] == 0.75 and m["recall"] == 2 / 3, m
    print("demo OK", {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(CLASSIFIER))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else main(a.lr, a.epochs, a.batch, a.out)
