"""Load the configured ettin encoder as a 2-class classifier on CPU.

  python -m classifier.sanity_check
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.paths import ETTIN_MODEL

m = ETTIN_MODEL
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForSequenceClassification.from_pretrained(
    m, num_labels=2, attn_implementation="sdpa"  # 0 = not purple, 1 = purple
)
print("ok", model.config.num_labels)
assert model.config.num_labels == 2
