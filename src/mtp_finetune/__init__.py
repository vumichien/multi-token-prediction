"""Medusa-style future-token heads on a frozen small base model.

Companion experiment to `mtp_benchmark`: instead of measuring an existing MTP
model, this package trains K extra prediction heads on top of a frozen
SmolLM2-1.7B-Instruct so a 10 GB consumer GPU can reproduce the MTP training
objective end to end.
"""

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Heads predict t+2 .. t+(NUM_HEADS+1); the frozen base LM head keeps t+1.
NUM_HEADS = 3
