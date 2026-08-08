# Multi-Token Prediction — Reproduce Experiments

Code, data, and sealed results for the MTP benchmark harness and Medusa-style future-token head fine-tune. Article prose and explanatory GIFs are intentionally omitted.

## Requirements

- Python 3.11
- CUDA GPU recommended (fine-tune was run on RTX 3080 10 GB)
- Hugging Face access for base models (`HuggingFaceTB/SmolLM2-1.7B-Instruct`, `google/gemma-4-E2B-it`)

## Setup

```bash
pip install -e .
# or, with uv + the CUDA torch index from pyproject.toml:
uv sync
```

## Tests

```bash
pytest
```

## Benchmark (Gemma vs assistant)

Dry run (no model download):

```bash
mtp-benchmark --dry-run --output-dir results/dry-run-local
```

Full run (matches sealed `results/rtx3080-gemma4-e2b-4bit-final/`):

```bash
mtp-benchmark \
  --target-model google/gemma-4-E2B-it \
  --policy 4bit \
  --output-dir results/rtx3080-gemma4-e2b-4bit-local
```

## Fine-tune future-token heads (SmolLM2)

Train (hyperparameters sealed in `results/finetune-smollm2-mtp-heads/train-config.json`):

```bash
python -m mtp_finetune.train_heads --run-name finetune-smollm2-mtp-heads-local
```

Evaluate / self-speculate / seal:

```bash
python -m mtp_finetune.eval_heads --run-name finetune-smollm2-mtp-heads-local
python -m mtp_finetune.self_speculate --run-name finetune-smollm2-mtp-heads-local
python -m mtp_finetune.seal_results --run-name finetune-smollm2-mtp-heads-local
```

## Sealed evidence

| Path | What it is |
|------|------------|
| `results/finetune-smollm2-mtp-heads/` | Authoritative fine-tune + self-speculation seal |
| `results/finetune-smollm2-run1-no-early-stop/` | Earlier overfit run (kept for comparison) |
| `results/rtx3080-gemma4-e2b-4bit-final/` | Gemma 4-bit benchmark seal |
| `data/prompts.json` | Benchmark prompts |
| `data/finetune/samples.jsonl` | Fine-tune corpus |

See `results/finetune-smollm2-mtp-heads/README.md` for headline numbers and caveats.
