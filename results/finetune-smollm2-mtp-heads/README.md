# Medusa-style future-token heads on a frozen SmolLM2-1.7B (RTX 3080, 10 GB)

Sealed evidence for the article's fine-tune section. Reproduce with
`src/mtp_finetune/` — see `train-config.json` for every hyperparameter.

## What is authoritative here

`manifest.json` digests `raw-results.json` and `summary.csv`. Those two are the
citable artifacts; every other file in this directory is also embedded inside
`raw-results.json`, so check a number against the sealed pair rather than
against the loose copies.

## Headline numbers

Held-out, token-weighted, on 288 sampled completions from **72 distinct
prompts** (`heldout-accuracy.json`). 72 prompts is the honest unit of
independent evidence; the 288 completions are four samples each.

| horizon | top-1 accuracy | cross-entropy |
|---------|---------------|---------------|
| t+1 (frozen base LM head, not trained) | 0.8371 | 0.5561 |
| t+2 | 0.5277 | 2.3431 |
| t+3 | 0.3832 | 3.4619 |
| t+4 | 0.2934 | 4.1590 |

Self-speculative greedy decoding (`self-speculation.json`), 2.207 tokens per
forward pass overall: code 3.686, structured JSON 1.940, instruction 1.855,
creative 1.347. Wall-clock tok/s is reported too but is implementation-bound —
eager fp32 heads in a Python loop are not a serving runtime.

## Two things a reader should know before quoting these

**Checkpoint selection used a separate slice.** Training picked
`heads-best.safetensors` by mean held-out loss on the first 32 held-out samples.
`eval_heads` skips exactly those (`dev_examples_excluded: 32`), so the table
above is measured on data the checkpoint was not selected against.

**`heldout-log.csv` and `heldout-accuracy.json` average differently.** The
in-training log in this run recorded a per-batch mean; the final evaluation is
token-weighted. Both are internally consistent, but do not compare a number from
one against a number from the other. The scoring code is now token-weighted, so
a re-run regenerates `heldout-log.csv` on the same basis.

## Files

| file | contents |
|------|----------|
| `raw-results.json`, `summary.csv`, `manifest.json` | sealed evidence + sha256 digests |
| `training-log.csv` | per-head training loss, 88 rows, steps 25-2200 (renders GIF G7) |
| `heldout-log.csv` | in-training held-out loss, 22 evaluations (see caveat above) |
| `heldout-accuracy.json` | final per-horizon accuracy, overall and per prompt family |
| `self-speculation.json` | draft/verify results, including divergence measurements |
| `train-summary.json`, `train-config.json` | run outcome and hyperparameters |

## Related directories

- `results/finetune-smollm2-run1-no-early-stop/` — the earlier run that overfit;
  kept because the divergence between its two logs is a finding in itself.
- `results/finetune-smollm2-mtp-heads-v1-superseded/` — the first seal of *this*
  run. Superseded after review: its `cross_entropy` was macro-averaged while
  sitting beside token-weighted accuracy, its evaluation set overlapped the
  32 samples used to pick the checkpoint, its `tokens_per_forward_pass` was
  ~1.5% low from counting a truncated final pass, and its `summary.csv` used
  the `trials` column as an index rather than a count. Training was unaffected,
  so the checkpoint and `training-log.csv` are shared with this directory.
