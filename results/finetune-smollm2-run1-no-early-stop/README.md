# Run 1 — head training without early stopping (kept as evidence of overfitting)

Stopped by hand at step 1,625 of a planned 11,000. Kept because the divergence
between the two logs here is the honest answer to "can you fine-tune MTP heads
on a 10 GB consumer GPU in an afternoon?" — you can train them, but with this
little data they memorise.

| step | train t+2 (`training-log.csv`) | held-out t+2 (`heldout-log.csv`) |
|-----:|------:|------:|
|   25 | 12.88 | — |
|  500 | ~1.9  | 2.721 |
| 1000 | ~0.9  | **2.689** (best) |
| 1500 | 0.71  | 2.823 |

Setup: SmolLM2-1.7B-Instruct frozen, K=3 heads (314.6M trainable), 1,440 training
samples / ~336k tokens, lr 5e-4, no weight decay, no early stopping. Held-out
loss stopped improving around step 1,000 while training loss kept falling, a
roughly 4x gap by the time the run was stopped.

`head_1_loss` is the frozen base LM head's t+1 loss, logged as the reference
floor; it is not trained and does not move.

The shipped run is `results/finetune-smollm2-mtp-heads/`, which uses more data,
weight decay, and early stopping on held-out loss.
