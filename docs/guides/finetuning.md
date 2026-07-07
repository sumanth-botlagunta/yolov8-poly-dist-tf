# Guide: Fine-tuning a trained model

Fine-tuning = take a model that is already trained and **adapt it to new/more data** with a gentle
schedule, *without* throwing away what it learned. You control how much it moves with the
**learning rate** (and augmentation), and — optionally — by **freezing** whole modules so they
don't update at all (Section 3).

## First: which workflow do you actually want?

Three things sound similar but are different. Pick the right one — they load different things:

| You want to… | Use | What it loads | Optimizer/EMA/step |
|---|---|---|---|
| **Continue an interrupted run** (same config, same data) | `--resume_from` / automatic | the run's own latest checkpoint | **kept** — picks up exactly where it stopped |
| **Adapt a trained model to new data** (this guide) | **`--finetune_from`** | the model's **EMA/deployed weights** | **fresh** — new LR schedule from step 0 |
| **Reuse a pretrained backbone for a different task** (new classes / new head) | `init_checkpoint` | only the selected modules (default backbone+decoder; **random head**) | fresh |

**Why fine-tune ≠ init_checkpoint** — the difference is **what's loaded**, not which weights:
both go through the same EMA-aware loader (`restore_eval_weights`, preferring the **EMA
shadows** — what `eval` and `export` use). `finetune_from` loads the **whole model** (you're
keeping the same task); `init_checkpoint` loads a **subset of modules** (default
backbone + decoder) and keeps the fresh random init for the rest (you're changing the task).

So: **same task, want it better on new data → `finetune_from`.** Different task / different classes
→ `init_checkpoint` (see the appendix).

## 1. Start the fine-tune

Make a fine-tune config first: **copy your tier YAML** (e.g. `yolov8_poly_dist.yaml`) to
`yolov8_poly_dist_finetune.yaml`, apply the Section 2 LR/epoch changes, and point its `train_data` at the
new data. Then:

```bash
python -m scripts.run_train \
    --config configs/experiments/yolo/yolov8_poly_dist_finetune.yaml \
    --output_dir /path/to/finetune_run \
    --finetune_from /path/to/source_run/ckpt-<step>
```
or set it in the YAML (the flag overrides the field):
```yaml
task:
  finetune_from: /path/to/source_run/ckpt-<step>
```
This loads the full model from the source's **EMA weights**, then builds a **fresh** optimizer +
EMA + `global_step = 0`, so the fine-tune LR schedule below applies from the start. The startup log
prints whether it loaded `ema` vs `raw` weights — confirm it says **`ema`**.

- Point at a **periodic `ckpt-N`** (it carries the EMA shadows). A `best_ckpt` or an exported
  SavedModel holds raw weights and loads as `'raw'` — usable, but not the EMA optimum.
- Use a **new `output_dir`** (don't overwrite the source run).
- `finetune_from` and `init_checkpoint` are **mutually exclusive** (rejected at startup).

## 2. Learning rate — the most important fine-tune setting

A from-scratch run starts at `initial_learning_rate: 0.01`. That LR on an **already-good** model
would wreck it in the first few steps. Fine-tuning uses a **much lower** LR so the model refines
instead of relearning.

| | From scratch | Fine-tune (recommended) |
|---|---|---|
| `initial_learning_rate` | `0.01` | **`0.001`** (gentle) … `0.0005` (very gentle) |
| `train_epochs` | 300 | **10–50** (it's already trained) |
| schedule (`learning_rate.type`) | `cosine` | `cosine` (decay to a small floor), or `constant` for a short run |
| `decay_steps` | `steps_per_loop × 300` | **`steps_per_loop × <your epochs>`** (so cosine hits its floor at the end) |
| `alpha` (cosine floor fraction) | `0.01` | `0.01` |
| `optimizer.sgd.warmup_steps` | `≈3 × steps_per_loop` (3 epochs) | **small** (e.g. ≤1 epoch) — a long warmup eats a short run |

Recommended fine-tune block (≈20 epochs example):
```yaml
trainer:
  train_epochs: 20
  optimizer_config:
    optimizer:
      sgd: { warmup_steps: 500 }             # short warmup for a short run
    learning_rate:
      type: cosine
      cosine:
        initial_learning_rate: 0.001         # 10× lower than from-scratch
        alpha: 0.01
        decay_steps: <steps_per_loop × 20>   # = train_total_examples // batch × epochs
```
> Rule of thumb: start at **1/10th** of the from-scratch LR. If the loss jumps or `F1score50` drops
> below the source model in the first epoch, the LR is still too high — halve it.

Also worth tuning:
- **Gentler augmentation** — lower `parser.mosaic.mosaic_frequency`, and set `close_mosaic_epochs`
  so the last epochs train on un-mosaicked images (the model settles on clean data).
- **Watch `train/update_ratio`** in TensorBoard (`lr·‖grad‖/‖weights‖`): a healthy fine-tune sits
  around `1e-3`; much higher means the LR is too aggressive for the trained weights.

## 3. Optionally freeze layers

Two granularities — frozen weights stop updating entirely, and their BatchNorm runs in inference
mode (frozen running stats):

**Whole modules** — freeze the entire backbone (or +decoder) to adapt only the head:
```yaml
task:
  freeze_modules: [backbone]            # subset of: backbone | decoder | head
```

**Partial (by depth)** — the standard "freeze the early layers, fine-tune the rest". Freeze the
**first N** backbone layers (in order: `stem_conv1, stem_conv2, stem_c2f, down1, c2f_p3, down2,
c2f_p4, down3, c2f_p5_pre, sppf` — 10 total):
```yaml
task:
  freeze_backbone_layers: 3             # freeze the stem; train the rest of the backbone + head
```
Early layers learn generic features that transfer; freezing more (`5`, `7`, …) keeps more of the
backbone fixed. The startup log lists exactly which layers were frozen. (You can combine both
fields.)

- Frozen weights are excluded from `model.trainable_variables` — no gradients, no optimizer slots,
  no EMA drift.
- Leave at least one module / some layers unfrozen (rejected at startup otherwise).
- Freezing is applied on **every** start (including resume), so it's a stable property of the run.
- Pairs naturally with `finetune_from`: freeze the early backbone + a low LR to refine the deeper
  layers + head on new data with minimal risk to the learned features.

When to freeze vs just use a low LR: freeze when you're confident the backbone features transfer
as-is (similar domain) and want speed + stability; use a low LR with nothing frozen when the new
domain differs enough that the backbone should still adapt a little.

## 4. Resuming a dropped / interrupted fine-tune

**Use the normal resume — there is nothing special to do.** A fine-tune run writes its own
checkpoints to its `output_dir` just like any run, so:

- **Just rerun the same command.** On restart the trainer finds the fine-tune run's own latest
  checkpoint and **auto-resumes** from it (model + optimizer + EMA + step intact) — it does **not**
  re-read the source checkpoint. `--finetune_from` is a **fresh-start-only seed**; once the run has
  a checkpoint it is ignored.
- You can also **drop `--finetune_from`** on the restart — same result.
- This means it's **safe even if the source checkpoint was moved/deleted** after the fine-tune
  started: resume never touches it.
- To resume from a specific checkpoint of the fine-tune run, use `--resume_from <prefix>` (same as
  any run).

In short: `--finetune_from` only matters on the **very first** start of a fresh `output_dir`.
After that, it's an ordinary run — resume normally.

## 5. Validate

Watch `<run>/val_history.jsonl` and the best checkpoint (see the [validation guide](validation.md)).
Compare `F1score50` against the **source** run to confirm the fine-tune actually helped — if it's
lower, the LR was too high or the new data is hurting.

## Pitfalls

- **Freeze vs LR.** To limit drift you can freeze modules (Section 3) and/or use a low LR + short
  schedule + gentle aug. Freezing everything is rejected — keep at least one module trainable.
- **Don't reuse the source `output_dir`** — a dir with existing checkpoints auto-resumes *that* run
  and ignores `finetune_from`.
- **Confirm `ema`, not `raw`** in the startup log — `raw` means you pointed at a `best_ckpt`/export
  and lost the EMA optimum.

---

## Appendix: transfer-init (different task / new head)

If you're changing the task — different class count, or you want the head to relearn — use
`init_checkpoint` (NOT `finetune_from`) to load just the feature extractor:

```yaml
task:
  init_checkpoint: /path/to/source_run/ckpt-100000
  init_checkpoint_modules: [backbone, decoder]   # head randomly initialized
```
The source must be a checkpoint **produced by this codebase** (trainer `ckpt-N` and `best_ckpt`
checkpoints both qualify). The full model is loaded through the EMA-aware loader
(`restore_eval_weights` — EMA shadows preferred), then the non-selected modules are restored to
their fresh random init; an unknown module name in `init_checkpoint_modules` is rejected at
startup. Add `head` only if its shape matches. Like `finetune_from`, this is a fresh-start-only
seed — a resumed run ignores it.

## Related
- Reference: [configuration.md](../configuration.md)
- See also: [training guide](training.md) · [validation guide](validation.md)
