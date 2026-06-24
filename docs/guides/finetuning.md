# Guide: Fine-tuning / warm-starting from a checkpoint

How to start a new run from an existing checkpoint — either continuing the same model on more/new
data, or transferring a pretrained backbone+decoder to a fresh head. This codebase has **no
layer-freezing mechanism**: the whole model is always trainable, so you steer how much it adapts
with the learning rate and augmentation, not by freezing weights.

> Fine-tune vs resume: **resume** continues the *same* run from its own latest checkpoint into a
> *new* `output_dir` (auto, or `--resume_from`). **Fine-tune** starts a *new* run that *warm-starts*
> its weights from *another* run's checkpoint via `init_checkpoint`. This guide is the latter.

## 1. Choose what to warm-start

In the experiment YAML (`task:` section):

```yaml
task:
  init_checkpoint: /path/to/source_run/ckpt-100000   # a checkpoint prefix
  init_checkpoint_modules: [backbone, decoder]        # which modules to load
```

| `init_checkpoint_modules` | When to use |
|---|---|
| `[backbone, decoder]` (default) | Transfer / new task: load the feature extractor, **randomly init the head** (e.g. different classes, or you want the head to relearn). |
| `[backbone, decoder, head]` | Continue the *same* architecture/task: load everything and keep training (closest to a soft resume from another run). |

`migrate_checkpoint` auto-detects the checkpoint kind. For a checkpoint produced by **this**
codebase it uses the `native` strategy (exact, complete restore of the requested modules); legacy
TF2-Vision checkpoints use `frozen`/`structural`. See
[checkpoint_migration.md](../checkpoint_migration.md). The migrated weights are written next to the
source under `migrated/ckpt`, and a coverage guard refuses to proceed if a requested module loaded
nothing.

## 2. Tune the hyperparameters for fine-tuning

Fine-tuning generally wants a **gentler** schedule than a from-scratch run. In the YAML:

- **Lower the LR.** Drop `learning_rate.initial_learning_rate` (e.g. 0.01 → 0.001–0.0005). For a
  short run a `constant` or short `cosine` schedule is common:
  ```yaml
  optimizer_config:
    learning_rate:
      type: cosine
      cosine: { initial_learning_rate: 0.001, alpha: 0.01, decay_steps: <steps_per_loop × epochs> }
  ```
  (Optimizer/LR are config-selectable — see [configuration.md](../configuration.md).)
- **Fewer epochs.** Set `trainer.train_epochs` to a small value, and keep `decay_steps =
  steps_per_loop × epochs` so the schedule lands its floor at the end.
- **Gentler augmentation.** Lower `parser.mosaic.mosaic_frequency`, and/or set
  `close_mosaic_epochs` to disable mosaic for the last epochs so the model settles on clean images.
- **Reduce momentum warmup** if the run is very short (`optimizer.sgd_torch.warmup_steps`), since a
  long warmup eats a short schedule.

## 3. Launch

Exactly like a normal run, with a **new** `output_dir` (so you don't overwrite the source run):

```bash
python -m scripts.run_train \
    --config configs/experiments/yolo/yolov8_poly_dist_finetune.yaml \
    --output_dir /path/to/finetune_run
```

At startup the log reports how many variables each module loaded from `init_checkpoint` — confirm
backbone/decoder (and head, if requested) loaded a non-zero count.

## 4. Validate

Same as any run — watch `<run>/val_history.jsonl` and the best checkpoint. See the
[validation guide](validation.md). Compare `F1score50` against the source run to confirm the
fine-tune helped.

## Notes / pitfalls

- **No freezing.** If you only want to adapt the head, the practical lever is a low LR + loading
  `[backbone, decoder, head]`; there is no `trainable=False` switch.
- **Head reinit when classes change.** If the new task has a different class count, the head shapes
  differ — use `[backbone, decoder]` so the head is freshly initialized at the new size.
- **Don't reuse the source `output_dir`** — auto-resume would pick up the source's optimizer state
  and step count instead of fine-tuning fresh.

## Related
- Reference: [checkpoint_migration.md](../checkpoint_migration.md) · [configuration.md](../configuration.md)
- See also: [training guide](training.md) · [validation guide](validation.md)
