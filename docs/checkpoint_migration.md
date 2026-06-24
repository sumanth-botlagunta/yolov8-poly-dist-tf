# Checkpoint Migration & Warm-start

`tools/checkpoint_migration.py` loads weights into the model from an existing checkpoint. It
auto-detects the checkpoint kind and picks a strategy — you normally just run `migrate` and
point `task.init_checkpoint` at the result. Done infrequently (once per new init source), so it
lives here rather than in the README.

## Two cases

| Source checkpoint | Auto strategy | What transfers |
|-------------------|---------------|----------------|
| **Legacy** TF2-Vision checkpoint (e.g. `ckpt-920304`) | `frozen` (or `structural`) | backbone + decoder by default; head too if class counts match |
| **This codebase's own** checkpoint (a `ckpt-N` / `best_*` from a previous run) | `native` | the requested modules, loaded from the EMA weights |

`migrate_checkpoint(strategy="auto")` inspects the keys: legacy object checkpoints
(`layer_with_weights` / `_head/`) → `frozen`; this codebase's checkpoints (EMA markers
`optimizer/_shadows` etc., or a `model/{backbone,decoder,head}` root) → `native`; otherwise
`structural`.

## Legacy migration (initial backbone+decoder)

```bash
# 1. Inspect the variables in the old checkpoint
python -m tools.checkpoint_migration list \
    --ckpt initial_checkpoint_folder/ckpt-920304

# 2. Dry-run: show which variables map cleanly and which are missed
python -m tools.checkpoint_migration map \
    --ckpt initial_checkpoint_folder/ckpt-920304 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml

# 3. Migrate and write the new checkpoint (modules auto-selected by class count;
#    pass --modules backbone decoder to force)
python -m tools.checkpoint_migration migrate \
    --ckpt   initial_checkpoint_folder/ckpt-920304 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output /tmp/migrated_ckpt/ckpt \
    --modules backbone decoder
```

Then set `task.init_checkpoint` in your YAML to the migrated checkpoint path; the trainer loads
it at startup. The migration assigns weights **in place** into the live model (the written
checkpoint is a re-serialisation), so training begins from the transferred weights.

`trace_shapes` is a useful pre-check that two model sources line up by variable shape/position:

```bash
python -m tools.trace_shapes \
    --src1 initial_checkpoint_folder/ckpt-920304 \
    --src2 configs/experiments/yolo/yolov8_poly_dist.yaml --only-mismatch
```

## Warm-starting a new run from a previous run (`native`)

> For **fine-tuning** (continue the same task on new data), prefer `task.finetune_from` /
> `--finetune_from` — it loads the full model from the **EMA/deployed** weights into a fresh
> optimizer, which is what you want. See [guides/finetuning.md](guides/finetuning.md). Use
> `init_checkpoint` below for **transfer-init** (a different task / partial-module warm-start).

To warm-start the selected modules from a checkpoint this codebase produced, point
`task.init_checkpoint` at a periodic `ckpt-N` or a `best_*` checkpoint. Auto-detection routes it to
the `native` strategy:

```bash
python -m tools.checkpoint_migration migrate \
    --ckpt   /previous_run/ckpt-100000 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output /tmp/warmstart/ckpt \
    --modules backbone decoder        # add 'head' to also warm-start the (same-class) head
```

### Important: warm-start from a checkpoint that carries EMA

The trainer stores the **complete** weights only in the **EMA shadows**. The plain `model/`
object graph omits the list-tracked C2f block variables (a Keras quirk), so a *model-only*
checkpoint cannot fully warm-start those blocks. The `native` strategy therefore loads via the
EMA path (the same one eval/export use). Use a periodic `ckpt-N` or a `best_*` checkpoint (both
carry EMA) — not a bare model export — as the init source. The excluded modules (e.g. `head`
when `--modules backbone decoder`) are kept at fresh init.

## Module selection rule

When `--modules` is omitted, the head is included only if the old and new classification-head
widths match (`num_classes`); otherwise just backbone + decoder transfer and the head is
re-trained. Pass `--modules` to override.
