# /migrate-ckpt — Migrate old checkpoint to new model

Maps variable names from the old TF checkpoint into the new model architecture
and saves a new checkpoint that the trainer can load via `init_checkpoint`.

## Usage

```
/migrate-ckpt list             # inspect what variables exist in the old checkpoint
/migrate-ckpt map              # dry-run: show old→new variable mapping
/migrate-ckpt run              # perform migration and save new checkpoint
```

## What to run

Old checkpoint path: `initial_checkpoint_folder/ckpt-920304`

```bash
# Step 1: inspect
python tools/checkpoint_migration.py list \
  --ckpt initial_checkpoint_folder/ckpt-920304

# Step 2: dry-run mapping
python tools/checkpoint_migration.py map \
  --ckpt initial_checkpoint_folder/ckpt-920304 \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --modules backbone decoder

# Step 3: migrate
python tools/checkpoint_migration.py migrate \
  --ckpt initial_checkpoint_folder/ckpt-920304 \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --output migrated_checkpoint/ckpt \
  --modules backbone decoder
```

## What to report

- Number of variables matched / skipped / not found
- Any backbone variables that FAILED to match (potential architecture mismatch)
- Parameter count comparison: old backbone vs new backbone (should be ~equal)
- Location of the migrated checkpoint

**Important:** Shape mismatches in HEAD variables are expected and benign — the head
is always randomly initialized. Only backbone+decoder mismatches are a problem.

After migration, update `task.init_checkpoint` in the experiment YAML to point
to the migrated checkpoint.
