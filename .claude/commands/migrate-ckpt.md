# /migrate-ckpt — Migrate old checkpoint to new model

Maps variable names from the old TF checkpoint into the new model architecture
and saves a new checkpoint that the trainer can load via `init_checkpoint`.

## Usage

```
/migrate-ckpt list             # inspect variables in the old checkpoint
/migrate-ckpt map              # dry-run: show old→new variable mapping (336/336)
/migrate-ckpt run              # perform migration and save new checkpoint
/migrate-ckpt compare          # compare two model architectures by variable position
```

## What to run

Old checkpoint path: `initial_checkpoint_folder/ckpt-920304`

```bash
# Step 1: inspect old checkpoint variables
python tools/checkpoint_migration.py list \
  --ckpt initial_checkpoint_folder/ckpt-920304

# Step 2: dry-run mapping (uses frozen hand-verified weight map)
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

# Optional: compare architectures by shape position (no name-matching)
python tools/trace_shapes.py \
  --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \
  --src2 initial_checkpoint_folder/ckpt-920304 \
  --only-mismatch
```

## What to report

- Number of variables matched / skipped / not found (expect 336/336 for backbone+decoder)
- Any backbone/decoder variables that FAILED to match (architectural difference)
- Location of the migrated checkpoint

**Notes:**
- The migration uses a frozen hand-verified positional map (`tools/legacy_weight_map_frozen.py`). The old-to-new name mapping is stored in Claude project memory.
- Shape mismatches in HEAD variables are expected and benign — the head is always randomly initialized. Only backbone+decoder mismatches are a problem.
- After migration, set `task.init_checkpoint` in the experiment YAML to point to the new checkpoint path.
