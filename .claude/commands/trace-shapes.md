# /trace-shapes — Compare model architectures by variable shape/position

Compares two model sources (YAML config builds a model, or a checkpoint path)
by variable shape at each architecture position. Critical for verifying checkpoint
compatibility before migration.

## Usage

```
/trace-shapes --src1 yaml --src2 ckpt     # compare new model vs old checkpoint
/trace-shapes --src1 yaml --src2 yaml2    # compare two configs
/trace-shapes --by-shape                  # sort both sides by shape before comparing
/trace-shapes --filter backbone           # backbone vars only
```

## What to run

```bash
# Compare new model (from YAML) vs old checkpoint
python tools/trace_shapes.py \
  --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \
  --src2 initial_checkpoint_folder/ckpt-920304 \
  --only-mismatch

# Sort by shape to find architectural differences regardless of name order
python tools/trace_shapes.py \
  --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \
  --src2 initial_checkpoint_folder/ckpt-920304 \
  --by-shape --filter backbone

# Shape histogram and module counts only
python tools/trace_shapes.py \
  --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \
  --src2 initial_checkpoint_folder/ckpt-920304 \
  --stats-only
```

## What to look for

- `MATCH` on every row with `--by-shape` → identical architecture
- `SHAPE MISMATCH` → true architectural difference (not just naming)
- Extra variables on one side → extra layers or heads
- Backbone should have 135 vars, decoder 90, head 111 (336 total)
