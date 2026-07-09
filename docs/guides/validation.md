# Guide: Validation & picking the best checkpoint

How to evaluate a checkpoint, read the metrics, and find the best one. For what each metric
means, see [metrics.md](../metrics.md).

## During training (automatic)

Every epoch the trainer validates with the EMA weights and **appends one report line** to
`<run>/val_history.jsonl` (no per-epoch file spam). The best checkpoint by `F1score50` is saved to
`output_dir/best_ckpt/` as training proceeds. You usually don't need to run eval manually — just
read the history.

## 1. Read the validation trend

```bash
python -m utils.reports.val_history /path/to/run_dir --list
```
Prints a table of every epoch: `epoch · step · F1score50 · mAP · mAP50 · AR100`. (Re-validated
epochs collapse to the latest; `--raw` shows the full append log.)

## 2. Pull the best epoch / a specific epoch

```bash
# the best epoch by F1score50, as the ckpt-format txt (best-conf-per-category table)
python -m utils.reports.val_history /path/to/run_dir --best --format txt -o best.txt

# a specific epoch as JSON or CSV
python -m utils.reports.val_history /path/to/run_dir --epoch 42 --format json
python -m utils.reports.val_history /path/to/run_dir --epoch 42 --format csv -o e42.csv

# whole history to one flat CSV (pandas if installed)
python -m utils.reports.val_history /path/to/run_dir --export-csv history.csv
```
`--best` already gives you the answer to "which checkpoint is best" — it's the epoch with the
highest `F1score50`, and the corresponding checkpoint is in `output_dir/best_ckpt/`.

## 3. Evaluate a checkpoint manually (offline)

To (re)evaluate a specific checkpoint on the val split:

```bash
python -m utils.eval \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-100000 \
    --split val --per_category --output_dir /tmp/eval_out
```
Prints `mAP / mAP50 / AR100 / F1score50` (+ polygon and distance metrics), and with
`--per_category` a per-class table. `--output_dir` also drops `metrics.json`, a per-category JSON,
and a **`<ckpt>_val.json` + `.txt`** — the same ckpt-format report the trainer writes.

Evaluate **every** checkpoint in a run (and append each to a log), or watch a live run:

```bash
python -m utils.eval --config <cfg> --all   --watch_dir /path/to/run_dir
python -m utils.eval --config <cfg> --watch --watch_dir /path/to/run_dir --interval 300
```

## 4. Understand the headline metric

`F1score50` is the **best-checkpoint selection metric**: per class, the best `2pr/(p+r)` over a
confidence-threshold sweep (`np.arange(0.1, 1.0, 0.05)`) at IoU 0.5 / maxDets 10, macro-averaged
over classes with a valid PR point (`eval/coco_eval_custom.py`). See [metrics.md](../metrics.md)
for the full glossary (mAP, AR, polygon mIoU, distance error).

## 5. See *why* a class is weak — failure mining

The `per_class/` TensorBoard metrics tell you *which* class is weak; `--dump_failures` shows you
*why*. It writes the worst predictions per class as annotated images:

```bash
python -m utils.eval --config <cfg> --checkpoint /run/ckpt-100000 \
    --output_dir /tmp/eval_out --dump_failures
```
For each class it keeps the worst few of each kind and writes them to
`/tmp/eval_out/failures/<NN_name>/`:
- **`fp_*`** — confident false positives (a detection with no matching GT; GT green, the FP red).
- **`fn_*`** — missed GT (no detection matched it; the missed GT in yellow).
- **`lowiou_*`** — correct class but poorly localized (matched at IoU `[0.5, 0.7)`; box in orange).

Open the folder for a class that's dragging the macro F1 down and you'll usually see the pattern
(confused with a similar class, consistently mislocalized, a labeling issue, …). Tune with
`--failures_per_class` and `--failures_dir`.

## 6. Render a single report JSON to txt

If you have a standalone report JSON (e.g. a `<ckpt>_val.json` from step 3):
```bash
python -m utils.reports.val_history /tmp/eval_out/ckpt-100000_val.json --best-only
```

## Related
- Reference: [metrics.md](../metrics.md) · [scripts.md](../scripts.md)
- Next: [fine-tuning](finetuning.md) · [deployment](deployment.md)
