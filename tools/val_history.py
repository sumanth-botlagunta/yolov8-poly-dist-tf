#!/usr/bin/env python3
"""Inspect / extract from a run's validation history (``val_history.jsonl``).

The trainer appends one validation report per epoch to ``<run>/val_history.jsonl``
(see ``eval/val_history.py``). This is the read side: list the trend, or pull any
single epoch/checkpoint (or the best) back into the exact ckpt-format **txt**, raw
**json**, or a best-conf **csv** — no SQL, no DB.

Usage:
    # trend table of every epoch (epoch / step / F1score50 / mAP / mAP50 / AR100)
    python -m tools.val_history <run_dir_or_jsonl>
    python -m tools.val_history <run> --list --sort F1score50

    # one epoch (or checkpoint, or the best) -> ckpt-format txt (default)
    python -m tools.val_history <run> --epoch 42
    python -m tools.val_history <run> --best
    python -m tools.val_history <run> --best --format json -o best.json
    python -m tools.val_history <run> --epoch 42 --format csv -o e42.csv

    # whole history -> one flat CSV (uses pandas if installed, else stdlib csv)
    python -m tools.val_history <run> --export-csv history.csv

Arguments:
    path                 run directory (containing val_history.jsonl) or the file.
    --list               print the trend table (default when no selector given).
    --sort COL           sort the trend table by this column (default: epoch).
    --epoch N            select the record for epoch N.
    --step N             select the record for global step N.
    --checkpoint SUBSTR  select the record whose checkpoint contains SUBSTR.
    --best               select the record with the highest --metric.
    --metric NAME        metric for --best / --sort headline (default: F1score50).
    --format txt|json|csv  output format for a selected record (default: txt).
    --best-only          txt: print only the best-conf table + mean (omit all-conf).
    -o, --out PATH       write the selected output here (default: stdout).
    --export-csv PATH    dump the whole history (headline metrics per epoch) to CSV.
"""
import argparse
import csv
import json
import os
import sys

from eval import metrics_report, val_history

_HEADLINE = ['F1score50', 'mAP', 'mAP50', 'AR100']
_ALL_CONF_MARKER = "\n=== all confidence thresholds per category ==="


def _headline_row(rec):
    m = rec.get('metrics') or {}
    row = {'epoch': rec.get('epoch'), 'step': rec.get('step')}
    for k in _HEADLINE:
        v = m.get(k)
        if v is None and k == 'F1score50':
            v = val_history.metric_of(rec, k)
        row[k] = v
    return row


def _print_table(records, sort_col):
    rows = [_headline_row(r) for r in records]
    try:
        rows.sort(key=lambda r: (r.get(sort_col) is None, r.get(sort_col)))
    except TypeError:
        rows.sort(key=lambda r: str(r.get(sort_col)))
    cols = ['epoch', 'step'] + _HEADLINE
    print('  '.join(f'{c:>10}' for c in cols))
    for r in rows:
        print('  '.join(
            (f'{r[c]:>10.4f}' if isinstance(r.get(c), float) else f'{str(r.get(c, "")):>10}')
            for c in cols))
    print(f"\n{len(rows)} epochs")


def _render_selected(rec, fmt, best_only):
    if fmt == 'json':
        return json.dumps(rec, indent=2) + '\n'
    if fmt == 'csv':
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['category', 'name', 'f1', 'precision', 'recall', 'conf_threshold'])
        for r in rec.get('best_conf', []):
            w.writerow([int(r['category']), metrics_report._name(r['category']),
                        f"{r['f1']:.6f}", f"{r['precision']:.6f}",
                        f"{r['recall']:.6f}", f"{r['conf_threshold']:.4f}"])
        return buf.getvalue()
    # txt — reuse the exact ckpt-format writer (record is a superset of the report)
    import tempfile
    with tempfile.NamedTemporaryFile('w+', suffix='.txt', delete=True) as tf:
        metrics_report.write_txt(rec, tf.name)
        tf.seek(0)
        text = tf.read()
    if best_only:
        text = text.split(_ALL_CONF_MARKER, 1)[0].rstrip() + '\n'
    return text


def _export_csv(records, out_path):
    rows = [_headline_row(r) for r in records]
    cols = ['epoch', 'step'] + _HEADLINE
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
        backend = 'pandas'
    except ImportError:
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        backend = 'stdlib csv'
    print(f"wrote {len(rows)} rows -> {out_path}  ({backend})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='run dir (with val_history.jsonl) or the jsonl file')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--sort', default='epoch')
    ap.add_argument('--epoch', type=int)
    ap.add_argument('--step', type=int)
    ap.add_argument('--checkpoint')
    ap.add_argument('--best', action='store_true')
    ap.add_argument('--metric', default='F1score50')
    ap.add_argument('--format', choices=['txt', 'json', 'csv'], default='txt')
    ap.add_argument('--best-only', action='store_true')
    ap.add_argument('-o', '--out')
    ap.add_argument('--export-csv')
    a = ap.parse_args()

    jsonl = val_history.resolve_path(a.path)
    records = val_history.load_records(jsonl)
    if not records:
        sys.exit(f"no records in {jsonl}")

    if a.export_csv:
        _export_csv(records, a.export_csv)
        return

    has_selector = a.epoch is not None or a.step is not None or a.checkpoint or a.best
    if a.list or not has_selector:
        _print_table(records, a.sort)
        return

    if a.best:
        rec = val_history.best_record(records, a.metric)
        if rec is None:
            sys.exit(f"no record has metric '{a.metric}'")
    else:
        rec = val_history.select(records, epoch=a.epoch, step=a.step,
                                 checkpoint=a.checkpoint)
        if rec is None:
            sys.exit("no record matches the given selector "
                     f"(epoch={a.epoch}, step={a.step}, checkpoint={a.checkpoint})")

    text = _render_selected(rec, a.format, a.best_only)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text)
        print(f"epoch={rec.get('epoch')} step={rec.get('step')} "
              f"{a.metric}={val_history.metric_of(rec, a.metric)} -> {a.out}",
              file=sys.stderr)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
