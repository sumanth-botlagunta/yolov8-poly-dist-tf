#!/usr/bin/env python3
"""Inspect / extract / export a run's validation history (``val_history.jsonl``).

The trainer appends one validation report per epoch to ``<run>/val_history.jsonl``
(see ``eval/val_history.py``). This is the read side: list the trend, pull any single
epoch/checkpoint (or the best) back into the exact ckpt-format **txt**, raw **json**, or
a best-conf **csv**, and export one or many reports to **xlsx** / **parquet** for trend
analysis. The input may also be a single report JSON (a ``<ckpt>_val.json`` from
``utils.eval --output_dir``); it is rendered / exported directly.

Usage:
    # trend table of every epoch (epoch / step / F1score50 / mAP / mAP50 / AR100)
    python -m utils.reports.val_history <run_dir_or_jsonl>
    python -m utils.reports.val_history <run> --list --sort F1score50

    # extract one epoch (or checkpoint, or the best) -> ckpt-format txt (default)
    python -m utils.reports.val_history <run> --epoch 42
    python -m utils.reports.val_history <run> --best --format json -o best.json

    # render a single standalone report JSON -> txt
    python -m utils.reports.val_history /run/ckpt-99000_val.json --best-only

    # export the best epoch to an xlsx workbook (best_conf / all_conf / mean sheets)
    python -m utils.reports.val_history <run> --best --format xlsx -o best.xlsx

    # export the WHOLE run to parquet (row per class x threshold x epoch)
    python -m utils.reports.val_history <run> --format parquet -o /tmp/run.parquet

Arguments:
    path                 run directory / ``val_history.jsonl``, or a single report JSON.
    --list               print the trend table (default when no selector given).
    --sort COL           sort the trend table by this column (default: epoch).
    --epoch N            select the record for epoch N.
    --step N             select the record for global step N.
    --checkpoint SUBSTR  select the record whose checkpoint contains SUBSTR.
    --best               select the record with the highest --metric.
    --metric NAME        metric for --best / --sort headline (default: F1score50).
    --format txt|json|csv|xlsx|parquet  output format (default: txt).
    --best-only          txt: print only the best-conf table + mean (omit all-conf).
    -o, --out PATH       write output here (required for xlsx/parquet; else stdout).
    --export-csv PATH    dump the whole history (headline metrics per epoch) to CSV.
    --raw                do not collapse re-validated epochs (show every appended line).
"""
import argparse
import csv
import json
import os
import sys

from eval import metrics_report, val_history

_HEADLINE = ['F1score50', 'mAP', 'mAP50', 'AR100']
_ALL_CONF_MARKER = "\n=== all confidence thresholds per category ==="
_REPORT_KEYS = ('best_conf', 'all_conf', 'mean')

_PANDAS_HINT = ("xlsx/parquet export needs pandas (+ openpyxl for xlsx, pyarrow for "
                "parquet): pip install pandas openpyxl pyarrow")


def _is_report(obj) -> bool:
    """True for a single validation report dict (mean / best_conf / all_conf)."""
    return isinstance(obj, dict) and all(k in obj for k in _REPORT_KEYS)


def _load_input(path):
    """Resolve ``path`` to a list of report records.

    Accepts a run directory / ``val_history.jsonl`` (many records) or a single report
    JSON file (one record). Returns ``(records, is_single_report)``.
    """
    if os.path.isfile(path) and path.endswith('.json') and not path.endswith('.jsonl'):
        with open(path) as f:
            obj = json.load(f)
        if not _is_report(obj):
            sys.exit(f"{path} is not a validation report (missing {_REPORT_KEYS}); "
                     f"point this at a <ckpt>_val.json or a val_history.jsonl")
        return [obj], True
    jsonl = val_history.resolve_path(path)
    return val_history.load_records(jsonl), False


def _headline_row(rec):
    m = rec.get('metrics') or {}
    row = {'epoch': rec.get('epoch'), 'step': rec.get('step')}
    for k in _HEADLINE:
        v = m.get(k)
        if v is None and k == 'F1score50':
            v = val_history.metric_of(rec, k)
        row[k] = v
    return row


def _print_table(records, sort_col, dropped=0):
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
    note = f"  ({dropped} re-validated epoch(s) collapsed to latest; --raw to show all)" if dropped else ""
    print(f"\n{len(rows)} epochs{note}")


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


# --- xlsx / parquet export (per class x threshold, one or many reports) --------------

def _best_df(pd, report):
    return pd.DataFrame([
        {'epoch': report.get('epoch'), 'step': report.get('step'),
         'category': int(r['category']), 'name': metrics_report._name(r['category']),
         'f1': r['f1'], 'precision': r['precision'], 'recall': r['recall'],
         'conf_threshold': r['conf_threshold']}
        for r in report.get('best_conf', [])])


def _all_df(pd, report):
    return pd.DataFrame([
        {'epoch': report.get('epoch'), 'step': report.get('step'),
         'category': int(r['category']), 'name': metrics_report._name(r['category']),
         'thresh': r['thresh'], 'f1': r['f1'],
         'precision': r['precision'], 'recall': r['recall']}
        for r in report.get('all_conf', [])])


def _mean_df(pd, report):
    m = report.get('mean', {})
    return pd.DataFrame([{'epoch': report.get('epoch'), 'step': report.get('step'),
                          'f1': m.get('f1'), 'precision': m.get('precision'),
                          'recall': m.get('recall')}])


def _export_tabular(records, fmt, out_path):
    try:
        import pandas as pd
    except ImportError:
        sys.exit(_PANDAS_HINT)
    best = pd.concat([_best_df(pd, r) for r in records], ignore_index=True)
    allc = pd.concat([_all_df(pd, r) for r in records], ignore_index=True)
    mean = pd.concat([_mean_df(pd, r) for r in records], ignore_index=True)

    if fmt == 'xlsx':
        try:
            with pd.ExcelWriter(out_path, engine='openpyxl') as xw:
                best.to_excel(xw, sheet_name='best_conf', index=False)
                allc.to_excel(xw, sheet_name='all_conf', index=False)
                mean.to_excel(xw, sheet_name='mean', index=False)
        except ImportError:
            sys.exit("xlsx export needs openpyxl: pip install openpyxl")
        written = [out_path]
    else:  # parquet — best_conf and all_conf have different schemas -> two files
        stem = os.path.splitext(out_path)[0]
        pb, pa = stem + '_best_conf.parquet', stem + '_all_conf.parquet'
        try:
            best.to_parquet(pb, index=False, compression='snappy')
            allc.to_parquet(pa, index=False, compression='snappy')
        except (ImportError, ValueError):
            sys.exit("parquet export needs pyarrow: pip install pyarrow")
        written = [pb, pa]
    for p in written:
        print(f"wrote {len(records)} report(s) -> {p}", file=sys.stderr)


def _select_record(records, a):
    if a.best:
        rec = val_history.best_record(records, a.metric)
        if rec is None:
            sys.exit(f"no record has metric '{a.metric}'")
        return rec
    rec = val_history.select(records, epoch=a.epoch, step=a.step, checkpoint=a.checkpoint)
    if rec is None:
        sys.exit("no record matches the given selector "
                 f"(epoch={a.epoch}, step={a.step}, checkpoint={a.checkpoint})")
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='run dir / val_history.jsonl, or a single report JSON')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--sort', default='epoch')
    ap.add_argument('--epoch', type=int)
    ap.add_argument('--step', type=int)
    ap.add_argument('--checkpoint')
    ap.add_argument('--best', action='store_true')
    ap.add_argument('--metric', default='F1score50')
    ap.add_argument('--format', choices=['txt', 'json', 'csv', 'xlsx', 'parquet'],
                    default='txt')
    ap.add_argument('--best-only', action='store_true')
    ap.add_argument('-o', '--out')
    ap.add_argument('--export-csv')
    ap.add_argument('--raw', action='store_true',
                    help='do not collapse re-validated epochs (show every appended line)')
    a = ap.parse_args()

    records, is_single = _load_input(a.path)
    if not records:
        sys.exit(f"no records in {a.path}")

    # Collapse re-validations to one canonical (latest) row per epoch for the trend /
    # export views; --raw keeps the full append-only log.
    view = records if a.raw else val_history.latest_per_epoch(records)
    dropped = len(records) - len(view)

    if a.export_csv:
        _export_csv(view, a.export_csv)
        return

    has_selector = a.epoch is not None or a.step is not None or a.checkpoint or a.best

    # xlsx/parquet: export the selected record, else the whole (collapsed) history.
    if a.format in ('xlsx', 'parquet'):
        if not a.out:
            sys.exit(f"--format {a.format} needs -o/--out (a file path)")
        recs = [_select_record(records, a)] if has_selector else view
        _export_tabular(recs, a.format, a.out)
        return

    # txt/json/csv of a single record.
    if not has_selector:
        if is_single:
            rec = records[0]
        else:
            _print_table(view, a.sort, dropped)
            return
    else:
        rec = _select_record(records, a)

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
