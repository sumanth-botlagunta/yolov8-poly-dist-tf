"""Export saved validation metrics to usable / space-efficient formats.

The trainer appends one validation report per epoch to ``<run>/val_history.jsonl``
(``eval/val_history.py``). This offline tool reads that store (or a single report
JSON) and turns one or many reports into:

  * ``.xlsx``    — ONE workbook, TWO sheets: ``best_conf`` (all classes at their best
                   confidence) and ``all_conf`` (all classes at every confidence).
                   (pandas + openpyxl.)
  * ``.parquet`` — columnar + compressed; the **space-efficient, fast-read** format for the
                   WHOLE run (one row per class×threshold×epoch) for trend analysis.
                   (pandas + pyarrow.)
  * ``.csv``     — plain (two files: ``*_best_conf.csv`` / ``*_all_conf.csv``).

For a quick txt/json/csv of a single epoch (no pandas), use ``tools/val_history.py``;
this tool is the xlsx/parquet trend exporter.

Whole run (recommended for trend analysis — adds an ``epoch`` column):
    python tools/pipeline/export_val_metrics.py --input <run>/val_history.jsonl --aggregate \
        --out_dir /tmp/metrics --formats parquet,xlsx

Single epoch (a report JSON, e.g. a <ckpt>_val.json or one extracted with
``tools/val_history.py --epoch N --format json``):
    python tools/pipeline/export_val_metrics.py --input epoch42.json \
        --out_dir /tmp/metrics --formats xlsx,csv

Requires: pandas, openpyxl (xlsx), pyarrow (parquet). Install: pip install pandas openpyxl pyarrow
"""

import argparse
import glob
import json
import os
from typing import List

try:
    from configs.class_map import DETECTION_CLASSES
    _NAMES = ({int(k): str(v) for k, v in DETECTION_CLASSES.items()}
              if isinstance(DETECTION_CLASSES, dict)
              else {i: str(n) for i, n in enumerate(DETECTION_CLASSES)})
except Exception:
    _NAMES = {}


def _name(cat):
    return _NAMES.get(int(cat), str(int(cat)))


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _best_df(pd, report: dict):
    rows = [{'epoch': report.get('epoch'), 'step': report.get('step'),
             'category': int(r['category']), 'name': _name(r['category']),
             'f1': r['f1'], 'precision': r['precision'], 'recall': r['recall'],
             'conf_threshold': r['conf_threshold']}
            for r in report.get('best_conf', [])]
    return pd.DataFrame(rows)


def _all_df(pd, report: dict):
    rows = [{'epoch': report.get('epoch'), 'step': report.get('step'),
             'category': int(r['category']), 'name': _name(r['category']),
             'thresh': r['thresh'], 'f1': r['f1'],
             'precision': r['precision'], 'recall': r['recall']}
            for r in report.get('all_conf', [])]
    return pd.DataFrame(rows)


def _mean_df(pd, report: dict):
    m = report.get('mean', {})
    return pd.DataFrame([{'epoch': report.get('epoch'), 'step': report.get('step'),
                          'f1': m.get('f1'), 'precision': m.get('precision'),
                          'recall': m.get('recall')}])


def _write_xlsx(pd, path, best, allc, mean):
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        best.to_excel(xw, sheet_name='best_conf', index=False)
        allc.to_excel(xw, sheet_name='all_conf', index=False)
        mean.to_excel(xw, sheet_name='mean', index=False)


def _write_parquet(path_best, path_all, best, allc):
    # snappy-compressed columnar; tiny + fast to read back for the whole run.
    best.to_parquet(path_best, index=False, compression='snappy')
    allc.to_parquet(path_all, index=False, compression='snappy')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='a val_history.jsonl, a run dir containing it, an older-format '
                         'val_metrics/ dir, or a single report JSON')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--formats', default='xlsx,csv',
                    help='comma list of: xlsx, csv, parquet')
    ap.add_argument('--aggregate', action='store_true',
                    help='if --input is a dir, combine ALL epochs into one set of '
                         'tables (with an epoch column). Best for parquet trend analysis.')
    ap.add_argument('--basename', default='val_metrics')
    a = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required: pip install pandas openpyxl pyarrow")

    formats = [f.strip() for f in a.formats.split(',') if f.strip()]
    os.makedirs(a.out_dir, exist_ok=True)

    # Resolve input -> a list of report dicts. Sources, in order of preference:
    #   * a val_history.jsonl (the run's append-only store; one report per line)
    #   * a run directory containing val_history.jsonl
    #   * a directory of older per-epoch epoch_*.json / step_*.json reports
    #   * a single report JSON (e.g. <ckpt>_val.json, or one extracted with
    #     `tools/val_history.py --format json`)
    from eval import val_history
    if os.path.isdir(a.input):
        jsonl = os.path.join(a.input, 'val_history.jsonl')
        if os.path.exists(jsonl):
            reports = val_history.load_records(jsonl)
        else:
            files = sorted(glob.glob(os.path.join(a.input, 'epoch_*.json')) +
                           glob.glob(os.path.join(a.input, 'step_*.json')))
            if not files:
                raise SystemExit(
                    f"no val_history.jsonl or epoch_*.json / step_*.json under {a.input}")
            reports = [_load(f) for f in files]
    elif a.input.endswith('.jsonl'):
        reports = val_history.load_records(a.input)
    else:
        reports = [_load(a.input)]
    if not reports:
        raise SystemExit(f"no reports loaded from {a.input}")
    print(f"loaded {len(reports)} report(s) from {a.input}")

    if a.aggregate or len(reports) > 1:
        best = pd.concat([_best_df(pd, r) for r in reports], ignore_index=True)
        allc = pd.concat([_all_df(pd, r) for r in reports], ignore_index=True)
        mean = pd.concat([_mean_df(pd, r) for r in reports], ignore_index=True)
        base = a.basename + '_all_epochs'
    else:
        best, allc, mean = _best_df(pd, reports[0]), _all_df(pd, reports[0]), _mean_df(pd, reports[0])
        ep = reports[0].get('epoch')
        base = a.basename + (f'_epoch_{int(ep):04d}' if ep is not None else '')

    written = []
    if 'xlsx' in formats:
        try:
            p = os.path.join(a.out_dir, base + '.xlsx')
            _write_xlsx(pd, p, best, allc, mean)
            written.append(p)
        except ImportError:
            print("  (xlsx skipped: pip install openpyxl)")
    if 'parquet' in formats:
        try:
            pb = os.path.join(a.out_dir, base + '_best_conf.parquet')
            pa = os.path.join(a.out_dir, base + '_all_conf.parquet')
            _write_parquet(pb, pa, best, allc)
            written += [pb, pa]
        except (ImportError, ValueError):
            print("  (parquet skipped: pip install pyarrow)")
    if 'csv' in formats:
        pb = os.path.join(a.out_dir, base + '_best_conf.csv')
        pa = os.path.join(a.out_dir, base + '_all_conf.csv')
        best.to_csv(pb, index=False); allc.to_csv(pa, index=False)
        written += [pb, pa]

    print("wrote:")
    for p in written:
        print("  " + p)


if __name__ == '__main__':
    main()
