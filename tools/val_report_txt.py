#!/usr/bin/env python3
"""Render a saved validation report JSON into the ckpt-dir ``.txt`` format.

This is the exact format the trainer drops next to checkpoints after each validation
(``eval/metrics_report.py:write_txt``): a "best confidence per category" table — each
category's best F1 / precision / recall over the ``arange(0.05, 1.0, 0.05)`` confidence
grid — plus the mean line and the full all-confidence sweep. Input is the canonical
report JSON (``val_metrics/epoch_NNNN.json`` or the ``<ckpt>_val.json`` written by
``tools/eval.py --output_dir``), which carries the ``mean`` / ``best_conf`` / ``all_conf``
keys.

Usage:
    python -m tools.val_report_txt <report.json> [-o out.txt] [--best-only]
    python -m tools.val_report_txt /run/val_metrics/epoch_0042.json
    python -m tools.val_report_txt /run/val_metrics        # every *.json in the dir
"""
import argparse
import glob
import json
import os
import sys

from eval import metrics_report

_REQUIRED = ('best_conf', 'all_conf', 'mean')


def _is_report(obj) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in _REQUIRED)


_ALL_CONF_MARKER = "\n=== all confidence thresholds per category ==="


def _render(report: dict, out_path: str | None, best_only: bool) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile('w+', suffix='.txt', delete=True) as tf:
        metrics_report.write_txt(report, tf.name)
        tf.seek(0)
        text = tf.read()
    if best_only:
        # keep only the best-per-category table + mean; drop the all-conf sweep section
        text = text.split(_ALL_CONF_MARKER, 1)[0].rstrip() + "\n"
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='a report .json or a directory of them')
    ap.add_argument('-o', '--out', help='write the .txt here (default: alongside each json)')
    ap.add_argument('--best-only', action='store_true',
                    help='print only the best-conf table + mean (omit the all-conf sweep)')
    ap.add_argument('--quiet', action='store_true', help='write files but do not print')
    a = ap.parse_args()

    if os.path.isdir(a.path):
        files = sorted(glob.glob(os.path.join(a.path, '**', '*.json'), recursive=True))
    else:
        files = [a.path]

    rendered = 0
    for fp in files:
        try:
            with open(fp) as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {fp}: {e}", file=sys.stderr)
            continue
        if not _is_report(obj):
            # tell the user precisely what's wrong instead of emitting junk
            if len(files) == 1:
                sys.exit(
                    f"{fp} is not a validation report: missing {_REQUIRED}.\n"
                    f"Point this at the canonical val report (val_metrics/epoch_*.json or "
                    f"<ckpt>_val.json from `tools/eval.py --output_dir`), NOT metrics.json "
                    f"/ per_category_metrics.json (those carry only headline AP numbers).")
            continue
        out_path = a.out if (a.out and len(files) == 1) else \
            (os.path.splitext(fp)[0] + ('_best.txt' if a.best_only else '.txt') if not a.out else None)
        text = _render(obj, out_path, a.best_only)
        rendered += 1
        if not a.quiet:
            if len(files) > 1:
                print(f"\n########## {os.path.basename(fp)} ##########")
            print(text, end='')
        if out_path:
            print(f"-> {out_path}", file=sys.stderr)

    if rendered == 0:
        sys.exit(f"no validation report JSON found under {a.path}")


if __name__ == '__main__':
    main()
