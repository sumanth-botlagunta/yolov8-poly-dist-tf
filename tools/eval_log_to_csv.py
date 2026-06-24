#!/usr/bin/env python3
"""Convert validation eval logs into a flat CSV (and print a compact table).

Companion to ``tools/eval.py``, whose ``--all`` / ``--watch`` modes append one JSON
record per checkpoint to ``<watch_dir>/eval_log.jsonl`` and whose ``--output_dir`` writes
a single ``metrics.json``. This flattens those into a spreadsheet-friendly CSV sorted by
training step.

Accepts any of:
  - an ``eval_log.jsonl`` (one JSON record per line)
  - a single ``metrics.json``
  - a directory (globs for ``eval_log.jsonl`` + ``*/metrics.json`` underneath)
  - a plain ``.txt`` log with one JSON object per line (a leading log prefix before the
    first ``{`` is stripped)

Usage:
    python -m tools.eval_log_to_csv <path> [-o out.csv] [--sort step]
    python -m tools.eval_log_to_csv /run/eval/eval_log.jsonl
    python -m tools.eval_log_to_csv /run/eval -o /tmp/val.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

# preferred leading columns when present; everything else appended alphabetically
PREFERRED = ['step', 'checkpoint', 'timestamp',
             'mAP', 'mAP50', 'F1score50', 'AR100']
_STEP_RE = re.compile(r'(?:ckpt|model|step)[-_]?(\d+)')


def _step_of(rec):
    ck = str(rec.get('checkpoint', ''))
    m = _STEP_RE.search(os.path.basename(ck)) or _STEP_RE.search(ck)
    return int(m.group(1)) if m else -1


def _records_from_file(path):
    """Yield dict records from a .jsonl / .json / .txt file."""
    with open(path) as f:
        text = f.read()
    # try whole-file JSON first (a single metrics.json, or a JSON array)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict):
                    yield o
            return
        if isinstance(obj, dict):
            yield obj
            return
    except json.JSONDecodeError:
        pass
    # else parse line by line; tolerate non-JSON lines and trailing log prefixes
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        start = line.find('{')
        if start == -1:
            continue
        try:
            yield json.loads(line[start:])
        except json.JSONDecodeError:
            continue


def _collect(path):
    files = []
    if os.path.isdir(path):
        files += sorted(glob.glob(os.path.join(path, '**', 'eval_log.jsonl'), recursive=True))
        files += sorted(glob.glob(os.path.join(path, '**', 'metrics.json'), recursive=True))
    else:
        files = [path]
    recs = []
    for fp in files:
        for r in _records_from_file(fp):
            r = {k: v for k, v in r.items() if not k.startswith('_')}
            if 'step' not in r:
                r['step'] = _step_of(r)
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='eval_log.jsonl / metrics.json / directory / .txt log')
    ap.add_argument('-o', '--out', help='output CSV (default: <path>.csv or eval_log.csv)')
    ap.add_argument('--sort', default='step', help='column to sort by (default: step)')
    a = ap.parse_args()

    recs = _collect(a.path)
    if not recs:
        sys.exit(f"no JSON records found in {a.path}")

    keys = list(dict.fromkeys(k for r in recs for k in r))
    cols = [c for c in PREFERRED if c in keys] + sorted(k for k in keys if k not in PREFERRED)

    try:
        recs.sort(key=lambda r: (r.get(a.sort) is None, r.get(a.sort)))
    except TypeError:
        recs.sort(key=lambda r: str(r.get(a.sort)))

    out = a.out
    if not out:
        base = a.path.rstrip('/')
        out = (base + '.csv') if os.path.isfile(base) else os.path.join(base, 'eval_log.csv')

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in recs:
            w.writerow(r)

    # also echo a compact table
    show = [c for c in ('step', 'mAP', 'mAP50', 'F1score50', 'AR100') if c in cols]
    if show:
        print('  '.join(f'{c:>10}' for c in show))
        for r in recs:
            print('  '.join(
                (f'{r[c]:>10.4f}' if isinstance(r.get(c), float) else f'{str(r.get(c, "")):>10}')
                for c in show))
    print(f"\n{len(recs)} rows, {len(cols)} cols -> {out}")


if __name__ == '__main__':
    main()
