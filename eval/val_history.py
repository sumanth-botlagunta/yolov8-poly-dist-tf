"""Append-only JSONL store for per-epoch validation reports.

One ``<run>/val_history.jsonl`` per run replaces the previous per-epoch
``val_metrics/epoch_NNNN.json`` + ``.txt`` pair (hundreds of files). Each
validation appends exactly **one** line — the full report dict from
``eval/metrics_report.build_report`` (``mean`` / ``best_conf`` / ``all_conf`` /
``per_category_ap``) augmented with ``epoch`` / ``step`` / ``checkpoint`` and the
headline scalar ``metrics`` (mAP / mAP50 / F1score50 / AR100).

Why JSONL: append is O(line) regardless of file size, so it has zero training-step
impact; the file is plain text (greppable, tailable, diffable, crash-safe — an
interrupted run loses at most the last partial line); and a record is a superset of
what ``metrics_report.write_txt`` consumes, so any epoch round-trips back to the exact
ckpt-format txt with no schema or SQL.

Read/extract with ``utils/reports/val_history.py`` (txt / json / csv, ``--best``, ``--list``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# Header keys written first on each line (purely cosmetic ordering; readers key by name).
_LEAD_KEYS = ('epoch', 'step', 'checkpoint', 'metrics')


def append_record(
    jsonl_path: str,
    report: dict,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    checkpoint: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> dict:
    """Append one validation report as a JSON line. Returns the written record.

    The record is the report dict (so it round-trips through
    ``metrics_report.write_txt`` unchanged) plus ``epoch`` / ``step`` /
    ``checkpoint`` / ``metrics``. ``metrics`` is coerced to plain floats.
    """
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

    record: Dict = dict(report)
    if epoch is not None:
        record['epoch'] = int(epoch)
    if step is not None:
        record['step'] = int(step)
    if checkpoint is not None:
        record['checkpoint'] = str(checkpoint)
    if metrics is not None:
        record['metrics'] = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    # Order the lead keys first for readability, keep the rest in insertion order.
    ordered = {k: record[k] for k in _LEAD_KEYS if k in record}
    ordered.update({k: v for k, v in record.items() if k not in _LEAD_KEYS})

    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(ordered) + '\n')
    return ordered


def load_records(jsonl_path: str) -> List[dict]:
    """Read all records (skips blank / partial trailing lines)."""
    out: List[dict] = []
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a partial last line from an interrupted write
                continue
    return out


def latest_per_epoch(records: List[dict]) -> List[dict]:
    """Collapse re-validations: keep only the LAST record per epoch (in file order).

    The store is append-only, so validating the same checkpoint twice writes two lines
    with the same ``epoch``. For trend / best / export views we want one canonical row
    per epoch — the most recent (a re-validation supersedes). Records without an
    ``epoch`` fall back to ``step`` then to a per-record identity, so nothing is dropped
    spuriously. Order is preserved (first appearance of each key).
    """
    out: Dict = {}
    order: List = []
    for i, r in enumerate(records):
        ep = r.get('epoch')
        if ep is not None:
            key = ('epoch', int(ep))
        elif r.get('step') is not None:
            key = ('step', int(r['step']))
        else:
            key = ('idx', i)
        if key not in out:
            order.append(key)
        out[key] = r       # later record wins
    return [out[k] for k in order]


def metric_of(record: dict, key: str = 'F1score50') -> Optional[float]:
    """Headline metric for a record. Prefers ``metrics[key]``; for the default
    F1score50 falls back to the report's ``mean.f1`` (they coincide by construction)."""
    m = record.get('metrics') or {}
    if key in m:
        return float(m[key])
    if key == 'F1score50':
        mean = record.get('mean') or {}
        if 'f1' in mean:
            return float(mean['f1'])
    return None


def best_record(records: List[dict], key: str = 'F1score50') -> Optional[dict]:
    """The record maximizing ``key`` (None if none have it).

    Collapses re-validations first (``latest_per_epoch``) so a stale duplicate of an
    epoch cannot win over its own re-validated value.
    """
    scored = [(metric_of(r, key), r) for r in latest_per_epoch(records)]
    scored = [(v, r) for v, r in scored if v is not None]
    if not scored:
        return None
    return max(scored, key=lambda vr: vr[0])[1]


def select(
    records: List[dict],
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    checkpoint: Optional[str] = None,
) -> Optional[dict]:
    """Return the last record matching the given selector (None if no match)."""
    match = None
    for r in records:
        if epoch is not None and r.get('epoch') != int(epoch):
            continue
        if step is not None and r.get('step') != int(step):
            continue
        if checkpoint is not None and checkpoint not in str(r.get('checkpoint', '')):
            continue
        match = r   # keep the latest match
    return match


def resolve_path(path: str) -> str:
    """Accept either the jsonl file or a run directory containing it."""
    if os.path.isdir(path):
        return os.path.join(path, 'val_history.jsonl')
    return path
