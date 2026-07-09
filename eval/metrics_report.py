"""Build and export the per-category F1 / precision / recall validation report.

The canonical store is **JSON** (full-precision, dependency-free). From a saved JSON
(or directly from a `COCOEvaluator`) this writes the human/usable formats:

  * ``.txt``  — console-style tables (best-conf per category + all-confidence sweep),
                matching the on-device eval print format.
  * ``.csv``  — two files: ``<base>_best_conf.csv`` and ``<base>_all_conf.csv``.
  * ``.xlsx`` — ONE workbook, TWO sheets: "best_conf" (all classes at their best
                confidence) and "all_conf" (all classes at every confidence).

Excel is written with the **standard library only** (a minimal OOXML zip), so it works
without pandas/openpyxl. Numbers are written as real numeric cells (Excel can sum/sort).

Typical use:
  * During training the trainer calls :func:`build_report` and appends the result
    (one line per validation) to ``<run>/val_history.jsonl`` via
    ``eval/val_history.py``; extract any epoch back to txt/json/csv with
    ``tools/val_history.py``. ``tools/eval.py --output_dir`` still uses
    :func:`save_canonical` to drop a ``<ckpt>_val.json`` + ``.txt`` pair for a
    single offline evaluation.
  * Offline, ``tools/pipeline/export_val_metrics.py`` reads one report JSON and
    calls :func:`write_csv` / :func:`write_xlsx` / :func:`write_txt`.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Dict, List, Optional

# Class-id → name (best effort; falls back to the bare id).
try:
    from configs.class_map import DETECTION_CLASSES
    _NAMES = {int(k): str(v) for k, v in DETECTION_CLASSES.items()} \
        if isinstance(DETECTION_CLASSES, dict) else \
        {i: str(n) for i, n in enumerate(DETECTION_CLASSES)}
except Exception:
    _NAMES = {}


def _name(cat: int) -> str:
    return _NAMES.get(int(cat), str(int(cat)))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_report(coco_ev, conf_grid=None, epoch=None, step=None,
                 extra: Optional[dict] = None, envelope_sweep: bool = False) -> dict:
    """Assemble the report dict from a COCOEvaluator (after ``evaluate()``).

    ``envelope_sweep=False`` (default) builds the all-conf table from the raw
    operating-point sweep so it matches the headline F1score50 / best-conf table;
    ``envelope_sweep=True`` uses COCO's interpolated envelope precision instead.
    """
    tables = coco_ev.metrics_tables(conf_grid=conf_grid, envelope_sweep=envelope_sweep)
    report = {
        'epoch': epoch,
        'step':  step,
        **tables,
    }
    if extra:
        report['extra'] = extra
    return report


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------

def write_json(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_canonical(report: dict, out_dir: str, basename: str) -> Dict[str, str]:
    """Write the per-validation canonical artifacts (json + txt). Returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    jp = write_json(report, os.path.join(out_dir, basename + '.json'))
    tp = write_txt(report, os.path.join(out_dir, basename + '.txt'))
    return {'json': jp, 'txt': tp}


# ---------------------------------------------------------------------------
# TXT (console-style)
# ---------------------------------------------------------------------------

def write_txt(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines: List[str] = []
    hdr = f"IoU={report.get('iou_thresh')}  area={report.get('area')}  maxDets={report.get('max_dets')}"
    ep = report.get('epoch'); st = report.get('step')
    tag = (f"epoch={ep} " if ep is not None else "") + (f"step={st}" if st is not None else "")
    m = report.get('mean', {})
    lines.append("=" * 78)
    lines.append(f" VALIDATION METRICS  {tag}".rstrip())
    lines.append(f" {hdr}")
    lines.append(f" MEAN over categories:  F1={m.get('f1',0):.4f}  "
                 f"precision={m.get('precision',0):.4f}  recall={m.get('recall',0):.4f}")
    lines.append("=" * 78)

    lines.append("\n=== best confidence per category ===")
    lines.append(f"{'cat':>4} {'name':<18}{'f1':>9}{'precision':>11}{'recall':>9}{'conf':>8}")
    lines.append("-" * 60)
    for r in report.get('best_conf', []):
        lines.append(f"{int(r['category']):>4} {_name(r['category']):<18}"
                     f"{r['f1']:>9.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}"
                     f"{r['conf_threshold']:>8.3f}")

    lines.append("\n=== all confidence thresholds per category ===")
    if report.get('sweep_source') == 'coco_envelope':
        lines.append(" NOTE: values below are COCO-interpolated (monotone-envelope) "
                     "precision, NOT raw operating-point values — they can disagree "
                     "with the best-conf table above.")
    lines.append(f"{'cat':>4} {'name':<18}{'thresh':>8}{'f1':>9}{'precision':>11}{'recall':>9}")
    lines.append("-" * 60)
    cur = None
    for r in report.get('all_conf', []):
        if r['category'] != cur:
            cur = r['category']
            lines.append(f"--- CATEGORY {int(cur)} ({_name(cur)}) ---")
        lines.append(f"{int(r['category']):>4} {_name(r['category']):<18}"
                     f"{r['thresh']:>8.2f}{r['f1']:>9.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}")

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# CSV (two files = the two "sheets")
# ---------------------------------------------------------------------------

def write_csv(report: dict, out_dir: str, base: str) -> Dict[str, str]:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    bp = os.path.join(out_dir, base + '_best_conf.csv')
    with open(bp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'name', 'f1', 'precision', 'recall', 'conf_threshold'])
        for r in report.get('best_conf', []):
            w.writerow([int(r['category']), _name(r['category']),
                        f"{r['f1']:.6f}", f"{r['precision']:.6f}",
                        f"{r['recall']:.6f}", f"{r['conf_threshold']:.4f}"])
    ap = os.path.join(out_dir, base + '_all_conf.csv')
    with open(ap, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'name', 'thresh', 'f1', 'precision', 'recall'])
        for r in report.get('all_conf', []):
            w.writerow([int(r['category']), _name(r['category']), f"{r['thresh']:.2f}",
                        f"{r['f1']:.6f}", f"{r['precision']:.6f}", f"{r['recall']:.6f}"])
    return {'best_conf': bp, 'all_conf': ap}


# ---------------------------------------------------------------------------
# XLSX — minimal OOXML, stdlib only, two sheets
# ---------------------------------------------------------------------------

def _xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


def _col_letter(idx0: int) -> str:
    s = ""
    n = idx0 + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _sheet_xml(rows: List[list]) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
           '<sheetData>']
    for ri, row in enumerate(rows, start=1):
        out.append(f'<row r="{ri}">')
        for ci, val in enumerate(row):
            ref = f'{_col_letter(ci)}{ri}'
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                out.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
                           f'{_xml_escape(str(val))}</t></is></c>')
        out.append('</row>')
    out.append('</sheetData></worksheet>')
    return "".join(out)


def _xlsx(path: str, sheets: List[tuple]) -> str:
    """sheets = [(name, rows), ...]; rows = list of lists (str or number cells)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(sheets)
    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                     '<Default Extension="xml" ContentType="application/xml"/>',
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(n):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
                             f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append('</Types>')

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                 '</Relationships>')

    sheets_xml = "".join(f'<sheet name="{_xml_escape(nm)}" sheetId="{i+1}" r:id="rId{i+1}"/>'
                         for i, (nm, _) in enumerate(sheets))
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{sheets_xml}</sheets></workbook>')

    wb_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(n):
        wb_rels.append(f'<Relationship Id="rId{i+1}" '
                       f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                       f'Target="worksheets/sheet{i+1}.xml"/>')
    wb_rels.append('</Relationships>')

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', "".join(content_types))
        z.writestr('_rels/.rels', root_rels)
        z.writestr('xl/workbook.xml', workbook)
        z.writestr('xl/_rels/workbook.xml.rels', "".join(wb_rels))
        for i, (_, rows) in enumerate(sheets):
            z.writestr(f'xl/worksheets/sheet{i+1}.xml', _sheet_xml(rows))
    return path


def write_xlsx(report: dict, path: str) -> str:
    """One workbook, two sheets: best_conf (per class at best conf) and all_conf."""
    best_rows = [['category', 'name', 'f1', 'precision', 'recall', 'conf_threshold']]
    for r in report.get('best_conf', []):
        best_rows.append([int(r['category']), _name(r['category']),
                          round(r['f1'], 6), round(r['precision'], 6),
                          round(r['recall'], 6), round(r['conf_threshold'], 4)])
    # mean row at the bottom of best_conf
    m = report.get('mean', {})
    best_rows.append(['MEAN', '', round(m.get('f1', 0), 6),
                      round(m.get('precision', 0), 6), round(m.get('recall', 0), 6), ''])

    all_rows = [['category', 'name', 'thresh', 'f1', 'precision', 'recall']]
    for r in report.get('all_conf', []):
        all_rows.append([int(r['category']), _name(r['category']), round(r['thresh'], 2),
                         round(r['f1'], 6), round(r['precision'], 6), round(r['recall'], 6)])

    return _xlsx(path, [('best_conf', best_rows), ('all_conf', all_rows)])
