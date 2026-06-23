# F1score50 definition change (design rationale — kept out of the codebase on purpose)

This note records WHY `eval/coco_metrics.py` was changed, so the repo code itself can stay
self-contained and professional (no external references). Full derivation + the reference
algorithm transcription live outside the repo at
`~/Documents/evaluation_scripts/extracted/_ANALYSIS.md`.

## What changed
`COCOEvaluator.F1score50` was redefined to match the reference validation pipeline the
checkpoints are selected under.

- BEFORE: per-category peak of `2PR/(P+R)` over pycocotools' 101-point INTERPOLATED PR curve,
  at maxDets=100; dont-care handled by `iscrowd=1`.
- AFTER: per-category MAX over a confidence-threshold grid (0.10..0.95 step 0.05) of F1 computed
  on the RAW cumulative precision/recall, at **maxDets=10**, with a duplicate-match (crowd) recall
  correction and dont-care-region absorption at IoU>=0.5; macro-averaged over categories with a
  valid F1. Implemented in `eval/coco_eval_custom.py::COCOevalCustom`.
- mAP / mAP50 / AR100 unchanged (stock pycocotools).
- GT annotations now carry a separate `dontcare` field (not collapsed into `iscrowd`).

## Confirmed configuration (from the reference validation)
- `find_best_score_thresh=True`, `ignore_dontcare=True`, `ignore_iscrowds=False`,
  `iscrowds_labels=[6, 13, 24, 36, 37]`, `params.maxDets=[1,10,100]`, score grid step 0.05 (0.1..0.95).
- mAP / mAP50 / AR100 ALSO come from the custom evaluator (not stock) — so the crowd/dont-care/
  duplicate-match corrections apply to them too. Routed through COCOevalCustom.
- maxDets per metric: mAP/mAP50/AR100 = 100; AR1/AR10 = 1/10; F1score50 = 10.
- Box scale: the reference uses original-image pixels (pre-letterbox). This affects only the
  area-stratified AP_small/medium/large (per-category); mAP/mAP50/AR100/F1score50 are
  scale-invariant and match without it. Original-px boxes NOT yet wired (deferred).

## Provenance
Derived from a reference evaluation implementation; transcription + analysis in the external
`evaluation_scripts/extracted/` folder. Do not reference any of that in the repo source.
