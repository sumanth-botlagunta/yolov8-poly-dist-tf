"""One-shot eval-parity diagnostic. Zero arguments — all paths baked in.

Answers, in a single run, WHICH input differs between the trainer's per-epoch
validation and tools.eval for the same checkpoint:

  1. CONFIG    — machine-diff of <run_dir>/params.yaml (the exact config the
                 trainer ran with) against the config file passed to tools.eval.
  2. CHECKPOINT— identity of the checkpoint: stored global_step / epochs /
                 EMA step, shadow count, and the trainer's recorded val metrics
                 for that step from val_history.jsonl.
  3. DATA      — the eval-mode validation dataset built twice: determinism +
                 content fingerprint (images, GT counts, crowd/dontcare).
  4. WEIGHTS   — the checkpoint restored through BOTH paths (tools.eval's
                 restore_eval_weights vs the trainer's Checkpoint+swap_in):
                 byte fingerprints must match.
  5. METRICS   — one forward pass over a bounded sample, the SAME predictions
                 fed through BOTH aggregation paths (trainer's aggregate/reduce
                 vs the tool's direct evaluator): both F1 tables printed.

Run from the repo root:  python check_eval_parity.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUN_DIR = os.environ.get(
    'EP_RUN_DIR', '/group-volume/bot.sumanth/new_codebase_tensorflow_experiments/nov_model_8')
CONFIG_PATH = os.environ.get(
    'EP_CONFIG', 'configs/experiments/yolo/yolov8_nov2_model.yaml')
CKPT = os.environ.get('EP_CKPT', os.path.join(RUN_DIR, 'ckpt-51498'))
SAMPLE_BATCHES = int(os.environ.get('EP_BATCHES', '50'))

SEP = '=' * 78


def _flatten(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f'{prefix}{k}.'))
    elif isinstance(d, list):
        out[prefix[:-1]] = json.dumps(d)
    else:
        out[prefix[:-1]] = repr(d)
    return out


def section_1_config_diff():
    print(SEP); print('SECTION 1 — config the trainer ran with  vs  config given to tools.eval')
    print(SEP)
    import yaml
    params_path = os.path.join(RUN_DIR, 'params.yaml')
    if not os.path.exists(params_path):
        print(f'!! {params_path} not found — cannot diff. SKIPPED'); return
    with open(params_path) as f:
        trained = _flatten(yaml.safe_load(f))
    with open(CONFIG_PATH) as f:
        raw_eval_cfg = yaml.safe_load(f)
    # The eval config file may be composed (defaults + includes) — diff what the
    # loader actually produces, converted back to plain data.
    import dataclasses
    from configs.yaml_loader import load_config
    evald = _flatten(dataclasses.asdict(load_config(CONFIG_PATH)))
    del raw_eval_cfg

    eval_relevant_prefixes = (
        'task.validation_data', 'task.model', 'task.num_classes',
        'task.ignore_dontcare', 'task.ignore_iscrowds', 'task.iscrowds_labels',
        'task.with_polygons', 'task.with_distance', 'task.find_best_score_thresh',
        'runtime.mixed_precision_dtype',
    )
    keys = sorted(set(trained) | set(evald))
    n_diff = n_relevant = 0
    for k in keys:
        a, b = trained.get(k, '<absent>'), evald.get(k, '<absent>')
        if a != b:
            n_diff += 1
            relevant = k.startswith(eval_relevant_prefixes)
            n_relevant += relevant
            tag = 'EVAL-RELEVANT >>>' if relevant else '(train-only)     '
            print(f'  {tag} {k}\n{"":21s}trained: {a}\n{"":21s}eval:    {b}')
    print(f'\n  config leaves compared: {len(keys)}   differing: {n_diff}   '
          f'EVAL-RELEVANT differing: {n_relevant}')
    if n_relevant:
        print('  >>> The two scripts are NOT evaluating under the same config. '
              'Fix the lines above first — this alone can explain the F1 gap.')


def section_2_checkpoint_identity():
    print(SEP); print(f'SECTION 2 — checkpoint identity: {CKPT}')
    print(SEP)
    import tensorflow as tf
    names = {n: s for n, s in tf.train.list_variables(CKPT)}
    # The saver canonicalizes each shared variable under ONE path; the model
    # weights usually appear as optimizer/_model_vars/N (aliased by model/).
    n_model = sum(1 for n in names
                  if n.startswith('model/') or '/_model_vars/' in n)
    n_shadow = sum(1 for n in names if '/_shadows/' in n)
    print(f'  variables total={len(names)}  model weights={n_model}  '
          f'EMA shadows={n_shadow}')
    if n_shadow == 0:
        print('  !! NO EMA shadows — restore_eval_weights would fall back to '
              'RAW weights for this file.')
    for key in names:
        base = key.split('/.ATTRIBUTES')[0]
        if base in ('global_step', 'completed_epochs') or base.endswith('_ema_step'):
            print(f'  {base:20s} = {tf.train.load_variable(CKPT, key)}')
    hist = os.path.join(RUN_DIR, 'val_history.jsonl')
    if os.path.exists(hist):
        want = os.path.basename(CKPT).rsplit('-', 1)[-1]
        found = False
        for line in open(hist):
            r = json.loads(line)
            if str(r.get('step')) == want:
                m = r.get('metrics', {})
                print(f"  trainer recorded @step {want}: "
                      f"F1score50={m.get('F1score50')}  P={m.get('precision50')}  "
                      f"R={m.get('recall50')}  best_conf={m.get('best_conf_thresh')}")
                found = True
        if not found:
            print(f'  !! no val_history row for step {want} — the trainer never '
                  'validated at this exact step; the comparison baseline may be '
                  'from a DIFFERENT step/checkpoint.')
    else:
        print(f'  ({hist} not found)')


def _image_hashes(images, labels):
    """Order-invariant: one hash per image (pixels + its GT rows)."""
    out = []
    ims = images.numpy()
    n_gt = labels['n_gt'].numpy()
    bbox = labels['bbox'].numpy()
    cls = labels['classes'].numpy()
    for i in range(ims.shape[0]):
        h = hashlib.sha256()
        h.update(ims[i].tobytes())
        ng = int(n_gt[i])
        h.update(bbox[i, :ng].tobytes())
        h.update(cls[i, :ng].tobytes())
        out.append(h.hexdigest()[:16])
    return out


def section_3_dataset(cfg, task):
    print(SEP); print('SECTION 3 — validation dataset: determinism + content')
    print(SEP)
    import dataclasses
    dc = dataclasses.replace(cfg.task.validation_data, is_training=False)
    per_build = []
    for build in range(2):
        ds = task.build_inputs(dc)
        hashes, n_img, n_gt, n_crowd, n_dc = [], 0, 0, 0, 0
        for i, (im, lb) in enumerate(ds):
            if i >= 3:
                break
            hashes += _image_hashes(im, lb)
            n_img += int(im.shape[0])
            n_gt += int(lb['n_gt'].numpy().sum())
            if 'is_crowd' in lb:
                n_crowd += int(lb['is_crowd'].numpy().sum())
            if 'is_dontcare' in lb:
                n_dc += int(lb['is_dontcare'].numpy().sum())
        per_build.append(hashes)
        print(f'  build {build}: images={n_img} GT={n_gt} crowd={n_crowd} '
              f'dontcare={n_dc}')
    same_order = per_build[0] == per_build[1]
    same_content = sorted(per_build[0]) == sorted(per_build[1])
    print(f'  content identical across builds: {same_content}'
          + ('' if same_content else '   <-- REAL nondeterminism (pixels/labels)'))
    print(f'  order identical across builds:   {same_order}'
          + ('' if same_order else '   (order-only difference is metric-invariant)'))


def _weights_fingerprint(model):
    h = hashlib.sha256()
    for v in model.variables:
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


def section_4_weights(cfg):
    print(SEP); print('SECTION 4 — weights: tools.eval restore vs trainer restore+swap')
    print(SEP)
    import tensorflow as tf
    import tools.eval as TE
    from train.task import YoloV8Task

    model_tool = TE._load_model_from_checkpoint(cfg, CKPT)
    fp_tool = _weights_fingerprint(model_tool)
    print(f'  tools.eval path (restore_eval_weights):        {fp_tool}')

    task_tr = YoloV8Task(cfg)
    model_tr = task_tr.build_model()
    ema = task_tr.build_optimizer()
    tf.train.Checkpoint(model=model_tr, optimizer=ema).restore(CKPT).expect_partial()
    ema.swap_in(model_tr)
    fp_tr = _weights_fingerprint(model_tr)
    print(f'  trainer path (Checkpoint restore + EMA swap):  {fp_tr}')
    print('  identical:', fp_tool == fp_tr)
    del model_tr, ema, task_tr
    return model_tool


def section_5_dual_aggregation(cfg, task, model):
    print(SEP)
    print(f'SECTION 5 — same forward pass through BOTH metric paths '
          f'({SAMPLE_BATCHES} batches sample)')
    print(SEP)
    import dataclasses
    import tensorflow as tf
    from train.task import normalize_images
    from eval.coco_metrics import COCOEvaluator

    dc = dataclasses.replace(cfg.task.validation_data, is_training=False)
    ds = task.build_inputs(dc)

    coco_ev = COCOEvaluator(
        num_classes=cfg.task.num_classes,
        image_size=tuple(cfg.task.model.input_size[:2]),
        ignore_dontcare=cfg.task.ignore_dontcare,
        ignore_iscrowds=cfg.task.ignore_iscrowds,
        iscrowds_labels=cfg.task.iscrowds_labels,
    )
    logs = None
    for i, (im, lb) in enumerate(ds):
        if i >= SAMPLE_BATCHES:
            break
        preds = model(normalize_images(im), training=False)
        coco_ev.update(preds, lb)                                   # tool path
        logs = task.aggregate_logs(logs, {'predictions': preds,    # trainer path
                                          'labels': lb})
        if (i + 1) % 10 == 0:
            print(f'  ... {i + 1}/{SAMPLE_BATCHES} batches')

    m_tool = coco_ev.evaluate()
    m_tr = task.reduce_aggregated_logs(logs)
    print('\n  metric               trainer-math   tool-math')
    for k in ('F1score50', 'precision50', 'recall50', 'best_conf_thresh',
              'mAP50', 'mAP'):
        a, b = float(m_tr.get(k, float('nan'))), float(m_tool.get(k, float('nan')))
        flag = '   <-- MISMATCH' if abs(a - b) > 5e-3 else ''
        print(f'  {k:20s} {a:12.4f} {b:12.4f}{flag}')
    print('\n  NOTE: sample-based numbers; they will not equal the full-split '
          'F1, but the two columns MUST match each other.')


def main():
    print(f'run_dir = {RUN_DIR}\nconfig  = {CONFIG_PATH}\nckpt    = {CKPT}\n')
    section_1_config_diff()
    section_2_checkpoint_identity()

    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    cfg = load_config(CONFIG_PATH)
    from tools.shared.runtime_setup import apply_eval_precision_policy
    apply_eval_precision_policy(cfg)
    task = YoloV8Task(cfg)

    section_3_dataset(cfg, task)
    model = section_4_weights(cfg)
    section_5_dual_aggregation(cfg, task, model)
    print(SEP)
    print('DONE. Interpretation: any EVAL-RELEVANT config diff (S1), unexpected '
          'checkpoint identity (S2), non-determinism (S3), weight-fingerprint '
          'mismatch (S4), or column mismatch (S5) is the cause. If ALL sections '
          'pass, the two scripts were not run on the same run_dir/config/'
          'checkpoint triple originally compared.')


if __name__ == '__main__':
    main()
