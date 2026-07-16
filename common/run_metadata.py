"""Writes a self-describing provenance record for a training run.

A checkpoint is the product of config + code + data; params.yaml already
captures the config. This records the other two, plus the invocation and
environment, so the run directory answers "what exactly produced this
checkpoint?" on its own:

  * code: git commit + branch + dirty flag.
  * data: each TFDS dataset's requested and resolved version (best-effort).
  * invocation: the command line + the seed-init / resume sources.
  * environment: TF / Python / platform / host / GPUs + start time.

Writes <output_dir>/run_metadata.json. Runs once at startup and never raises
(provenance must not break training): every probe is wrapped.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from typing import Optional

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(args) -> Optional[str]:
    try:
        out = subprocess.run(['git', '-C', _REPO_ROOT, *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_info(output_dir: str) -> dict:
    commit = _git(['rev-parse', 'HEAD'])
    if commit is None:
        return {'available': False}
    status = _git(['status', '--porcelain'])
    return {
        'available': True,
        'commit': commit,
        'branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'dirty': bool(status),
    }


def _env_info() -> dict:
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'host': platform.node()}
    try:
        import tensorflow as tf
        env['tensorflow'] = tf.__version__
        env['gpus'] = [d.name for d in tf.config.list_physical_devices('GPU')]
        env['mixed_precision'] = tf.keras.mixed_precision.global_policy().name
    except Exception:
        pass
    return env


def _dataset_specs(config) -> list:
    """Collects every (role, name, requested_version) the config references."""
    td = config.task.train_data
    vd = config.task.validation_data
    sources = [('train', getattr(td, 'tfds_name', None)),
               ('val', getattr(vd, 'tfds_name', None)),
               ('copy_paste', getattr(td, 'tfds_for_cnp', None))]
    dist = getattr(td, 'distance_data', None)
    if dist is not None:
        sources.append(('distance', getattr(dist, 'tfds_name', None)))

    data_dir = getattr(td, 'tfds_data_dir', None)
    specs = []
    for role, raw in sources:
        if not raw:
            continue
        for part in str(raw).split(','):
            part = part.strip()
            if not part:
                continue
            name, _, ver = part.partition(':')
            specs.append({'role': role, 'name': name, 'requested': ver or None,
                          'resolved': _resolve_version(part, data_dir)})
    return specs


def _resolve_version(name_with_ver: str, data_dir) -> Optional[str]:
    """Returns the version TFDS would actually load (best-effort; catches unpinned names)."""
    try:
        import tensorflow_datasets as tfds
        builder = tfds.builder(name_with_ver, data_dir=data_dir)
        return str(builder.info.version)
    except Exception:
        return None


def write_run_metadata(output_dir: str, config, resume_from: Optional[str] = None,
                       started_at: Optional[str] = None) -> Optional[str]:
    """Writes run_metadata.json to output_dir.

    Never raises (provenance must not break a training run).

    Returns:
      The written path, or None on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        task = config.task
        meta = {
            'started_at': started_at,
            'command': ' '.join([sys.executable] + sys.argv),
            'git': _git_info(output_dir),
            'env': _env_info(),
            'run': {
                'output_dir': output_dir,
                'finetune_from': getattr(task, 'finetune_from', None),
                'init_checkpoint': getattr(task, 'init_checkpoint', None),
                'init_checkpoint_modules': getattr(task, 'init_checkpoint_modules', None),
                'freeze_modules': getattr(task, 'freeze_modules', None),
                'resume_from': resume_from,
                'grad_accum_steps': getattr(config.trainer, 'grad_accum_steps', 1),
            },
            'datasets': _dataset_specs(config),
        }
        path = os.path.join(output_dir, 'run_metadata.json')
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)
        g = meta['git']
        log.info("Run provenance -> %s (commit %s%s)", path,
                 (g.get('commit') or '?')[:8], ', DIRTY' if g.get('dirty') else '')
        return path
    except Exception as e:                      # pragma: no cover - defensive
        log.warning("Could not write run metadata: %s", e)
        return None
