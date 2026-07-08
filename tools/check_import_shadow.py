"""Detect stale installed copies of this codebase shadowing the working tree.

Prints, with zero TensorFlow involvement:
  1. where each top-level package actually resolves from (must be the repo)
  2. any copy of these packages present in site-packages (a shadow waiting to
     fire whenever a process runs from a different cwd)
  3. installed distributions whose names look like this project

Run from anywhere:  python tools/check_import_shadow.py
"""

import importlib
import os
import site
import sys

PKGS = ("models", "losses", "data_pipeline", "eval", "train",
        "configs", "optimizers", "tools")

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"repo root      : {repo}")
print(f"cwd            : {os.getcwd()}")
print(f"sys.path[0:3]  : {sys.path[0:3]}")

print("\n=== 1. import resolution (every path must be inside the repo) ===")
sys.path.insert(0, repo)
bad = []
for name in PKGS:
    try:
        m = importlib.import_module(name)
        f = getattr(m, "__file__", None) or str(getattr(m, "__path__", "?"))
        ok = repo in str(f)
        if not ok:
            bad.append((name, f))
        print(f"  {name:<14} -> {f}   {'ok' if ok else '<-- SHADOWED'}")
    except Exception as e:
        print(f"  {name:<14} -> import failed: {e}")

print("\n=== 2. copies present in site-packages (shadow risk even if 1 is clean) ===")
dirs = list(site.getsitepackages())
if site.getusersitepackages():
    dirs.append(site.getusersitepackages())
found = []
for d in dirs:
    if not os.path.isdir(d):
        continue
    for name in PKGS:
        p = os.path.join(d, name)
        if os.path.isdir(p):
            found.append(p)
            print(f"  FOUND: {p}")
if not found:
    print("  none — no installed copies of these packages")

print("\n=== 3. suspicious installed distributions ===")
try:
    from importlib import metadata
    hits = [f"  {d.metadata['Name']}=={d.version}" for d in metadata.distributions()
            if any(k in (d.metadata['Name'] or '').lower()
                   for k in ('rvc', 'yolo', 'vision', 'polygon'))]
    print("\n".join(hits) if hits else "  none")
except Exception as e:
    print(f"  (could not list: {e})")

print("\n=== VERDICT ===")
if bad or found:
    print("  SHADOW DETECTED. Fix: pip uninstall the distribution owning the paths")
    print("  above (see section 3), then re-run one eval — numbers should snap to")
    print("  the trainer's values.")
else:
    print("  No shadowing: both processes execute the working-tree code. The next")
    print("  (and final) divergence layer to check is a weight checksum.")
