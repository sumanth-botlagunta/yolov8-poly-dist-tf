#!/usr/bin/env bash
# Build the illustrated guide PDF with tectonic.
#   bash docs/codebase_guide/build.sh
# Regenerate the real-pipeline figures first with:
#   PYTHONPATH=. python docs/codebase_guide/gen_figures.py
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v tectonic >/dev/null 2>&1; then
  TECTONIC="$(command -v tectonic)"
elif [ -x /opt/homebrew/Caskroom/miniconda/base/envs/texenv/bin/tectonic ]; then
  TECTONIC=/opt/homebrew/Caskroom/miniconda/base/envs/texenv/bin/tectonic
else
  echo "tectonic not found. Install it, e.g.: conda create -n texenv -c conda-forge tectonic" >&2
  exit 1
fi

cd "$DIR"
"$TECTONIC" --keep-logs --outdir "$DIR" main.tex
mv -f "$DIR/main.pdf" "$DIR/codebase_guide.pdf"
echo "wrote $DIR/codebase_guide.pdf"
