# Codebase Guide (illustrated PDF)

An illustrated, beginner-friendly explainer of this codebase, built as an editable LaTeX
source that compiles to `codebase_guide.pdf`.

- `main.tex` — document root (title, TOC, chapter includes).
- `chapters/` — one `.tex` file per chapter; edit these to change the content.
- `figures/` — generated figures (real pipeline renders) + diagram assets.
- `gen_figures.py` — regenerates the real-pipeline render figures from the actual code.
- `build.sh` — compiles the PDF with `tectonic`.

## Build

```bash
bash docs/codebase_guide/build.sh
```
