# `.claude/` — Claude Code project configuration

This folder is **committed** so the whole team gets the same Claude Code setup
(slash commands, permissions, and subagents) with zero manual configuration.

## Contents

| Path | What it is |
|------|------------|
| `commands/` | Project **slash commands** (skills). Each `*.md` is invocable as `/<name>` in Claude Code — e.g. `/train`, `/eval`, `/test`, `/benchmark`, `/export`, `/check-env`, `/migrate-ckpt`, `/visualize-aug`. |
| `agents/` | Project **subagents** — specialized assistants the main agent can delegate to (see below). |
| `settings.json` | **Shared** settings: a Bash permission allowlist for common dev commands (pytest, git, python tools, pip/conda) so routine actions don't prompt. Safe to edit and commit. |
| `settings.local.json` | **Per-user, git-ignored** overrides (personal paths/permissions). Never committed. |

## Slash commands (`commands/`)
Run any of these in Claude Code:
- `/train` — launch a training run
- `/eval` — standalone evaluation on a checkpoint
- `/export` — export to SavedModel (+ optional TFLite)
- `/benchmark` — profile data-pipeline throughput
- `/test` — run the pytest suite
- `/check-env` — verify the training environment + dataset availability
- `/migrate-ckpt` — migrate an old checkpoint to the new model
- `/visualize-aug` — dump augmentation-stage visualizations

## Subagents (`agents/`)
Delegate focused work to these:
- **tf-test-writer** — writes/fixes pytest tests following repo conventions (eager mode, synthetic fixtures, no TFDS for unit/integration).
- **loss-reviewer** — audits the loss / TAL assignment code against the Ultralytics YOLOv8 + PolyYOLO reference recipes.
- **pipeline-debugger** — diagnoses tf.data pipeline, augmentation, and polygon-format issues.

## Project rules & knowledge
- **`/CLAUDE.md`** (repo root) is the authoritative project guide Claude loads automatically: architecture, polygon formats, loss conventions, file layout, dependencies. Keep it current when the code changes.
- **`/docs/`** holds longer-form documentation (architecture, data pipeline, losses, training, testing).

## Notes
- `settings.json` may contain an absolute `additionalDirectories` path from the machine it was created on; it's harmless on other machines (it just grants extra read access that may not exist). Adjust if needed.
