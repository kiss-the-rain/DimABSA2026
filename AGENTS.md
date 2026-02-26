# Language & Output Policy
- 与我沟通、计划、评审、变更说明一律**使用中文（简体）**。
- 提交信息（commit）与 PR 描述默认中文：标题一句话，正文包含“动机/改动/验证方式/影响范围”四点。
- 代码内注释优先中文；如涉及官方 API/报错信息可保留英文原文并附中文解释。
- **Shell/Python 等可执行命令必须是有效英文指令**；路径尽量使用英文与 ASCII 字符。
- 遇到不确定术语，先给中文解释，再在括号中给英文原词。


# Repository Guidelines

## Project Structure & Module Organization
The `src/` tree mirrors the two Kaggle subtasks: `task_1/` holds the valence–arousal regressor, `task_2/` the ASTE extractor, and `utils/` the shared helpers (`paths.py`, `clean_text.py`). Raw competition inputs stay in `data/track_a/`, derived parquet/JSONL files go under `data/output/track_a/`, and scratch experiments belong in `data/trial/`. Downloaded checkpoints live in `models/` and should always be referenced through `Config.local_model_dir` plus `src.utils.paths.output_path` so scripts remain portable.

## Build, Test, and Development Commands
- `python src/task_1/save_roberta_base.py` — fetch `roberta-base` once (needs internet) and populate `models/roberta-base`.
- `python -m src.task_1.model` — train/validate Subtask 1; writes `best_model.pt` and `data/submit/task1/pred_dev.jsonl`.
- `python -m src.task_2.model` — train Subtask 2 with EMA + early stop; produces `aste_pa_best.pt` and `data/submit/task2/pred_dev.jsonl`.
- `python -m src.task_2.eval --gold ... --pred ... --lower --strip-punct` — offline scoring with strict and AO F1 plus VA metrics.
- `python src/test.py` — sanity-check text lengths before changing tokenizer limits or truncation rules.

## Coding Style & Naming Conventions
Target Python 3.10+, four spaces, snake_case for functions/tensors, and CapWords for models/datasets. Keep hyperparameters inside dataclasses near the top of each module, prefer explicit type hints, and route all filesystem logic through `src.utils.paths` rather than custom `Path` joins. CLI-ready modules must expose `main()` behind the usual guard so they can run via `python -m ...`.

## Testing Guidelines
There is no automated test suite, so enforce manual checks: (1) call `set_seed` before every run, (2) perform a short epoch on a 5–10% slice to confirm the loss trend, (3) validate JSONL outputs with `src/task_2/eval.py`, and (4) log exploratory stats through `src/test.py`, storing helper files under `data/trial/`. Document heuristic knobs (pair windows, VA weights, thresholds) and expected metric deltas directly in your PR.

## Commit & Pull Request Guidelines
Prefer Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) plus a brief imperative summary—the current history mixes this style with bare numerals. Each PR should outline motivation, reproduction commands, headline metrics (RMSE for Subtask 1, AO/strict F1 for Subtask 2), and any data or weight artifacts touched. Include before/after JSONL snippets whenever schemas or decoding logic change.

## Security & Configuration Tips
Set `DIMABSA_ROOT` (and optional `DIMABSA_OUT`) when running from notebooks so `paths.project_root()` resolves correctly. Keep `TRANSFORMERS_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false` in your environment to avoid surprise downloads or tokenizer contention. Store large checkpoints in Git LFS and keep Kaggle API tokens out of the repo—use the user-level kaggle.json or environment variables instead.
