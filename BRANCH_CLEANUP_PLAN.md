# Branch Cleanup Plan: `exp/streamingtom-eval-lat`

**Goal:** Produce a clean PR-ready branch that can merge into `main` — no data blobs, no vendored subtrees, no scratch docs, squashed history. The two eval method families (`streaming/ReKV/` and `streaming/StreamingTom/`) stay exactly as-is; only outputs, scripts, and git hygiene are touched.

---

## Inventory of Problems

| # | Problem | Scope | Impact |
|---|---------|-------|--------|
| 1 | Evaluation result JSONs + plots committed | ~400 files across both output dirs | Repo bloat; data belongs on disk, not git |
| 2 | Two output roots for the same eval data | `outputs/evaluations_streaming/` (ReKV + SLURM ST) vs `outputs/evaluations_streamingtom/` (RunPod default) | One bad default in `run_eval_runpod.sh` line 70; everything else already correct |
| 3 | Training optimizer checkpoints committed | `outputs/train/**/*.pt` (~50 files × 2 ranks) | Binary blobs; already gitignored as `*.pt` but tracked anyway |
| 4 | Intermediate Duo head weight TSVs committed | `outputs/train/**/full_attention_heads_step=*.tsv` | Only `full_attention_heads_latest.tsv` per run needs to be in git |
| 5 | Vendored `LLaVA-NeXT` committed wholesale | 230 files in `streaming/StreamingTom/LLaVA-NeXT/` | Docs, examples, and unmodified upstream content |
| 6 | `MMDUO_STREAMINGTOM_WORKLOG.md` in repo root | 1 file | Scratch worklog, not a deliverable |
| 7 | Noisy 7-commit history with duplicates | All 7 commits | Makes `git log` on main unreadable |
| 8 | `.gitignore` missing rules for eval outputs | `.gitignore` | Missing coverage lets data sneak back in |
| 9 | Operational one-off scripts in `scripts/` | 7 scripts | Cluster-personal paths, dead experiments |

---

## What Does NOT Change

- `streaming/ReKV/` — untouched, stays exactly as-is
- `streaming/StreamingTom/` — untouched, stays exactly as-is
- All 6 method implementations, eval runners, merge scripts, guide

---

## Step-by-Step Cleanup

### Step 1 — Fix `.gitignore` first (before any `git rm`)

Add these rules so that after untracking, the files can never be accidentally re-staged:

```gitignore
# Evaluation outputs — data, not code; all 6 methods write here
outputs/evaluations_streaming/
outputs/evaluations_streamingtom/   # legacy RunPod default; fixed in Step 2

# Training checkpoints — binary blobs
outputs/train/**/*.pt
outputs/train/**/optimizer_scheduler_state*/

# Intermediate Duo head weight checkpoints — keep only _latest
outputs/train/**/full_attention_heads_step=*.tsv

# Scratch worklog
MMDUO_STREAMINGTOM_WORKLOG.md
```

**Why first:** If you `git rm` before updating `.gitignore`, a subsequent `git add .` silently re-stages the files.

Verify the rule works:
```bash
git check-ignore -v outputs/evaluations_streaming/rvs-ego/full_eval/full_streaming/chunk_000.json
# should print: .gitignore:N:outputs/evaluations_streaming/  <path>
```

---

### Step 2 — Fix the one divergent output path in `run_eval_runpod.sh`

**Root cause:** Every script and both `run_eval.py` files already write to `outputs/evaluations_streaming/`. The single exception is [streaming/StreamingTom/scripts/eval/run_eval_runpod.sh](streaming/StreamingTom/scripts/eval/run_eval_runpod.sh) line 70:

```bash
# Before
OUTPUT_BASE=${OUTPUT_BASE:-${ROOT}/outputs/evaluations_streamingtom}

# After
OUTPUT_BASE=${OUTPUT_BASE:-${ROOT}/outputs/evaluations_streaming}
```

This is a one-line code fix. After it, all 6 methods — ReKV (methods 1-4) via SLURM array, StreamingTom (methods 5-6) via SLURM or RunPod — write to the same root. The smoke test (`streaming/run_smoke_all_methods.sh`) already uses `outputs/evaluations_streaming/smoke_all_methods/` for all 6 methods.

**Note on dataset naming:** `run_eval.py` in both families converts `rvs_ego` → `rvs-ego` via `.replace("_", "-")` (line 122 in both files). The RunPod script passes `DATASET=rvs_ego`, so output lands at `outputs/evaluations_streaming/rvs-ego/...` — consistent with the SLURM scripts.

**Migrate existing RunPod results on disk** (one-time, before untracking):
```bash
mkdir -p outputs/evaluations_streaming/rvs-ego/full_eval/run2
cp -r outputs/evaluations_streamingtom/rvs_ego/full_eval/run2/* \
      outputs/evaluations_streaming/rvs-ego/full_eval/run2/

mkdir -p outputs/evaluations_streaming/rvs-movie/full_eval/run2
cp -r outputs/evaluations_streamingtom/rvs_movie/full_eval/run2/* \
      outputs/evaluations_streaming/rvs-movie/full_eval/run2/ 2>/dev/null || true
```

---

### Step 3 — Untrack all eval outputs from git

```bash
git rm --cached -r outputs/evaluations_streaming/
git rm --cached -r outputs/evaluations_streamingtom/
```

Files stay on disk. Only git tracking is removed. Verify:
```bash
git ls-files outputs/evaluations_streaming/ | wc -l   # expect 0
git ls-files outputs/evaluations_streamingtom/ | wc -l  # expect 0
ls outputs/evaluations_streaming/rvs-ego/              # should still exist
```

---

### Step 4 — Untrack training checkpoints, keep what matters

Remove optimizer checkpoints (binary, already gitignored but somehow tracked):
```bash
git ls-files outputs/train | grep '\.pt$' | xargs git rm --cached
```

Remove intermediate step TSVs:
```bash
git ls-files outputs/train | grep 'full_attention_heads_step=' | xargs git rm --cached
```

**Keep tracked** (these are small text files, reproducibility-critical):
- `outputs/train/*/config.json`
- `outputs/train/*/full_attention_heads_latest.tsv`
- `outputs/train/*/full_attention_heads.tsv` (final weight file, not a step checkpoint)

Verify:
```bash
git ls-files outputs/train/
# Expected: only config.json and full_attention_heads{,_latest}.tsv per run dir
```

---

### Step 5 — Prune vendored `LLaVA-NeXT` content

The 230 files in `streaming/StreamingTom/LLaVA-NeXT/` are a full upstream copy. Only one file is locally modified: `llava/__init__.py` (fixes `ImportError` for `LlavaLlamaForCausalLM`).

Add to `.gitignore`:
```gitignore
streaming/StreamingTom/LLaVA-NeXT/docs/
streaming/StreamingTom/LLaVA-NeXT/llava-critic-r1/
streaming/StreamingTom/LLaVA-NeXT/playground/
streaming/StreamingTom/LLaVA-NeXT/scripts/
streaming/StreamingTom/LLaVA-NeXT/trl/
streaming/StreamingTom/LLaVA-NeXT/cog.yaml
streaming/StreamingTom/LLaVA-NeXT/predict.py
```

Untrack:
```bash
git rm --cached -r \
  streaming/StreamingTom/LLaVA-NeXT/docs/ \
  streaming/StreamingTom/LLaVA-NeXT/llava-critic-r1/ \
  streaming/StreamingTom/LLaVA-NeXT/playground/ \
  streaming/StreamingTom/LLaVA-NeXT/scripts/ \
  streaming/StreamingTom/LLaVA-NeXT/trl/ \
  streaming/StreamingTom/LLaVA-NeXT/cog.yaml \
  streaming/StreamingTom/LLaVA-NeXT/predict.py 2>/dev/null || true
```

**Keep tracked:** `streaming/StreamingTom/LLaVA-NeXT/llava/` (patched `__init__.py`), `pyproject.toml`, `requirements.txt`, `LICENSE`.

---

### Step 6 — Remove scratch worklog

```bash
git rm --cached MMDUO_STREAMINGTOM_WORKLOG.md
mv MMDUO_STREAMINGTOM_WORKLOG.md untracked/   # untracked/ is already gitignored
```

---

### Step 7 — Delete operational one-off scripts

| Script | Reason to delete |
|--------|-----------------|
| `scripts/check_mi300x_access.sh` | Cluster-specific audit, dead end |
| `scripts/backup_streaming_outputs_to_gdrive.sh` | Personal gdrive path |
| `scripts/sync_streaming_outputs.sh` | Personal rsync |
| `scripts/run_streaming_rocm_audit_local.sh` | ROCm/MI300X dead end |
| `scripts/run_streaming_subsample5_local.sh` | One-off subsample run |
| `scripts/run_streaming_subsample_matrix_local.sh` | One-off matrix sweep |
| `scripts/run_streaming_subset3_slurm.sh` | One-off subset run |

```bash
git rm \
  scripts/check_mi300x_access.sh \
  scripts/backup_streaming_outputs_to_gdrive.sh \
  scripts/sync_streaming_outputs.sh \
  scripts/run_streaming_rocm_audit_local.sh \
  scripts/run_streaming_subsample5_local.sh \
  scripts/run_streaming_subsample_matrix_local.sh \
  scripts/run_streaming_subset3_slurm.sh
```

**Keep:** all other `scripts/` entries (SLURM array, judge, smoke, env setup, 7B eval, validate, resume helper, profiling, full eval local).

---

### Step 8 — Commit the cleanup

```bash
git add .gitignore
git status --short  # review all staged changes before committing
git commit -m "chore: untrack eval outputs, checkpoints, and scratch files; fix RunPod output path"
```

This creates one cleanup commit on top of the existing 7. Squash in Step 9.

---

### Step 9 — Squash history with interactive rebase

Current 7 commits collapse into 3 logical commits:

| Commit | Content |
|--------|---------|
| `feat: streaming video-QA eval framework (6 methods: ReKV + DuoAttention + StreamingTom)` | `streaming/ReKV/`, `streaming/StreamingTom/streamingtom/`, `streaming/merge_all_results.py`, `duo_attn/patch/`, `streaming/rekv_st_duo.md`, `streaming/run_smoke_all_methods.sh` |
| `feat: evaluation scripts and environment setup` | `scripts/` (pruned), `streaming/StreamingTom/scripts/`, `setup.sh`, `environment.yml` |
| `feat: add synthetic two-needle retrieval example` | `images/synthetic_two_needle_example.png` + associated code |

```bash
git rebase -i main
# Mark commits 2-8 as 'squash' or 'fixup' under their logical parent
```

**Important:** Rebase rewrites history. Confirm with the team before `git push --force-with-lease`.

---

### Step 10 — Final verification checklist

```bash
# No large blobs in index
git ls-files | xargs -I{} git cat-file -s :{} 2>/dev/null | sort -rn | head -10
# All should be <100KB

# No eval output files tracked
git ls-files outputs/evaluations_streaming/ | wc -l   # expect 0
git ls-files outputs/evaluations_streamingtom/ | wc -l  # expect 0

# Correct train artifacts only
git ls-files outputs/train/
# Expected: config.json and full_attention_heads{,_latest}.tsv per run

# No .pt files tracked
git ls-files | grep '\.pt$'   # expect empty

# Worklog not tracked
git ls-files MMDUO_STREAMINGTOM_WORKLOG.md   # expect empty

# Data still on disk
ls outputs/evaluations_streaming/rvs-ego/full_eval/merged_all/merged/

# Smoke test — all 6 methods, 1 video each
bash streaming/run_smoke_all_methods.sh
# Results land in outputs/evaluations_streaming/smoke_all_methods/ (all 6 methods)

# Env validation
conda activate envs/duo
python -m streaming.ReKV.validate_runtime_env
# Must show: "streaming_attn_backend_actual": "blocksparse"

conda activate envs/duo-st
python -c "import torch,flash_attn,flashinfer,llava,duo_attn,streamingtom; print('all OK')"
```

---

## Canonical Output Structure After Cleanup

All 6 methods — regardless of how they are launched (SLURM array, RunPod, local) — write to:

```
outputs/evaluations_streaming/
  <dataset>/                          # rvs-ego  or  rvs-movie  (always hyphen)
    smoke_all_methods/
      full_streaming/chunk_000.json   # methods 1-4 from run_smoke_all_methods.sh
      duo_streaming/chunk_000.json
      rekv/chunk_000.json
      duo_plus_rekv/chunk_000.json
      streamingtom/chunk_000.json     # methods 5-6 from same script
      duo_plus_streamingtom/chunk_000.json
    full_eval/
      run1/<method>/chunk_*.json      # ReKV methods 1-4 (existing complete results)
      run2/<method>/chunk_*.json      # StreamingTom methods 5-6
      merged_all/
        merged/<method>.json          # 6-method merged output
        comparison/summary.md
        plots/*.png
        plots_judge/*.png
```

---

## What Stays in Git After Cleanup

```
streaming/
  ReKV/                              ← eval framework methods 1-4 (unchanged)
  StreamingTom/
    streamingtom/                    ← eval framework methods 5-6 (unchanged)
    LLaVA-NeXT/llava/               ← patched llava package only
    scripts/eval/                    ← ST eval runner + SLURM + RunPod scripts
    scripts/setup_*.sh               ← env setup
  merge_all_results.py               ← unified 6-method merge
  run_smoke_all_methods.sh           ← unified smoke test (all 6 methods → one dir)
  rekv_st_duo.md                     ← full guide
duo_attn/patch/                      ← DuoAttention patches
scripts/                             ← reusable SLURM/eval scripts (pruned)
outputs/train/*/
  config.json                        ← training config
  full_attention_heads_latest.tsv    ← final Duo head weights
setup.sh                             ← env build script
environment.yml                      ← conda spec
streaming/ReKV/prompt.txt            ← system prompt reference
images/                              ← synthetic example image
```

## What Lives on Disk Only

```
outputs/evaluations_streaming/       ← all eval results (ReKV + ST, consolidated)
outputs/train/**/*.pt                ← optimizer/scheduler checkpoints
outputs/train/**/full_attention_heads_step=*.tsv
envs/                                ← conda environments
logs/                                ← SLURM job logs
untracked/MMDUO_STREAMINGTOM_WORKLOG.md
```

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Files changed vs `main` | ~700 | ~80 |
| Tracked eval output files | 434 | 0 |
| Tracked `.pt` binary blobs | 50+ | 0 |
| Tracked intermediate TSVs | ~40 | 0 |
| Vendored LLaVA files | 230 | ~20 (patched `llava/` only) |
| Commit count | 7 (noisy) | 3 (logical) |
| Divergent output paths | 2 | 1 (`outputs/evaluations_streaming/`) |
