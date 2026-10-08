# TEXGen refactor — plan

Turn the fork into a clean TEXGen-Emission: our changes as direct edits on upstream's modules, only
the code behind the paper and release runs, generic entry points for LightgenBench users.

## Progress

- [x] 1. Housekeeping
- [x] 2. Dead root scripts and the Blender call
- [x] 3. The existing `deprecated/` tree and stale ignore rules
- [x] 4. Upstream demo material
- [x] 5. Launchers
- [x] 5b. Move the kept cluster scripts into `scripts/internal/`
- [x] 6. Configs
- [x] 7. Check setup: the training check and the noise baselines (no code change)
- [x] 7b. Release checks on Hugging Face data, and a fan-out driver
- [x] 8. Fold our modules into upstream's originals
- [x] 9. Coverage run (no code change)
- [x] 10. `inference_specific_samples.py`: output-only code and hardcoded paths
- [x] 11. Render-only data in our own code (done in 8a)
- [x] 12. The memory tracker
- [x] 13. `launch.py`
- [x] 14. Environment files
- [x] 15. Release layer: generic entry points for LightgenBench users
- [ ] 16. Docs
- [ ] 17. Merge into `main` and rename to `TEXGen-Emission/` (needs Dongchen's go)
- [ ] 18. Cut-over (optional, per D8)

## How we work

- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change. Cleanups change no behavior; fixes go under "Later", one step each, only if wanted.
- Steps run one at a time, together with Dongchen: he gives the go and any **Instructions** for each
  step; when its checks pass, commit it, fill in its **Result**, report, and wait for the next go.
- **Result** is at most three lines (what changed, commits, verdict); details live in CLAUDE.md,
  commit messages and the step's report.
- Inside a step, stop and ask for: (1) an open decision the step needs; (2) a check that fails, unless the
  cause is an obvious slip in the step; (3) something the plan does not cover (a needed fix, a
  behavior change, a file outside the step's list); (4) an action outside the worktree and our
  compute: a push, a parent-repo commit, the merge into `main`, a write to jupiter's shared tree.
  Jobs on our machines for the checks need no go.
- After each step that touches code the gate must pass; steps that touch training code also need the
  training check; `missing` must be 0. Gate images (all four files): a shape that differs by at most
  1/255 per pixel, on at most 0.5% of each image's channel values, is noise and passes; any other
  shape must lie within the envelope, the largest difference per measure (pixels, IoU/MAE/PSNR)
  between any two of at least 4 runs of tag `pre-refactor` on the same GPU type (Dongchen's rule of
  2026-10-06). IoU/MAE/PSNR deltas are reported but do not fail a shape within the tolerance.
  Training check: every recorded value within its envelope, the structure (step order, batch ids,
  keys, pickled classes) exact. A result outside is rerun once on another GPU of that type, never the
  one it ran on (on Solar another node), before it counts as failed; with no such GPU free it stays
  unresolved. A paper piece outside also runs as `pre-refactor` on the same shapes in the same order
  (output depends on the shapes before it in the process) and passes if within the tolerance of
  that control (Dongchen 2026-10-08). Every run is reported (`fanout.py` does this; step 7b).
- Run checks in parallel across our machines (workstation, lab 4090s, Solar, Alliance; the
  compute-access skill): split the full gate by shape list, run a training check's before, after and
  noise runs at once, prepare the next step while checks run (commit it only after the step before
  passes). Skip a machine that needs a 2FA login.
- Never wait for a GPU: right before dispatch, probe every candidate (free GPU memory and free host
  RAM) and send work only to the ones free now; a busy GPU or a held lock is skipped, not waited on;
  a Solar job is submitted only when free L40S show up and must start at once or fail. If nothing is
  free, report it and stop.
- Compare only within one GPU type and environment: outputs differ between GPU types. Each GPU type
  gets its own baseline and envelope runs from `pre-refactor` (steps 7, 7b; `scripts/internal/gate/baselines.json`).
- Our changes are edits to upstream's original modules; a file of ours that only keeps a published
  class path alive is a one-line alias. Upstream code we do not use stays untouched, even inside files
  we edit. `git diff upstream/main -- <file>` shows what we changed.
- A doc that names a removed file is fixed in the same step.
- Each step updates TEXGen's own docs it affects (`CLAUDE.md`, `README.md`, `LIGHTGEN_WORKFLOW.md`)
  in the same commit. Parent docs (`AGENTS.md`, `.claude/rules/texgen.md`, `docs/baselines/texgen.md`)
  describe `main` and change only at the merge: a step that makes one stale adds an item to step 17's
  parent-docs list.
- All edits happen in the worktree `.claude/worktrees/refactor` (branch `refactor`, from tag
  `pre-refactor`); `git diff pre-refactor` shows every edit. The main checkout stays on `main`.
- In the main checkout: never `git add -A` (the `texgen-v2` worktree is untracked there) and never
  `git clean -x`/`-X` (its ignored `outputs/` holds the paper checkpoint, 12 GB).
- Parent pointer bumps are path-limited (`git commit -- TEXGen`): the parent carries other sessions'
  uncommitted work. Nothing is pushed without Dongchen's go.

## Runs and the gate

| run | config | checkpoint |
|---|---|---|
| paper | the checkpoint's own `configs/parsed.yaml` | main checkout `outputs/texgen_alpha_74k_v2_agentic/ckpts/epoch=124-step=35500.ckpt` |
| release retrain | `configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml` via `scripts/internal/fir/train_lightgenbench_alpha_fir.sh` (batch 16 × accumulate 2) | jupiter `outputs/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt` |

The gate: inference with the paper checkpoint, seed 0, deterministic mode, 4090 lock. Paths must be
absolute (the driver `chdir`s); `<texgen_root>` is the worktree.

```bash
L=/local-scratch/localhome/dya78/lightgen_eval_local
bash scripts/internal/gate/gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file>]   # prints GATE_DONE
python ../evaluation/newdata_eval/compare_pred_dirs.py --a $L/pred_gate_det1_seed00 --b <out_dir> \
    --shas_file <same shas_file> [--filename gt_emission.png|mask.png|input_albedo.png]
```

- Routine steps: `scripts/internal/gate/subset20.txt` (20 shapes, 164 s). Full: all 204
  (`$L/test_agentic_clean.txt`, about 24 min), for steps that touch the loader or the model and
  before the merge.
- Compare all four files. Pass: `missing=0`, and every shape within 1/255 per pixel on at most 0.5% of
  the values in all four files, or within the envelope (`compare_numeric.py --envelope`, How we work),
  after at most one rerun on another GPU of the type. The release checkpoint gets its own gate (step 7).
- The 4090 baseline `$L/pred_gate_det1_seed00` exists only on the workstation.

## Keep stable (other code depends on these)

1. `inference_samples(checkpoint_path, sample_ids, output_dir, data_root=None, parquet_file=None,
   seed=0)` in `inference_specific_samples.py`, imported after a `chdir` and called by keyword from
   `../evaluation/newdata_eval/run_texgen_infer.py:40-46` and
   `scripts/internal/fir/infer_lightgenbench_fir.sh:81-84`. Legacy caller:
   `../evaluation/pointsampled/texgen_infer.py` (keep or retire in step 16).
2. Class paths in each run's `configs/parsed.yaml` (read as `ckpt/../../configs/parsed.yaml`):
   `spuv.models.tokenizers.clip.ClipTokenizer`; a path we retire is mapped by
   `scripts/convert_config.py` instead (8a: the loader's, 8b: the backbone's, 8c: the system's).
   Pickled in the paper and release checkpoints: `spuv.systems.texgen_base.LossConfig` (stays) and
   `spuv.systems.lightgen_system.LightGenSystem`, which `convert_config.py --ckpt` rewrites (8c).
3. Every key in the published `parsed.yaml` files stays accepted (`spuv/utils/config.py:151` merges in
   struct mode): keys may stop being read; a key deleted from the dataclasses is dropped by
   `scripts/convert_config.py` (8a: `data.use_alpha`).
4. Attribute names `backbone`, `backbone_ema`, `image_tokenizer` (inference loads with `strict=False`,
   so a rename drops weights silently) and the checkpoint keys `_scheduler_states`, `wandb_run_id`
   (`spuv/systems/base.py:137,160`).
5. Output layout `<out>/<sha>/pred_emission.png` (+ `gt_emission`, `mask`, `input_albedo`): 256²,
   linear, [0, 1], masked, the input atlas's UV layout.
6. The `TEXGEN_ENABLE_FLASH` switch in `spuv/models/sparse_networks/texgen_network.py:49` (star2 sbatch
   files grep for it).
7. The loader contract (`spuv/data/mesh_uv.py` since 8a):
   `<data_root>/<ditem_dir>/somage.npz`,
   `<data_root>/thumbnails/<sha>.png` (the CLIP input), parquet with `ditem_dir` and `success`, split
   indices = positions over `success==True` rows, alpha from the npz,
   positions decoded from [−2, 2].
8. `sample_seed` (`inference_specific_samples.py:26-32`): signature and return line identical to
   `../evaluation/newdata_eval/seedutil.py`.
9. Branch `texgen-74k-v2-venus05` stays frozen at `370818f`: the release launcher
   (`scripts/internal/fir/train_lightgenbench_alpha_fir.sh:61,157`) and
   `../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:78-91` check it out, and the
   `texgen-v2` worktree sits on it (`../evaluation/newdata_eval/sweeps/job_texgen.sh:16`,
   `stage_texgen_host.sh:22`, `tests/test_run_texgen_infer.py:16`). Inside a fir clone of that branch
   the fir scripts still live at the old paths (`scripts/fir/`, `scripts/lightgenbench/stage_hf.sh`).
10. The directory `TEXGen/` until step 17 renames it; the `outputs/<run>/{ckpts,configs}` layout.

## Decisions

- D1 Removed files are deleted, not moved to `deprecated/`; git history and the tags keep them.
- D2 Checks cover both the paper and the release checkpoints.
- D3 Cleanup steps change no behavior.
- D4 The directory becomes `TEXGen-Emission/` (step 17). Class paths and the `inference_samples`
  signature stay.
- D5 Steps that touch training code get the training check: a short deterministic run, loss per
  step compared before and after, plus one resume from a real checkpoint. Reseed
  (`pl.seed_everything`) right before `trainer.fit`.
- D6 Every loss branch and config-switched option stays (mask classification, GT-mask conditioning,
  dark region, `use_precomputed_clip`, ...); only their configs were deleted.
- D7 Work happens on branch `refactor`; the parent pointer is bumped once, at the merge (step 17).
- D9 Upstream code we do not use stays untouched, files and blocks alike; cleanups touch only our code.
- [ ] D8 Cut-over (before step 18): after the merge, do the launchers, the `texgen-v2` worktree and
  the staged `TEXGen.tar` move to `main`, or stay on the frozen branch?
- [ ] D10 Internal scripts in the published code (before publishing): `scripts/internal/` holds our
  Slurm accounts and paths. Publish as is, publish a snapshot without it, or move it to the parent
  repo? Recommended: a published snapshot without it.

## Done

1. `texgen-74k-v2-venus05` merged into `main` (`2b578af`, pushed); `.claude/worktrees/` ignored
   (`edcae73`); the gate moved into the repo (`7408f06`).
2. The Blender call removed (`b0ea68a`, gate identical); gate subset option (`8460b3b`); nine dead root
   scripts deleted (`8fdba12`; parent `2895ff4`).
3. `deprecated/` (54 files) and its ignore rules deleted (`7e0dfab`; parent `2466f82`).
4. Upstream's `tools/`, `assets/` and two unused images deleted (`26eb97b`).
5. 18 launchers not used for the release results deleted (`b0f7904`; parent `af2fd7e`).
   5b. The kept cluster scripts and the gate moved under `scripts/internal/` (`71a2e3d`; parent
   `3e7f2bd`).
6. Every config except the release config deleted (`b3dc916`).

## Steps

### 7. Check setup: the training check and the noise baselines (no code change)

All runs on tag `pre-refactor`, in parallel across machines.

- [x] A small local training root (N shapes: parquet, split JSON, thumbnails, alpha) and a local-paths
  config override (the release config points at `${SLURM_TMPDIR}`). Or reuse
  `../.superpowers/sdd/2026-09-16-texgen-restructure/cpu_training_smoke/smoke.py`.
- [x] Training-check noise: two runs on unchanged code, per GPU type used.
- [x] A numeric gate comparison under `scripts/internal/gate/` (per shape: mean and max absolute pixel
  difference; the evaluator's IoU, MAE and PSNR).
- [x] Gate noise: two runs on unchanged code, per GPU type used (the 4090 has one baseline already).
- [x] The release checkpoint: pull it with its `configs/parsed.yaml` (rsync + sha256), pick a
  release-shape subset, make its baseline and noise runs.

**Instructions:**

**Result:** The check tooling in `scripts/internal/gate/` (numeric comparison, release gate, training
check, Solar sbatch) and the first `pre-refactor` baselines; commits `a1d0459`, `2223ff5`. Verdict:
done; its zero-noise finding, release root and release baselines were superseded in step 7b.

### 7b. Release checks on Hugging Face data, and a fan-out driver

Runs after 8a. The paper gate keeps the paper-era test bake (`$L/out_eval_gt`): its outputs tie to the
published paper numbers, and the release has different test shapes, bake and GT rule.

- [x] The release gate and the training check read files downloaded from Hugging Face
  (`3dlg-hcvc/LightgenBench`), not jupiter's `dataset_release`: fetch only the tars that hold the 38
  check shapes, verify them against the release's `checksums.sha256`, and rebuild the check root the
  release run's way (untar, then `scripts/lightgenbench/build_index.py`).
- [x] New `pre-refactor` baselines and noise runs on that root, per GPU type used; then re-check the
  8a head against them.
- [x] A fan-out driver: one command that probes every candidate GPU (How we work: never wait), splits
  the paper gate across every free 4090 (workstation, sp4090, xg4090 after a one-subset match check
  against the 4090 baseline) and every free Solar L40S, compares each shard with its own GPU type's
  baseline, and runs the release gate and both training checks at the same time on other free GPUs.
  `gate-det.sh`'s lock is tried once instead of waited on when the driver calls it. Alliance GPUs only
  with an already open login and their own baselines.

**Instructions:** 2026-10-06: "we should use the huggingface data if we are not"; "run them all
across the avaialble devices"; "dont have them waiting for the GPU all the time... just find the
avaialbe ones and running them."

**Result:** Release check root from Hugging Face, the fan-out driver, the pass rule (1/255 on ≤0.5% of
the values, envelope, rerun once on another GPU) and its baselines; commits `9054f22`, `7f0586e`,
`b72221a` and the review fix after them. Verdict: head (code = 8a) PASS (re-scored); negative control FAIL.

### 8. Fold our modules into upstream's originals

| ours | upstream original |
|---|---|
| `spuv/data/lightgen_uv.py` `LightGenDataModule` | `spuv/data/mesh_uv.py` `ObjaverseDataModule` |
| `spuv/models/sparse_networks/lightgen_pointuvnet.py` `LightGenPointUVNet` (overrides `forward`, skips the image-feature bake) | `spuv/models/sparse_networks/texgen_network.py` `PointUVNet` |
| `spuv/systems/lightgen_system.py` `LightGenSystem` (subclass of `texgen_test.TEXGenDiffusion`) | `spuv/systems/texgen_test.py` `TEXGenDiffusion` |

Each fold moves our code into the original file, editing upstream's classes and methods in place;
our file is deleted and `scripts/convert_config.py` maps its class path in old configs. One commit per fold, full gate after
each, in this order. Needs step 7 first.

- [x] 8a. Data: `lightgen_uv.py` → `mesh_uv.py`. Full gate on all four files plus the training check.
- [x] 8b. Backbone: `LightGenPointUVNet.forward` → `PointUVNet.forward`. Full gate.
- [x] 8c. System: our conditioning, `forward` and `prepare_diffusion_data` replace upstream's in
  `texgen_test.py`, in place; the training part (training and validation step, losses, `configure`'s
  additions), which upstream did not release (its `texgen_base.py` hooks are `pass` stubs), goes to a
  new `spuv/systems/texgen_train.py` (Dongchen 2026-10-08); `lightgen_system.py` is deleted and
  `convert_config.py` maps its path in configs and, with `--ckpt`, in checkpoints. Every loss
  branch moves unchanged. Unchanged: upstream's render branch of `on_check_train` (`:219-242`), our
  disabled 3D-render block in `test_step` (from `:444`), the `texgen_base.py` methods that
  `texgen_test.py` overrides. Inference runs through `test_pipeline` (`texgen_test.py:496-544`) and
  `ema_scope` (`texgen_base.py:232`); `test_step` (`:269-494`) runs in validation and `trainer.test`.
  `LossConfig` stays in `texgen_base.py`. Full gate, training check and a resume from a real checkpoint.
- [x] 8d. Delete `spuv/models/simple_uv_unet.py` (ours, imported by nothing).

During each fold, check what the alias changes observably (the class `__name__` in logs, the path new
checkpoints pickle, Lightning hparams) and name in the commit message which upstream behavior the
edit replaces (for example `ObjaverseDataModule`'s mesh loading, `PointUVNet`'s bake). Branch
`restructure-2026-09-16` did the same folds (its tasks T5–T7) as a reference; take neither its
`texgen_emission_*` renames nor its behavior changes.

**Instructions:**

**Result:**

- 8a. Our loader written into upstream's `spuv/data/mesh_uv.py`, `lightgen_uv.py` deleted, old configs
  through `scripts/convert_config.py`; commits `4b148ce` (docs), `e64db55`. Full gate PASS: 204/204
  paper (59 matched by their `pre-refactor` control), release, fresh, resume IDENTICAL.
- 8b. Our forward written into upstream's `PointUVNet.forward`, `lightgen_pointuvnet.py` deleted. Full
  gate PASS: 204/204 paper (21 matched by their control), release, fresh, resume IDENTICAL.
- 8c. Inference path edited into upstream's `texgen_test.py`, training into a new `texgen_train.py`,
  `lightgen_system.py` deleted, checkpoints converted (`convert_config.py --ckpt`), normal dropped.
  Full gate PASS: 204/204 paper (117 matched by their control), release, fresh, resume IDENTICAL.
- 8d. `spuv/models/simple_uv_unet.py` deleted; nothing imported it (the step-9 coverage run exercises
  the code at this commit).

### 9. Coverage run (no code change)

A training config (the paper checkpoint's `configs/parsed.yaml` or the release config) under a
line-coverage tracer, `TEXGEN_ENABLE_FLASH=0`, 1 GPU,
`data.num_workers=0`, batch limits so that training, validation, a checkpoint save, a resume and the
test pass all run; plus inference on a few gate shapes and a few release shapes. Output: per file,
the lines that never run. "Never ran" is not "dead" for the flash branch or the in-npz alpha branch
(used only by the release).

**Instructions:**

**Result:** `scripts/internal/gate/coverage.sh` (fresh, resume, 5 paper + 5 release shapes) at `17d0f3e`:
48% of 6,251 lines ran; report `C/coverage/step9-20261008-073600/report.txt`. Never imported: upstream's
`models/{camera,isosurface,networks,timestep}.py`, `utils/rasterize.py` (D9: kept). Ours not run: the wandb
logging (`texgen_train.py:89-152`, `launch.py:264-445`), loss branches off in the recipe (D6: kept).

### 10. `inference_specific_samples.py`: output-only code and hardcoded paths

Keep the `inference_samples(...)` signature and `sample_seed`.

- [x] `:84` falls back to the missing `configs/lightgen_pointuv_256_batch32_pretrained.yaml`.
- [x] `:96-97` default data root and parquet under `/localhome/dya78/code/lightgen/data/baked_uv`
  (bypassed when the driver passes both).
- [x] `:245` thumbnail dir under `/localhome/.../baked_uv_local/thumbnails`: output-only, and the dir
  does not exist (it falls back to the batch thumbnail).
- [x] `create_overall_visualization` (`:318-321`, `:388-484`; a 24 MB `overall_visualization.png` per
  run), the per-shape `comparison.png` and `thumbnail.png` (no reader found).
- [x] `:487` checkpoint in the `__main__` block.

**Instructions:**

**Result:** No fallback config; `data_root`/`parquet_file` required; the batch thumbnail only;
`comparison.png`, `thumbnail.png`, `overall_visualization.png` dropped (four files per shape); a generic
CLI replaces the hardcoded `__main__`. Paper subset20 and release gate IDENTICAL.

### 11. Render-only data in our own code

**Result:** Done in 8a: `get_camera_info`, the dummy UV-grid mesh and the `mesh` / `mvp_mtx_cond` batch
keys are gone from the loader, the system and the backbone; upstream's lines stay (D9).

### 12. The memory tracker

- [x] Memory logging: `lightgen_system.py:62-64` (`init_tracker(enabled=True, log_interval=1)`),
  `:284-303`; `spuv/utils/memory_tracker.py`.
- [x] Keep the `gc.collect()` / `torch.cuda.empty_cache()` every 5 steps (`texgen_base.py:221-229`)
  unless measured on an H100 (the release run already had to drop to batch 16).

**Instructions:**

**Result:** `spuv/utils/memory_tracker.py` deleted with every `log_memory` call (`base.py`,
`texgen_base.py`, `texgen_train.py`) and the overrides that only logged (`configure`, `validation_step`,
`on_train_epoch_start`); every `gc.collect()`/`empty_cache()` stays. Quick check IDENTICAL.

### 13. `launch.py`

332 of its 571 lines are ours: the `torch.load` patch and env vars (`:9-28`), safe globals
(`:89-104`), auto-resume and the wandb run-id restore (`:145-199`), the checkpoint dirpath
(`:217-225`), the wandb config and stale-cache cleanup (`:263-341`), `define_metric` and the
`log_metrics` patch (`:343-404`), signal and `atexit` handling (`:406-445`, `:474-509`). Decide what
stays; any change here changes how training runs behave.

**Instructions:**

**Result:** Kept: the `torch.load` patch, auto-resume, the wandb restore, cache cleanup, x-axis
fix and shutdown handlers (the release run used each; their comments record why). Removed: the
`add_safe_globals` block (inert: the patch forces `weights_only=False`). Fresh and resume IDENTICAL.

### 14. Environment files

- [x] `requirements.txt` (upstream's list plus three pins, no torch line) and `requirements_125.txt`
  (torch cu126 next to cu118 xformers and spconv): one file that matches what runs.
- [x] Not imported by live code: xformers, open3d, pointops, torchdyn, torch-geometric.
- [x] The envs that run: the workstation `texgen` (copied to star2 `~/envs/texgen` and the lab 4090s,
  packed as `texgen_env.tar.gz` in jupiter `solar_texgen_stage/`); `texgen-bw` on cs-venus-05/19,
  where the paper run trained (recipe on branch `texgen-74k-v2-venus05`,
  `scripts/star2/bootstrap_texgen_bw.sh`); the fir venv plus pin overlay, where the release run
  trained (`../docs/status/checkpoints.md:1637-1660`).
- [x] The invariant: the four conditioning pins (transformers 4.28.1, diffusers 0.28.0,
  huggingface_hub 0.25.2, tokenizers 0.13.3).
- [x] If a dependency goes, update the import probes: `scripts/internal/fir/train_lightgenbench_alpha_fir.sh:220`,
  `../evaluation/newdata_eval/sweeps/star2/texgen_worker.sbatch`.

**Instructions:**

**Result:** One `requirements.txt`: what the code imports at the `texgen` env's versions, the four
pins required, torch/CUDA extensions per GPU; xformers, open3d, pointops, torchdyn, torch-geometric
dropped; `requirements_125.txt` deleted; the three envs in CLAUDE.md. No probe imports a dropped one.

### 15. Release layer: generic entry points for LightgenBench users

The code will be published for LightgenBench users to train and run. No user or cluster paths in
this layer. Needs step 10; its README commands are written in step 16.

- [x] `scripts/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>`: the generic half of
  `scripts/internal/fir/stage_hf.sh` (untar the Hugging Face release into `<data_root>/<uuid>/somage.npz`
  and `thumbnails/<uuid>.png`), then `build_index.py`.
- [x] The release config carries the recipe values (batch 16 × accumulate 2) and says that
  `TEXGEN_ENABLE_FLASH=0` and the four conditioning pins are required.
- [x] The internal fir launchers stay as they are.

**Instructions:**
- Decided: the script checks the download against `checksums.sha256` first, unpacks only the tars
  it checked, and writes `<data_dir>/texgen_root/<uuid>/somage.npz`,
  `texgen_root/thumbnails/<uuid>.png` and the index beside it. The release config reads that
  directory from `LIGHTGENBENCH_DATA` alone (omegaconf 2.3.0 resolves a nested `oc.env` default
  even when the variable is set, so a `$SLURM_TMPDIR` fallback fails without SLURM); the fir
  launcher gains one line exporting it, its only change besides comments.
- Decided: the config names no cluster path: `exp_root_dir` and `checkpoint.dirpath` go (defaults
  `outputs/` and `<trial_dir>/ckpts`); the fir launcher passes both, as before.
- Decided: the training check keeps one micro-batch per step (`accumulate_grad_batches: 1` in
  `traincheck.yaml`), as its baselines ran; the release config now says 2.

**Result:** `prepare_lightgenbench.sh` run on an 826-shape slice of the real release: index check
OK, 8 npz and 8 thumbnails byte-identical to the check root. The config resolves with only
`LIGHTGENBENCH_DATA` set and raises without it; fresh and resume IDENTICAL against `pre-refactor`.

### 16. Docs

One doc of record, keeping upstream's attribution and citation (`README.md:103`).

- [ ] Inside: `README.md` (upstream's; its command uses the missing `configs/texgen_test.yaml`;
  `static/teaser.jpg` goes with it), `CLAUDE.md` (no alpha channel, paper or release run; the
  `requirements_125.txt` claim at `:15` and the DDP sampler claim at `:186-190` are wrong),
  `LIGHTGEN_WORKFLOW.md`, the "4 Pro 6000" comment in the release config (`:89`).
- [ ] The release commands in the README: single-node training (`python launch.py --config ...
  --gpu 0,1,2,3 --train data.data_root=...`) and inference (`inference_specific_samples.py`) on the
  release test split.
- [ ] Keep or retire `../evaluation/pointsampled/texgen_infer.py`. Keeping it needs more than its
  imports (`lightgen_uv`, `lightgen_system`): its config and checkpoint need
  `scripts/convert_config.py`, and its checkpoint is 12-channel (no alpha), which the current code
  (alpha always read) cannot run.

**Instructions:**

**Result:**

### 17. Merge into `main` and rename to `TEXGen-Emission/` (needs Dongchen's go)

- [ ] The full gate and the training check on the branch head, against the `pre-refactor` baselines.
- [ ] Merge `refactor` into `main`.
- [ ] In the parent, one commit: the pointer bump, the rename `TEXGen/` → `TEXGen-Emission/`
  (`.gitmodules`, the gitlink, every parent reference: search again; on 2026-10-06 about 20 files, among
  them `../evaluation/newdata_eval/run_texgen_infer.py` and its test, `sweeps/job_texgen.sh`,
  `sweeps/stage_texgen_host.sh`, `sweeps/star2/texgen_worker.sbatch`, `infer_texgen_v2_venus05.sbatch`,
  `seedutil.py`, `../data_processing/data_preparation/README.md` and its
  `tests/test_encoding_parity.py`, `../.claude/rules/texgen.md` with its `paths:` front matter,
  `../AGENTS.md`). The `texgen-v2` worktree moves with the directory.
- [ ] Parent docs: `../docs/baselines/texgen.md` (the config table `:12-30,48` names deleted configs;
  `:53-61` says the flash switch is only on the branch), `../.claude/rules/texgen.md` (deleted configs
  in its commands and Losses; `:38` names the backbone `LightGenPointUVNet`, now upstream's
  `PointUVNet`; `:34,45` name `LightGenSystem` / `lightgen_system.py`, now `texgen_train.py` and
  `texgen_test.py`), `../AGENTS.md:64,212,228`, `../docs/status/refactor.md` (the TEXGen
  entry); the stale "main has no flash gate" comments in
  `../evaluation/newdata_eval/run_texgen_infer.py:15-18`, `infer_texgen_v2_venus05.sbatch:6`,
  `sweeps/stage_texgen_host.sh:46`.
- [ ] Loader pointers (since 8a the loader is `spuv/data/mesh_uv.py` and `spuv/data/lightgen_uv.py`
  is gone): comments and mirrored constants in
  `../data_processing/data_preparation/{atlas.py:5,contract.py:8,multiview.py:335}` and
  `tests/test_encoding_parity.py:5,17,53`; docs `../data_processing/data_preparation/README.md:45,56`,
  `../docs/baselines/texgen.md:35`, `../.claude/rules/texgen.md:72`. The import in
  `../evaluation/pointsampled/texgen_infer.py:23` breaks; parent scripts that run the current code
  on a published run's `parsed.yaml` (`infer_texgen_v2_venus05.sbatch`, `texgen_worker.sbatch`,
  `sweeps/job_texgen.sh`, the smoke command `sweeps/stage_texgen_host.sh` prints) need
  `scripts/convert_config.py` first, as `gate-det.sh` does; converting once in
  `run_texgen_infer.py` covers them all.
- [ ] Decide here: rename the GitHub repository `dongchen-yang/TEXGen` too?

**Instructions:**

**Result:**

### 18. Cut-over (optional, per D8)

- [ ] Move the consumers to `main`: the `texgen-v2` worktree, the `BRANCH=` lines in
  `scripts/internal/fir/train_lightgenbench_alpha_fir.sh:61,157` (and the old-layout paths the fir
  scripts use), the branch check in `../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:78-91`;
  rebuild the staged `TEXGen.tar` on jupiter (it unpacks to `TEXGen/`) and delete the Solar nodes'
  `.tree_ok` so they unpack it again.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **Flash default.** An unset `TEXGEN_ENABLE_FLASH` means on (`texgen_network.py:49`); both
  published runs used dense attention (`=0`). Forgetting it silently changes numbers on a 4090 or an
  H100 and asserts on Blackwell. Default to dense, or fail when unset.
- [ ] **Release batch.** The release config says batch 32; the run used 16 × 2 from the launcher.
- [ ] **Fixes from `restructure-2026-09-16`** (`b4e423f`, `e20b015`, `4ef35a7`; re-apply by hand):
  EMA weights on the test pass, the wandb directory, a non-zero exit on interrupt, the resume order
  (one checkpoint read instead of two), validation metrics synced across GPUs, the training preview
  panel, the removed noise schedule; the loss guard (`309d3ab`, `bfb2135`: rejects a nonzero loss
  weight no code reads), after step 8c.
- [ ] **Tests from `restructure-2026-09-16`** (12 files, including a toy CPU fit/resume test).

## Out of scope

The position frame (atlas decoded as [−2, 2], voxels in [−0.5, 0.5]): a data-format change, tracked
in `../docs/status/refactor.md`.
