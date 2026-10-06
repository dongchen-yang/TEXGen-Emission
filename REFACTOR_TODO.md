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
- [ ] 8. Fold our modules into upstream's originals
- [ ] 9. Coverage run (no code change)
- [ ] 10. `inference_specific_samples.py`: output-only code and hardcoded paths
- [ ] 11. Render-only data in our own code
- [ ] 12. The memory tracker
- [ ] 13. `launch.py`
- [ ] 14. Environment files
- [ ] 15. Release layer: generic entry points for LightgenBench users
- [ ] 16. Docs
- [ ] 17. Merge into `main` and rename to `TEXGen-Emission/` (needs Dongchen's go)
- [ ] 18. Cut-over (optional, per D8)

## How we work

- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change. Cleanups change no behavior; fixes go under "Later", one step each, only if wanted.
- Steps run automatically, in order: when a step's checks pass, commit it, fill in its **Result**,
  report it in one line, start the next. Dongchen can add **Instructions** to any step at any time.
- Stop and ask only for: (1) an open decision the step needs; (2) a check that fails, unless the
  cause is an obvious slip in the step; (3) something the plan does not cover (a needed fix, a
  behavior change, a file outside the step's list); (4) an action outside the worktree and our
  compute: a push, a parent-repo commit, the merge into `main`, a write to jupiter's shared tree.
  Jobs on our machines for the checks need no go.
- After each step that touches code the gate must pass; steps that touch training code also need the
  training check. A check passes when outputs are identical, or every difference is within the noise
  of two runs of tag `pre-refactor` on the same GPU type; `missing` must be 0.
- Run checks in parallel across our machines (workstation, lab 4090s, Solar, Alliance; the
  compute-access skill): split the full gate by shape list, run a training check's before, after and
  noise runs at once, prepare the next step while checks run (commit it only after the step before
  passes). Skip a machine that needs a 2FA login.
- Compare only within one GPU type and environment: outputs differ between GPU types. Each GPU type
  gets its own baseline and noise run from `pre-refactor` (step 7).
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
- Compare all four files. Pass: `same=N different=0 missing=0`, or every difference within noise
  (the numeric comparison from step 7). The release checkpoint gets its own gate (step 7).
- The 4090 baseline `$L/pred_gate_det1_seed00` exists only on the workstation.

## Keep stable (other code depends on these)

1. `inference_samples(checkpoint_path, sample_ids, output_dir, data_root=None, parquet_file=None,
   seed=0)` in `inference_specific_samples.py`, imported after a `chdir` and called by keyword from
   `../evaluation/newdata_eval/run_texgen_infer.py:40-46` and
   `scripts/internal/fir/infer_lightgenbench_fir.sh:71-74`. Legacy caller:
   `../evaluation/pointsampled/texgen_infer.py` (keep or retire in step 16).
2. Class paths in each run's `configs/parsed.yaml` (read as `ckpt/../../configs/parsed.yaml`):
   `spuv.systems.lightgen_system.LightGenSystem`, `spuv.data.lightgen_uv.LightGenDataModule`,
   `spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet`,
   `spuv.models.tokenizers.clip.ClipTokenizer`. Pickled in the paper checkpoint:
   `spuv.systems.lightgen_system.LightGenSystem`, `spuv.systems.texgen_base.LossConfig`.
3. Every key in the published `parsed.yaml` files stays accepted (`spuv/utils/config.py:151` merges in
   struct mode): keys may stop being read, never be deleted from the dataclasses.
4. Attribute names `backbone`, `backbone_ema`, `image_tokenizer` (inference loads with `strict=False`,
   so a rename drops weights silently) and the checkpoint keys `_scheduler_states`, `wandb_run_id`
   (`spuv/systems/base.py:137,160`).
5. Output layout `<out>/<sha>/pred_emission.png` (+ `gt_emission`, `mask`, `input_albedo`): 256²,
   linear, [0, 1], masked, the input atlas's UV layout.
6. The `TEXGEN_ENABLE_FLASH` switch in `spuv/models/sparse_networks/texgen_network.py:49` (star2 sbatch
   files grep for it).
7. The loader contract (`spuv/data/lightgen_uv.py`): `<data_root>/<ditem_dir>/somage.npz`,
   `<data_root>/thumbnails/<sha>.png` (the CLIP input), parquet with `ditem_dir` and `success`, split
   indices = positions over `success==True` rows, alpha from the npz or an `alpha.npy` sidecar,
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

**Result:** Tooling only, in `scripts/internal/gate/`; data and run outputs in
`$L/refactor_checks/` (`C` below), outside the repo. Every run used tag `pre-refactor`.

- Built: `compare_numeric.py` (per shape and file: byte identity, mean and max pixel difference; the
  evaluator's IoU, MAE and PSNR from `evaluation/pointsampled/metrics.py`, on the UV texels against
  the baseline's GT, not point-sampled; verdict IDENTICAL, WITHIN_NOISE, DIFFERENT or OUTSIDE_NOISE).
  `gate-det.sh` takes an optional data root and parquet (5th, 6th) and host overrides (`GATE_LOCK`,
  `GATE_ACTIVATE`, `GATE_REPO`, `GATE_L`); its defaults are unchanged. `make_check_root.py` builds
  the release check root `C/release_root`: 16 train, 2 val, 20 test release shapes, fixed in
  `check_root_splits.json`, alpha inside the npz. The training check: `train-check.sh fresh|resume`,
  `traincheck/traincheck.yaml` (extends the release config: the check root, batch 1, 2 epochs, one
  save), `traincheck/sitecustomize.py` (deterministic mode, `pl.seed_everything(42)` right before
  `trainer.fit` and a recording callback, all through PYTHONPATH), `traincheck/summarize_ckpt.py`,
  `compare_traincheck.py`. `star2/checks.sbatch` runs all of it on one Solar L40S node.
- The training check runs the real `launch.py --train`. fresh: from scratch, 32 steps, validation
  after each epoch, `trainer.test`, one save. resume: from the release checkpoint, 17 steps
  (Lightning first ends the checkpoint's own epoch with one batch), validation, test, one save. It
  resumes the release checkpoint, not one the run writes: only a published checkpoint shows that the
  code under test still loads its pickled classes, `_scheduler_states`, optimizer and EMA state.
  Recorded: per step the batch ids, lr, loss, gradient norm and weight sum; per pass the metrics and
  the EMA sum; for the saved file its keys, scheduler state, pickled classes and state sum. Batch 2
  runs out of memory on a 4090. Not the CPU `smoke.py`: its toy backbone skips the real model and it
  bypasses `launch.py` (steps 8b, 13).
- Release checkpoint: main checkout `outputs/texgen_alpha_lightgenbench_v1/{ckpts,configs}`, sha256
  `6c42a311…92aa` on both ends.
- Noise of two runs: zero everywhere (every file byte-identical, every recorded number equal).

  | check | 4090, workstation | 4090, sp4090 | L40S, cs-venus-16 (4 GPUs) |
  |---|---|---|---|
  | paper gate, 204 | identical (`det1`, `det2`) | identical to the workstation | identical, each shape on another GPU |
  | release gate, 20 | identical | identical to the workstation | identical |
  | training check fresh | identical | out of memory (a desktop session holds 0.7 GB) | identical |
  | training check resume | identical | identical to the workstation | identical |

  Across GPU types outputs differ, as expected: L40S against 4090, 180 of 204 paper predictions
  differ by at most 4/255 (mean IoU change 2e-5), and training differs from its first step.
- Baselines. 4090 (workstation and sp4090): paper `$L/pred_gate_det1_seed00`, release
  `C/gate/rtx4090-ws/release_test20_run1`, training `C/train/rtx4090-ws/{fresh,resume}_run1/record.jsonl`.
  L40S: `C/gate/l40s-venus16/{paper_test204,release_test20}_run1`, `C/train/l40s-venus16/train_{fresh,resume}_run1.jsonl`.
- Run (absolute paths; `T` the tree under test, e.g. a `git archive` of the commit; python from the
  `texgen` env):

  ```bash
  G=scripts/internal/gate
  bash $G/gate-det.sh T <out> <paper ckpt> [$G/subset20.txt]                       # paper gate
  bash $G/gate-det.sh T <out> <release ckpt> C/release_root/release_test20.txt \
      C/release_root C/release_root/df_lightgenbench.parquet                        # release gate
  python $G/compare_numeric.py --a <baseline> --b <out> --shas_file <shas>         # all four files
  bash $G/train-check.sh T <out> fresh       # then resume
  python $G/compare_traincheck.py --a <baseline record> --b <out>/record.jsonl
  ```

  sp4090 (gates and resume only): the inputs sit at the same paths under `C`; set
  `GATE_LOCK=/local-scratch/localhome/dya78/lightgen_4090.lock GATE_ACTIVATE=C/tooling/activate_sp4090.sh
  GATE_REPO=C/tooling/repo`. L40S: the header of `star2/checks.sbatch` (the tree goes to
  `star2:refactor_checks/trees/`). Times: on a 4090 the paper gate 23 min, the release gate 2.5 min,
  fresh 2 min, resume 1.5 min; on the L40S node the paper gate 16.5 min (4 GPUs), the release gate
  7 min, fresh and resume together 5 min.
- Found: (1) the shared Solar `texgen` env runs on sm_89 only (torchsparse fails on an A40;
  `checks.sbatch` refuses other cards); (2) Lightning saves a checkpoint through a temp file in
  `$TMPDIR` and the workstation's `/` has 7 GB free, so `train-check.sh` points `TMPDIR` at a short
  directory on `/local-scratch` (a long one breaks the dataloader's sockets); (3) the paper
  checkpoint's `configs/parsed.yaml` on jupiter differs from the workstation copy in `resume:` only,
  which inference does not read.

### 8. Fold our modules into upstream's originals

| ours | upstream original |
|---|---|
| `spuv/data/lightgen_uv.py` `LightGenDataModule` | `spuv/data/mesh_uv.py` `ObjaverseDataModule` |
| `spuv/models/sparse_networks/lightgen_pointuvnet.py` `LightGenPointUVNet` (overrides `forward`, skips the image-feature bake) | `spuv/models/sparse_networks/texgen_network.py` `PointUVNet` |
| `spuv/systems/lightgen_system.py` `LightGenSystem` (subclass of `texgen_test.TEXGenDiffusion`) | `spuv/systems/texgen_test.py` `TEXGenDiffusion` |

Each fold moves our code into the original file, editing upstream's classes and methods in place;
our file becomes a one-line alias for the published class path. One commit per fold, full gate after
each, in this order. Needs step 7 first.

- [ ] 8a. Data: `lightgen_uv.py` → `mesh_uv.py`. Full gate on all four files plus the training check.
- [ ] 8b. Backbone: `LightGenPointUVNet.forward` → `PointUVNet.forward`. Full gate.
- [ ] 8c. System: `LightGenSystem` → `texgen_test.TEXGenDiffusion` (and `texgen_base.py` where ours
  overrides a method only it defines). Our methods replace upstream's in place (for example
  `prepare_diffusion_data`, `forward`, `prepare_condition_info`, `texgen_test.py:35-182`); every loss
  branch moves unchanged. Unchanged: upstream's render branch of `on_check_train` (`:219-242`), our
  disabled 3D-render block in `test_step` (from `:444`), the `texgen_base.py` methods that
  `texgen_test.py` overrides. Inference runs through `test_pipeline` (`texgen_test.py:496-544`) and
  `ema_scope` (`texgen_base.py:232`); `test_step` (`:269-494`) runs in validation and `trainer.test`.
  `LossConfig` stays in `texgen_base.py`. Full gate, training check and a resume from a real checkpoint.
- [ ] 8d. Delete `spuv/models/simple_uv_unet.py` (ours, imported by nothing).

During each fold, check what the alias changes observably (the class `__name__` in logs, the path new
checkpoints pickle, Lightning hparams) and name in the commit message which upstream behavior the
edit replaces (for example `ObjaverseDataModule`'s mesh loading, `PointUVNet`'s bake). Branch
`restructure-2026-09-16` did the same folds (its tasks T5–T7) as a reference; take neither its
`texgen_emission_*` renames nor its behavior changes.

**Instructions:**

**Result:**

### 9. Coverage run (no code change)

A training config (the paper checkpoint's `configs/parsed.yaml` or the release config) under a
line-coverage tracer, `TEXGEN_ENABLE_FLASH=0`, 1 GPU,
`data.num_workers=0`, batch limits so that training, validation, a checkpoint save, a resume and the
test pass all run; plus inference on a few gate shapes and a few release shapes. Output: per file,
the lines that never run. "Never ran" is not "dead" for the flash branch or the in-npz alpha branch
(used only by the release).

**Instructions:**

**Result:**

### 10. `inference_specific_samples.py`: output-only code and hardcoded paths

Keep the `inference_samples(...)` signature and `sample_seed`.

- [ ] `:84` falls back to the missing `configs/lightgen_pointuv_256_batch32_pretrained.yaml`.
- [ ] `:96-97` default data root and parquet under `/localhome/dya78/code/lightgen/data/baked_uv`
  (bypassed when the driver passes both).
- [ ] `:245` thumbnail dir under `/localhome/.../baked_uv_local/thumbnails`: output-only, and the dir
  does not exist (it falls back to the batch thumbnail).
- [ ] `create_overall_visualization` (`:318-321`, `:388-484`; a 24 MB `overall_visualization.png` per
  run), the per-shape `comparison.png` and `thumbnail.png` (no reader found).
- [ ] `:487` checkpoint in the `__main__` block.

**Instructions:**

**Result:**

### 11. Render-only data in our own code

- [ ] The unused `camera_functions` import (`spuv/data/lightgen_uv.py:28`), `get_camera_info` (`:439`,
  called at `:405`, dummy matrices) and the `mesh` dict (`:380-387`): batch keys that
  `lightgen_system.py:115-116` reads and the backbone ignores; remove them together. After step 8
  they sit in `mesh_uv.py` and `texgen_test.py` (line numbers are from before the folds).

Depends on step 9. Full gate plus the training check.

**Instructions:**

**Result:**

### 12. The memory tracker

- [ ] Memory logging: `lightgen_system.py:62-64` (`init_tracker(enabled=True, log_interval=1)`),
  `:284-303`; `spuv/utils/memory_tracker.py`.
- [ ] Keep the `gc.collect()` / `torch.cuda.empty_cache()` every 5 steps (`texgen_base.py:221-229`)
  unless measured on an H100 (the release run already had to drop to batch 16).

**Instructions:**

**Result:**

### 13. `launch.py`

332 of its 571 lines are ours: the `torch.load` patch and env vars (`:9-28`), safe globals
(`:89-104`), auto-resume and the wandb run-id restore (`:145-199`), the checkpoint dirpath
(`:217-225`), the wandb config and stale-cache cleanup (`:263-341`), `define_metric` and the
`log_metrics` patch (`:343-404`), signal and `atexit` handling (`:406-445`, `:474-509`). Decide what
stays; any change here changes how training runs behave.

**Instructions:**

**Result:**

### 14. Environment files

- [ ] `requirements.txt` (upstream's list plus three pins, no torch line) and `requirements_125.txt`
  (torch cu126 next to cu118 xformers and spconv): one file that matches what runs.
- [ ] Not imported by live code: xformers, open3d, pointops, torchdyn, torch-geometric.
- [ ] The envs that run: the workstation `texgen` (copied to star2 `~/envs/texgen` and the lab 4090s,
  packed as `texgen_env.tar.gz` in jupiter `solar_texgen_stage/`); `texgen-bw` on cs-venus-05/19,
  where the paper run trained (recipe on branch `texgen-74k-v2-venus05`,
  `scripts/star2/bootstrap_texgen_bw.sh`); the fir venv plus pin overlay, where the release run
  trained (`../docs/status/checkpoints.md:1637-1660`).
- [ ] The invariant: the four conditioning pins (transformers 4.28.1, diffusers 0.28.0,
  huggingface_hub 0.25.2, tokenizers 0.13.3).
- [ ] If a dependency goes, update the import probes: `scripts/internal/fir/train_lightgenbench_alpha_fir.sh:220`,
  `../evaluation/newdata_eval/sweeps/star2/texgen_worker.sbatch`.

**Instructions:**

**Result:**

### 15. Release layer: generic entry points for LightgenBench users

The code will be published for LightgenBench users to train and run. No user or cluster paths in
this layer. Needs step 10; its README commands are written in step 16.

- [ ] `scripts/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>`: the generic half of
  `scripts/internal/fir/stage_hf.sh` (untar the Hugging Face release into `<data_root>/<uuid>/somage.npz`
  and `thumbnails/<uuid>.png`), then `build_index.py`.
- [ ] The release config carries the recipe values (batch 16 × accumulate 2) and says that
  `TEXGEN_ENABLE_FLASH=0` and the four conditioning pins are required.
- [ ] The internal fir launchers stay as they are.

**Instructions:**

**Result:**

### 16. Docs

One doc of record, keeping upstream's attribution and citation (`README.md:103`).

- [ ] Inside: `README.md` (upstream's; its command uses the missing `configs/texgen_test.yaml`;
  `static/teaser.jpg` goes with it), `CLAUDE.md` (no alpha channel, paper or release run; the
  `requirements_125.txt` claim at `:15` and the DDP sampler claim at `:186-190` are wrong),
  `LIGHTGEN_WORKFLOW.md`, the "4 Pro 6000" comment in the release config (`:89`).
- [ ] The release commands in the README: single-node training (`python launch.py --config ...
  --gpu 0,1,2,3 --train data.data_root=...`) and inference (`inference_specific_samples.py`) on the
  release test split.
- [ ] Keep or retire `../evaluation/pointsampled/texgen_infer.py`.

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
  in its commands and Losses), `../AGENTS.md:64,212,228`, `../docs/status/refactor.md` (the TEXGen
  entry); the stale "main has no flash gate" comments in
  `../evaluation/newdata_eval/run_texgen_infer.py:15-18`, `infer_texgen_v2_venus05.sbatch:6`,
  `sweeps/stage_texgen_host.sh:46`.
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
