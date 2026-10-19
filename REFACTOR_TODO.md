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
- [x] 16. Docs
- [x] 16b. `base.py` back to upstream's text: our training hooks move to `texgen_train.py`
- [x] 16c. `texgen_base.on_train_batch_end` back to upstream's text
- [x] 17. Merge into `main` and rename to `TEXGen-Emission/` (needs Dongchen's go)
- [x] 18. Cut-over (optional, per D8): skipped, D8 = stay
- [x] 19. Release cut: the check tooling leaves the released tree
- [x] 20. `convert_config.py --ckpt` rewrites only real pickled class references

Cleanup (2026-10-18): only the code stays, like upstream's repository.

- [x] 21. Check data on the other machines (before step 24: it uses the tooling)
- [x] 22. Check data on the workstation
- [x] 23. What CLAUDE.md knows moves to the parent (per D12)
- [ ] 24. `main` gets the release cut: tag `pre-cleanup`, then the internal files leave (per D11)
- [ ] 25. Local git leftovers: worktrees, merged branches, generated files
- [ ] 26. GitHub: tags pushed, retired branches deleted (per D14; needs Dongchen's go)

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
   (`on_save_checkpoint` / `on_load_checkpoint` in `spuv/systems/texgen_train.py`; in `base.py` until 16b).
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
- [x] D8 Cut-over (before step 18): after the merge, do the launchers, the `texgen-v2` worktree and
  the staged `TEXGen.tar` move to `main`, or stay on the frozen branch? Decided 2026-10-10
  (Dongchen): stay.
- [x] D10 Internal scripts in the published code (before publishing): `scripts/internal/` holds our
  Slurm accounts and paths. Decided 2026-10-09 (Dongchen): the release drops the check tooling, and
  with it the other internal files (step 19).
- [x] D11 (before step 24) `main` itself becomes the clean tree: after step 24 its tree is `release`'s,
  `release` retires, and the tooling lives on at tag `pre-cleanup`. This reverses step 19's "`main`
  keeps the internal files". Decided 2026-10-18 (Dongchen): clean `main` (not: publish `release`).
- [x] D12 (before step 23) CLAUDE.md's content: what is still true and still needed moves to the
  parent's `docs/baselines/texgen.md` (the runs behind the results, the data format and input
  channels, the DDP sampler and `val/psnr` notes, the `shuffle_orders` seeding note, the `save_top_k`
  trap, the memory sizing); the rest stays at the tag (the 1k/74k data history, the overfit tables,
  § Checking a code change). Decided 2026-10-18 (Dongchen): move.
- [x] D13 (before step 25) The two published checkpoints in `outputs/` (2 × 12.4 GB, the gate's
  local copies). Decided 2026-10-18 (Dongchen): keep them as the local cache AGENTS.md describes.
- [x] D14 (before step 26) GitHub: delete the branches `release`, `restructure-2026-09-16` (in
  `main`'s history; tag `archive/restructure-2026-09-16` keeps the name the docs cite) and
  `texgen-74k-v2-vulcan` (= tag `archive/texgen-74k-v2-vulcan`, already pushed); keep
  `texgen-74k-v2-venus05` (Keep stable 9); push the tags `pre-refactor`, `checks-tooling`,
  `pre-cleanup` and the `archive/` ones (docs cite them; today the first two are local only).
  Decided 2026-10-18 (Dongchen): yes, as listed; the push itself still waits for his go at step 26.

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

- [x] Inside: `README.md` (upstream's; its command uses the missing `configs/texgen_test.yaml`;
  `static/teaser.jpg` goes with it), `CLAUDE.md` (no alpha channel, paper or release run; the
  `requirements_125.txt` claim at `:15` and the DDP sampler claim at `:186-190` are wrong),
  `LIGHTGEN_WORKFLOW.md`, the "4 Pro 6000" comment in the release config (`:89`).
- [x] The release commands in the README: single-node training (`python launch.py --config ...
  --gpu 0,1,2,3 --train data.data_root=...`) and inference (`inference_specific_samples.py`) on the
  release test split.
- [x] Keep or retire `../evaluation/pointsampled/texgen_infer.py`. Keeping it needs more than its
  imports (`lightgen_uv`, `lightgen_system`): its config and checkpoint need
  `scripts/convert_config.py`, and its checkpoint is 12-channel (no alpha), which the current code
  (alpha always read) cannot run.

**Instructions:**
- Decided: `README.md` is the doc of record for users (what differs from TEXGen, setup, data, training,
  inference on the test split, upstream's attribution and BibTeX); the teaser image goes with
  upstream's demo. `LIGHTGEN_WORKFLOW.md` is deleted: its lines are in the README or CLAUDE.md.
- Found while checking CLAUDE.md: the loader's explicit `SequentialSampler` does not stop Lightning
  2.5 from sharding val/test under DDP (`_requires_distributed_sampler` replaces any sampler that is
  not a `DistributedSampler`), and `val/psnr` (per-batch, no `sync_dist`) is rank 0's quarter. Docs
  fixed; the code stays (it only shapes the logged `val/*` and panels; the scores come from
  separate inference over the full split).
- Decided: retire `../evaluation/pointsampled/texgen_infer.py` at the merge (nothing references it;
  hardcoded paths, the old 12-channel 1k checkpoint and the deleted classes). The parent's driver is
  `../evaluation/newdata_eval/run_texgen_infer.py`; its call still matches `inference_samples`.

**Result:** README rewritten (every command checked against the code and config; the data commands are
the ones step 15 tested), CLAUDE.md fixed (alpha channel, runs behind the results, DDP sampler,
`val/psnr` formula, config keys, output layout), workflow map and teaser deleted. Docs only.

### 16b. `base.py` back to upstream's text: our training hooks move to `texgen_train.py`

Dongchen 2026-10-09: our hooks in upstream's `base.py` move into our training class, so `base.py`
reads like upstream's.

- [x] Moved verbatim: the wandb run id (its default now a class attribute), `on_save_checkpoint`,
  `on_load_checkpoint`, `on_train_start` (the `on_fit_start` note joins its docstring),
  `on_train_epoch_end`, and the every-10-batches cleanup from `on_train_batch_end` (now after
  `texgen_base`'s EMA update). `base.py` keeps one change: the `loss` default (`field(default_factory=…)`),
  which newer Python requires.

**Result:** `base.py` differs from upstream by one line. Quick check wt-16b-20261009-033714 (paper 20,
release, fresh, resume) IDENTICAL; Codex and a three-lens review found no behavior change, 4 stale comments fixed.

### 16c. `texgen_base.on_train_batch_end` back to upstream's text (Dongchen 2026-10-09: continue)

- [x] Upstream's method is the EMA update alone. Ours (`7d544d4`) adds a `super()` call and a cleanup
  every 5 steps: the cleanup moves to `texgen_train.on_train_batch_end`; the `super()` call goes (it
  runs only `update_step_end`, which no class in `spuv/` overrides). Quick check.

**Result:** `texgen_base.py`'s method is upstream's again. Full check wt-16c-full-20261009-040031:
204/204 paper (one L40S piece WITHIN_NOISE), release, fresh, resume IDENTICAL.

### 17. Merge into `main` and rename to `TEXGen-Emission/` (needs Dongchen's go)

- [x] The full gate and the training check on the branch head, against the `pre-refactor` baselines
  (wt-16c-full-20261009-040031, the 16c tree: PASS; later commits change only docs).
- [x] Merge `refactor` into `main`.
- [x] In the parent, one commit: the pointer bump, the rename `TEXGen/` → `TEXGen-Emission/`
  (`.gitmodules`, the gitlink, every parent reference: search again; on 2026-10-06 about 20 files, among
  them `../evaluation/newdata_eval/run_texgen_infer.py` and its test, `sweeps/job_texgen.sh`,
  `sweeps/stage_texgen_host.sh`, `sweeps/star2/texgen_worker.sbatch`, `infer_texgen_v2_venus05.sbatch`,
  `seedutil.py`, `../data_processing/data_preparation/README.md` and its
  `tests/test_encoding_parity.py`, `../.claude/rules/texgen.md` with its `paths:` front matter,
  `../AGENTS.md`). The `texgen-v2` worktree moves with the directory.
- [x] Parent docs: `../docs/baselines/texgen.md` (the config table `:12-30,48` names deleted configs;
  `:53-61` says the flash switch is only on the branch), `../.claude/rules/texgen.md` (deleted configs
  in its commands and Losses; `:38` names the backbone `LightGenPointUVNet`, now upstream's
  `PointUVNet`; `:34,45` name `LightGenSystem` / `lightgen_system.py`, now `texgen_train.py` and
  `texgen_test.py`), `../AGENTS.md:64,212,228`, `../docs/status/refactor.md` (the TEXGen
  entry); the stale "main has no flash gate" comments in
  `../evaluation/newdata_eval/run_texgen_infer.py:15-18`, `infer_texgen_v2_venus05.sbatch:6`,
  `sweeps/stage_texgen_host.sh:46`.
- [x] Loader pointers (since 8a the loader is `spuv/data/mesh_uv.py` and `spuv/data/lightgen_uv.py`
  is gone): comments and mirrored constants in
  `../data_processing/data_preparation/{atlas.py:5,contract.py:8,multiview.py:335}` and
  `tests/test_encoding_parity.py:5,17,53`; docs `../data_processing/data_preparation/README.md:45,56`,
  `../docs/baselines/texgen.md:35`, `../.claude/rules/texgen.md:72`. The import in
  `../evaluation/pointsampled/texgen_infer.py:23` breaks; parent scripts that run the current code
  on a published run's `parsed.yaml` (`infer_texgen_v2_venus05.sbatch`, `texgen_worker.sbatch`,
  `sweeps/job_texgen.sh`, the smoke command `sweeps/stage_texgen_host.sh` prints) need
  `scripts/convert_config.py` first, as `gate-det.sh` does (the config and, since 8c, the checkpoint:
  `--ckpt`); converting once in `run_texgen_infer.py` covers them all.
- [x] From step 16: delete `../evaluation/pointsampled/texgen_infer.py`; `../.claude/rules/texgen.md`
  also names `requirements_125.txt` (`:12`), `LIGHTGEN_WORKFLOW.md` (`:15`), `memory_tracker.py`
  (`:90`), the SequentialSampler fix as working, 12/13 input channels and per-GPU batch 32;
  `../AGENTS.md:228` and `../docs/baselines/texgen.md:5,11` link `LIGHTGEN_WORKFLOW.md` (now gone:
  point at `TEXGen/README.md`).
- [x] Decide here: rename the GitHub repository `dongchen-yang/TEXGen` too? Kept: renaming it is a
  push-side change for Dongchen; the submodule keeps the name `TEXGen` and its URL.

**Instructions:** (Dongchen 2026-10-09: "lets just continue the plan")
- Runs from a NEW session started at the lightgen root, not from the `refactor` worktree: that
  session is barred from git in the main checkout, and the rename moves its own directory.
- Merge: `main` is at `pre-refactor` (`de78afc`), so `cd TEXGen && git status --short` (must be clean)
  then `git merge --ff-only refactor`.
- Parent: other sessions leave uncommitted edits there (34 files on 2026-10-09, `AGENTS.md` among
  them). Stage only this step's hunks (`git add -p <file>` or `git apply --cached`); never `git add -A`.
- Rename: `git mv TEXGen TEXGen-Emission` (the submodule's name stays `TEXGen`), then
  `git -C TEXGen-Emission worktree repair .claude/worktrees/refactor .claude/worktrees/texgen-v2`.
  25 live files name the path (`git grep -l -E "(^|[^A-Za-z_-])TEXGen/|/TEXGen\b"`, leaving out
  `docs/status/`, `docs/superpowers/`, `data_processing/archive/`, html galleries: records keep
  the old name).
- The GitHub repository keeps its name unless Dongchen says otherwise (a push is his call).

**Result:** `main` fast-forwarded to `refactor` (`463860c`); the directory is `TEXGen-Emission/`, its
two worktrees repaired. Parent: the rename with the pointer and the local path references; the docs;
the loader pointers, `texgen_infer.py` retired, `run_texgen_infer.py` converting published runs
(smoke: the paper checkpoint converted on the fly, 2 shapes IDENTICAL to the 4090 baseline). Codex:
4 findings (cache key, nested layout, the `_v2` launcher's checkout, a doc claim), all fixed.

### 18. Cut-over (optional, per D8)

- [ ] Move the consumers to `main`: the `texgen-v2` worktree, the `BRANCH=` lines in
  `scripts/internal/fir/train_lightgenbench_alpha_fir.sh:61,157` (and the old-layout paths the fir
  scripts use), the branch check in `../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:78-91`;
  rebuild the staged `TEXGen.tar` on jupiter (it unpacks to `TEXGen/`) and delete the Solar nodes'
  `.tree_ok` so they unpack it again.

**Instructions:** Skipped (D8: stay, Dongchen 2026-10-10).

**Result:** The consumers keep branch `texgen-74k-v2-venus05`, the code the published numbers ran;
`main` gives the same outputs (the gate) and cannot run the 12-channel `_v2` checkpoint. No push, no
jupiter or cluster change.

### 19. Release cut: the check tooling leaves the released tree (Dongchen 2026-10-09)

The released repository carries no regression-check code. This runs after step 17's final check,
which needs the tooling.

- [x] Tag the last commit that has it (for example `checks-tooling`), so a later check can restore it.
- [x] Delete `scripts/internal/gate/` and CLAUDE.md's § Checking a code change (one line pointing at
  the tag stays).
- [x] With it, decide the other internal files: `scripts/internal/fir/` (cluster launchers with our
  paths), `REFACTOR_TODO.md`, `CLAUDE.md`.

**Instructions:** Dongchen 2026-10-10: the release is a separate branch; `main` keeps the internal
files (the checks and these docs keep working there). Tag `checks-tooling` marks the `main` commit
the release was cut from. Also asked: one data script with an explicit name, no code change, so
`prepare_lightgenbench.sh` and `build_index.py` became `scripts/lightgenbench/unpack_hf_dataset.sh`
(`aeb404e`; index files byte-identical on an 826-shape slice of the release, the check root rebuilt
identical).

**Result:** Branch `release` (`08d8130`, from `aeb404e`): `scripts/internal/`, `REFACTOR_TODO.md` and
`CLAUDE.md` removed, README and config comments no longer name them; code files are `main`'s. Its
tree through `main`'s checks: release and resume IDENTICAL (sp4090), fresh IDENTICAL (L40S, cs-venus-17).

### 20. `convert_config.py --ckpt` rewrites only real pickled class references (Dongchen 2026-10-17)

Found by Codex in the Hunyuan fork's converter, which copied this design: a raw bytes replace in
`data.pkl` corrupted a pickled string holding the old GLOBAL's bytes, and the "still referenced"
check aborted on strings that merely name the old path.

- [x] `_rewrite_globals` parses the pickle (`pickletools.genops`) and splices only GLOBAL opcodes
  that name an old class; a protocol-4 pickle (FRAME or STACK_GLOBAL) that mentions an old module
  is refused; `--ckpt` refuses src == dst; the count sums over `data.pkl` entries. The copy is still
  written when nothing changes (`ckpt_for_tree.sh` and the parent's driver expect it).
- [x] Tests: `scripts/internal/tests/test_convert_config.py` (7).

**Result:** On both published checkpoints the output is sha256-identical to the old implementation's
(1 class reference each), so no gate result moves. Codex: 2 findings (framed pickles, the count),
fixed.

## Cleanup: only the code stays (steps 21–26)

Dongchen 2026-10-18: "eventually, we want a clean repo like the original repo"; "just remove the
unneeded or testing files that we generated or the gated files". Files only: no code file changes.

Where it stands: branch `release` (`d8b8f6a`) is the clean tree already (step 19). `main`
(`bed1e8f`, on GitHub) still has 24 internal files: `scripts/internal/` (`gate/` 17, `fir/` 4,
`tests/` 1), `REFACTOR_TODO.md`, `CLAUDE.md`. Apart from those, `main` and `release` differ only in the
README's `scripts/internal/` row and three config comments. The check data sits outside the repo
on the workstation, star2, the Solar nodes and the lab 4090s.

- Work in the main checkout on `main`. One commit per step where git changes. Outside git: list
  each path with its size, confirm it is TEXGen's check data and nothing live reads it (`git grep`
  in the parent; on 2026-10-18 only `scripts/internal/` did), then delete it. Each step waits for
  Dongchen's go.
- Checks: steps 21, 22 and 25 change no tracked file; 23 is docs. Step 24 deletes files and edits
  comments: `git diff release main` must come out empty, the config must resolve and
  `python launch.py --help` must run. No GPU check: the code files are byte-identical to
  `release`'s, which passed release, resume and fresh in step 19.
- From step 24 on this file is gone (its last version is at tag `pre-cleanup`): Results go to the
  commit messages and the TEXGen entry of `../docs/status/refactor.md`.
- Parent commits stage only their own hunks (`AGENTS.md` carries another session's edit).

### 21. Check data on the other machines (before step 24)

- [x] No checks job is running: `ssh star2 "squeue --me -h -o '%i %j %T %N'"`, none named
  `refactor_checks_*`.
- [x] Solar `/localscratch/dya78/refactor_checks` and `/localscratch/dya78/tmp/tc.*`: list with
  `python3 scripts/internal/gate/fanout.py --solar_clean cs-venus-15,cs-venus-16,cs-venus-17,cs-venus-18,cs-venus-20 --plan`.
  `--solar_clean` removes only `out/*` on our three nodes (~30 GB staging per node stays), so delete
  there with the same 1-CPU `srun -w <node> --immediate=60` job and the full `rm` it runs on the
  theia-lab nodes.
- [x] star2 home `~/refactor_checks/` (staged trees, tooling, `paper_root/`, `hf_root/`, `out/`).
- [x] sp4090 (and xg4090 if it was ever staged): `/local-scratch/localhome/dya78/lightgen_eval_local/refactor_checks/`
  (tooling, a repo copy, checkpoint copies, `paper_root/`). The copied `texgen` env stays.

**Instructions:** Dongchen 2026-10-18: "go 21–25".

**Result:** Deleted: Solar `refactor_checks/` on cs-venus-15/16/17/18 (~114 GB; `solar_clean` with the full
`rm` on every node; no `tc.*` left, cs-venus-20 empty), star2 `~/refactor_checks/` (3.5 GB), sp4090's
(52 GB). xg4090 was never staged (not admitted, still unresolvable). Log `../log/20261018-213021-texgen-cleanup-solar.log`.

### 22. Check data on the workstation

- [x] `/local-scratch/localhome/dya78/lightgen_eval_local/refactor_checks/` (42 GB: `ckpt_cache/` 24,
  `ckpt_test/` 12, `paper_root/` 3.3, `hf/` 1.6, the rest under 1).
- [x] The older TEXGen gate outputs beside it: `pred_gate_*_seed00/` (10 dirs, ~480 MB, among them the
  4090 baseline `pred_gate_det1_seed00`) and `gate_{final,fold}_different.txt`. A later check would
  rebuild its baselines from tag `pre-refactor` (step 7's recipe).
- Stays: the eval lists (`test_agentic_clean.txt`, `val_agentic_clean.txt`), every other `pred_*`,
  the other lanes' `*_refactor_checks/`, the parent's `log/`.

**Instructions:**

**Result:** Deleted `refactor_checks/`, the 10 `pred_gate_*_seed00/` and both `gate_*_different.txt` (42 GB; no
process or live file used them). `/local-scratch` 301 → 345 GB free.

### 23. What CLAUDE.md knows moves to the parent (per D12)

- [x] Its still-true sections go to `../docs/baselines/texgen.md` (shortened, nothing restated that
  the README says).
- [x] References that name `TEXGen-Emission/CLAUDE.md` point there: `../.claude/rules/texgen.md:15`,
  `../AGENTS.md:230`, `../docs/baselines/texgen.md:5`. Historical logs keep the old name
  (`../evaluation/newdata_eval/EXECUTION_LOG.md`).
- One parent commit, docs only.

**Instructions:**

**Result:** Four sections in `../docs/baselines/texgen.md` (runs, code map, data contract, traps), checked
against the code and by Codex (6 of 8 findings applied); rule and index repointed. Parent `625e630`.

### 24. `main` gets the release cut (per D11)

- [ ] Tag `pre-cleanup` on the last commit that has the internal files (this file's final version).
- [ ] Delete `scripts/internal/`, `REFACTOR_TODO.md`, `CLAUDE.md`; take `release`'s README row and config
  comments. Then `git diff release main` is empty.
- [ ] Parent, one commit: the pointer, and the fir-launcher references as tag paths
  (`git -C TEXGen-Emission show pre-cleanup:scripts/internal/fir/<file>`):
  `../docs/baselines/texgen.md:26`, `../docs/evaluation.md:1089`; the TEXGen entry of
  `../docs/status/refactor.md` (still lists steps 18 and 19 as open).
- The converter keeps working without its tests (`scripts/internal/tests/`), as on `release`.

**Instructions:**

**Result:**

### 25. Local git leftovers

- [ ] Worktrees `refactor` (= `main`) and `release` (= `main`'s tree after 24): check clean, then
  `git worktree remove`. `texgen-v2` and its branch stay (Keep stable 9), and with it the
  `.claude/worktrees/` ignore rule.
- [ ] Branches: tag `archive/release` and `archive/restructure-2026-09-16`, then delete the local
  `refactor`, `release`, `restructure-2026-09-16`.
- [ ] Generated files (ignored, not in git): every `__pycache__/`, `.pytest_cache/`, the leftover
  `tests/` (only `__pycache__/` inside), and the two run stubs without checkpoints
  `outputs/texgen_alpha_74k_v2/`, `outputs/texgen_alpha_74k_v2_nonzero_nocopy/` (a `parsed.yaml`
  each, one `SHA256SUMS.pulled`). The two checkpoints per D13.

**Instructions:**

**Result:**

### 26. GitHub (per D14; a push, needs Dongchen's go)

- [ ] The submodule first: push `main` and the tags, delete the remote branches D14 names. Then the
  parent (its pointer), so a fresh clone always finds the gitlink.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **Flash default.** An unset `TEXGEN_ENABLE_FLASH` means on (`texgen_network.py:49`); both
  published runs used dense attention (`=0`). Forgetting it silently changes numbers on a 4090 or an
  H100 and asserts on Blackwell. Default to dense, or fail when unset.
- [x] **Release batch.** Done in step 15: the release config carries 16 × 2.
- [ ] **Fixes from `restructure-2026-09-16`** (`b4e423f`, `e20b015`, `4ef35a7`; re-apply by hand):
  EMA weights on the test pass, the wandb directory, a non-zero exit on interrupt, the resume order
  (one checkpoint read instead of two), validation metrics synced across GPUs, the training preview
  panel, the removed noise schedule; the loss guard (`309d3ab`, `bfb2135`: rejects a nonzero loss
  weight no code reads), after step 8c.
- [ ] **Tests from `restructure-2026-09-16`** (12 files, including a toy CPU fit/resume test).

## Out of scope

The position frame (atlas decoded as [−2, 2], voxels in [−0.5, 0.5]): a data-format change, tracked
in `../docs/status/refactor.md`.

Rewriting git history. Steps 21–26 clean the tree, not the history: `main`'s public history keeps
the internal files (our Slurm launchers and paths) and our 254 commits above upstream's 11. A
squashed history would change every commit id the parent's gitlinks and the docs cite.
