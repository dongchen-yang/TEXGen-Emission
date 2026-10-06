# TEXGen refactor — to-do

Started 2026-09-29 from `main` = `6096a63` (the paper code, tag `pre-trim-2026-09-16`, plus the
LightgenBench retrain files). The 2026-09-16 restructure was reverted on 2026-09-19 (`0286f23`)
because, in Dongchen's words, doing it as a whole "will make the refactoring lost track" and he could
not tell whether the agent added anything not intended; this list does it in small steps instead.
That attempt stays on branch `restructure-2026-09-16` for reference.

## Progress

On 2026-10-06 the old steps 11, 13, 14, 15 and 17 were removed (emptied by D9 and D6, or absorbed by
the system fold) and the open steps were renumbered in the order they run. Old to new, for commit
messages written before that: 7 → 8 (7a–7d → 8a–8d), 8 → 7, 12 → 11, 16 → 12, 18 → 13, 19 → 14,
20 → 16, 20b → 15, 21 → 17; 9 and 10 kept their numbers. The removed text is in this file's history.

- [x] 1. Housekeeping
- [x] 2. Dead root scripts and the Blender call
- [x] 3. The existing `deprecated/` tree and stale ignore rules
- [x] 4. Upstream demo material
- [x] 5. Launchers (before the configs they use)
- [x] 5b. Move the kept cluster scripts into `scripts/internal/`
- [x] 6. Configs
- [ ] 7. Training-check setup (no code change)
- [ ] 8. Fold our modules into upstream's originals (direct edits)
- [ ] 9. Coverage run (no code change)
- [ ] 10. `inference_specific_samples.py`: output-only code and hardcoded paths
- [ ] 11. Render-only data in our own code
- [ ] 12. The memory tracker
- [ ] 13. `launch.py`
- [ ] 14. Environment files
- [ ] 15. Release layer: generic entry points for LightgenBench users
- [ ] 16. Docs
- [ ] 17. Cut-over (optional, per D8)

## How we work

- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change.
- The steps run automatically, in order, now that the plan is agreed (Dongchen, 2026-10-06): when a
  step's checks pass, commit it, record the result, report it in a short line, and start the next
  step without waiting for a go. Dongchen can add instructions under any step's **Instructions** at
  any time; a step reads them before it starts.
- Stop and ask only when: (1) a step needs an open decision (each D says which step it is due
  before); (2) a check fails (the gate is not `same=N different=0 missing=0`, or the training check
  moves outside its measured noise), and the cause is not an obvious slip in the step itself; (3) a
  step meets something the plan does not cover (a fix the step would need, a behavior change, a file
  outside the step's list); (4) the next action leaves the worktree: a push, a commit in the parent
  repo, the merge into `main`, a write to jupiter or a cluster job.
- After each step that touches code: the byte gate (below) must read `same=N different=0
  missing=0` (N = 20 on the subset, 204 on the full set). Steps that touch training code also need
  the training check (D5).
- A doc that names a removed file is fixed in the same step as the removal.
- In `TEXGen/`: never `git add -A` (the `texgen-v2` worktree is untracked there) and never
  `git clean -x`/`-X` (the ignored `outputs/` holds the gate's only local checkpoint, 12 GB).
- Parent pointer bumps use a path-limited commit, `git commit -- TEXGen`: the parent working tree
  carries other sessions' uncommitted work.
- Nothing is pushed without Dongchen's go.
- Our changes are edits to upstream's original modules, not new parallel modules (Dongchen,
  2026-10-06), so `git diff upstream/main -- <file>` shows exactly what was modified; a file of ours
  that only keeps a published class path alive is a one-line alias. `git diff --stat upstream/main
  HEAD -- spuv launch.py` lists every modified upstream file.
- From step 6 on, all edits happen in the worktree `.claude/worktrees/refactor` (branch `refactor`);
  `git diff pre-refactor` there shows every edit since the tag. The main checkout stays on `main`, so
  the parent's scripts that default to `TEXGen/` keep running the unchanged code. The gate's
  `<texgen_root>` is the worktree; the paper checkpoint lives only in the main checkout's ignored
  `outputs/`, so pass its absolute path.
- Status: tick a step under **Progress** when it is done (commit hashes in its **Result**), each item
  inside a step as it lands, and a decision once it is answered.

Live runs:

| run | config | checkpoint |
|---|---|---|
| paper | the checkpoint's own `configs/parsed.yaml` (the config file `…_74k_v2_alpha_venus19_agentic.yaml` was deleted in step 6; tag `pre-refactor` and branch `texgen-74k-v2-venus05` keep it) | local `outputs/texgen_alpha_74k_v2_agentic/ckpts/epoch=124-step=35500.ckpt` |
| release retrain | `configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml`, run by `scripts/internal/fir/train_lightgenbench_alpha_fir.sh` (overrides batch 16 × accumulate 2) | jupiter `outputs/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt` |

Byte gate (paper checkpoint, 204 test shapes, seed 0, PyTorch deterministic mode, 4090 lock;
inference only, not training; all paths absolute because the driver `chdir`s):

```bash
L=/local-scratch/localhome/dya78/lightgen_eval_local
bash scripts/internal/gate/gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file>]   # prints GATE_DONE
python ../evaluation/newdata_eval/compare_pred_dirs.py --a $L/pred_gate_det1_seed00 --b <out_dir> \
    --shas_file <same shas_file> [--filename gt_emission.png|mask.png|input_albedo.png]
```

Routine steps pass `scripts/internal/gate/subset20.txt` (20 shapes, 164 s); without it the gate runs all 204
(`$L/test_agentic_clean.txt`, about 24 min), for the final check before the D7 merge and for steps
that touch the loader or the model. Each shape's seed depends only on its sha, so a subset gives the
same bytes. The compare checks `pred_emission.png` by default; compare all four files. The baseline `$L/pred_gate_det1_seed00` exists only on this workstation's disk.

## Keep stable (other code depends on these)

1. `inference_samples(checkpoint_path, sample_ids, output_dir, data_root=None, parquet_file=None,
   seed=0)` in `inference_specific_samples.py`, imported after a `chdir` and called by keyword from
   `../evaluation/newdata_eval/run_texgen_infer.py:40-46` (the gate's driver) and from
   `scripts/internal/fir/infer_lightgenbench_fir.sh:71-74` (the release inference, branch
   `texgen-74k-v2-venus05`). Legacy caller: `../evaluation/pointsampled/texgen_infer.py` imports
   `spuv` directly (decide keep or retire in step 16).
2. Class paths loaded from each run's `configs/parsed.yaml` (read as `ckpt/../../configs/parsed.yaml`,
   `inference_specific_samples.py:65-68`): `spuv.systems.lightgen_system.LightGenSystem`,
   `spuv.data.lightgen_uv.LightGenDataModule`,
   `spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet`,
   `spuv.models.tokenizers.clip.ClipTokenizer`. Pickled inside the paper checkpoint itself (so
   `torch.load` imports them): `spuv.systems.lightgen_system.LightGenSystem` and
   `spuv.systems.texgen_base.LossConfig`.
3. Every key in the published `parsed.yaml` files stays accepted: `spuv/utils/config.py:151` merges
   them into the dataclasses in struct mode, which rejects unknown keys. Keys may stop being read;
   they may not be deleted from the dataclasses.
4. Model attribute names `backbone`, `backbone_ema`, `image_tokenizer` (inference loads with
   `strict=False`, `inference_specific_samples.py:135`, so a rename drops weights silently) and the
   custom checkpoint keys `_scheduler_states`, `wandb_run_id` (`spuv/systems/base.py:137,160`).
5. The output layout `<out>/<sha>/pred_emission.png` (+ `gt_emission`, `mask`, `input_albedo`):
   256², linear, [0, 1], masked, same UV layout as the input atlas.
6. The `TEXGEN_ENABLE_FLASH` gate in `spuv/models/sparse_networks/texgen_network.py:49` (star2 sbatch
   files grep for it).
7. The loader contract in `spuv/data/lightgen_uv.py`: `<data_root>/<ditem_dir>/somage.npz`,
   `<data_root>/thumbnails/<sha>.png` (the CLIP input), parquet with `ditem_dir` and `success`,
   split indices that are positions over the `success==True` rows, alpha from the npz or an
   `alpha.npy` sidecar, positions decoded from [−2, 2].
8. `sample_seed` (`inference_specific_samples.py:26-32`): signature and return line byte-identical to
   `../evaluation/newdata_eval/seedutil.py` (the docstrings differ).
9. Branch `texgen-74k-v2-venus05` stays, frozen at `370818f`: the release launcher
   (`scripts/internal/fir/train_lightgenbench_alpha_fir.sh:61,157`), the paper launcher (deleted from `main` in
   step 5, kept on this branch) and `../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:78-91` all check it out, and
   the worktree `.claude/worktrees/texgen-v2` sits on it (used by
   `../evaluation/newdata_eval/sweeps/job_texgen.sh:16`, `stage_texgen_host.sh:22`, and pinned in
   `../evaluation/newdata_eval/tests/test_run_texgen_infer.py:16`). Moving them to `main` is step 17.
10. The directory name `TEXGen/` and the `outputs/<run>/{ckpts,configs}` layout.

## Open decisions

- [x] **D1 Removal** (before step 2). Delete (tag `pre-trim-2026-09-16` keeps everything) or move to
  `deprecated/` (the current convention; it already holds 54 files)? Recommended: delete.
  **Answer (2026-09-29):** delete.
- [ ] **D2 Live runs** (before step 7). Paper run only, or paper and release retrain? The inference code
  is the same; the release row is in `../docs/evaluation.md:1051`, not in the paper. Recommended:
  both.
- [ ] **D3 Behavior** (before step 8). Cleanup steps change no behavior (gate `same=204` each time);
  fixes come later, one per step, only if wanted. Recommended: yes.
- [ ] **D4 Names** (before step 8). Keep `TEXGen/`, the class paths and the inference signature.
  Recommended: keep.
- [ ] **D5 Training check** (before step 7). For steps that touch training code: a short deterministic
  GPU run, loss at step N compared before and after, plus one resume from a real checkpoint. Reseed
  (`pl.seed_everything`) right before `trainer.fit`, because removing LPIPS or a CLIP build shifts the
  CPU random stream. Recommended: yes.
- [x] **D6 Paper ablation variants** (before step 6). Keep or drop the GT-mask conditioning and
  mask-classification code paths? Only the 1k and overfit configs use them; the paper mentions the
  GT-mask variant only in commented-out text (`../paper_draft/sec/6_experiment.tex:20`). If kept,
  keep one config per path.
  **Answer (2026-10-06):** the configs went in step 6; the code paths stay ("we can still keep the
  losses there. since they can just be adjusted by the config"). GT-mask conditioning is an input
  option rather than a loss, but it is switched by config the same way
  (`use_gt_emission_mask_cond`), so it stays too.
- [x] **D7 Where commits go** (before step 1). Directly on `main`, or on a branch merged when you are
  happy (one merge to undo)? Is this file committed? Pointer bump per step or only at the merge?
  Recommended: a branch; commit this file; bump once at the merge.
  **Answer (2026-10-06):** from step 6 on, a branch: `refactor`, checked out in the worktree
  `.claude/worktrees/refactor`, started at tag `pre-refactor` (`de78afc` = `main` with steps 1–5b and
  this file committed). Steps 1–5b were committed directly on `main` before that (only the step-1
  merge is pushed). Open: when the parent pointer is bumped (still `6096a63`).
- [ ] **D8 Cut-over** (before step 17). After the refactor, do the launchers, the worktree and the staged
  `TEXGen.tar` move to `main`, or stay on the frozen branch?
- [x] **D9 Upstream code we do not use** (before step 8). Under direct editing, upstream files and
  blocks our runs never reach (`camera.py`, `isosurface.py`, `networks.py`, `timestep.py`,
  `spuv/utils/rasterize.py`, upstream's render stack, its star imports): leave them untouched, so
  `git diff upstream/main` shows only real changes, or delete them? Step 4 already deleted
  upstream's demo files (`tools/`, `assets/`, two images). Recommended: leave untouched from step 8 on.
  This narrowed steps 11 and 12.
  **Answer (2026-10-06):** untouched. Upstream code changes only where our runs need a change; step
  4's deletions stay as they are. Extended the same day ("for other steps, we can just leave the
  original TEXGen's code there if they are not used"): this also covers unused code *inside* upstream
  files, even files we edit. Cleanups apply only to our own code.
- [ ] **D10 Our internal scripts in the published code** (before publishing). `scripts/internal/`
  holds our Slurm accounts, `/scratch/dya78` and workstation paths, and step 15's no-paths rule
  covers only the release layer. Publish it as it is, publish a snapshot without it, or move it into
  the parent repo? Recommended: a published snapshot without `scripts/internal/`; our working copy
  keeps it.

## Steps

### 1. Housekeeping

- [x] Merge branch `texgen-74k-v2-venus05` into `main`: a true merge (main is 37 commits ahead), clean,
  the only difference is `370818f` adding `scripts/fir/infer_lightgenbench_fir.sh`, which
  `../docs/evaluation.md:1073` cites. The branch itself stays (Keep stable #9).
- [x] Add `.claude/worktrees/` to `.gitignore` (the parent ignores the same path). `edcae73`, not
  pushed.
- [x] Move the gate tooling (`gate-det.sh`, `detmode/sitecustomize.py`) out of the parent's git-ignored
  `.superpowers/` into `scripts/gate/`. `7408f06`, not pushed.

**Instructions:** 2026-09-29: "lets merge the branch to main now."; push it; add `.claude/worktrees`
to the gitignore; move the gate tooling "here".

**Result:** 2026-09-29 merge `2b578af` on `main`, pushed to GitHub (`6096a63..2b578af`); parent
pointer not bumped.
`main`'s tree is now identical to `texgen-74k-v2-venus05`'s.

### 2. Dead root scripts and the Blender call

- [x] `render_inference_blender.py` together with its caller: `inference_specific_samples.py:323-325`
  and `render_blender_multiview` (`:335-386`, the Blender binary at `:347`). Every inference run,
  the gate included, starts Blender, which fails on the `xgutils` import; the error is swallowed, so
  outputs do not change.
- [x] `render_inference_multiview.py` (needs the suspended `xgutils`).
- [x] `main.py` and `pyproject.toml` (empty `uv init` stubs), `slurm_train.sh`, `slurm_train_74k.sh`,
  `batch_sweep.sh`, `filter_dataset.py`, `filter_local_subset_emission.py`,
  `precompute_clip_embeddings.py` (no config sets `use_precomputed_clip`). `detar_progress.py`
  (called only by the superseded 74k and vulcan launchers) moved to step 5, with its callers.

**Instructions:** 2026-09-29: D1 = delete. Commit A = the Blender call, gated; commit B = the rest.

**Result:** Commit A `b0ea68a` (the call, `render_blender_multiview`, `render_inference_blender.py`):
byte-identical on all four files, 63/63 shapes from a stopped full run and 20/20 on `subset20.txt`.
The gate's subset option came in `8460b3b`. Commit B `8fdba12`: the nine root files, plus the lines
naming them in `CLAUDE.md` and `LIGHTGEN_WORKFLOW.md`; parent commit `2895ff4` fixes
`../.claude/rules/texgen.md` and `../docs/baselines/texgen.md`. Comments naming them remain only in
configs and vulcan launchers that steps 5 and 6 remove. Nothing pushed.

### 3. The existing `deprecated/` tree and stale ignore rules

- [x] `deprecated/` (54 files) per D1.
- [x] `.gitignore` entries that go stale: `deprecated/comparisons/generated/`, `comparison_overfit/`
  (plus `render_outputs/` and `deprecated/scripts/**/.claude/`); the ignored stray
  `deprecated/scripts/star2/.claude/` folder, which would survive a `git rm -r deprecated`.

**Instructions:** 2026-09-29: go (D1 = delete). Keep the unused `use_precomputed_clip` loading branch.

**Result:** TEXGen `7e0dfab`: `deprecated/` deleted (54 files, all in tag `pre-trim-2026-09-16`);
the ignored `deprecated/scripts/star2/.claude/settings.local.json` (a Notion MCP switch) removed by
hand; `.gitignore` drops `comparison_overfit/`, `render_outputs/`, `deprecated/comparisons/generated/`,
`deprecated/scripts/**/.claude/`; `CLAUDE.md` and `LIGHTGEN_WORKFLOW.md` updated. Parent commit `2466f82` fixes
`../.claude/rules/texgen.md` and `../docs/baselines/texgen.md`. Nothing pushed.

### 4. Upstream demo material

- [x] `tools/` (`export_to_glb.py`, `render_all.py`, `render_one.py`, `render_one.blend`).
- [x] `assets/` (`checkpoints/.gitkeep`, `input_list/`, 18 sample-mesh files under `models/`), and its
  `.gitignore` rule `assets/checkpoints/*.ckpt` (moved here from step 3; `*.ckpt` still covers
  checkpoints).
- [x] `static/blocks.jpg` and `static/overview.jpg` (unreferenced). `static/teaser.jpg` stays until the
  README changes in step 16.

**Instructions:** 2026-09-29: go.

**Result:** TEXGen `26eb97b`: 26 files deleted (`tools/` 4, `assets/` 20, 2 images), the
`assets/checkpoints/*.ckpt` ignore rule dropped, a note under README's Usage that upstream's demo
does not run in this fork. Nothing referenced them outside README. Nothing pushed.

### 5. Launchers (before the configs they use)

Keep only what produced the release results: `scripts/fir/train_lightgenbench_alpha_fir.sh`,
`scripts/fir/infer_lightgenbench_fir.sh`, `scripts/fir/save_janitor.sh`, `scripts/lightgenbench/`.

- [x] Remove the 5 `scripts/fir/texgen_256_batchsize32_filter_emission*.sh`, `scripts/vulcan/` (4 files),
  the 4 `scripts/star2/train_74k_v2*_venus05.sh`, `scripts/star2/stage_alpha_venus05.sh` and
  `detar_progress.py`.
- [x] Also remove (not used for the release results): the paper launcher
  `scripts/star2/train_74k_v2_alpha_agentic_venus19.sh`, `scripts/star2/bootstrap_texgen_bw.sh`, and
  `scripts/fir/train_74k_v2_alpha_agentic_fir.sh` (the release launcher's header now points at it with
  `git show texgen-74k-v2-venus05:scripts/fir/train_74k_v2_alpha_agentic_fir.sh` instead of copying it in).

**Instructions:** 2026-09-29/30: keep only the scripts used for the release results; the kept ones are
cluster-specific, so split them into a release layer and an internal layer (steps 5b and 15).

**Result:** TEXGen `b0f7904`: 18 files deleted, all still on branch `texgen-74k-v2-venus05`; the release
launcher's comments say where the deleted files live, and its usage line, which named the old
fir launcher instead of itself, is fixed; `CLAUDE.md` and `LIGHTGEN_WORKFLOW.md` drop the wrapper
commands. Parent `af2fd7e`: `../.claude/rules/texgen.md` and `../docs/baselines/texgen.md`. Nothing
pushed.

### 5b. Move the kept cluster scripts into `scripts/internal/`

Pure moves, no content change: `scripts/fir/*` → `scripts/internal/fir/`, the sbatch
`scripts/lightgenbench/stage_hf.sh` → `scripts/internal/fir/`, `scripts/gate/` →
`scripts/internal/gate/`. `scripts/lightgenbench/build_index.py` stays (it is generic). Fix every
path that cites them in the same commit (for example `../docs/evaluation.md:1089`, the gate command
above, the release launcher's own comments).

- [x] Moves.
- [x] Path fixes in docs and comments.

**Instructions:** 2026-09-30: go.

**Result:** TEXGen `71a2e3d`: 7 renames plus comment edits (no executable line changed; a read-only
reviewer confirmed it). The fir scripts run inside a fir clone of branch `texgen-74k-v2-venus05`,
where the old layout still exists, so their operational paths and fir-side usage lines keep it, with a
note in each file; workstation-side usage lines and descriptions use the new paths. Parent `3e7f2bd`:
`../docs/evaluation.md:1089`, `../docs/baselines/texgen.md:34`. Nothing pushed.

### 6. Configs

Keep only the release config, `configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml`
(the rule from step 5: keep what produced the release results).

- [x] Delete the other 19 YAML files and `configs/data_splits_overfit_single.json`: the 1k variants
  (5), the overfit set (4), the 74k pinned (2), the 74k_v2 vanilla (3), the alpha chain (4), and the
  paper config `…_74k_v2_alpha_venus19_agentic.yaml` (the paper checkpoint carries its own
  `configs/parsed.yaml`, so the gate does not need it).
- [x] Fix `CLAUDE.md` and `LIGHTGEN_WORKFLOW.md`, which named them.
- [ ] Parent docs that name them (`../docs/baselines/texgen.md:12-30,48`, `../.claude/rules/texgen.md`)
  describe `TEXGen/` on `main`, which keeps every config until `refactor` merges: fixed at the merge
  (see "Due at the merge").

**Instructions:** 2026-10-06: "delete the unneeded configs". D6: the ablation configs went with the
rest; keeping or dropping their code paths was still open then (later answered: they stay, D6).

**Result:** TEXGen `refactor` `b3dc916`: 20 files deleted, all kept by tag `pre-refactor` (the
paper config also on branch `texgen-74k-v2-venus05`). No code reads any of them (the only code path
under `configs/` is the release launcher's `CONFIG`, which names the kept file). Nothing pushed.

### 7. Training-check setup (no code change)

- [ ] A small local training root (N shapes: parquet, split JSON, thumbnails, alpha) and a local-paths
  override config; the live configs point at `/localscratch/...` and `${SLURM_TMPDIR}`, which do not
  exist on the workstation. Or reuse
  `../.superpowers/sdd/2026-09-16-texgen-restructure/cpu_training_smoke/smoke.py`.
- [ ] Run the check twice on unchanged code to measure its noise before comparing anything.
- [ ] If D2 = both: pull the release checkpoint and its `configs/parsed.yaml` (rsync + sha256).

**Instructions:**

**Result:**

### 8. Fold our modules into upstream's originals (direct edits)

Our three main modules are parallel copies of upstream's. Upstream's own config
(`configs/texgen_test.yaml` on `upstream/main`) names the originals:

| ours (named in every config and `parsed.yaml`) | upstream original |
|---|---|
| `spuv/data/lightgen_uv.py` `LightGenDataModule` | `spuv/data/mesh_uv.py` `ObjaverseDataModule` |
| `spuv/models/sparse_networks/lightgen_pointuvnet.py` `LightGenPointUVNet` (subclass, overrides `forward` to skip the image-feature bake) | `spuv/models/sparse_networks/texgen_network.py` `PointUVNet` |
| `spuv/systems/lightgen_system.py` `LightGenSystem` (subclass of `texgen_test.TEXGenDiffusion`) | `spuv/systems/texgen_test.py` `TEXGenDiffusion` (on `texgen_base.py`) |

Each fold moves our code into the original file, editing upstream's classes and methods in place,
so `git diff upstream/main -- <original>` shows exactly what we changed. Our file then shrinks to a
one-line alias, because the published class paths (Keep stable #2) must still resolve. One commit per
fold, full gate after each. Order: data, backbone, system.

- [ ] 8a. Data: `lightgen_uv.py` → `mesh_uv.py`. Touches the loader: full gate on all four files,
  plus the training check (the gate does not exercise the training split).
- [ ] 8b. Backbone: `LightGenPointUVNet.forward` → `PointUVNet.forward`. Full gate.
- [ ] 8c. System: `LightGenSystem` → `texgen_test.TEXGenDiffusion` (and `texgen_base.py` where ours
  overrides a method only it defines). Our methods replace upstream's versions in place (for example
  `prepare_diffusion_data`, `forward`, `prepare_condition_info`, `texgen_test.py:35-182`); every loss
  branch moves unchanged (D6). Stays as it is: upstream's render branch of `on_check_train`
  (`:219-242`), our disabled 3D-render block in `test_step` (from `:444`), and the `texgen_base.py`
  methods that `texgen_test.py` overrides (upstream's own split). Afterwards the chain is upstream's
  own, `BaseSystem` → `TEXGenDiffusion` (`texgen_base.py`) → `TEXGenDiffusion` (`texgen_test.py`), with
  `LightGenSystem` only an alias; merging upstream's two levels is not planned. Inference (the gate)
  runs through `test_pipeline` (`texgen_test.py:496-544`) and `ema_scope` (`texgen_base.py:232`);
  `test_step` (`:269-494`) runs in validation and `trainer.test`. Keep stable #2 (including
  `spuv.systems.texgen_base.LossConfig`, which stays where it is: the paper checkpoint pickles it),
  #3 and #4. The riskiest fold: full gate, training check, and a resume from a real checkpoint.
- [ ] 8d. Our other added files: `spuv/models/simple_uv_unet.py` (ours, imported by nothing):
  delete. `spuv/utils/memory_tracker.py` (ours, used): step 12.

Check during each fold (not yet looked at): what an alias changes that can be observed (the class
`__name__` in logs, the path new checkpoints pickle, Lightning hparams), and which upstream behavior
the edit replaces (for example `ObjaverseDataModule`'s mesh loading, `PointUVNet`'s bake). Replacing it
is allowed under direct editing; each commit message says what was replaced. Reference: branch
`restructure-2026-09-16` did the same folds (its tasks T5–T7), but renamed the files
`texgen_emission_*` and mixed in 7 behavior changes (listed under Later); take neither.

Upstream modules our code does not use at all (`spuv/models/camera.py`, `isosurface.py`,
`networks.py`, `timestep.py`, `spuv/utils/rasterize.py`): stay untouched (D9).

Needs step 7 (the training check) first.

**Instructions:** 2026-10-06: "what we did was creating new modules for our TEXGen-Emission. I want
it to be more like the direct editing on the original modules. in this way we can also easily track
what has been modified." (Replaces the earlier plan for this step: delete the orphaned modules.)

**Result:**

### 9. Coverage run (no code change)

Paper config under a line-coverage tracer, `TEXGEN_ENABLE_FLASH=0`, 1 GPU, `data.num_workers=0`, with
batch limits so that training, validation, a checkpoint save, a resume and the test pass all run;
plus inference on a few gate shapes and (D2) a few release shapes with the release checkpoint.
Output: per file, the functions and lines that never run. "Never ran" is not "dead" for the flash
branch and the in-npz alpha branch (`spuv/data/lightgen_uv.py:153-158`, used only by the release).

**Instructions:**

**Result:**

### 10. `inference_specific_samples.py`: output-only code and hardcoded paths

- [ ] `:84` falls back to `configs/lightgen_pointuv_256_batch32_pretrained.yaml`, which does not exist.
- [ ] `:96-97` default data root and parquet under `/localhome/dya78/code/lightgen/data/baked_uv`
  (bypassed when the driver passes both).
- [ ] `:245` thumbnail dir under `/localhome/.../baked_uv_local/thumbnails`: output-only (writes
  `thumbnail.png` and column 0 of `overall_visualization.png`), never the model input; the dir does
  not exist, so it already falls back to the batch thumbnail.
- [ ] `create_overall_visualization` (`:318-321`, `:388-484`; a 24 MB `overall_visualization.png` per
  run), the per-shape `comparison.png` and `thumbnail.png`. Removing them changes what lands in the
  output folder (no parent reader found).
- [ ] `:487` checkpoint in the `__main__` block.

Keep the `inference_samples(...)` signature and `sample_seed`.

**Instructions:**

**Result:**

### 11. Render-only data in our own code

Nothing on the pre-baked UV path renders, but our loader builds render-only data for each batch.
Upstream's render pieces stay (D9), even the ones our runs build and never use.

- [ ] The unused `camera_functions` import at `spuv/data/lightgen_uv.py:28`; `get_camera_info`
  (`:439`, called at `:405`, returns dummy matrices) and the `mesh` dict (`:380-387`) build batch keys
  that `lightgen_system.py:115-116` reads and the backbone ignores; remove them together. After step 8
  they sit in `mesh_uv.py` and `texgen_test.py` (line numbers above are from before the folds).

Depends on step 9. Loader and system code: full gate plus the training check.

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

332 of its 571 lines are ours (upstream's has 251). Our parts: the `torch.load` patch and env vars
(`:9-28`), safe globals (`:89-104`), auto-resume and the wandb run-id restore (`:145-199`), the
checkpoint dirpath (`:217-225`), the wandb config and stale-cache cleanup (`:263-341`),
`define_metric` and the `log_metrics` patch (`:343-404`), signal and `atexit` handling (`:406-445`,
`:474-509`). Decide what stays; any change here changes how training runs behave.

**Instructions:**

**Result:**

### 14. Environment files

- [ ] `requirements.txt`: upstream's list plus three pins we added (`775268b`), no torch line.
  `requirements_125.txt`: torch cu126 next to cu118 xformers and spconv.
- [ ] Not imported by live code: xformers, open3d, pointops, torchdyn, torch-geometric.
- [ ] The envs that actually run: the workstation `texgen` (copied to star2 `~/envs/texgen` and the lab
  4090s, and packed as `texgen_env.tar.gz` in jupiter `solar_texgen_stage/`); `texgen-bw` on
  cs-venus-05/19, where the paper run trained (`scripts/star2/bootstrap_texgen_bw.sh`, now only on branch
  `texgen-74k-v2-venus05`); the fir venv
  plus pin overlay, where the release run trained (`../docs/status/checkpoints.md:1637-1660`).
- [ ] The invariant across all of them: the four conditioning pins (transformers 4.28.1, diffusers
  0.28.0, huggingface_hub 0.25.2, tokenizers 0.13.3).
- [ ] Import probes to update if a dependency goes: `scripts/internal/fir/train_lightgenbench_alpha_fir.sh:220`,
  the probe in
  `../evaluation/newdata_eval/sweeps/star2/texgen_worker.sbatch`.

**Instructions:**

**Result:**

### 15. Release layer: generic entry points for LightgenBench users

The TEXGen code will be published for LightgenBench users to train and run (Dongchen, 2026-10-06),
so this layer is what they use. No user or cluster paths anywhere in it. Needs step 10 (the inference
script takes its paths as arguments); its README commands are written in step 16 (Docs).

- [ ] `scripts/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>`: the generic half of
  `stage_hf.sh` (untar the Hugging Face release into `<data_root>/<uuid>/somage.npz` and
  `thumbnails/<uuid>.png`), then `build_index.py`.
- [ ] One release config with the recipe values (batch 16 × accumulate 2, and the note that
  `TEXGEN_ENABLE_FLASH=0` and the four conditioning pins are required).
- [ ] The internal fir launchers stay as they are (turning them into thin wrappers would need a fir
  test run).

**Instructions:** 2026-10-06: "it will be published for lightgenbench users to train and run".

**Result:**

### 16. Docs

One doc of record, keeping upstream's attribution and citation (`README.md:103`).

- [ ] Inside: `README.md` (upstream; its command uses the missing `configs/texgen_test.yaml`;
  `static/teaser.jpg` goes with it), `CLAUDE.md` (no alpha channel, paper or release run; the
  `requirements_125.txt` claim at `:15` and the DDP sampler claim at `:186-190` are wrong),
  `LIGHTGEN_WORKFLOW.md` (stale), the "4 Pro 6000" comment at
  `configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml:89`.
- [ ] Parent: `../docs/baselines/texgen.md` (the whole file describes the vanilla `_v2` run as current;
  `:53-61` says the flash gate is only on the branch), `../.claude/rules/texgen.md`,
  `../AGENTS.md:64,212,228` (has uncommitted edits from another session),
  `../docs/status/refactor.md` (the TEXGen entry).
- [ ] The stale "main has no flash gate" comments: `../evaluation/newdata_eval/run_texgen_infer.py:15-18`,
  `../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:6`,
  `../evaluation/newdata_eval/sweeps/stage_texgen_host.sh:46`.
- [ ] Keep or retire `../evaluation/pointsampled/texgen_infer.py` (Keep stable #1).
- [ ] The release layer's commands (step 15) in the README: single-node training (`python launch.py
  --config ... --gpu 0,1,2,3 --train data.data_root=...`) and inference (`inference_specific_samples.py`)
  on the release test split.

**Instructions:**

**Result:**

### 17. Cut-over (optional, per D8)

Move the consumers to the refactored `main`: the `texgen-v2` worktree, the `BRANCH=` lines in
`scripts/internal/fir/train_lightgenbench_alpha_fir.sh:61,157` (the paper launcher's copy lives only on the
branch), the branch check in
`../evaluation/newdata_eval/infer_texgen_v2_venus05.sbatch:78-91`; rebuild the staged `TEXGen.tar`
on jupiter and delete the Solar nodes' `.tree_ok` so they unpack it again.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **Flash default.** `spuv/models/sparse_networks/texgen_network.py:49` treats an unset
  `TEXGEN_ENABLE_FLASH` as on. Both published runs used the dense path (`=0`); where flash_attn
  imports (4090, H100) forgetting the variable silently changes the numbers, and on Blackwell it
  asserts (`ptv3_model_texgen.py:294`). Options: default to dense, or fail when unset.
- [ ] **Release batch.** The release YAML says batch 32; the run used 16 × 2 from the launcher.
- [ ] **Fixes from `restructure-2026-09-16`** (`b4e423f`, `e20b015`, `4ef35a7`; re-apply by hand, the
  files differ): EMA weights on the test pass, the wandb directory, a non-zero exit on interrupt, the
  resume order (and one checkpoint read instead of two), validation metrics synced across GPUs, the
  training preview panel, the removed noise schedule; the loss guard (`309d3ab`, `bfb2135`;
  it rejects a nonzero loss weight that no code reads), which fits after step 8c.
- [ ] **Tests from `restructure-2026-09-16`** (12 files, including a toy CPU fit/resume test).

## Due at the merge into `main` (parent repo)

The parent's docs describe `TEXGen/` on `main`; they change when `refactor` merges and the pointer is
bumped, not before.

- [ ] Step 6: `../docs/baselines/texgen.md:12-30,48` (the config table and the `_v2` config row) and
  `../.claude/rules/texgen.md` (the 74k and overfit commands, the overfit configs line, the
  λ = 0.01 config in Losses) name deleted configs.

## Out of scope

The position frame (atlas [−1, 1] decoded as [−2, 2] vs voxels [−0.5, 0.5]): a data-format change,
tracked in `../docs/status/refactor.md`.
