# LightGen @ TEXGen — active workflow map

Short map of what to use for emission training/inference. Upstream TEXGen
docs remain in `README.md`; agent notes in `CLAUDE.md`. Archived experiments
were deleted 2026-09-29; git history keeps them (tag `pre-trim-2026-09-16`).

## Active configs (`configs/`)

- `lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml` — the release retrain on
  LightgenBench, run by `scripts/internal/fir/train_lightgenbench_alpha_fir.sh`.

Every other config (the three 1k baseline variants, the overfit set, the 74k runs, the paper run's)
was deleted on 2026-10-06; tag `pre-refactor` keeps them, and the paper checkpoint carries its own
`configs/parsed.yaml`.

## Train

```bash
# The release retrain on fir (4× H100); submits the job, usage and probe run in its header
bash scripts/internal/fir/train_lightgenbench_alpha_fir.sh
```

## Infer / evaluate samples

```bash
conda run -n texgen bash -c 'python launch.py --config <config.yaml> --gpu 0 --test system.resume=/path/to/checkpoint.ckpt'
conda run -n texgen bash -c 'python inference_specific_samples.py --ckpt <run>/ckpts/<file>.ckpt --data_root <root> --parquet <root>/<index>.parquet --shas_file <shapes.txt> --out_dir <out>'
```

## Check a code change

The inference gate, its numeric comparison and the training check are in `scripts/internal/gate/`;
`python3 scripts/internal/gate/fanout.py --rev <commit>` runs all of them on every free GPU and
prints one verdict. Details in `CLAUDE.md` § Checking a code change.

## Do not change casually

- `spuv/data/mesh_uv.py` (the loader) — atlas encoding contract (somage / uv_voxel_pipeline parity).
  Published checkpoints' configs name it by its old path; `scripts/convert_config.py` converts them.
- Production checkpoints live on fir / Jupiter (`/cs/3dlg-jupiter-project/lightgen/TEXGen/outputs`), not in local `outputs/`.

## Env

- Training/inference: `conda run -n texgen ...`
- Parent LightGen evaluation (Blender/VLM): `lightgen` env — outside this submodule.
