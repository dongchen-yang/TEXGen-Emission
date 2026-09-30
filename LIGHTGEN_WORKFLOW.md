# LightGen @ TEXGen — active workflow map

Short map of what to use for emission training/inference. Upstream TEXGen
docs remain in `README.md`; agent notes in `CLAUDE.md`. Archived experiments
were deleted 2026-09-29; git history keeps them (tag `pre-trim-2026-09-16`).

## Active configs (`configs/`)

### Three baseline variants (1k filtered + overfit)

Variant→config table: canonical copy in the parent repo at
[`docs/baselines/texgen.md`](../docs/baselines/texgen.md) (do not restate here).

### 74k production

- `lightgen_pointuv_256_batch32_emissive_74k.yaml` — primary 74k vanilla
- `lightgen_pointuv_256_batch32_emissive_74k_paperHP.yaml` — paper HP variant

### Mask-cls λ=0.01 (still relevant)

- `lightgen_pointuv_256_batch32_emission_filtered_mask_cls_lambda001.yaml`
- `lightgen_pointuv_256_batch32_emission_filtered_mask_cls_lambda001_balanced.yaml`
- `lightgen_pointuv_overfit_mask_cls_lambda001.yaml` — overfit recipe for λ=0.01

Also keep `configs/data_splits_overfit_single.json` for overfit splits.

## Train

```bash
# From TEXGen/, conda env: texgen
conda run -n texgen bash -c 'python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --train'

# Full 1k-style filtered baseline
conda run -n texgen bash -c 'python launch.py --config configs/lightgen_pointuv_256_batch32_emission_filtered.yaml --gpu 0 --train --wandb'

```

## Infer / evaluate samples

```bash
conda run -n texgen bash -c 'python launch.py --config <config.yaml> --gpu 0 --test system.resume=/path/to/checkpoint.ckpt'
conda run -n texgen bash -c 'python inference_specific_samples.py'
```

## Do not change casually

- `spuv/data/lightgen_uv.py` — atlas encoding contract (somage / uv_voxel_pipeline parity).
- Production checkpoints live on fir / Jupiter (`/cs/3dlg-jupiter-project/lightgen/TEXGen/outputs`), not in local `outputs/`.

## Env

- Training/inference: `conda run -n texgen ...`
- Parent LightGen evaluation (Blender/VLM): `lightgen` env — outside this submodule.
