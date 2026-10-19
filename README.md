# TEXGen for emission maps

This repository adapts **TEXGen** from albedo to **emission**. Given a mesh's geometry and PBR
materials baked into a UV atlas, and an image of the object, it generates the RGB emission map in UV
space. It is the TEXGen baseline of LightgenBench
([Hugging Face `3dlg-hcvc/LightgenBench`](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)).

TEXGen and most of this code are the work of its authors:

> **TEXGen: a Generative Diffusion Model for Mesh Textures** (SIGGRAPH Asia 2024, Journal Track, Best
> Paper Honorable Mention). Xin Yu, Ze Yuan, Yuan-Chen Guo, Ying-Tian Liu, JianHui Liu, Yangguang Li,
> Yan-Pei Cao, Ding Liang, Xiaojuan Qi. The University of Hong Kong, VAST, Beihang University,
> Tsinghua University.
> [Paper](https://arxiv.org/abs/2411.14740) · [Project page](https://cvmi-lab.github.io/TEXGen/) ·
> [Code](https://github.com/CVMI-Lab/TEXGen)

If you use this code, please cite TEXGen ([below](#citation)).

## What differs from TEXGen

- **Input.** Upstream bakes image features onto the mesh's UV map at run time. Here the loader
  (`spuv/data/mesh_uv.py`) reads pre-baked 256 × 256 UV atlases (`somage.npz`), and the backbone
  (`PointUVNet.forward`, `spuv/models/sparse_networks/texgen_network.py`) takes the noisy emission with
  position, albedo, metallic, roughness, alpha and the occupancy mask: 13 channels.
- **Target.** The emission map. Training (`spuv/systems/texgen_train.py`) uses MSE and L1 on the
  flow-matching velocity, masked to occupied texels.
- **Condition.** The CLIP embedding of the object's thumbnail, dropped 10% of the time for
  classifier-free guidance.
- **Timesteps.** The backbone receives t in [0, 1], not upstream's t × 1000; our models were trained so.

The upstream files we changed are edited in place, so `git diff` against upstream shows every change:

```bash
git remote add upstream https://github.com/CVMI-Lab/TEXGen && git fetch upstream
git diff upstream/main -- spuv/
```

## Setup

```bash
conda create -n texgen python=3.10 -y && conda activate texgen
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

torchsparse and nvdiffrast are compiled for your GPU (set `TORCH_CUDA_ARCH_LIST`), and torchsparse
needs the sparsehash headers (`apt-get install libsparsehash-dev`); spconv is a prebuilt wheel per CUDA
version (`spconv-cu118` in the file). The four pins at the top of
`requirements.txt` are required. Our runs used torch 2.1.0 + CUDA 11.8 (RTX 4090), torch 2.9.1 + CUDA
12.8 (RTX PRO 6000 Blackwell) and torch 2.9.0 + CUDA 12.6 (H100).

Run every command below from the repository root, with

```bash
export TEXGEN_ENABLE_FLASH=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`TEXGEN_ENABLE_FLASH=0` selects the attention all our models were trained and evaluated with;
`expandable_segments` avoids out-of-memory errors from fragmentation. On first use the code downloads
two models from Hugging Face: `lambdalabs/sd-image-variations-diffusers` (CLIP image encoder, noise
schedule) and the CLIP tokenizer and text encoder of `stabilityai/stable-diffusion-3.5-large`, which is
gated: accept its license on its page first.

## Data

LightgenBench is gated: accept its terms on the dataset page, then log in (`huggingface-cli login`).

```bash
huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
    --include "splits.json" "checksums.sha256" "data/*/atlas/*.tar" "data/*/thumbnail/*.tar"
bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
export LIGHTGENBENCH_DATA=<data_dir>
```

The script checks every tar against `checksums.sha256`, unpacks the atlases and thumbnails, writes the
two index files the loader reads (a table of the shapes and their folders, and the train/val/test
row numbers into it, made from `splits.json`), and checks that every shape of the split is on disk. That is 36,426 train /
200 val / 200 test shapes: about 70 GB to download and 75 GB unpacked.

## Training

```bash
python launch.py --config configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml \
    --gpu 0,1,2,3 --train
```

This is the recipe of our release run: 4 GPUs × micro-batch 16 × gradient accumulation 2 = global
batch 128, 125 epochs (35,625 steps), EMA, bf16. On 4 × H100 80 GB an epoch took about 18 minutes before
validation, with a peak of 51 GB per GPU. Outputs go to `outputs/lightgen/texgen_alpha_lightgenbench_v1/`:
a checkpoint every 5 epochs in `ckpts/` (12.4 GB each, about 310 GB in all) plus `last.ckpt`, from
which the same command resumes. Training logs to TensorBoard (`tb_logs/`); add `--wandb` to log to
Weights & Biases as well (project `LightGen`; set `wandb.project=<yours>`).

With another number or size of GPUs, keep the global batch at 128: GPUs × `data.batch_size` ×
`trainer.accumulate_grad_batches` = 128, for example `trainer.accumulate_grad_batches=1` on 8 GPUs, or a
smaller `data.batch_size` with more accumulation on smaller cards. Any `key=value` after the command
overrides the config.

Do not choose a checkpoint by `val/psnr`: emission is sparse, so predicting little light scores well,
and the metric peaks in the first epochs. We score the last epoch.

## Inference on the test split

```bash
python -c "import json, sys; print('\n'.join(json.load(open(sys.argv[1]))['test']))" \
    <download_dir>/splits.json > test_uuids.txt
python inference_specific_samples.py \
    --ckpt outputs/lightgen/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt \
    --data_root "$LIGHTGENBENCH_DATA/texgen_root" --parquet "$LIGHTGENBENCH_DATA/df_lightgenbench.parquet" \
    --shas_file test_uuids.txt --out_dir <out_dir> --seed 0
```

This writes `<out_dir>/<uuid>/{pred_emission,gt_emission,mask,input_albedo}.png`, 256 × 256 UV maps,
using the EMA weights, 50 Euler steps and guidance scale 2.0. The model's config is read from the
run's `configs/parsed.yaml`, beside `ckpts/`. Each shape's noise comes from its uuid and `--seed`, not
from its place in the list, but outputs are not bit-reproducible: a rerun can differ by 1/255 on a few
texels, and the same shape in a different list or order can differ more. A GPU with 24 GB is enough. The script prints each shape's PSNR and SSIM on the UV map; the
benchmark's own scoring is not part of this repository.

A checkpoint trained with this code before its October 2026 refactor (such as ours) names older
classes: convert its config and checkpoint first with `scripts/convert_config.py` (usage in its header).

## Repository

| path | contents |
|---|---|
| `configs/` | the release config |
| `spuv/` | the model, the loader, training and inference (upstream's package, edited in place) |
| `launch.py`, `inference_specific_samples.py` | training and test entry point; inference on chosen shapes |
| `scripts/lightgenbench/` | `unpack_hf_dataset.sh`: the Hugging Face dataset to the data folder |
| `scripts/convert_config.py` | older config and checkpoint class names to the current ones |

## Citation

```bib
@article{10.1145/3687909,
    author = {Yu, Xin and Yuan, Ze and Guo, Yuan-Chen and Liu, Ying-Tian and Liu, Jianhui and Li, Yangguang and Cao, Yan-Pei and Liang, Ding and Qi, Xiaojuan},
    title = {TEXGen: a Generative Diffusion Model for Mesh Textures},
    journal = {ACM Trans. Graph.},
    volume = {43},
    number = {6},
    year = {2024},
    issn = {0730-0301},
    doi = {10.1145/3687909},
    articleno = {213},
    numpages = {14},
    keywords = {generative model, texture generation}
}
```
