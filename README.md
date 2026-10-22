# TEXGen-Emission

TEXGen-Emission generates a mesh's emission map in UV space from its geometry, its PBR materials and an image of the object, and is built on [TEXGen](https://github.com/CVMI-Lab/TEXGen).


## Environment Setup
Create the environment and install the packages.
Run every command in this README from the root of the repository.
```bash
# Create and activate the environment
conda create -n texgen python=3.10 -y
conda activate texgen

# torchsparse needs the sparsehash headers; torchsparse and nvdiffrast compile for your GPU with the CUDA toolkit (nvcc)
sudo apt-get install libsparsehash-dev
export TORCH_CUDA_ARCH_LIST=<arch>    # your GPU's compute capability, such as 8.9

# Install torch first, then the rest
# (for another CUDA version, change the index URL here and spconv-cu118 in requirements.txt)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Set these before training and prediction: the dense attention branch, and less memory fragmentation
export TEXGEN_ENABLE_FLASH=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```


## Data Setup
The dataset download is described at https://github.com/3dlg-hcvc/lightgenbench; this code reads `splits.json`, `checksums.sha256` and the `atlas` and `thumbnail` tars.
Unpack the download into a data folder:
```bash
# Check the tars, unpack the atlases and thumbnails, and write the split index files
bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>

# Training and prediction read the data folder from this variable
export LIGHTGENBENCH_DATA=<data_dir>
```


## Prediction with Our Checkpoint
This needs the Environment Setup and Data Setup above; our checkpoint is in the Hugging Face repo [`3dlg-hcvc/LightgenBench-baselines`](https://huggingface.co/3dlg-hcvc/LightgenBench-baselines/tree/main/TEXGen-Emission).
Predict the emission of the test split with it; the script writes predictions to `<out_dir>`.
```bash
# Download our checkpoint and its config into checkpoints/TEXGen-Emission/
huggingface-cli download 3dlg-hcvc/LightgenBench-baselines --include "TEXGen-Emission/*" --local-dir checkpoints

# List the test shapes
python -c "import json, sys; print('\n'.join(json.load(open(sys.argv[1]))['test']))" \
    <download_dir>/splits.json > test_uuids.txt

# Predict the test shapes
python inference_specific_samples.py \
    --ckpt checkpoints/TEXGen-Emission/ckpts/texgen-emission.ckpt \
    --data_root "$LIGHTGENBENCH_DATA/texgen_root" --parquet "$LIGHTGENBENCH_DATA/df_lightgenbench.parquet" \
    --shas_file test_uuids.txt --out_dir <out_dir> --seed 0
```
Score the predictions with the LightGenBench evaluation: https://github.com/3dlg-hcvc/lightgenbench


## Training
The network trains from scratch, not from TEXGen's weights; only its CLIP image and text encoders are pretrained, and they download from Hugging Face on first use and stay frozen.
Checkpoints go to `outputs/lightgen/texgen_alpha_lightgenbench_v1/ckpts/`, and the same command resumes from the last one.
```bash
# Train on 4 GPUs (global batch 128)
python launch.py --config configs/lightgen_pointuv_256_batch32_lightgenbench_alpha.yaml --gpu 0,1,2,3 --train

# On another number of GPUs, keep GPUs x data.batch_size x trainer.accumulate_grad_batches = 128, for example on 8 GPUs
python launch.py --config configs/lightgen_pointuv_256_batch32_lightgenbench_alpha.yaml --gpu 0,1,2,3,4,5,6,7 --train \
    trainer.accumulate_grad_batches=1
```


## Prediction with Your Checkpoint
The same as Prediction with Our Checkpoint, with a checkpoint from Training (choose it on the validation split).
The script reads the model's config from the run folder, so keep the checkpoint in its `ckpts/` folder.
```bash
# Predict the test shapes (test_uuids.txt as in Prediction with Our Checkpoint)
python inference_specific_samples.py \
    --ckpt outputs/lightgen/texgen_alpha_lightgenbench_v1/ckpts/<checkpoint>.ckpt \
    --data_root "$LIGHTGENBENCH_DATA/texgen_root" --parquet "$LIGHTGENBENCH_DATA/df_lightgenbench.parquet" \
    --shas_file test_uuids.txt --out_dir <out_dir> --seed 0
```


## License
TEXGen's code has no license from its authors, so they keep all rights to it; our changes in this fork are released under the MIT license.
nvdiffrast, which this code uses, has its own license, the [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).


## Citation
If you use this code, please cite TEXGen:
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
