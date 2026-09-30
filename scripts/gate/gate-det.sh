set -euo pipefail
# One GPU job at a time on the workstation 4090, shared with other sessions' gates (flock waits up to 3 h).
exec 9>/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock
flock -w 10800 9 || { echo 'ABORT: could not take the 4090 lock within 3 h'; exit 7; }
echo "[gpu-lock] taken $(date +%H:%M:%S)"
ROOT=${1:?texgen root}; OUT=${2:?out dir}; CKPT=${3:?checkpoint}
REPO=/local-scratch/localhome/dya78/code/lightgen
L=/local-scratch/localhome/dya78/lightgen_eval_local
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +u; source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/texgen; set -u
export TEXGEN_GATE_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/detmode"
python $REPO/evaluation/newdata_eval/run_texgen_infer.py \
  --texgen_root $ROOT --ckpt $CKPT \
  --data_root $L/atlas_root --parquet $L/atlas_root/eval_parquet_test204.parquet \
  --shas_file $L/test_agentic_clean.txt --seed 0 --out_dir $OUT
echo GATE_DONE
