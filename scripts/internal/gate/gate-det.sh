set -euo pipefail
# usage: gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file> [<data_root> <parquet>]]   (absolute paths)
# Defaults: the workstation, its 4090 lock and the paper gate data. Another host or data set
# overrides them (all optional): GATE_LOCK (a lock file, or "none" inside a Slurm job),
# GATE_ACTIVATE (a script to source that activates the texgen env), GATE_REPO (the parent lightgen
# checkout holding evaluation/newdata_eval/run_texgen_infer.py), GATE_L (the eval-local dir).
# 3, 4 or 6 arguments: a data root without its parquet would pair it with the paper parquet.
case $# in 3|4|6) ;; *) echo "usage: gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file> [<data_root> <parquet>]]"; exit 2 ;; esac
# One GPU job at a time on the workstation 4090, shared with other sessions' gates (flock waits up to 3 h).
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"
  flock -w 10800 9 || { echo "ABORT: could not take the GPU lock $LOCK within 3 h"; exit 7; }
  echo "[gpu-lock] taken $(date +%H:%M:%S)"
fi
ROOT=${1:?texgen root}; OUT=${2:?out dir}; CKPT=${3:?checkpoint}
REPO=${GATE_REPO:-/local-scratch/localhome/dya78/code/lightgen}
L=${GATE_L:-/local-scratch/localhome/dya78/lightgen_eval_local}
# Optional 4th argument: a shapes list (one sha per line). Default: all 204 test shapes (~24 min);
# scripts/internal/gate/subset20.txt runs 20 of them (~3 min). Each shape's seed depends only on its sha.
SHAS=${4:-$L/test_agentic_clean.txt}
# Optional 5th and 6th: the data root and its parquet. Default: the paper gate data. The release
# gate passes the root make_check_root.py builds and its df_lightgenbench.parquet.
DATA=${5:-$L/atlas_root}; PARQUET=${6:-$L/atlas_root/eval_parquet_test204.parquet}
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/texgen; fi
set -u
export TEXGEN_GATE_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/detmode"
python $REPO/evaluation/newdata_eval/run_texgen_infer.py \
  --texgen_root $ROOT --ckpt $CKPT \
  --data_root $DATA --parquet $PARQUET \
  --shas_file $SHAS --seed 0 --out_dir $OUT
echo GATE_DONE
