set -euo pipefail
# usage: gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file> [<data_root> <parquet>]]   (absolute paths)
# Defaults: the workstation, its 4090 lock and the paper gate data. Another host or data set
# overrides them (all optional): GATE_LOCK (a lock file, or "none" inside a Slurm job),
# GATE_ACTIVATE (a script to source that activates the texgen env), GATE_REPO (the parent lightgen
# checkout holding evaluation/newdata_eval/run_texgen_infer.py), GATE_L (the eval-local dir),
# GATE_PAPER_ROOT (the paper gate root, default $GATE_L/refactor_checks/paper_root),
# GATE_LOCK_WAIT (seconds to wait for the lock, default 10800; 0 tries once and exits 7 if held).
# 3, 4 or 6 arguments: a data root without its parquet would pair it with the paper parquet.
case $# in 3|4|6) ;; *) echo "usage: gate-det.sh <texgen_root> <out_dir> <ckpt> [<shas_file> [<data_root> <parquet>]]"; exit 2 ;; esac
# One GPU job at a time on the workstation 4090, shared with other sessions' gates (flock waits up to 3 h).
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
WAIT=${GATE_LOCK_WAIT:-10800}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"
  if [ "$WAIT" = 0 ]; then flock -n 9 || { echo "ABORT: the GPU lock $LOCK is held"; exit 7; }
  else flock -w "$WAIT" 9 || { echo "ABORT: could not take the GPU lock $LOCK within $WAIT s"; exit 7; }; fi
  echo "[gpu-lock] taken $(date +%H:%M:%S)"
fi
ROOT=${1:?texgen root}; OUT=${2:?out dir}; CKPT=${3:?checkpoint}
REPO=${GATE_REPO:-/local-scratch/localhome/dya78/code/lightgen}
L=${GATE_L:-/local-scratch/localhome/dya78/lightgen_eval_local}
# The paper gate data: the paper-era test bake with alpha inside each npz (make_paper_root.py).
P=${GATE_PAPER_ROOT:-$L/refactor_checks/paper_root}
# Optional 4th argument: a shapes list (one sha per line). Default: all 204 test shapes (~24 min);
# scripts/internal/gate/subset20.txt runs 20 of them (~3 min). Each shape's seed depends only on its sha.
SHAS=${4:-$P/test_agentic_clean.txt}
# Optional 5th and 6th: the data root and its parquet. Default: the paper gate root. The release
# gate passes the root make_check_root.py builds and its df_lightgenbench.parquet.
DATA=${5:-$P}; PARQUET=${6:-$P/eval_parquet_test204.parquet}
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/texgen; fi
set -u
export TEXGEN_GATE_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/detmode"
# The device this run used: compare a run only with a baseline from the same GPU type.
mkdir -p "$OUT"
echo "host=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}")" | tee "$OUT/gpu.txt"
# A tree that ships scripts/convert_config.py reads current names only: inference gets a temporary
# copy of the run directory, its configs/parsed.yaml converted and the checkpoint linked. Older
# trees (pre-refactor) read the run's own config.
if [ -f "$ROOT/scripts/convert_config.py" ]; then
  RUN=$(mktemp -d); trap 'rm -rf "$RUN"' EXIT
  mkdir -p "$RUN/ckpts" "$RUN/configs"
  python "$ROOT/scripts/convert_config.py" "$(dirname "$(dirname "$CKPT")")/configs/parsed.yaml" "$RUN/configs/parsed.yaml"
  SRC=$(bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/ckpt_for_tree.sh" "$ROOT" "$CKPT")   # converted, cached
  ln -s "$SRC" "$RUN/ckpts/$(basename "$CKPT")"
  CKPT=$RUN/ckpts/$(basename "$CKPT")
  echo "[config] converted with $ROOT/scripts/convert_config.py: $CKPT"
fi
python $REPO/evaluation/newdata_eval/run_texgen_infer.py \
  --texgen_root $ROOT --ckpt $CKPT \
  --data_root $DATA --parquet $PARQUET \
  --shas_file $SHAS --seed 0 --out_dir $OUT
echo GATE_DONE
