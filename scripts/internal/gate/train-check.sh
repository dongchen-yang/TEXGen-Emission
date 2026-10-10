set -euo pipefail
# usage: train-check.sh <texgen_root> <out_dir> fresh|resume [<resume_ckpt>]   (absolute paths; prints TRAINCHECK_DONE)
#   fresh   launch.py --train from scratch: 2 epochs x 16 steps, validation after each, trainer.test, one save
#   resume  launch.py --train resume=<ckpt>: one more epoch (16 steps) from a real checkpoint, default the
#           release checkpoint epoch=124-step=35625, then validation, trainer.test and one save
# Real launch.py, real config (traincheck/traincheck.yaml extends the release config), deterministic
# mode, and a reseed right before trainer.fit; traincheck/sitecustomize.py records every step to
# <out_dir>/record.jsonl. Compare two records with compare_traincheck.py.
# Host overrides as in gate-det.sh: GATE_LOCK, GATE_LOCK_WAIT, GATE_ACTIVATE, GATE_L; TRAINCHECK_ROOT
# (the data root, default the Hugging Face check root make_check_root.py builds), TRAINCHECK_TMPBASE
# (a short directory on a disk with room for a 12.4 GB temp file).
# KEEP_CKPT=1 keeps the 12.4 GB checkpoint the run writes (deleted by default, also when the run aborts).
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
WAIT=${GATE_LOCK_WAIT:-10800}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"
  if [ "$WAIT" = 0 ]; then flock -n 9 || { echo "ABORT: the GPU lock $LOCK is held"; exit 7; }
  else flock -w "$WAIT" 9 || { echo "ABORT: could not take the GPU lock $LOCK within $WAIT s"; exit 7; }; fi
  echo "[gpu-lock] taken $(date +%H:%M:%S)"
fi
ROOT=${1:?texgen root}; OUT=${2:?out dir}; MODE=${3:?fresh or resume}
L=${GATE_L:-/local-scratch/localhome/dya78/lightgen_eval_local}
CKPT=${4:-/local-scratch/localhome/dya78/code/lightgen/TEXGen-Emission/outputs/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt}
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export TRAINCHECK_ROOT=${TRAINCHECK_ROOT:-$L/refactor_checks/hf_root} TRAINCHECK_OUT=$OUT
export TEXGEN_TRAINCHECK_OUT=$OUT/record.jsonl
[ -e "$TEXGEN_TRAINCHECK_OUT" ] && { echo "ABORT: $TEXGEN_TRAINCHECK_OUT exists; use a new out dir"; exit 2; }
mkdir -p "$OUT"
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/texgen; fi
set -u
export TEXGEN_GATE_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$HERE/traincheck"
# The device this run used: compare a run only with a baseline from the same GPU type.
echo "host=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")" | tee "$OUT/gpu.txt"
# Lightning writes a checkpoint through a temp file in $TMPDIR, and the workstation's /tmp has < 12 GB
# free. The dataloader workers put sockets there, so the path must stay short (AF_UNIX limit).
mkdir -p "${TRAINCHECK_TMPBASE:=/local-scratch/localhome/dya78/tmp}"
TMPDIR=$(mktemp -d "$TRAINCHECK_TMPBASE/tc.XXXXXX") && [ -d "$TMPDIR" ] \
  || { echo "ABORT: mktemp -d under $TRAINCHECK_TMPBASE failed"; exit 2; }
export TMPDIR
# On every exit, the aborts below included: the temp dir goes, and so does the saved checkpoint
# (12.4 GB) unless KEEP_CKPT=1; its summary is in the record by then.
trap 'rm -rf "$TMPDIR"; [ "${KEEP_CKPT:-0}" = 1 ] || find "$OUT/ckpts" -name "*.ckpt" -delete 2>/dev/null || true' EXIT
# A tree that converts checkpoints (scripts/convert_config.py --ckpt) resumes from a converted copy
if [ "$MODE" = resume ]; then
  CKPT=$(bash "$HERE/ckpt_for_tree.sh" "$ROOT" "$CKPT") || { echo "ABORT: checkpoint conversion failed"; exit 2; }
  echo "[ckpt] resume from $CKPT"
fi
cd "$ROOT"
case "$MODE" in
  fresh)  python launch.py --config "$HERE/traincheck/traincheck.yaml" --gpu 0 --train ;;
  resume) python launch.py --config "$HERE/traincheck/traincheck.yaml" --gpu 0 --train \
            resume="$CKPT" trainer.max_epochs=126 checkpoint.every_n_epochs=1 ;;
  *) echo "mode must be fresh or resume"; exit 2 ;;
esac
grep -q '"kind": "test"' "$TEXGEN_TRAINCHECK_OUT" || { echo "ABORT: no test record in $TEXGEN_TRAINCHECK_OUT"; exit 3; }
# A NaN loss does not stop Lightning; a record that holds one is not a usable run.
env -u TEXGEN_TRAINCHECK_OUT python -c 'import json, math, sys
bad = [r["gs"] for r in map(json.loads, open(sys.argv[1])) if r["kind"] == "train" and not math.isfinite(r["loss"])]
sys.exit(f"ABORT: non-finite training loss at global steps {bad}" if bad else 0)' "$TEXGEN_TRAINCHECK_OUT"
SAVED=$(find "$OUT/ckpts" -name '*.ckpt' | sort)
[ "$(echo "$SAVED" | grep -c .)" = 1 ] || { echo "ABORT: expected one checkpoint in $OUT/ckpts, got: $SAVED"; exit 3; }
env -u TEXGEN_TRAINCHECK_OUT python "$HERE/traincheck/summarize_ckpt.py" "$SAVED" >> "$TEXGEN_TRAINCHECK_OUT"
echo TRAINCHECK_DONE
