#!/bin/bash
# Line coverage of the code that produced the paper and release results (REFACTOR_TODO.md step 9).
#   coverage.sh <texgen_root> <out_dir>
# Runs under `python -m coverage`, on one GPU, TEXGEN_ENABLE_FLASH=0, no dataloader workers:
#   1. fresh training with the training-check config (2 epochs: training, validation, a checkpoint
#      save, the test pass) on the release check root;
#   2. a resume from the release checkpoint (converted for this tree), one more epoch;
#   3. inference on 5 paper-gate shapes (paper checkpoint) and 5 release shapes (release checkpoint).
# Then <out_dir>/report.txt lists, per file of spuv/, launch.py and inference_specific_samples.py,
# the lines that never ran; <out_dir>/coverage.json holds the same.
set -uo pipefail
ROOT=$(readlink -f "${1:?texgen root}"); OUT=$(readlink -f -m "${2:?out dir}")
GT=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
L=${GATE_L:-/local-scratch/localhome/dya78/lightgen_eval_local}
C=$L/refactor_checks
REPO=${GATE_REPO:-/local-scratch/localhome/dya78/code/lightgen}
CK=${COVERAGE_CKPTS:-$REPO/TEXGen-Emission/outputs}
PCK=$CK/texgen_alpha_74k_v2_agentic/ckpts/epoch=124-step=35500.ckpt
RCK=$CK/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT" 2>/dev/null)" ] || { echo "ABORT: $OUT is not empty"; exit 2; }
set +u   # conda's activation scripts read unset variables
source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh
conda activate /localhome/dya78/software/miniconda3/envs/texgen
set -u
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TRAINCHECK_ROOT=$C/hf_root
export COVERAGE_FILE=$OUT/.coverage
mkdir -p "${COVERAGE_TMPBASE:=/local-scratch/localhome/dya78/tmp}"
export TMPDIR=$(mktemp -d "$COVERAGE_TMPBASE/cov.XXXXXX")
trap 'rm -rf "$TMPDIR"; find "$OUT" -name "*.ckpt" -delete 2>/dev/null || true' EXIT
COV=(python -m coverage run --parallel-mode --source="$ROOT")
cd "$ROOT"

say() { echo "[coverage $(date +%H:%M:%S)] $*"; }
run() { local n=$1; shift; say "$n"; "$@" > "$OUT/$n.log" 2>&1; local rc=$?; say "$n exit=$rc"; return $rc; }

# 1. fresh training: training steps, validation, the save, the test pass
TRAINCHECK_OUT=$OUT/fresh run fresh "${COV[@]}" launch.py --config "$GT/traincheck/traincheck.yaml" --gpu 0 --train \
  data.num_workers=0

# 2. resume from the release checkpoint (converted for this tree when it ships the converter)
RES=$(bash "$GT/ckpt_for_tree.sh" "$ROOT" "$RCK")
TRAINCHECK_OUT=$OUT/resume run resume "${COV[@]}" launch.py --config "$GT/traincheck/traincheck.yaml" --gpu 0 --train \
  data.num_workers=0 resume="$RES" trainer.max_epochs=126 checkpoint.every_n_epochs=1

# 3. inference: 5 paper-gate shapes, 5 release shapes, each through a converted copy of its run dir
infer() {  # name ckpt data_root parquet shas
  local run=$TMPDIR/run_$1
  mkdir -p "$run/ckpts" "$run/configs"
  python "$ROOT/scripts/convert_config.py" "$(dirname "$(dirname "$2")")/configs/parsed.yaml" "$run/configs/parsed.yaml"
  ln -s "$(bash "$GT/ckpt_for_tree.sh" "$ROOT" "$2")" "$run/ckpts/$(basename "$2")"
  head -5 "$5" > "$TMPDIR/$1.shas"
  run "infer_$1" "${COV[@]}" "$REPO/evaluation/newdata_eval/run_texgen_infer.py" --texgen_root "$ROOT" \
    --ckpt "$run/ckpts/$(basename "$2")" --data_root "$3" --parquet "$4" --shas_file "$TMPDIR/$1.shas" \
    --seed 0 --out_dir "$OUT/infer_$1"
}
infer paper "$PCK" "$C/paper_root" "$C/paper_root/eval_parquet_test204.parquet" "$GT/subset20.txt"
infer release "$RCK" "$C/hf_root" "$C/hf_root/df_lightgenbench.parquet" "$C/hf_root/release_test20.txt"

python -m coverage combine "$OUT"
INC="$ROOT/spuv/*,$ROOT/launch.py,$ROOT/inference_specific_samples.py"
python -m coverage report --include="$INC" --show-missing --skip-empty > "$OUT/report.txt"
python -m coverage json --include="$INC" -o "$OUT/coverage.json" > /dev/null
say "report: $OUT/report.txt"
tail -1 "$OUT/report.txt"
