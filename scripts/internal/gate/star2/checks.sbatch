#!/bin/bash
# The refactor checks on one Solar node: the paper gate split across its GPUs, the release gate and
# the training check, for one TEXGen tree. GPU type = whatever the node has; compare a run only with
# runs on the same type (REFACTOR_TODO.md, How we work).
#
# Staging (once per node, flock'd, markers): inputs come from jupiter (read-only) onto the node's
# /localscratch/dya78/refactor_checks: the paper gate data (solar_texgen_stage/eval_local, checked
# equal to the workstation's atlas_root for all 204 test shapes on 2026-10-06), both checkpoints
# with their configs (sha256-checked), and the release check root (make_check_root.py). The tree
# under test and this tooling live on the shared $HOME:
#   rsync -a <tree>/ star2:refactor_checks/trees/<name>/          (e.g. git archive of a commit)
#   rsync -a --delete scripts/ star2:refactor_checks/tooling/scripts/          (this tooling, repo layout)
#   rsync -a ../evaluation/newdata_eval/run_texgen_infer.py star2:refactor_checks/tooling/repo/evaluation/newdata_eval/
#
# Submit: never `sbatch --export` on star2. Bake the placeholders with sed, feed sbatch on stdin:
#   sed -e 's|__TREE__|pre-refactor|' -e 's|__LABEL__|l40s_pre-refactor|' -e 's|__NGPU__|4|' \
#       -e 's|__PAPER_RUNS__|1 2|' -e 's|__RELEASE_RUNS__|1 2|' -e 's|__TRAIN_RUNS__|fresh:1 fresh:2 resume:1 resume:2|' \
#       -e 's|__GRES__|l40s:4|' -e 's|__NCPU__|16|' -e 's|__MEM__|300G|' -e 's|__TIME__|02:00:00|' \
#       scripts/internal/gate/star2/checks.sbatch > /tmp/checks.sbatch
#   ssh star2 'sbatch -w cs-venus-16' < /tmp/checks.sbatch
# __TREE__ is a name under ~/refactor_checks/trees/; __PAPER_RUNS__ / __RELEASE_RUNS__ are run numbers
# (run r sends shape k of the list to GPU (k + r) mod NGPU, so two runs put each shape on different
# GPUs); __TRAIN_RUNS__ are mode:run pairs; "" skips a kind.
# Output: ~/refactor_checks/out/__LABEL__/ — paper_test204_run<r>/ and release_test20_run<r>/ (the
# four gate PNGs per shape), train_<mode>_run<r>.jsonl (the training record), logs/.
#SBATCH --job-name=refactor_checks___LABEL__
#SBATCH --gres=gpu:__GRES__
#SBATCH --cpus-per-task=__NCPU__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=/home/dya78/refactor_checks/logs/checks-%j.out
set -uo pipefail
TREE_NAME="__TREE__"; LABEL="__LABEL__"; NGPU=__NGPU__
PAPER_RUNS="__PAPER_RUNS__"; RELEASE_RUNS="__RELEASE_RUNS__"; TRAIN_RUNS="__TRAIN_RUNS__"
H=$HOME/refactor_checks; TREE=$H/trees/$TREE_NAME; TOOL=$H/tooling; GT=$TOOL/scripts/internal/gate
LS=/localscratch/dya78/refactor_checks; EL=$LS/eval_local; RROOT=$LS/release_root
JUP=/3dlg-jupiter-project/lightgen; EVIN=$JUP/trellis2_bw/eval_inputs
OUT=$LS/out/$LABEL; HOUT=$H/out/$LABEL
PAPER_CKPT=$LS/ckpts/texgen_alpha_74k_v2_agentic/ckpts/epoch=124-step=35500.ckpt
REL_CKPT=$LS/ckpts/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt
say() { echo "[checks $(date +%H:%M:%S)] $*"; }
die() { echo "CHECKS_FAIL label=$LABEL node=$(hostname -s) job=${SLURM_JOB_ID:-} reason=$*"; exit 1; }
say "node=$(hostname -s) job=${SLURM_JOB_ID:-} tree=$TREE label=$LABEL gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
# The shared env's torchsparse carries sm_89 kernels only: an A40 (sm_86) fails with "no kernel image".
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[ "$CC" = 8.9 ] || die "compute capability $CC; this env runs on sm_89 only (L40S, RTX 6000 Ada)"
[ -d "$TREE" ] && [ -f "$GT/gate-det.sh" ] && [ -f "$TOOL/repo/evaluation/newdata_eval/run_texgen_infer.py" ] || die "tree or tooling missing under $H"
[ -e "$OUT" ] && die "$OUT exists; use a new label"
mkdir -p "$LS/ckpts" "$OUT/logs" "$HOUT/logs" || die "mkdir"

# ---- environment: the shared texgen env (a conda-pack of the workstation env) ----
ACT=$OUT/activate.sh
cat > "$ACT" <<EOF
source $HOME/envs/texgen/bin/activate
export PATH=$HOME/envs/texgen/bin:\$PATH LD_LIBRARY_PATH=$HOME/envs/texgen/lib:\${LD_LIBRARY_PATH:-}
export HF_HOME=$EVIN/hf_home HF_HUB_CACHE=$EVIN/hf_home/hub TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCH_HOME=$HOME/torch_home_texgen TORCH_EXTENSIONS_DIR=$HOME/torch_ext_texgen MAX_JOBS=4
EOF

# ---- staging (once per node) ----
exec 8>"$LS/.stage.lock"; flock 8
if [ ! -f "$EL/.ok" ]; then
  say "stage: paper gate data"; rm -rf "$EL"; cp -dR "$EVIN/solar_texgen_stage/eval_local" "$EL" || die "eval_local copy"; touch "$EL/.ok"
fi
stage_ckpt() {  # run file sha256
  local d=$LS/ckpts/$1
  [ -f "$d/ckpts/.ok_$2" ] && return 0
  mkdir -p "$d/ckpts" "$d/configs"
  cp "$JUP/outputs/$1/configs/parsed.yaml" "$d/configs/" || return 1
  say "stage: $1/$2"; cp "$JUP/outputs/$1/ckpts/$2" "$d/ckpts/$2.part" && mv "$d/ckpts/$2.part" "$d/ckpts/$2" || return 1
  [ "$(sha256sum "$d/ckpts/$2" | cut -d' ' -f1)" = "$3" ] || { echo "sha256 mismatch $2"; return 1; }
  touch "$d/ckpts/.ok_$2"
}
stage_ckpt texgen_alpha_74k_v2_agentic epoch=124-step=35500.ckpt 2e52305fd352a931f20d03e942b2b4e85665218d198d9e29cfe5c56ab61249f6 || die "paper ckpt"
stage_ckpt texgen_alpha_lightgenbench_v1 epoch=124-step=35625.ckpt 6c42a311f5925f5bcfd87a1bb68592a76ae71d53ce0efe3c90e880bf8e3892aa || die "release ckpt"
( set +u; source "$ACT"; set -u; python "$GT/make_check_root.py" --release "$JUP/datasets/dataset_release" --out "$RROOT" ) || die "release root"
flock -u 8
( set +u; source "$ACT"; set -u; python -c "import torch; print('[probe] torch', torch.__version__, torch.cuda.device_count(), 'GPUs')" ) || die "env probe"

# ---- tasks, run in waves of NGPU (task j of a wave on GPU j) ----
TASKS=()
SHAS=$EL/test_agentic_clean.txt
for r in $PAPER_RUNS; do
  mkdir -p "$OUT/shas"
  for g in $(seq 0 $((NGPU-1))); do
    awk -v r="$r" -v g="$g" -v n="$NGPU" 'NF && ((NR - 1 + r) % n) == g' "$SHAS" > "$OUT/shas/paper_run${r}_g$g.txt"
    TASKS+=("paper_run${r}_g$g|gate|$OUT/paper_test204_run$r.parts/g$g|$PAPER_CKPT|$OUT/shas/paper_run${r}_g$g.txt")
  done
done
for mr in $TRAIN_RUNS; do TASKS+=("train_${mr%%:*}_run${mr##*:}|train|$OUT/train_${mr%%:*}_run${mr##*:}|${mr%%:*}"); done
for r in $RELEASE_RUNS; do TASKS+=("release_run$r|gate|$OUT/release_test20_run$r|$REL_CKPT|$RROOT/release_test20.txt|$RROOT|$RROOT/df_lightgenbench.parquet"); done
# SLURM_JOB_NAME=bash: Lightning then ignores the Slurm job (as scripts/internal/fir/infer_lightgenbench_fir.sh does)
# and every task runs single-process exactly as on the workstation.
export GATE_LOCK=none GATE_ACTIVATE=$ACT GATE_REPO=$TOOL/repo GATE_L=$EL TRAINCHECK_ROOT=$RROOT TRAINCHECK_TMPBASE=/localscratch/dya78/tmp SLURM_JOB_NAME=bash
RC=0
for ((w = 0; w < ${#TASKS[@]}; w += NGPU)); do
  PIDS=()
  for ((j = 0; j < NGPU && w + j < ${#TASKS[@]}; j++)); do
    IFS='|' read -r name kind dest a1 a2 a3 a4 <<< "${TASKS[$((w + j))]}"
    say "launch $name on gpu $j"
    if [ "$kind" = gate ]; then
      CUDA_VISIBLE_DEVICES=$j bash "$GT/gate-det.sh" "$TREE" "$dest" "$a1" "$a2" ${a3:+"$a3"} ${a4:+"$a4"} > "$OUT/logs/$name.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$j bash "$GT/train-check.sh" "$TREE" "$dest" "$a1" "$REL_CKPT" > "$OUT/logs/$name.log" 2>&1 &
    fi
    PIDS+=("$!:$name")
  done
  for p in "${PIDS[@]}"; do
    wait "${p%%:*}"; rc=$?; say "${p#*:} exit=$rc $(tail -1 "$OUT/logs/${p#*:}.log")"; [ $rc -eq 0 ] || RC=1
  done
done

# ---- collect: merge the paper parts, copy the four gate PNGs per shape and the records to $HOME ----
for r in $PAPER_RUNS; do
  mkdir -p "$OUT/paper_test204_run$r"
  for g in $(seq 0 $((NGPU-1))); do mv "$OUT/paper_test204_run$r.parts/g$g"/*/ "$OUT/paper_test204_run$r/" 2>/dev/null; done
done
for d in "$OUT"/paper_test204_run* "$OUT"/release_test20_run*; do
  case "$d" in *.parts) continue ;; esac
  [ -d "$d" ] || continue
  for s in "$d"/*/; do
    s=${s%/}; mkdir -p "$HOUT/$(basename "$d")/$(basename "$s")"
    cp "$s"/{pred_emission,gt_emission,mask,input_albedo}.png "$HOUT/$(basename "$d")/$(basename "$s")/" 2>/dev/null
  done
done
for mr in $TRAIN_RUNS; do
  n=train_${mr%%:*}_run${mr##*:}; cp "$OUT/$n/record.jsonl" "$HOUT/$n.jsonl" 2>/dev/null || RC=1
done
cp "$OUT"/logs/*.log "$HOUT/logs/"
for d in "$HOUT"/paper_test204_run* "$HOUT"/release_test20_run*; do
  [ -d "$d" ] && say "$(basename "$d"): $(find "$d" -name pred_emission.png | wc -l) pred_emission.png"
done
[ $RC -eq 0 ] && echo "CHECKS_DONE label=$LABEL" || { echo "CHECKS_INCOMPLETE label=$LABEL"; exit 2; }
