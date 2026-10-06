#!/bin/bash
# The refactor checks on one Solar node's GPUs, for one TEXGen tree: the paper gate, the release gate
# and the training check. GPU type = whatever the node has; compare a run only with runs on the same
# type (REFACTOR_TODO.md, How we work). fanout.py (one directory up) writes, stages and submits this
# file for a check; by hand it serves baselines and noise runs.
#
# Staging (once per node, flock'd, markers): the paper gate data (jupiter solar_texgen_stage/eval_local,
# checked equal to the workstation's atlas_root for all 204 test shapes on 2026-10-06) and both
# checkpoints with their configs (sha256-checked) come from jupiter (read-only) onto the node's
# /localscratch/dya78/refactor_checks. The release check root (make_check_root.py, built from the
# Hugging Face release on the workstation) comes from the shared $HOME and is checked against its
# MANIFEST.sha256 on every run. The tree under test and this tooling live on the shared $HOME:
#   rsync -a <tree>/ star2:refactor_checks/trees/<name>/          (e.g. git archive of a commit)
#   rsync -a --delete scripts/ star2:refactor_checks/tooling/scripts/          (this tooling, repo layout)
#   rsync -a ../evaluation/newdata_eval/run_texgen_infer.py star2:refactor_checks/tooling/repo/evaluation/newdata_eval/
#   rsync -a --delete $L/refactor_checks/hf_root/ star2:refactor_checks/hf_root/
#
# Submit: never `sbatch --export` on star2. Bake the placeholders with sed, feed sbatch on stdin:
#   sed -e 's|__TREE__|pre-refactor|' -e 's|__LABEL__|l40s_pre-refactor|' -e 's|__NGPU__|4|' \
#       -e 's|__TASKS__||' -e 's|__PAPER_RUNS__|1 2|' -e 's|__RELEASE_RUNS__|1 2|' \
#       -e 's|__TRAIN_RUNS__|fresh:1 fresh:2 resume:1 resume:2|' \
#       -e 's|__GRES__|l40s:4|' -e 's|__NCPU__|24|' -e 's|__MEM__|256G|' -e 's|__TIME__|02:00:00|' \
#       scripts/internal/gate/star2/checks.sbatch > /tmp/checks.sbatch
#   python3 ~/.claude/skills/compute-access/scripts/hostprobe.py      # a node with that many free L40S, now
#   ssh star2 'sbatch --deadline=now+<time + 2 min>' < /tmp/checks.sbatch   # starts within 2 min or is cancelled
#   ssh star2 'sacct -N <node> -a -S now-15minutes -X -n -s PR'      # any line = a preemption: tell Dongchen
# No `-w`: Slurm picks a node with free L40S (`sbatch --test-only` routes it to 3dlg-hcvc-lab-debug).
# A GPU job pinned to one of our nodes with `-w` evicts another lab's running job there instead of
# waiting. Staging is per node, so a run on a node it has not used yet first copies ~30 GB (~2 min).
# Tasks: __TASKS__ names a task file under ~/refactor_checks/ (fanout.py writes it), one task per line
#   <gpu> <name> paper|release|fresh|resume [<shas file under ~/refactor_checks/>]
# and the run placeholders are then ignored. Without one: __PAPER_RUNS__ / __RELEASE_RUNS__ are run
# numbers (paper run r sends shape k of the list to GPU (k + r) mod NGPU, so two runs put each shape
# on different GPUs); __TRAIN_RUNS__ are mode:run pairs; "" skips a kind; task j of that list goes to
# GPU j mod NGPU. Each GPU runs its own tasks one after another.
# Output: ~/refactor_checks/out/__LABEL__/ — per task <name>/ (the four gate PNGs per shape, or
# record.jsonl; gpu.txt, the device it ran on), the run mode's paper parts merged into
# paper_test204_run<r>/, logs/, and status.txt (one line per task, then CHECKS_DONE or CHECKS_INCOMPLETE).
#SBATCH --job-name=refactor_checks___LABEL__
#SBATCH --gres=gpu:__GRES__
#SBATCH --cpus-per-task=__NCPU__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=/home/dya78/refactor_checks/logs/checks-%j.out
set -uo pipefail
TREE_NAME="__TREE__"; LABEL="__LABEL__"; NGPU=__NGPU__; TASKFILE="__TASKS__"
PAPER_RUNS="__PAPER_RUNS__"; RELEASE_RUNS="__RELEASE_RUNS__"; TRAIN_RUNS="__TRAIN_RUNS__"
H=$HOME/refactor_checks; TREE=$H/trees/$TREE_NAME; TOOL=$H/tooling; GT=$TOOL/scripts/internal/gate
LS=/localscratch/dya78/refactor_checks; EL=$LS/eval_local; RROOT=$LS/hf_root
JUP=/3dlg-jupiter-project/lightgen; EVIN=$JUP/trellis2_bw/eval_inputs
OUT=$LS/out/$LABEL; HOUT=$H/out/$LABEL
PAPER_CKPT=$LS/ckpts/texgen_alpha_74k_v2_agentic/ckpts/epoch=124-step=35500.ckpt
REL_CKPT=$LS/ckpts/texgen_alpha_lightgenbench_v1/ckpts/epoch=124-step=35625.ckpt
say() { echo "[checks $(date +%H:%M:%S)] $*"; }
SYNC="rsync -rlt"   # transfers use rsync, never cp (parent AGENTS.md); -rlt: no owner or permission bits
die() { echo "CHECKS_FAIL label=$LABEL node=$(hostname -s) job=${SLURM_JOB_ID:-} reason=$*"
        mkdir -p "$HOUT" && echo "CHECKS_FAIL $*" > "$HOUT/status.txt"; exit 1; }
say "node=$(hostname -s) job=${SLURM_JOB_ID:-} tree=$TREE label=$LABEL gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
# The shared env's torchsparse carries sm_89 kernels only: an A40 (sm_86) fails with "no kernel image".
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[ "$CC" = 8.9 ] || die "compute capability $CC; this env runs on sm_89 only (L40S, RTX 6000 Ada)"
[ -d "$TREE" ] && [ -f "$GT/gate-det.sh" ] && [ -f "$TOOL/repo/evaluation/newdata_eval/run_texgen_infer.py" ] || die "tree or tooling missing under $H"
[ -f "$H/hf_root/MANIFEST.sha256" ] || die "no release check root at $H/hf_root"
[ -z "$TASKFILE" ] || [ -f "$H/$TASKFILE" ] || die "no task file $H/$TASKFILE"
command -v rsync > /dev/null || die "no rsync on $(hostname -s)"
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

# ---- staging (once per node; the release check root on every run) ----
exec 8>"$LS/.stage.lock"; flock 8
if [ ! -f "$EL/.ok" ]; then
  say "stage: paper gate data"; rm -rf "$EL"; $SYNC "$EVIN/solar_texgen_stage/eval_local/" "$EL/" || die "eval_local copy"; touch "$EL/.ok"
fi
stage_ckpt() {  # run file sha256
  local d=$LS/ckpts/$1
  [ -f "$d/ckpts/.ok_$2" ] && return 0
  mkdir -p "$d/ckpts" "$d/configs"
  $SYNC "$JUP/outputs/$1/configs/parsed.yaml" "$d/configs/" || return 1
  say "stage: $1/$2"; $SYNC --inplace "$JUP/outputs/$1/ckpts/$2" "$d/ckpts/$2.part" && mv "$d/ckpts/$2.part" "$d/ckpts/$2" || return 1
  [ "$(sha256sum "$d/ckpts/$2" | cut -d' ' -f1)" = "$3" ] || { echo "sha256 mismatch $2"; return 1; }
  touch "$d/ckpts/.ok_$2"
}
stage_ckpt texgen_alpha_74k_v2_agentic epoch=124-step=35500.ckpt 2e52305fd352a931f20d03e942b2b4e85665218d198d9e29cfe5c56ab61249f6 || die "paper ckpt"
stage_ckpt texgen_alpha_lightgenbench_v1 epoch=124-step=35625.ckpt 6c42a311f5925f5bcfd87a1bb68592a76ae71d53ce0efe3c90e880bf8e3892aa || die "release ckpt"
$SYNC --delete "$H/hf_root/" "$RROOT/" && (cd "$RROOT" && sha256sum -c --quiet MANIFEST.sha256) || die "release check root"
flock -u 8
( set +u; source "$ACT"; set -u; python -c "import torch; print('[probe] torch', torch.__version__, torch.cuda.device_count(), 'GPUs')" ) || die "env probe"

# ---- tasks: "gpu|name|kind|dest|args..." ----
TASKS=()
if [ -n "$TASKFILE" ]; then
  while read -r g name kind shas; do
    [ -n "${g:-}" ] || continue
    case $kind in
      paper)   TASKS+=("$g|$name|gate|$OUT/$name|$PAPER_CKPT|$H/$shas") ;;
      release) TASKS+=("$g|$name|gate|$OUT/$name|$REL_CKPT|$RROOT/release_test20.txt|$RROOT|$RROOT/df_lightgenbench.parquet") ;;
      fresh|resume) TASKS+=("$g|$name|train|$OUT/$name|$kind") ;;
      *) die "task file: unknown kind $kind" ;;
    esac
  done < "$H/$TASKFILE"
else
  SHAS=$EL/test_agentic_clean.txt
  for r in $PAPER_RUNS; do
    mkdir -p "$OUT/shas"
    for g in $(seq 0 $((NGPU-1))); do
      awk -v r="$r" -v g="$g" -v n="$NGPU" 'NF && ((NR - 1 + r) % n) == g' "$SHAS" > "$OUT/shas/paper_run${r}_g$g.txt"
      TASKS+=("$g|paper_run${r}_g$g|gate|$OUT/paper_test204_run$r.parts/g$g|$PAPER_CKPT|$OUT/shas/paper_run${r}_g$g.txt")
    done
  done
  j=0
  for mr in $TRAIN_RUNS; do
    TASKS+=("$((j % NGPU))|train_${mr%%:*}_run${mr##*:}|train|$OUT/train_${mr%%:*}_run${mr##*:}|${mr%%:*}"); j=$((j + 1))
  done
  for r in $RELEASE_RUNS; do
    TASKS+=("$((j % NGPU))|release_test20_run$r|gate|$OUT/release_test20_run$r|$REL_CKPT|$RROOT/release_test20.txt|$RROOT|$RROOT/df_lightgenbench.parquet"); j=$((j + 1))
  done
fi
# SLURM_JOB_NAME=bash: Lightning then ignores the Slurm job (as scripts/internal/fir/infer_lightgenbench_fir.sh does)
# and every task runs single-process exactly as on the workstation.
export GATE_LOCK=none GATE_ACTIVATE=$ACT GATE_REPO=$TOOL/repo GATE_L=$EL TRAINCHECK_ROOT=$RROOT TRAINCHECK_TMPBASE=/localscratch/dya78/tmp SLURM_JOB_NAME=bash
run_queue() {  # gpu: this GPU's tasks one after another; one "<name> exit=<rc> secs=<s> <last log line>" line each
  local gq=$1 t g name kind dest a1 a2 a3 a4 rc s
  for t in "${TASKS[@]}"; do
    IFS='|' read -r g name kind dest a1 a2 a3 a4 <<< "$t"
    [ "$g" = "$gq" ] || continue
    say "launch $name on gpu $g"; s=$(date +%s)
    if [ "$kind" = gate ]; then
      CUDA_VISIBLE_DEVICES=$g bash "$GT/gate-det.sh" "$TREE" "$dest" "$a1" "$a2" ${a3:+"$a3"} ${a4:+"$a4"} > "$OUT/logs/$name.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$g bash "$GT/train-check.sh" "$TREE" "$dest" "$a1" "$REL_CKPT" > "$OUT/logs/$name.log" 2>&1
    fi
    rc=$?; echo "$name exit=$rc secs=$(( $(date +%s) - s )) $(tail -1 "$OUT/logs/$name.log")" | tee -a "$OUT/status.g$gq"
  done
}
PIDS=()
for g in $(seq 0 $((NGPU-1))); do run_queue "$g" & PIDS+=("$!"); done
for p in "${PIDS[@]}"; do wait "$p"; done
cat "$OUT"/status.g* > "$OUT/status.txt" 2>/dev/null
RC=0
[ "$(grep -c . "$OUT/status.txt")" = "${#TASKS[@]}" ] || RC=1
grep -v -q -E ' exit=0 secs=[0-9]+ (GATE_DONE|TRAINCHECK_DONE)$' "$OUT/status.txt" && RC=1

# ---- collect: the four gate PNGs per shape, gpu.txt and the records go to $HOME ----
for r in $PAPER_RUNS; do
  [ -z "$TASKFILE" ] || break
  mkdir -p "$OUT/paper_test204_run$r"
  for g in $(seq 0 $((NGPU-1))); do
    mv "$OUT/paper_test204_run$r.parts/g$g"/*/ "$OUT/paper_test204_run$r/" 2>/dev/null
    cat "$OUT/paper_test204_run$r.parts/g$g/gpu.txt" >> "$OUT/paper_test204_run$r/gpu.txt" 2>/dev/null
  done
done
for d in "$OUT"/*/; do
  d=${d%/}; n=$(basename "$d")
  case "$n" in logs|shas|*.parts) continue ;; esac
  mkdir -p "$HOUT/$n"
  $SYNC "$d"/gpu.txt "$HOUT/$n/" 2>/dev/null
  if [ -f "$d/record.jsonl" ]; then $SYNC "$d/record.jsonl" "$HOUT/$n/" || RC=1; continue; fi
  for s in "$d"/*/; do
    s=${s%/}; mkdir -p "$HOUT/$n/$(basename "$s")"
    $SYNC "$s"/{pred_emission,gt_emission,mask,input_albedo}.png "$HOUT/$n/$(basename "$s")/" 2>/dev/null
  done
  say "$n: $(find "$HOUT/$n" -name pred_emission.png | wc -l) pred_emission.png"
done
$SYNC "$OUT"/logs/*.log "$HOUT/logs/"
[ $RC -eq 0 ] && echo "CHECKS_DONE label=$LABEL" >> "$OUT/status.txt" || echo "CHECKS_INCOMPLETE label=$LABEL" >> "$OUT/status.txt"
$SYNC "$OUT/status.txt" "$HOUT/status.txt.part" && mv "$HOUT/status.txt.part" "$HOUT/status.txt"
cat "$OUT/status.txt"
[ $RC -eq 0 ] || exit 2
