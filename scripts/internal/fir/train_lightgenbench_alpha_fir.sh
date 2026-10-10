#!/bin/bash
# TEXGen alpha on the RELEASED dataset (Hugging Face 3dlg-hcvc/LightgenBench) on **fir** —
# 1 node x 4 H100 80 GB, per-GPU batch 32 -> GLOBAL BATCH 128. The paper's recipe and the
# paper's code; only the data is new.
#
# THIS IS A DATA PORT OF scripts/fir/train_74k_v2_alpha_agentic_fir.sh, the launcher of the
# paper run's fir port (deleted from main 2026-09-29; read it with
# `git show texgen-74k-v2-venus05:scripts/fir/train_74k_v2_alpha_agentic_fir.sh`). Everything outside the data section is that file's, unchanged, and
# its header explains each piece (the pin overlay, TEXGEN_ENABLE_FLASH=0,
# SLURM_JOB_NAME=bash, the offline HF cache, the sick-node guard, 24 h segments with
# --requeue + auto_resume). Deltas, all forced by the data:
#   * the archives are the release's own tars, downloaded and sha256-checked once by
#     scripts/internal/fir/stage_hf.sh into ${DATA}/lightgenbench (80 files);
#   * every segment unpacks the 39 atlas + 39 thumbnail tars (~75 GB) into
#     $SLURM_TMPDIR, renaming <uuid>/atlas.npz -> <uuid>/somage.npz and
#     <uuid>/thumbnail.png -> thumbnails/<uuid>.png as it goes, which is the layout the
#     loader reads. No symlinks, no copy of the data on /scratch's inode quota;
#   * the parquet and the split JSON are written from splits.json by
#     scripts/lightgenbench/build_index.py at the start of every segment, and its --check
#     replaces the md5 gates: the split's indices are POSITIONAL into the parquet, and the
#     check proves they select exactly the uuids of splits.json, whose sha256 is pinned;
#   * alpha is a key inside each npz, so the sidecar count gate becomes a key check;
#   * wandb goes online when the node has a login and reaches the API, else offline;
#   * per-GPU MICRO-batch 16 x accumulate_grad_batches 2, not per-GPU batch 32. The
#     release's atlases are heavier than the paper run's: at per-GPU 32 job 60564907
#     (fir, 2026-09-19) peaked at 76.15 GB by step ~50 and OOM'd an 80 GB H100 at step ~57,
#     where the paper run peaked at 65.56 GB on the same card. 4 ranks x 16 x 2 is still
#     GLOBAL BATCH 128, and Lightning counts OPTIMIZER steps as global_step, so
#     steps/epoch (285), T_max and max_epochs keep their meaning (the deleted vulcan
#     launcher's pattern). EMA needs no change: LitEma updates once per micro-batch, but its decay is
#     min(0.9999, (1+n)/(10+n)) and the cap never binds in this run (n < 89,990), so two
#     updates per optimizer step give 1 - 9/(5+s) where the paper run had 1 - 9/(10+s).
#     The config carries the same two values; this launcher still passes them on the
#     command line.
#
# Usage (runs from the workstation OR from fir; it submits either way):
#   # probe first — separate output dir, so auto_resume can never find it later:
#   OUT_SUFFIX=_probe WALLTIME=3:00:00 \
#     EXTRA="trainer.max_steps=30 trainer.limit_val_batches=2" \
#     bash TEXGen-Emission/scripts/internal/fir/train_lightgenbench_alpha_fir.sh
#   # full run:
#   bash TEXGen-Emission/scripts/internal/fir/train_lightgenbench_alpha_fir.sh
# (In a fir clone on branch texgen-74k-v2-venus05 this file is scripts/fir/train_lightgenbench_alpha_fir.sh;
# the job body below checks out that branch, so its paths use that layout.)
#
# Score at epoch 124 / step 35,625 (125 epochs x 285 steps).
set -euo pipefail

NAME=texgen_alpha_lightgenbench_v1
TAG=fir
CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_lightgenbench_alpha_alliance.yaml}
REPO=${REPO:-/scratch/dya78/lightgen/TEXGen_agentic}
DATA=${DATA:-/scratch/dya78/lightgen/data}
RUNS=${RUNS:-/scratch/dya78/lightgen/texgen_runs}
OVERLAY=${OVERLAY:-/scratch/dya78/lightgen/texgen_pin_overlay}
VENV=${VENV:-/scratch/dya78/lightgen/env}
# Leave empty to inherit ~/.cache/huggingface (correct for dya78, whose home holds the CLIP
# snapshots). A DIFFERENT submitting user has a different home and almost certainly no cache,
# so point this at the group-readable copy: HF_CACHE=/scratch/dya78/lightgen/hf_cache
HF_CACHE=${HF_CACHE:-}
BRANCH=texgen-74k-v2-venus05
OUT_SUFFIX=${OUT_SUFFIX:-}
EXTRA=${EXTRA:-}
NUM_GPUS=${NUM_GPUS:-4}
BS=${BS:-16}                  # per-GPU MICRO-batch (see the header)
ACCUM=${ACCUM:-2}             # accumulate_grad_batches
GLOBAL_BATCH=${GLOBAL_BATCH:-128}
CPUS=${CPUS:-48}
MEM=${MEM:-1024G}
WALLTIME=${WALLTIME:-24:00:00}
ACCOUNT=${ACCOUNT:-rrg-msavva_gpu}
MAIL_USER=${MAIL_USER:-yangdongchen1@gmail.com}
# fir's H100 pool is usually congested and these are BYNODE partitions, so a 4-GPU request
# waits for a whole free node. Queue the real run behind the probe so it accrues queue age
# while the probe runs, instead of starting that wait from scratch afterwards:
#   DEPENDENCY=afterok:<probe_jobid> bash TEXGen-Emission/scripts/internal/fir/train_lightgenbench_alpha_fir.sh
DEPENDENCY=${DEPENDENCY:-}

# sha256 of the release's splits.json (2026-09-18), the same value stage_hf.sh pins.
SPLITS_SHA256=6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e
N_TARS=39                    # per representation: 37 train + 1 val + 1 test
EXPECT=36826                 # 36,426 train + 200 val + 200 test
OUTPUT_DIR=${RUNS}/output_lightgenbench_alpha_v1${OUT_SUFFIX}

# The global batch is the thing being held equal across sites and arms. Refuse rather than
# quietly train a different effective batch than the paper run.
ACTUAL=$((NUM_GPUS * BS * ACCUM))
if [ "${ACTUAL}" -ne "${GLOBAL_BATCH}" ]; then
    echo "REFUSING TO SUBMIT: NUM_GPUS(${NUM_GPUS}) x BS(${BS}) x ACCUM(${ACCUM}) = ${ACTUAL}, expected ${GLOBAL_BATCH}." >&2
    exit 1
fi

echo "Submitting ${NAME}${OUT_SUFFIX}_${TAG}"
echo "  gpus     : ${NUM_GPUS} x h100  (micro-batch ${BS} x accumulate ${ACCUM} -> global ${ACTUAL})"
echo "  config   : ${CONFIG}"
echo "  outdir   : ${OUTPUT_DIR}"
echo "  walltime : ${WALLTIME}   account: ${ACCOUNT}"
echo "  extra    : '${EXTRA}'"
echo "  depend   : '${DEPENDENCY:-<none>}'"

DEP_ARG=()
[ -n "${DEPENDENCY}" ] && DEP_ARG=(--dependency="${DEPENDENCY}")
if command -v sbatch >/dev/null 2>&1; then
    SUBMIT=(sbatch "${DEP_ARG[@]}")
else
    SUBMIT=(ssh -o BatchMode=yes fir sbatch "${DEP_ARG[@]}")
fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH --gpus-per-node=h100:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@900

set -uo pipefail

handle_timeout() {
    echo "[trap] wall-clock signal (~15 min left); stopping for requeue..."
    if [ -n "\${TRAIN_PID:-}" ] && kill -0 "\${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "\${TRAIN_PID}" 2>/dev/null
        wait "\${TRAIN_PID}" || true
    fi
    scontrol requeue "\${SLURM_JOB_ID}" || true
    exit 0
}
trap handle_timeout SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  restart_count=\${SLURM_RESTART_COUNT:-0}"
echo "Started: \$(date -Iseconds)  Node: \$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "======================================"

# fc10218 was found with an orphaned PID holding 69 GB on GPU 0 (2026-08-20). Refuse to
# start on a card that is already occupied — at per-GPU batch 32 this run needs ~69 GB of
# the 80 GB card and would OOM in a way that looks like a config problem.
BUSY=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1 > 2048' | wc -l)
if [ "\${BUSY}" -gt 0 ]; then
    echo "FATAL: \${BUSY} GPU(s) on \$(hostname) already hold >2 GB before training; node is sick. Requeueing."
    scontrol requeue "\${SLURM_JOB_ID}"; sleep 60; exit 0
fi

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
echo "TEXGen HEAD: \$(git log --oneline -1)"

VENV=${VENV}
[ -f "\${VENV}/bin/activate" ] || { echo "FATAL: no venv at \${VENV}"; exit 4; }
source "\${VENV}/bin/activate"
export EXPECT_VENV=${VENV}

# ---- environment gates -------------------------------------------------------------
# fir has no conda, so the venus launcher's 'envs/texgen-bw' interpreter assertion is
# replaced by this venv's own prefix. Deliberately NOT \`assert\` — an inherited
# PYTHONOPTIMIZE strips asserts and the gate would pass silently on the wrong python.
export TEXGEN_ENABLE_FLASH=0
export SLURM_JOB_NAME=bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
# wandb goes online only where it can: a wandb login (an api.wandb.ai entry in ~/.netrc, or
# WANDB_API_KEY) AND a reachable API; otherwise offline — 'wandb sync <trial
# dir>/wandb/offline-run-*' later from a login node. The mode is set on both branches, so
# an inherited WANDB_MODE cannot make this line lie.
if { [ -n "\${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" "\$HOME/.netrc"; } \\
   && curl -sSI -m 8 https://api.wandb.ai >/dev/null 2>&1; then
    export WANDB_MODE=online; echo "[wandb] online"
else
    export WANDB_MODE=offline; echo "[wandb] OFFLINE (no login or no internet on this node)"
fi
# A cache MISS on the gated CLIP repos is a 401, not a download (that crashed job 237299).
if [ -n "${HF_CACHE}" ]; then
    [ -d "${HF_CACHE}/hub" ] || { echo "FATAL: HF_CACHE=${HF_CACHE} has no hub/ subdir"; exit 4; }
    export HF_HOME=${HF_CACHE}
    echo "[hf] HF_HOME=\$HF_HOME (explicit; not this user's home cache)"
else
    echo "[hf] HF_HOME unset -> \$HOME/.cache/huggingface"
fi
export TMPDIR=\${SLURM_TMPDIR}/tmp
mkdir -p "\$TMPDIR"

# Shadow the shared venv's newer conditioning stack with venus's hard pins. PYTHONPATH
# precedes site-packages in sys.path, so these win without touching the shared venv.
[ -d "${OVERLAY}/transformers" ] || { echo "FATAL: pin overlay missing at ${OVERLAY} (see the header for the pip line)"; exit 4; }
export PYTHONPATH=${OVERLAY}\${PYTHONPATH:+:\$PYTHONPATH}

python - <<'ENVCHK' || { echo "FATAL bad env"; exit 4; }
import sys, os
want_venv = os.environ["EXPECT_VENV"]
if not sys.executable.startswith(want_venv):
    sys.exit("WRONG INTERPRETER: %s (want it under %s)" % (sys.executable, want_venv))
if os.environ.get("TEXGEN_ENABLE_FLASH") != "0":
    sys.exit("TEXGEN_ENABLE_FLASH must be 0 on fir: flash_attn imports here but not on the venus arms")
import transformers, diffusers, huggingface_hub, omegaconf
# The four packages bootstrap_texgen_bw.sh (deleted from main 2026-09-29) hard-pins on venus. Anything else in this env
# is allowed to differ (this is an acknowledged cross-site run), but the conditioning
# stack the model actually consumes must match, or the comparison loses its meaning.
for name, mod, want in (("transformers", transformers, "4.28.1"),
                        ("diffusers", diffusers, "0.28.0"),
                        ("huggingface_hub", huggingface_hub, "0.25.2"),
                        ("omegaconf", omegaconf, "2.3.0")):
    got = mod.__version__.split("+")[0]
    if got != want:
        sys.exit("%s is %s but venus pins %s -- pin overlay not shadowing (PYTHONPATH=%s)"
                 % (name, got, want, os.environ.get("PYTHONPATH", "<unset>")))
import torch, spconv.pytorch, torchsparse, nvdiffrast.torch, pytorch_lightning
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False -- no usable GPU in this allocation")
print("[env] pins ok: transformers %s / diffusers %s / hub %s / omegaconf %s"
      % (transformers.__version__, diffusers.__version__, huggingface_hub.__version__, omegaconf.__version__))
print("[env] %s" % sys.executable)
print("[env] torch %s  cuda %s  gpus %d  lightning %s"
      % (torch.__version__, torch.version.cuda, torch.cuda.device_count(), pytorch_lightning.__version__))
ENVCHK
# ---- content gates on the source data (cheap, on /scratch, before the extract) ---------
HFD=${DATA}/lightgenbench
[ -f "\$HFD/splits.json" ] || { echo "FATAL missing \$HFD/splits.json — run scripts/lightgenbench/stage_hf.sh"; exit 2; }
GOT=\$(sha256sum "\$HFD/splits.json" | cut -d' ' -f1)
if [ "\$GOT" != "${SPLITS_SHA256}" ]; then
    echo "FATAL sha256 mismatch \$HFD/splits.json"; echo "  got  \$GOT"; echo "  want ${SPLITS_SHA256}"; exit 2
fi
echo "[sha256] ok  splits.json  \$GOT"

for rep in atlas thumbnail; do
    N=\$(find "\$HFD"/data/{train,val,test}/\$rep -name "\$rep-*.tar" -size +0 2>/dev/null | wc -l)
    if [ "\$N" -ne "${N_TARS}" ]; then
        echo "FATAL: \$N non-empty \$rep tars under \$HFD/data, expected ${N_TARS} — staging incomplete"
        exit 2
    fi
done
echo "[data] all \$(( 2 * ${N_TARS} )) archives present on /scratch"

# ---- stage into \$SLURM_TMPDIR ---------------------------------------------------------
D=\${SLURM_TMPDIR}/lightgen/data
export LIGHTGENBENCH_DATA=\$D   # the config's data paths read it
ROOT=\$D/texgen_root
mkdir -p "\$ROOT/thumbnails"

AVAIL_KB=\$(df -Pk "\${SLURM_TMPDIR}" | awk 'NR==2{print \$4}')
echo "[stage] \\\$SLURM_TMPDIR=\${SLURM_TMPDIR}  free=\$(( AVAIL_KB / 1024 / 1024 )) GB"
if [ "\$AVAIL_KB" -lt 125829120 ]; then      # 120 GB
    echo "FATAL: \\\$SLURM_TMPDIR has < 120 GB free; the training root needs ~75 GB"
    exit 2
fi

# The tar members are <uuid>/atlas.npz and <uuid>/thumbnail.png (files only). --transform
# renames them on the way out to the two paths spuv/data/mesh_uv.py reads.
echo "[stage] extracting ${N_TARS} atlas tars, 8 in parallel  \$(date -Iseconds)"
find "\$HFD"/data/{train,val,test}/atlas -name 'atlas-*.tar' \\
  | xargs -P 8 -I{} tar xf {} -C "\$ROOT" --transform 's|/atlas\\.npz\$|/somage.npz|' \\
  || { echo "FATAL: atlas extraction failed"; exit 2; }
echo "[stage] extracting ${N_TARS} thumbnail tars  \$(date -Iseconds)"
find "\$HFD"/data/{train,val,test}/thumbnail -name 'thumbnail-*.tar' \\
  | xargs -P 8 -I{} tar xf {} -C "\$ROOT/thumbnails" --transform 's|^\\([^/]*\\)/thumbnail\\.png\$|\\1.png|' \\
  || { echo "FATAL: thumbnail extraction failed"; exit 2; }
echo "[stage] extraction done  \$(date -Iseconds)"

# Counts, not samples: a partially unpacked root is otherwise silent — a missing npz is
# dropped from its batch and a missing thumbnail falls back to the albedo UV map.
NPZ=\$(find "\$ROOT" -mindepth 2 -maxdepth 2 -name somage.npz | wc -l)
THM=\$(find "\$ROOT/thumbnails" -maxdepth 1 -name '*.png' | wc -l)
echo "[stage] somage.npz=\$NPZ  thumbnails=\$THM  (expect ${EXPECT} each)"
if [ "\$NPZ" -ne "${EXPECT}" ] || [ "\$THM" -ne "${EXPECT}" ]; then
    echo "FATAL: staged tree is incomplete"
    exit 2
fi

# ---- the index files, and the split -> parquet POSITIONAL MAPPING gate -----------------
# build_index.py writes the parquet and the split JSON from splits.json, then re-reads
# both the way the loader does and requires the indices to select exactly those uuids.
python scripts/lightgenbench/build_index.py --splits "\$HFD/splits.json" --out-dir "\$D" \\
    || { echo "FATAL split/parquet positional mapping"; exit 2; }

# Every uuid of the split must be on disk with its thumbnail, and the npz must carry the
# keys the loader reads, alpha included, as the uint8 (H, W, 1) plane its /255 assumes.
python - <<'DATACHK' || { echo "FATAL staged data does not match the split"; exit 3; }
import json, os, random, sys
import numpy as np
d = os.path.join(os.environ["SLURM_TMPDIR"], "lightgen", "data")
root = os.path.join(d, "texgen_root")
import pandas as pd
ids = [str(x) for x in pd.read_parquet(os.path.join(d, "df_lightgenbench.parquet")).index]
missing = [u for u in ids if not os.path.isfile(os.path.join(root, u, "somage.npz"))
           or not os.path.isfile(os.path.join(root, "thumbnails", u + ".png"))]
if missing:
    sys.exit("%d of %d split uuids lack a somage.npz or a thumbnail, e.g. %s"
             % (len(missing), len(ids), missing[0]))
want = ("occupancy", "position", "objnormal", "color", "metal", "rough", "emission_color", "alpha")
random.seed(0)
for u in random.sample(ids, 200):
    with np.load(os.path.join(root, u, "somage.npz")) as z:
        lack = [k for k in want if k not in z.files]
        if lack:
            sys.exit("%s/somage.npz lacks %s" % (u, lack))
        a = z["alpha"]
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 1:
        sys.exit("BAD ALPHA %s: shape=%s dtype=%s" % (u, a.shape, a.dtype))
print("[data] ok - all %d split uuids staged; 200 random npz carry every key, alpha uint8 (H, W, 1)" % len(ids))
DATACHK

mkdir -p ${OUTPUT_DIR} ${RUNS}
echo "[inodes] /scratch usage before training:"; diskusage_report 2>/dev/null | grep scratch || true

# No \`| tee\`: --output already persists everything, and piping would make TRAIN_PID the
# tee process, so \`wait\` would report TEE's status and a crashed trainer would read as
# success. Backgrounded bare so the SIGTERM trap can reach it.
# exp_root_dir MUST be overridden too, not just checkpoint.dirpath: launch.py builds
# trial_dir = exp_root_dir/<name>/<trial> and writes logs, configs/, progress/, tb_logs/,
# save/, cmd.txt AND the offline wandb directory there. RUNS alone does not move it, so a
# submitter who cannot write the config's baked-in path fails at startup.
python launch.py --config ${CONFIG} --gpu \$(seq -s, 0 \$(( ${NUM_GPUS} - 1 ))) --train --wandb \\
    exp_root_dir=${RUNS} checkpoint.dirpath=${OUTPUT_DIR}/ \\
    data.batch_size=${BS} trainer.accumulate_grad_batches=${ACCUM} ${EXTRA} &
TRAIN_PID=\$!
wait "\$TRAIN_PID"
EXIT_CODE=\$?

echo "[done] train exit \${EXIT_CODE} at \$(date -Iseconds)"
if [ "\${EXIT_CODE}" -ne 0 ]; then
    if [ "\${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
        echo "[done] crash — requeueing (restart_count=\${SLURM_RESTART_COUNT:-0})"
        scontrol requeue "\${SLURM_JOB_ID}" || true
        exit 0
    fi
    echo "[done] crash — restart cap reached, NOT requeueing; inspect the log"
fi
exit \$EXIT_CODE
EOF

echo "Submitted: ${NAME}${OUT_SUFFIX}_${TAG}"
echo "Watch with:"
echo "  ssh -o BatchMode=yes fir 'squeue -u dya78 -o \"%.10i %.34j %.2t %.11M %.11L %R\"'"
echo "  ssh -o BatchMode=yes fir 'tail -40 ~/*-<JOBID>.out'"
