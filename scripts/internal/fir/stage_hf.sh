#!/bin/bash
#SBATCH -J lightgenbench_stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Download the part of the LightgenBench release that TEXGen trains on, and verify it.
# Run ONCE per cluster, where the internet is reachable: as a CPU job on fir (its compute
# nodes have internet), on the login node of a cluster whose compute nodes do not.
#
#   sbatch --account=def-msavva_cpu scripts/lightgenbench/stage_hf.sh     # fir
#   bash scripts/lightgenbench/stage_hf.sh                                # a login node
# (On main this file is scripts/internal/fir/stage_hf.sh; the fir clone checks out branch
# texgen-74k-v2-venus05, where it is still scripts/lightgenbench/stage_hf.sh, so the usage above uses that path.)
#
# What lands in $DEST (80 files, so the /scratch inode quota does not notice):
#   splits.json, checksums.sha256
#   data/{train,val,test}/atlas/atlas-*.tar            39 tars, members <uuid>/atlas.npz
#   data/{train,val,test}/thumbnail/thumbnail-*.tar    39 tars, members <uuid>/thumbnail.png
# The voxel and multiview tars are not downloaded. The tars stay packed: the training
# launcher unpacks them into $SLURM_TMPDIR at the start of every job segment.
#
# The repo is public, so no token is read. Any python with huggingface_hub works; on fir
# that is the project venv.
set -euo pipefail

REPO_ID=3dlg-hcvc/LightgenBench
# sha256 of splits.json as released 2026-09-18 (the repo's own checksums.sha256 row). Every
# cluster must train on this split, so it is pinned here and not read from the download.
SPLITS_SHA256=6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e
N_FILES=79                   # splits.json + 39 atlas tars + 39 thumbnail tars

DEST=${DEST:-/scratch/dya78/lightgen/data/lightgenbench}
VENV=${VENV:-/scratch/dya78/lightgen/env}
WORKERS=${WORKERS:-8}

if [ -f "${VENV}/bin/activate" ]; then
    module load StdEnv/2023 gcc python/3.11 arrow/21.0.0 2>/dev/null || true
    source "${VENV}/bin/activate"
fi
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "${DEST}"
echo "=== stage LightgenBench -> ${DEST}  $(hostname)  $(date -Iseconds) ==="

DEST="${DEST}" REPO_ID="${REPO_ID}" WORKERS="${WORKERS}" python - <<'PY'
import os
import huggingface_hub
from huggingface_hub import snapshot_download
print("[hf] huggingface_hub", huggingface_hub.__version__)
snapshot_download(
    repo_id=os.environ["REPO_ID"], repo_type="dataset", local_dir=os.environ["DEST"],
    allow_patterns=["splits.json", "checksums.sha256",
                    "data/*/atlas/*.tar", "data/*/thumbnail/*.tar"],
    max_workers=int(os.environ["WORKERS"]),
)
PY

cd "${DEST}"
echo "${SPLITS_SHA256}  splits.json" | sha256sum -c -
grep -E '  (splits\.json|data/(train|val|test)/(atlas|thumbnail)/[^/]+\.tar)$' checksums.sha256 > .want.sha256
GOT=$(wc -l < .want.sha256)
if [ "${GOT}" -ne "${N_FILES}" ]; then
    echo "FATAL: checksums.sha256 lists ${GOT} files for TEXGen, expected ${N_FILES}"
    exit 2
fi
echo "[sha256] checking ${GOT} files with ${WORKERS} workers  $(date -Iseconds)"
# one sha256sum per line ("<hash>  <path>" -> $0 $1); xargs exits 123 if any check fails
xargs -P "${WORKERS}" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < .want.sha256
echo "[sha256] ok - ${GOT}/${N_FILES} files match checksums.sha256  $(date -Iseconds)"
du -sh "${DEST}/data" | sed 's/^/[size] /'
