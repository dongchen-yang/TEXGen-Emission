#!/bin/bash
#SBATCH -J texgen_save_janitor
#SBATCH --account=def-msavva_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/dya78/lightgen/texgen_runs/save_janitor.log
#SBATCH --open-mode=append
# Keeps a training run's validation images from filling fir's /scratch FILE quota.
#
# Every validation writes save/it<step>-test/ with 6 PNGs per val shape: 1,200 files per
# epoch on the LightgenBench split, 150K over 125 epochs, against a 1,000K-file project
# quota that stood at 885K on 2026-09-20. A full quota stops every lane's checkpoint saves.
#
# One pass: each finished it<step>-test dir becomes it<step>-test.tar (1 file), the dir is
# removed only after the tar lists exactly the dir's file count. The newest dir is left
# alone while the training job exists, since validation may still be writing it. The pass
# then resubmits itself for an hour later while the training job is in the queue (running
# or waiting to be requeued); once it is gone, a last pass tars the newest dir too.
#
#   sbatch scripts/fir/save_janitor.sh        # from the repo clone on fir
# (On main this file is scripts/internal/fir/save_janitor.sh; the fir clone checks out branch
# texgen-74k-v2-venus05, where it is still scripts/fir/save_janitor.sh, so the usage above uses that path.)
set -uo pipefail

SAVE=${SAVE:-/scratch/dya78/lightgen/texgen_runs/lightgen/texgen_alpha_lightgenbench_v1/save}
TRAIN_JOB_NAME=${TRAIN_JOB_NAME:-texgen_alpha_lightgenbench_v1_fir}
SELF=${SELF:-/scratch/dya78/lightgen/TEXGen_agentic/scripts/fir/save_janitor.sh}

echo "=== janitor pass  $(date -Iseconds)  job ${SLURM_JOB_ID:-none} ==="
cd "${SAVE}" || { echo "FATAL: no ${SAVE}"; exit 1; }

if squeue -h -u "$USER" -n "${TRAIN_JOB_NAME}" -o %i | grep -q .; then TRAINING=1; else TRAINING=0; fi

mapfile -t DIRS < <(find . -maxdepth 1 -type d -name 'it*-test' -printf '%f\n' | sort -V)
[ "${TRAINING}" -eq 1 ] && [ "${#DIRS[@]}" -gt 0 ] && unset 'DIRS[-1]'

for d in "${DIRS[@]}"; do
    n=$(find "$d" -type f | wc -l)
    rm -f "$d.tar.part"
    if tar -cf "$d.tar.part" "$d" && [ "$(tar -tf "$d.tar.part" | grep -vc '/$')" -eq "$n" ]; then
        mv "$d.tar.part" "$d.tar" && rm -rf "$d" && echo "[tar] $d: $n files -> $d.tar"
    else
        rm -f "$d.tar.part"; echo "[tar] FAILED $d ($n files); left in place"
    fi
done
echo "[save] $(find . -type f | wc -l) files now under save/"
diskusage_report 2>/dev/null | grep scratch || true

if [ "${TRAINING}" -eq 1 ]; then
    sbatch --begin=now+1hour "${SELF}" || echo "WARNING: could not resubmit the janitor"
else
    echo "[done] ${TRAIN_JOB_NAME} is no longer in the queue; last pass, not resubmitting"
fi
