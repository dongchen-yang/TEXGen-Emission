#!/bin/bash
# TEXGen inference on the LightgenBench release TEST split (200 shapes) on fir: one seed per job,
# on a MIG slice (fir's eval rule), the paper run's code and venv, TEXGEN_ENABLE_FLASH=0.
#
#   SEED=0 sbatch scripts/fir/infer_lightgenbench_fir.sh     # from the TEXGen clone on fir
# (On main this file is scripts/internal/fir/infer_lightgenbench_fir.sh; the fir clone checks out branch
# texgen-74k-v2-venus05, where it is still scripts/fir/infer_lightgenbench_fir.sh, so the usage above uses that path;
# the body, REPO's `../..` included, assumes that layout too.)
#
# Reads CKPT (default: the epoch-124 checkpoint of texgen_alpha_lightgenbench_v1, with its
# configs/parsed.yaml two levels up, which inference_samples loads) and the release's test tars
# staged by scripts/internal/fir/stage_hf.sh. Writes $OUT/pred_seed0N/<uuid>/pred_emission.png and
# refuses to report success unless all 200 exist.
#SBATCH -J texgen_lightgenbench_infer
#SBATCH --account=def-msavva_gpu
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/dya78/lightgen/texgen_runs/lightgenbench_eval/logs/infer-%j.out
set -uo pipefail
SEED=${SEED:?set SEED}
SPLIT=${SPLIT:-test}
E=/scratch/dya78/lightgen/texgen_runs/lightgenbench_eval
CKPT=${CKPT:-$E/ckpts/epoch=124-step=35625.ckpt}
HFD=${HFD:-/scratch/dya78/lightgen/data/lightgenbench}
VENV=${VENV:-/scratch/dya78/lightgen/env}
OVERLAY=${OVERLAY:-/scratch/dya78/lightgen/texgen_pin_overlay}
REPO=$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd); [ -f "$REPO/inference_specific_samples.py" ] || REPO=/scratch/dya78/lightgen/TEXGen_agentic
OUT=$E/pred_${SPLIT}_seed$(printf '%02d' "$SEED")
echo "=== infer seed=$SEED split=$SPLIT job=${SLURM_JOB_ID:-} node=$(hostname) $(date -Iseconds)"
echo "repo=$REPO ($(git -C "$REPO" log --oneline -1)) ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$CKPT" ] || { echo "FATAL no $CKPT"; exit 2; }
[ -f "$(dirname "$(dirname "$CKPT")")/configs/parsed.yaml" ] || { echo "FATAL no configs/parsed.yaml beside $CKPT"; exit 2; }

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash
source "$VENV/bin/activate"
export TEXGEN_ENABLE_FLASH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SLURM_JOB_NAME=bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
[ -n "${HF_CACHE:-}" ] && export HF_HOME=$HF_CACHE
export PYTHONPATH=$OVERLAY${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=$SLURM_TMPDIR/tmp; mkdir -p "$TMPDIR"

# ---- stage the split's tars into $SLURM_TMPDIR under the names the loader reads -------------
D=$SLURM_TMPDIR/lightgen/data; ROOT=$D/texgen_root; mkdir -p "$ROOT/thumbnails"
for t in "$HFD"/data/$SPLIT/atlas/atlas-*.tar; do tar xf "$t" -C "$ROOT" --transform 's|/atlas\.npz$|/somage.npz|' || exit 2; done
for t in "$HFD"/data/$SPLIT/thumbnail/thumbnail-*.tar; do tar xf "$t" -C "$ROOT/thumbnails" --transform 's|^\([^/]*\)/thumbnail\.png$|\1.png|' || exit 2; done
python - "$HFD/splits.json" "$SPLIT" "$D" <<'PY' || { echo "FATAL split file"; exit 2; }
import json, sys
s = json.load(open(sys.argv[1])); split, d = sys.argv[2], sys.argv[3]
only = {"train": [], "val": [], "test": []}; only[split] = s[split]
json.dump(only, open(d + "/splits_only.json", "w"))
open(d + "/uuids.txt", "w").write("\n".join(s[split]) + "\n")
print("[split]", split, len(s[split]))
PY
python "$REPO/scripts/lightgenbench/build_index.py" --splits "$D/splits_only.json" --out-dir "$D" || { echo "FATAL index"; exit 2; }
N=$(wc -l < "$D/uuids.txt")
NPZ=$(find "$ROOT" -mindepth 2 -maxdepth 2 -name somage.npz | wc -l); THM=$(find "$ROOT/thumbnails" -name '*.png' | wc -l)
echo "[stage] somage.npz=$NPZ thumbnails=$THM expect=$N"
[ "$NPZ" -eq "$N" ] && [ "$THM" -eq "$N" ] || { echo "FATAL staged tree incomplete"; exit 2; }

# ---- a repo that ships scripts/convert_config.py reads current names only: inference gets a copy
# of the run directory, its configs/parsed.yaml converted and the checkpoint linked ----------
if [ -f "$REPO/scripts/convert_config.py" ]; then
  RUN=$SLURM_TMPDIR/run; mkdir -p "$RUN/ckpts" "$RUN/configs"
  python "$REPO/scripts/convert_config.py" "$(dirname "$(dirname "$CKPT")")/configs/parsed.yaml" "$RUN/configs/parsed.yaml" \
    || { echo "FATAL convert_config"; exit 2; }
  if grep -q -- "--ckpt" "$REPO/scripts/convert_config.py"; then   # the checkpoint's pickled class paths too
    python "$REPO/scripts/convert_config.py" --ckpt "$(readlink -f "$CKPT")" "$RUN/ckpts/$(basename "$CKPT")" \
      || { echo "FATAL convert_config --ckpt"; exit 2; }
  else
    ln -s "$(readlink -f "$CKPT")" "$RUN/ckpts/$(basename "$CKPT")"
  fi
  CKPT=$RUN/ckpts/$(basename "$CKPT")
fi

# ---- inference: the paper harness's call, per-shape seed sha256(uuid:seed) -----------------
mkdir -p "$OUT"
cd "$REPO"
python - "$CKPT" "$D" "$OUT" "$SEED" <<'PY'
import sys
ckpt, d, out, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
from inference_specific_samples import inference_samples
uuids = [l.strip() for l in open(d + "/uuids.txt") if l.strip()]
inference_samples(checkpoint_path=ckpt, sample_ids=uuids, output_dir=out,
                  data_root=d + "/texgen_root", parquet_file=d + "/df_lightgenbench.parquet", seed=seed)
PY
echo "[infer] python exit $?"
GOT=$(find "$OUT" -mindepth 2 -maxdepth 2 -name pred_emission.png | wc -l)
echo "[done] pred_emission.png=$GOT expect=$N  $(date -Iseconds)"
[ "$GOT" -eq "$N" ] && echo "INFER_OK seed=$SEED" || { echo "INFER_INCOMPLETE seed=$SEED"; exit 1; }
