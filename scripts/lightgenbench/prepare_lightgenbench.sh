#!/bin/bash
# LightgenBench (Hugging Face 3dlg-hcvc/LightgenBench) -> the data this code trains and runs on.
#
#   bash scripts/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_dir>
#
# <download_dir> holds a download of the dataset with at least splits.json, checksums.sha256 and
# data/{train,val,test}/{atlas,thumbnail}/*.tar, for example
#   huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
#     --include "splits.json" "checksums.sha256" "data/*/atlas/*.tar" "data/*/thumbnail/*.tar"
# The tars are checked against checksums.sha256, then unpacked into
#   <data_dir>/texgen_root/<uuid>/somage.npz         (the release's atlas.npz)
#   <data_dir>/texgen_root/thumbnails/<uuid>.png      (the CLIP image condition)
# and the two index files the loader reads are written:
#   <data_dir>/df_lightgenbench.parquet, <data_dir>/data_splits_lightgenbench.json
# The release config reads these from LIGHTGENBENCH_DATA=<data_dir>.
# JOBS (default 8) tars are checked and unpacked in parallel. The full release needs about 75 GB.
set -euo pipefail
HFD=$(readlink -f "${1:?download dir}")
D=$(readlink -f -m "${2:?data dir}")
ROOT=$D/texgen_root
JOBS=${JOBS:-8}
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
for f in splits.json checksums.sha256; do
  [ -f "$HFD/$f" ] || { echo "ABORT: no $f in $HFD"; exit 2; }
done

mkdir -p "$ROOT/thumbnails"
cd "$HFD"
WANT=$D/.want.sha256
grep -E '  (splits\.json|data/(train|val|test)/(atlas|thumbnail)/[^/]+\.tar)$' checksums.sha256 > "$WANT"
echo "[check] $(wc -l < "$WANT") files against checksums.sha256"
xargs -P "$JOBS" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < "$WANT"

# Only the tars just checked are unpacked; any other tar in <download_dir> is ignored.
echo "[unpack] atlas tars -> $ROOT/<uuid>/somage.npz"
awk '$2 ~ /^data\/[^\/]+\/atlas\//{print $2}' "$WANT" \
  | xargs -P "$JOBS" -I{} tar xf {} -C "$ROOT" --transform 's|/atlas\.npz$|/somage.npz|'
echo "[unpack] thumbnail tars -> $ROOT/thumbnails/<uuid>.png"
awk '$2 ~ /^data\/[^\/]+\/thumbnail\//{print $2}' "$WANT" \
  | xargs -P "$JOBS" -I{} tar xf {} -C "$ROOT/thumbnails" --transform 's|^\([^/]*\)/thumbnail\.png$|\1.png|'
rm -f "$WANT"

echo "[index] $D/df_lightgenbench.parquet, $D/data_splits_lightgenbench.json"
python "$HERE/build_index.py" --splits "$HFD/splits.json" --out-dir "$D"

# Every uuid of the split must be on disk with its thumbnail: a missing npz is dropped from its batch
# and a missing thumbnail falls back to the albedo map, both silently.
python - "$D" <<'PY'
import os, sys
import pandas as pd
d = sys.argv[1]
root = os.path.join(d, "texgen_root")
ids = [str(x) for x in pd.read_parquet(os.path.join(d, "df_lightgenbench.parquet")).index]
missing = [u for u in ids if not os.path.isfile(os.path.join(root, u, "somage.npz"))
           or not os.path.isfile(os.path.join(root, "thumbnails", u + ".png"))]
if missing:
    sys.exit(f"{len(missing)} of {len(ids)} uuids lack a somage.npz or a thumbnail, e.g. {missing[0]}")
print(f"[ok] {len(ids)} shapes with somage.npz and thumbnail under {root}")
PY
