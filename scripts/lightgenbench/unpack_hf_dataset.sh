#!/bin/bash
# Unpack the Hugging Face dataset (3dlg-hcvc/LightgenBench) into the data folder this code trains and
# runs on:
#
#   bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
#
# <download_dir> holds a download of the dataset with at least splits.json, checksums.sha256 and
# data/{train,val,test}/{atlas,thumbnail}/*.tar, for example
#   huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
#     --include "splits.json" "checksums.sha256" "data/*/atlas/*.tar" "data/*/thumbnail/*.tar"
# The tars are checked against checksums.sha256, then unpacked into
#   <data_dir>/texgen_root/<uuid>/somage.npz         (the release's atlas.npz)
#   <data_dir>/texgen_root/thumbnails/<uuid>.png      (the CLIP image condition)
# and the two index files the loader (spuv/data/mesh_uv.py) reads are written from splits.json:
#   <data_dir>/df_lightgenbench.parquet         one row per shape: its uuid and folder
#   <data_dir>/data_splits_lightgenbench.json   train/val/test as row numbers into that parquet
# The release config reads these from LIGHTGENBENCH_DATA=<data_dir>.
# JOBS (default 8) tars are checked and unpacked in parallel. The full release needs about 75 GB.
set -euo pipefail
HFD=$(readlink -f "${1:?download dir}")
D=$(readlink -f -m "${2:?data dir}")
ROOT=$D/texgen_root
JOBS=${JOBS:-8}
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

# The index: rows train, val, test in splits.json order, so each split is a consecutive range of row
# numbers. It is then re-read the way the loader reads it (ObjaverseDataset._load_from_parquet,
# _apply_indices), and every uuid of the split must be on disk with its thumbnail: a missing npz is
# dropped from its batch and a missing thumbnail falls back to the albedo map, both silently.
echo "[index] $D/df_lightgenbench.parquet, $D/data_splits_lightgenbench.json"
python - "$HFD/splits.json" "$D" <<'PY'
import json, os, sys
import pandas as pd

splits_json, d = sys.argv[1], sys.argv[2]
SPLITS = ("train", "val", "test")
with open(splits_json) as f:
    splits = json.load(f)
if sorted(splits) != sorted(SPLITS):
    sys.exit(f"{splits_json}: keys {sorted(splits)}, expected {sorted(SPLITS)}")
uuids = [u for s in SPLITS for u in splits[s]]
if len(set(uuids)) != len(uuids):
    sys.exit(f"{splits_json}: a uuid appears more than once across the splits")
parquet, split_json = os.path.join(d, "df_lightgenbench.parquet"), os.path.join(d, "data_splits_lightgenbench.json")
pd.DataFrame({"ditem_dir": uuids, "success": True}, index=pd.Index(uuids, name="uuid")).to_parquet(parquet)
indices, start = {}, 0
for s in SPLITS:
    indices[s] = {"indices": list(range(start, start + len(splits[s])))}
    start += len(splits[s])
with open(split_json, "w") as f:
    json.dump(indices, f)

df = pd.read_parquet(parquet)
df = df[df["success"] == True]  # noqa: E712
ids, dirs = [str(x) for x in df.index], list(df["ditem_dir"])
with open(split_json) as f:
    indices = json.load(f)
for s in SPLITS:
    idx = indices[s]["indices"]
    if [ids[i] for i in idx] != splits[s] or [dirs[i] for i in idx] != splits[s]:
        sys.exit(f"MISMATCH: the {s} indices do not select the {s} uuids of splits.json")
root = os.path.join(d, "texgen_root")
missing = [u for u in ids if not os.path.isfile(os.path.join(root, u, "somage.npz"))
           or not os.path.isfile(os.path.join(root, "thumbnails", u + ".png"))]
if missing:
    sys.exit(f"{len(missing)} of {len(ids)} uuids lack a somage.npz or a thumbnail, e.g. {missing[0]}")
print(f"[ok] {len(ids)} shapes (train/val/test = {'/'.join(str(len(splits[s])) for s in SPLITS)}) "
      f"with somage.npz and thumbnail under {root}")
PY
