#!/bin/bash
# usage: ckpt_for_tree.sh <texgen_root> <ckpt>   -> prints the checkpoint that tree should load
# A tree whose scripts/convert_config.py converts checkpoints (--ckpt) reads current class paths only,
# so the checkpoint is converted once per host into GATE_CKPT_CACHE (default
# $GATE_L/refactor_checks/ckpt_cache), keyed by the converter and the file's size and mtime, under a
# lock so that parallel tasks convert it once; that copy's path is printed. An older tree (pre-refactor)
# gets the original path back.
set -euo pipefail
ROOT=${1:?texgen root}; CKPT=${2:?checkpoint}
CONV=$ROOT/scripts/convert_config.py
if ! grep -q -- "--ckpt" "$CONV" 2>/dev/null; then echo "$CKPT"; exit 0; fi
CACHE=${GATE_CKPT_CACHE:-${GATE_L:-/local-scratch/localhome/dya78/lightgen_eval_local}/refactor_checks/ckpt_cache}
SRC=$(readlink -f "$CKPT")
KEY=$(sha256sum "$CONV" | cut -c1-12)-$(stat -c %s-%Y "$SRC")
DST=$CACHE/$KEY/$(basename "$SRC")
mkdir -p "$CACHE/$KEY"
exec 6>"$CACHE/$KEY.lock"; flock 6
[ -f "$DST" ] || python "$CONV" --ckpt "$SRC" "$DST" >&2
echo "$DST"
