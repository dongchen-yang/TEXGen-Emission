"""Build the paper gate's data root: the paper-era test bake with alpha inside each somage.npz (the
loader reads alpha from the npz only; there is no alpha.npy sidecar fallback).

  python make_paper_root.py --out <root>     (default source: $L/atlas_root, read only)

The source, $L/atlas_root, keeps alpha beside the npz: <sha>/somage.npz is a symlink to
$L/out_eval_gt/<sha>/atlas.npz (no alpha key) and <sha>/alpha.npy is the sidecar. Other evaluations
read both trees, so they are never written. For every row of the source parquet (the 204 test shapes
and the 381 validation shapes it also lists) the root gets:

  <root>/<ditem_dir>/somage.npz          every array of the source npz, in its order, then `alpha`
                                         (the sidecar: uint8 (H, W, 1)); stored uncompressed, as the
                                         source; checked array by array against both sources
  <root>/thumbnails/<sha>.png            the CLIP input, copied byte for byte
  <root>/eval_parquet_test204.parquet    the source parquet, copied byte for byte
  <root>/test_agentic_clean.txt          the 204 test shapes, the paper gate's shapes file
  <root>/SOURCE.json                     the source paths and the sha256 of every source file read
  <root>/MANIFEST.sha256                 sha256 of every file above; a copy on another host is
                                         checked with `sha256sum -c MANIFEST.sha256` inside the root

The root is built in <root>.part and renamed into place; an existing root is never touched.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

L = "/local-scratch/localhome/dya78/lightgen_eval_local"
PARQUET = "eval_parquet_test204.parquet"
SHAS = "test_agentic_clean.txt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pack(src_npz, sidecar, dst_npz):
    """Write dst_npz = every array of src_npz, then `alpha` = the sidecar; check it read back."""
    with np.load(src_npz) as z:
        arrays = {k: z[k] for k in z.files}
    if "alpha" in arrays:
        sys.exit(f"{src_npz} already has an alpha key")
    alpha = np.load(sidecar)
    hw = next(iter(arrays.values())).shape[:2]
    if alpha.dtype != np.uint8 or alpha.shape != hw + (1,):
        sys.exit(f"{sidecar}: alpha is {alpha.dtype} {alpha.shape}, expected uint8 {hw + (1,)}")
    np.savez(dst_npz, **arrays, alpha=alpha)
    with np.load(dst_npz) as z:
        if z.files != list(arrays) + ["alpha"]:
            sys.exit(f"{dst_npz}: keys {z.files}, expected {list(arrays) + ['alpha']}")
        for k, a in list(arrays.items()) + [("alpha", alpha)]:
            b = z[k]
            if b.dtype != a.dtype or b.shape != a.shape or not np.array_equal(a, b):
                sys.exit(f"{dst_npz}: {k} differs from its source")
    return len(arrays)


def build(src, out):
    if os.path.exists(out):
        sys.exit(f"{out} exists; build a new root elsewhere (or remove it first)")
    df = pd.read_parquet(os.path.join(src, PARQUET))
    shas = [l.strip() for l in open(os.path.join(L, SHAS)) if l.strip()]
    if not set(shas) <= set(df.index):
        sys.exit(f"{SHAS}: {len(set(shas) - set(df.index))} shapes are not in {PARQUET}")
    part = out + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(os.path.join(part, "thumbnails"))
    hashes = {}
    for f, d in ((PARQUET, src), (SHAS, L)):
        shutil.copyfile(os.path.join(d, f), os.path.join(part, f))
        hashes[os.path.join(d, f)] = sha256(os.path.join(d, f))
    if "ditem_dir" not in df.columns or df["ditem_dir"].isna().any():
        sys.exit(f"{PARQUET}: every row needs a ditem_dir")
    n_arrays = set()
    for i, (sid, row) in enumerate(df.iterrows()):
        ditem = row["ditem_dir"]
        src_npz = os.path.realpath(os.path.join(src, ditem, "somage.npz"))
        sidecar = os.path.realpath(os.path.join(src, ditem, "alpha.npy"))
        thumb = os.path.join(src, "thumbnails", f"{sid}.png")
        os.makedirs(os.path.join(part, ditem))
        n_arrays.add(pack(src_npz, sidecar, os.path.join(part, ditem, "somage.npz")))
        shutil.copyfile(thumb, os.path.join(part, "thumbnails", f"{sid}.png"))
        for p in (src_npz, sidecar, thumb):
            hashes[p] = sha256(p)
        if hashes[thumb] != sha256(os.path.join(part, "thumbnails", f"{sid}.png")):
            sys.exit(f"{thumb}: the copy differs")
        if (i + 1) % 100 == 0:
            print(f"[pack] {i + 1}/{len(df)}", flush=True)
    for f in (PARQUET, SHAS):
        d = src if f == PARQUET else L
        if sha256(os.path.join(part, f)) != hashes[os.path.join(d, f)]:
            sys.exit(f"{f}: the copy differs")
    with open(os.path.join(part, "SOURCE.json"), "w") as f:
        json.dump({"source_root": src, "shapes_file": os.path.join(L, SHAS), "rows": len(df),
                   "test_shapes": len(shas), "sha256": hashes}, f, indent=1)
        f.write("\n")
    rel = sorted(os.path.relpath(os.path.join(d, n), part) for d, _, names in os.walk(part) for n in names)
    with open(os.path.join(part, "MANIFEST.sha256"), "w") as f:
        f.writelines(f"{sha256(os.path.join(part, r))}  {r}\n" for r in rel)
    os.rename(part, out)
    print(f"[root] {out}: {len(df)} shapes ({len(shas)} test), source npz arrays per shape {sorted(n_arrays)} "
          f"+ alpha, every array checked equal to its source; {len(rel)} files in MANIFEST.sha256")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", default=f"{L}/atlas_root", help="the paper-era root with alpha.npy sidecars (read only)")
    ap.add_argument("--out", required=True, help="root to build, e.g. $L/refactor_checks/paper_root")
    a = ap.parse_args()
    build(os.path.abspath(a.src), os.path.abspath(a.out))


if __name__ == "__main__":
    main()
