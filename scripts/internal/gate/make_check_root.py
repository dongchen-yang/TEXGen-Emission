"""Build the small release data root the release gate and the training check read, from the
LightgenBench release on Hugging Face (3dlg-hcvc/LightgenBench, gated: needs a login on this host).

  python make_check_root.py --hf <download dir> --fetch           # download what the check shapes need
  python make_check_root.py --hf <download dir> --out <root>      # verify, untar, index
  python make_check_root.py --hf <download dir> --select          # (re)write check_root_splits.json

The shapes are fixed by `check_root_splits.json` beside this file (written once with --select):
16 train shapes, a fixed stride over the sorted uuids of the release's last train shard (426 shapes,
the smallest tar), and, as in step 7, 2 val and 20 test shapes, fixed strides over the sorted val and
test splits (one shard each). So the root needs six tars, about 1.7 GB: the atlas and thumbnail tars
of train shard 36, val shard 0 and test shard 0 (which tar holds a shape: the release's
metadata.parquet `shard` column).

--fetch downloads, at the pinned REVISION, splits.json, checksums.sha256, metadata.parquet and those
six tars with the `hf` CLI (--hf_cli; the CLI reads its own login token, this script never does).
--out first checks every file it reads against the release's checksums.sha256 (splits.json also
against the sha256 the release runs pin), then builds the root the release run's way
(scripts/internal/fir/train_lightgenbench_alpha_fir.sh): the tar members are extracted with the
same renames, and the index step of scripts/lightgenbench/unpack_hf_dataset.sh writes the parquet
and split JSON:

  <root>/<uuid>/somage.npz              tar member <uuid>/atlas.npz (alpha inside)
  <root>/thumbnails/<uuid>.png          tar member <uuid>/thumbnail.png (the CLIP input)
  <root>/df_lightgenbench.parquet       rows train, val, test (the index step)
  <root>/data_splits_lightgenbench.json train 0-15, val 16-17, test 18-37
  <root>/data_splits_traincheck.json    the same, test cut to its first 2 shapes (the training
                                        check runs trainer.test after fit; 20 would cost 2.5 min)
  <root>/release_test20.txt             the 20 test uuids, the release gate's shapes file
  <root>/SOURCE.json                    repo, revision and the sha256 of every release file used
  <root>/MANIFEST.sha256                sha256 of every file above; a copy on another host is
                                        checked with `sha256sum -c MANIFEST.sha256` inside the root

The root is built in <root>.part and renamed into place; an existing root is never touched.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_FILE = os.path.join(HERE, "check_root_splits.json")
REPO_ID = "3dlg-hcvc/LightgenBench"
REVISION = "d152558a9b2adbfcaa305db78ffa468248c67da3"  # the Hub commit the check root was built from
# sha256 of splits.json as released 2026-09-18, the value stage_hf.sh and the release launcher pin
SPLITS_SHA256 = "6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e"
ROOT_FILES = ("splits.json", "checksums.sha256", "metadata.parquet")
TRAIN_STRIDE, VAL_STRIDE, TEST_STRIDE = 27, 100, 10  # 426 -> 16, 200 -> 2, 200 -> 20
NPZ_KEYS = ("occupancy", "position", "objnormal", "color", "metal", "rough", "emission_color", "alpha")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_metadata(hf):
    import pandas as pd
    return pd.read_parquet(os.path.join(hf, "metadata.parquet"))


def read_splits(path):
    with open(path) as f:
        return json.load(f)


def tars_for(hf, splits):
    """{tar path relative to the download: [uuids it must yield]} for the atlas and thumbnail tars."""
    meta = load_metadata(hf).set_index("uuid")
    out = {}
    for split in ("train", "val", "test"):
        for u in splits[split]:
            row = meta.loc[u]
            if row["split"] != split:
                sys.exit(f"{u}: metadata.parquet says split {row['split']}, check_root_splits.json says {split}")
            for kind in ("atlas", "thumbnail"):
                out.setdefault(f"data/{split}/{kind}/{kind}-{int(row['shard']):05d}.tar", []).append(u)
    return out


def select(hf):
    """Write check_root_splits.json: strides over sorted uuids, train from the last train shard."""
    if sha256(os.path.join(hf, "splits.json")) != SPLITS_SHA256:
        sys.exit("splits.json does not have the pinned sha256")
    s = read_splits(os.path.join(hf, "splits.json"))
    meta = load_metadata(hf)
    train = meta[meta["split"] == "train"]
    last = int(train["shard"].max())
    shard_uuids = sorted(train.loc[train["shard"] == last, "uuid"])
    assert set(shard_uuids) <= set(s["train"]), "metadata.parquet and splits.json disagree on train"
    pick = {"train": shard_uuids[::TRAIN_STRIDE], "val": sorted(s["val"])[::VAL_STRIDE],
            "test": sorted(s["test"])[::TEST_STRIDE]}
    assert [len(pick[k]) for k in ("train", "val", "test")] == [16, 2, 20], {k: len(v) for k, v in pick.items()}
    with open(SPLITS_FILE, "w") as f:
        json.dump(pick, f, indent=1)
        f.write("\n")
    print(f"[select] wrote {SPLITS_FILE}: train from shard {last} ({len(shard_uuids)} shapes)")


def hf_download(cli, hf, files):
    cmd = [cli, "download", REPO_ID, *files, "--repo-type", "dataset", "--revision", REVISION, "--local-dir", hf]
    print("[fetch]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(cli, hf):
    os.makedirs(hf, exist_ok=True)
    hf_download(cli, hf, list(ROOT_FILES))
    hf_download(cli, hf, sorted(tars_for(hf, read_splits(SPLITS_FILE))))


def verify(hf, files):
    """sha256 of every file against the release's checksums.sha256; returns {file: sha256}."""
    want = {}
    with open(os.path.join(hf, "checksums.sha256")) as f:
        for line in f:
            h, name = line.rstrip("\n").split("  ", 1)
            want[name] = h
    got = {}
    for name in files:
        got[name] = sha256(os.path.join(hf, name))
        if name not in want:
            sys.exit(f"{name} is not listed in checksums.sha256")
        if got[name] != want[name]:
            sys.exit(f"sha256 mismatch {name}: got {got[name]}, checksums.sha256 says {want[name]}")
    if got["splits.json"] != SPLITS_SHA256:
        sys.exit("splits.json does not have the pinned sha256")
    print(f"[sha256] ok - {len(files)} files match checksums.sha256")
    return got


PARQUET, SPLIT_JSON = "df_lightgenbench.parquet", "data_splits_lightgenbench.json"
INDEX_SPLITS = ("train", "val", "test")


def build_index(splits, out_dir):
    """The index step of scripts/lightgenbench/unpack_hf_dataset.sh, kept the same: rows train, val,
    test in splits.json order, so each split is a consecutive range of row numbers."""
    import pandas as pd
    uuids = [u for s in INDEX_SPLITS for u in splits[s]]
    pd.DataFrame({"ditem_dir": uuids, "success": True}, index=pd.Index(uuids, name="uuid")).to_parquet(
        os.path.join(out_dir, PARQUET))
    indices, start = {}, 0
    for s in INDEX_SPLITS:
        indices[s] = {"indices": list(range(start, start + len(splits[s])))}
        start += len(splits[s])
    with open(os.path.join(out_dir, SPLIT_JSON), "w") as f:
        json.dump(indices, f)


def check_index(splits, out_dir):
    """Re-read the index the way the loader does (spuv/data/mesh_uv.py): the row numbers must select
    exactly the uuids of each split, in order. 0: ok, 1: mismatch."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(out_dir, PARQUET))
    df = df[df["success"] == True]  # noqa: E712
    ids, dirs = [str(x) for x in df.index], list(df["ditem_dir"])
    with open(os.path.join(out_dir, SPLIT_JSON)) as f:
        indices = json.load(f)
    for s in INDEX_SPLITS:
        idx = indices[s]["indices"]
        if [ids[i] for i in idx] != splits[s] or [dirs[i] for i in idx] != splits[s]:
            print(f"MISMATCH: the {s} indices do not select the {s} uuids of splits.json")
            return 1
    print(f"[index] ok: {len(ids)} rows")
    return 0


def untar(tar, dest, members, transform):
    """The release run's extraction (its --transform), limited to the members the root needs."""
    subprocess.run(["tar", "xf", tar, "-C", dest, "--transform", transform, *members], check=True)


def build(hf, out):
    import numpy as np

    if os.path.exists(out):
        sys.exit(f"{out} exists; build a new root elsewhere (or remove it first)")
    splits = read_splits(SPLITS_FILE)
    tars = tars_for(hf, splits)
    hashes = verify(hf, ["splits.json", "metadata.parquet"] + sorted(tars))
    part = out + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(os.path.join(part, "thumbnails"))
    for tar, uuids in sorted(tars.items()):
        if "/atlas/" in tar:
            untar(os.path.join(hf, tar), part, [f"{u}/atlas.npz" for u in uuids], r"s|/atlas\.npz$|/somage.npz|")
        else:
            untar(os.path.join(hf, tar), os.path.join(part, "thumbnails"), [f"{u}/thumbnail.png" for u in uuids],
                  r"s|^\([^/]*\)/thumbnail\.png$|\1.png|")
    uuids = splits["train"] + splits["val"] + splits["test"]
    for u in uuids:
        if not os.path.isfile(os.path.join(part, "thumbnails", f"{u}.png")):
            sys.exit(f"{u}: no thumbnail after extraction")
        with np.load(os.path.join(part, u, "somage.npz")) as z:
            lack = [k for k in NPZ_KEYS if k not in z.files]
            a = z["alpha"] if "alpha" in z.files else None
        if lack or a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 1:
            sys.exit(f"{u}/somage.npz: lacks {lack} or its alpha is not uint8 (H, W, 1)")
    build_index(splits, part)
    if check_index(splits, part) != 0:
        sys.exit("index check failed")
    with open(os.path.join(part, SPLIT_JSON)) as f:
        idx = json.load(f)
    idx["test"]["indices"] = idx["test"]["indices"][:2]
    with open(os.path.join(part, "data_splits_traincheck.json"), "w") as f:
        json.dump(idx, f)
    with open(os.path.join(part, "release_test20.txt"), "w") as f:
        f.write("\n".join(splits["test"]) + "\n")
    with open(os.path.join(part, "SOURCE.json"), "w") as f:
        json.dump({"repo": REPO_ID, "revision": REVISION, "sha256": hashes,
                   "check_root_splits.json": sha256(SPLITS_FILE)}, f, indent=1)
        f.write("\n")
    rel = sorted(os.path.relpath(os.path.join(d, n), part) for d, _, names in os.walk(part) for n in names)
    with open(os.path.join(part, "MANIFEST.sha256"), "w") as f:
        f.writelines(f"{sha256(os.path.join(part, r))}  {r}\n" for r in rel)
    os.rename(part, out)
    print(f"[root] {out}: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} shapes from "
          f"{len(tars)} tars, {len(rel)} files in MANIFEST.sha256")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--hf", required=True, help="the release download dir (splits.json, metadata.parquet, data/...)")
    ap.add_argument("--out", help="root to build")
    ap.add_argument("--fetch", action="store_true", help="download the files the check shapes need first")
    ap.add_argument("--select", action="store_true", help="(re)write check_root_splits.json first")
    ap.add_argument("--hf_cli", default=shutil.which("hf") or "/localhome/dya78/software/miniconda3/bin/hf")
    a = ap.parse_args()
    if a.select:
        select(a.hf)
    if a.fetch:
        fetch(a.hf_cli, a.hf)
    if a.out:
        build(a.hf, a.out)


if __name__ == "__main__":
    main()
