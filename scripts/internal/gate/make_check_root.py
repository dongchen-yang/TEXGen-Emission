"""Build the small release data root the release gate and the training check read.

  python make_check_root.py --release <dataset_release dir> --out <root>

The shapes are fixed by `check_root_splits.json` beside this file (written once with --select):
16 release train shapes, 2 release val shapes and 20 release test shapes, each list a fixed
stride over the sorted release split. The root has the loader's layout (Keep stable #7):

  <root>/<uuid>/somage.npz              the release's <uuid>/atlas.npz (alpha inside), copied
  <root>/thumbnails/<uuid>.png          the release's <uuid>/thumbnail.png
  <root>/df_lightgenbench.parquet       rows train, val, test (scripts/lightgenbench/build_index.py)
  <root>/data_splits_lightgenbench.json train 0-15, val 16-17, test 18-37
  <root>/data_splits_traincheck.json    the same, test cut to its first 2 shapes (the training
                                        check runs trainer.test after fit; 20 would cost 2.5 min)
  <root>/release_test20.txt             the 20 test uuids, the release gate's shapes file

Idempotent: a file whose size already matches is not copied again.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_FILE = os.path.join(HERE, "check_root_splits.json")
TRAIN_STRIDE, VAL_STRIDE, TEST_STRIDE = 2277, 100, 10  # 36,426 -> 16, 200 -> 2, 200 -> 20


def select(release):
    with open(os.path.join(release, "splits.json")) as f:
        s = json.load(f)
    pick = {"train": sorted(s["train"])[::TRAIN_STRIDE], "val": sorted(s["val"])[::VAL_STRIDE],
            "test": sorted(s["test"])[::TEST_STRIDE]}
    assert [len(pick[k]) for k in ("train", "val", "test")] == [16, 2, 20], {k: len(v) for k, v in pick.items()}
    with open(SPLITS_FILE, "w") as f:
        json.dump(pick, f, indent=1)
        f.write("\n")
    print(f"[select] wrote {SPLITS_FILE}")


def copy_if_needed(src, dst):
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    shutil.copyfile(src, dst + ".part")
    os.replace(dst + ".part", dst)
    return True


def build(release, out):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "lightgenbench"))
    import build_index  # the release run's own index writer, unchanged

    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    os.makedirs(os.path.join(out, "thumbnails"), exist_ok=True)
    n_copied = 0
    for uuid in splits["train"] + splits["val"] + splits["test"]:
        os.makedirs(os.path.join(out, uuid), exist_ok=True)
        n_copied += copy_if_needed(os.path.join(release, uuid, "atlas.npz"), os.path.join(out, uuid, "somage.npz"))
        n_copied += copy_if_needed(os.path.join(release, uuid, "thumbnail.png"),
                                   os.path.join(out, "thumbnails", f"{uuid}.png"))
    build_index.build(splits, out)
    if build_index.check(splits, out) != 0:
        sys.exit("index check failed")
    with open(os.path.join(out, build_index.SPLIT_JSON)) as f:
        idx = json.load(f)
    idx["test"]["indices"] = idx["test"]["indices"][:2]
    with open(os.path.join(out, "data_splits_traincheck.json"), "w") as f:
        json.dump(idx, f)
    with open(os.path.join(out, "release_test20.txt"), "w") as f:
        f.write("\n".join(splits["test"]) + "\n")
    print(f"[root] {out}: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} shapes, "
          f"{n_copied} files copied")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--release", required=True, help="dataset_release dir (per-uuid dirs + splits.json)")
    ap.add_argument("--out", help="root to build")
    ap.add_argument("--select", action="store_true", help="(re)write check_root_splits.json first")
    a = ap.parse_args()
    if a.select:
        select(a.release)
    if a.out:
        build(a.release, a.out)


if __name__ == "__main__":
    main()
