"""Numeric gate comparison: how far a candidate prediction root is from a baseline, and whether
that distance is within the noise of two runs of unchanged code on the same GPU type.

  python compare_numeric.py --a <baseline> --b <candidate> --shas_file <shas>
      [--noise_a <run1> --noise_b <run2>] [--csv <per-shape.csv>]

Both roots are laid out <root>/<sha>/{pred_emission,gt_emission,mask,input_albedo}.png. Per shape
and per file it reports byte identity and the mean and max absolute pixel difference (in 1/255
steps). For pred_emission it also scores both runs against the BASELINE's gt_emission.png over the
baseline's mask.png texels with the evaluator's own metric functions
(`evaluation/pointsampled/metrics.py`: emission mask at eps 1/255, IoU, MAE, PSNR with its 40 dB
cap), so a metric change reflects the prediction alone. These are UV-space numbers: the published
evaluator samples the same PNG at mesh surface points, which this script does not do.

A shape whose mask has no occupied texel has no MAE (the evaluator's point_mae gives NaN; IoU is 1
and PSNR is the 40 dB cap there). Such a shape is reported as a count, left out of the MAE means,
and its |dMAE| counts as 0 when MAE is undefined on both sides and as infinite (a failure) when on
one side only. Every other NaN or infinity also counts as a failure, never as a pass.

Verdict (exit 0 for the first two):
  IDENTICAL     every file byte-identical, nothing missing
  WITHIN_NOISE  nothing missing, and for every file and every statistic (per-shape mean and max
                pixel difference; |dIoU|, |dMAE|, |dPSNR|; the change of the mean IoU, MAE, PSNR)
                the candidate's mean and max over shapes are <= the noise pair's, on the same shapes
  DIFFERENT     no noise pair given and something differs, or something is missing
  OUTSIDE_NOISE a statistic exceeds the noise pair's
  EMPTY         the shapes file lists no shape
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from PIL import Image

REPO = os.environ.get("LIGHTGEN_REPO", "/local-scratch/localhome/dya78/code/lightgen")
sys.path.insert(0, REPO)
from evaluation.pointsampled.metrics import emission_mask, point_iou, point_mae, point_psnr  # noqa: E402

FILES = ("pred_emission.png", "gt_emission.png", "mask.png", "input_albedo.png")
EPS = 1.0 / 255.0  # evaluation/pointsampled/pipeline.py EPS_DEFAULT
METRICS = ("iou", "mae", "psnr")


def absdiff(x, y):
    """|x - y| that never hides a NaN: both NaN -> 0, one NaN -> inf, equal infinities -> 0."""
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return 0.0 if math.isnan(x) and math.isnan(y) else math.inf
    return 0.0 if x == y else abs(x - y)


def finite_mean(values):
    """Mean over the defined (non-NaN) values; NaN when none is defined."""
    v = [x for x in values if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def score(pred_u8, gt_u8, occ):
    """The evaluator's metrics on the occupied texels, values in the canonical linear space."""
    p = pred_u8[occ].astype(np.float64) / 255.0
    g = gt_u8[occ].astype(np.float64) / 255.0
    return {"iou": point_iou(emission_mask(g, EPS), emission_mask(p, EPS)),
            "mae": point_mae(p, g), "psnr": point_psnr(p, g)}


def compare_pair(root_a, root_b, shas):
    """Per-shape rows {sha, <file>_{state,mean,max}, iou_a, iou_b, ...}."""
    rows = []
    for sha in shas:
        row = {"sha": sha}
        imgs = {}
        for fn in FILES:
            k = fn[:-4]
            pa, pb = os.path.join(root_a, sha, fn), os.path.join(root_b, sha, fn)
            if not (os.path.isfile(pa) and os.path.isfile(pb)):
                row[f"{k}_state"] = "missing"
                continue
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                same_bytes = fa.read() == fb.read()
            a, b = load(pa), load(pb)
            imgs[k] = (a, b)
            d = np.abs(a - b) if a.shape == b.shape else None
            row[f"{k}_state"] = "identical" if same_bytes else ("shape" if d is None else "different")
            row[f"{k}_mean"] = float(d.mean()) if d is not None else float("inf")
            row[f"{k}_max"] = float(d.max()) if d is not None else float("inf")
        if all(k in imgs for k in ("pred_emission", "gt_emission", "mask")) and row["pred_emission_state"] != "shape":
            occ = imgs["mask"][0].max(axis=-1) > 0
            gt = imgs["gt_emission"][0]
            sa, sb = score(imgs["pred_emission"][0], gt, occ), score(imgs["pred_emission"][1], gt, occ)
            for m in METRICS:
                row[f"{m}_a"], row[f"{m}_b"] = sa[m], sb[m]
                row[f"d{m}"] = absdiff(sb[m], sa[m])
        rows.append(row)
    return rows


def summarize(rows):
    """Statistic name -> (mean over shapes, max over shapes), plus counts."""
    s, counts = {}, {}
    for fn in FILES:
        k = fn[:-4]
        states = [r.get(f"{k}_state", "missing") for r in rows]
        counts[k] = {st: states.count(st) for st in ("identical", "different", "shape", "missing")}
        for stat in ("mean", "max"):
            v = [r[f"{k}_{stat}"] for r in rows if f"{k}_{stat}" in r]
            s[f"{k}_{stat}"] = (float(np.mean(v)), float(np.max(v))) if v else (0.0, 0.0)
    for m in METRICS:
        v = [r[f"d{m}"] for r in rows if f"d{m}" in r]
        s[f"d{m}"] = (float(np.mean(v)), float(np.max(v))) if v else (0.0, 0.0)
        a = [r[f"{m}_a"] for r in rows if f"{m}_a" in r]
        b = [r[f"{m}_b"] for r in rows if f"{m}_b" in r]
        s[f"mean_{m}_a"], s[f"mean_{m}_b"] = finite_mean(a), finite_mean(b)
        delta = absdiff(s[f"mean_{m}_b"], s[f"mean_{m}_a"]) if a else 0.0
        s[f"mean_{m}_change"] = (delta, delta)
        counts[f"undefined_{m}"] = (sum(math.isnan(x) for x in a), sum(math.isnan(x) for x in b))
    return s, counts


def report(tag, s, counts):
    for fn in FILES:
        k = fn[:-4]
        c = counts[k]
        print(f"[{tag}] {fn:18s} identical={c['identical']} different={c['different'] + c['shape']} "
              f"missing={c['missing']}  |d| per shape (1/255 steps): mean {s[k + '_mean'][0]:.4g} "
              f"(worst shape {s[k + '_mean'][1]:.4g}), max {s[k + '_max'][1]:.4g}")
    for m in METRICS:
        ua, ub = counts[f"undefined_{m}"]
        print(f"[{tag}] {m.upper():4s} mean a={s['mean_' + m + '_a']:.6f} b={s['mean_' + m + '_b']:.6f} "
              f"change {s['mean_' + m + '_change'][0]:.3g};  per shape |d{m}| mean {s['d' + m][0]:.3g} "
              f"max {s['d' + m][1]:.3g}"
              + (f";  undefined on {ua} / {ub} shapes (no occupied texel)" if ua or ub else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline root (tag pre-refactor)")
    ap.add_argument("--b", required=True, help="candidate root")
    ap.add_argument("--shas_file", required=True)
    ap.add_argument("--noise_a", help="noise pair: first run of pre-refactor on the same GPU type")
    ap.add_argument("--noise_b", help="noise pair: second run")
    ap.add_argument("--csv", help="write the candidate's per-shape rows here")
    a = ap.parse_args(argv)
    shas = [l.strip() for l in open(a.shas_file) if l.strip()]
    if not shas:
        print(f"[numeric] n=0 verdict=EMPTY ({a.shas_file} lists no shape)")
        return 1
    rows = compare_pair(a.a, a.b, shas)
    s, counts = summarize(rows)
    report("candidate", s, counts)
    if a.csv:
        keys = sorted({k for r in rows for k in r} - {"sha"})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sha"] + keys)
            w.writeheader()
            w.writerows(rows)
    missing = sum(counts[fn[:-4]]["missing"] for fn in FILES)
    identical = all(counts[fn[:-4]]["identical"] == len(shas) for fn in FILES)
    if identical:
        verdict = "IDENTICAL"
    elif missing or not (a.noise_a and a.noise_b):
        verdict = "DIFFERENT"
    else:
        ns, ncounts = summarize(compare_pair(a.noise_a, a.noise_b, shas))
        report("noise", ns, ncounts)
        if sum(ncounts[fn[:-4]]["missing"] for fn in FILES):
            sys.exit("the noise pair is missing files for these shapes")
        keys = [k for k in s if not k.startswith("mean_") or k.endswith("_change")]
        if any(not math.isfinite(v) for k in keys for v in ns[k]):
            sys.exit("the noise pair has a NaN or infinity in its statistics; it is not a noise baseline")
        # `not (<=)`, so that a NaN in the candidate counts as over the noise
        over = [k for k in keys if not (s[k][0] <= ns[k][0] and s[k][1] <= ns[k][1])]
        verdict = "WITHIN_NOISE" if not over else "OUTSIDE_NOISE (" + ", ".join(sorted(over)) + ")"
    print(f"[numeric] n={len(shas)} verdict={verdict}")
    return 0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1


if __name__ == "__main__":
    sys.exit(main())
