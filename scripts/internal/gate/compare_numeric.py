"""Numeric gate comparison: how far a candidate prediction root is from a baseline, and whether
that distance is within the measured noise of unchanged code on the same GPU type.

  python compare_numeric.py --a <baseline> --b <candidate> --shas_file <shas>
      [--envelope <run> <run> ... [--envelope_shas <shas>] [--envelope_json <cache>]]
      [--csv <per-shape.csv>]

The envelope (REFACTOR_TODO.md, How we work) comes from several runs of tag pre-refactor on that GPU
type: every pair of them is compared like a candidate, over --envelope_shas (default --shas_file;
give the check's full shape list, so that every shard of a check meets the same envelope), and per
measure the largest per-shape difference of any pair is the bound. The measures, per shape: for
each file the mean and the max absolute pixel difference, and |dIoU|, |dMAE|, |dPSNR|. A candidate
is within the envelope when, for every measure, no shape of it differs from the baseline by more.
(The change of the mean IoU, MAE, PSNR is printed, not tested: it is bounded by the per-shape ones.)
--envelope_json caches the envelope; it is recomputed when the runs, the shapes, any file the
envelope reads (path, size, mtime) or the cache's schema (ENVELOPE_SCHEMA) differ.
--noise_a <run1> --noise_b <run2> is the same as --envelope <run1> <run2>.

Tolerance first (Dongchen, 2026-10-07): a shape passes as noise, whatever the envelope says, when in
each of its four files every pixel differs from the baseline by at most 1/255 (TOLERANCE) and at most
0.5% of the file's values differ (TOLERANCE_MAX_FRACTION; values = the H x W x 3 channel values of
the whole image, so 0.5% of a 256 x 256 RGB image is 983 values). Real second outputs differ on
about 0.1% of values (worst shape seen 0.67%); +1/255 on every emitting texel of 5 release shapes
differed on 0.50-9.4%. The shape's IoU, MAE and PSNR deltas are still reported. Every other shape
meets the envelope.

Both roots are laid out <root>/<sha>/{pred_emission,gt_emission,mask,input_albedo}.png. Per shape
and per file it reports byte identity, the mean and max absolute pixel difference (in 1/255 steps)
and the fraction of values that differ. For pred_emission it also scores both runs against the
BASELINE's gt_emission.png over the baseline's mask.png texels with the evaluator's own metric functions
(`evaluation/pointsampled/metrics.py`: emission mask at eps 1/255, IoU, MAE, PSNR with its 40 dB
cap), so a metric change reflects the prediction alone. These are UV-space numbers: the published
evaluator samples the same PNG at mesh surface points, which this script does not do.

A shape whose mask has no occupied texel has no MAE (the evaluator's point_mae gives NaN; IoU is 1
and PSNR is the 40 dB cap there). Such a shape is reported as a count, left out of the MAE means,
and its |dMAE| counts as 0 when MAE is undefined on both sides and as infinite (a failure) when on
one side only. Every other NaN or infinity also counts as a failure, never as a pass.

Verdict (exit 0 for the first two):
  IDENTICAL     every file byte-identical, nothing missing
  WITHIN_NOISE  nothing missing, every shape within the tolerance or, if over it, within the envelope
  DIFFERENT     something is missing, or a shape is over the tolerance and no envelope was given
  OUTSIDE_NOISE a shape over the tolerance exceeds the envelope in a measure
  EMPTY         the shapes file lists no shape
"""
import argparse
import csv
import hashlib
import itertools
import json
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
            row[f"{k}_frac"] = float(np.count_nonzero(d) / d.size) if d is not None else float("inf")
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
        for stat in ("mean", "max", "frac"):
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
              f"(worst shape {s[k + '_mean'][1]:.4g}), max {s[k + '_max'][1]:.4g};  values differing: "
              f"worst shape {100 * s[k + '_frac'][1]:.3g}%")
    for m in METRICS:
        ua, ub = counts[f"undefined_{m}"]
        print(f"[{tag}] {m.upper():4s} mean a={s['mean_' + m + '_a']:.6f} b={s['mean_' + m + '_b']:.6f} "
              f"change {s['mean_' + m + '_change'][0]:.3g};  per shape |d{m}| mean {s['d' + m][0]:.3g} "
              f"max {s['d' + m][1]:.3g}"
              + (f";  undefined on {ua} / {ub} shapes (no occupied texel)" if ua or ub else ""))


TOLERANCE = 1                   # 1/255 steps: a larger per-pixel difference in any file meets the envelope
TOLERANCE_MAX_FRACTION = 0.005  # and so does a file with more than this share of its channel values differing
MEASURES = [f"{fn[:-4]}_{stat}" for fn in FILES for stat in ("mean", "max")] + [f"d{m}" for m in METRICS]
ENVELOPE_SCHEMA = 2             # bump when MEASURES or their definition change: older caches are recomputed


def within_tolerance(row):
    """A shape is noise when, in each of the four files, no pixel differs by more than TOLERANCE and at
    most TOLERANCE_MAX_FRACTION of the values differ. Missing or reshaped files never are."""
    return all(row.get(f"{fn[:-4]}_max", math.inf) <= TOLERANCE
               and row.get(f"{fn[:-4]}_frac", math.inf) <= TOLERANCE_MAX_FRACTION for fn in FILES)


def files_digest(runs, shas):
    """sha256 over path, size and mtime of every file the envelope reads: a rewritten run is noticed."""
    h = hashlib.sha256()
    for r in runs:
        for sha in shas:
            for fn in FILES:
                p = os.path.join(os.path.abspath(r), sha, fn)
                try:
                    st = os.stat(p)
                    h.update(f"{p} {st.st_size} {st.st_mtime_ns}\n".encode())
                except FileNotFoundError:
                    h.update(f"{p} missing\n".encode())
    return h.hexdigest()


def envelope(runs, shas, cache=None):
    """Per measure, the largest per-shape difference between any two runs; cached in `cache`."""
    key = {"schema": ENVELOPE_SCHEMA, "measure_names": MEASURES, "runs": [os.path.abspath(r) for r in runs],
           "n": len(shas), "shas_sha256": hashlib.sha256("\n".join(shas).encode()).hexdigest(),
           "files_sha256": files_digest(runs, shas)}
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            e = json.load(f)
        if all(e.get(k) == v for k, v in key.items()):
            return e
    pairs = {}
    for i, j in itertools.combinations(range(len(runs)), 2):
        s, counts = summarize(compare_pair(runs[i], runs[j], shas))
        if sum(counts[fn[:-4]]["missing"] for fn in FILES):
            sys.exit(f"envelope runs {runs[i]} and {runs[j]} are missing files for these shapes")
        pairs[f"{i}-{j}"] = {"identical": all(counts[fn[:-4]]["identical"] == len(shas) for fn in FILES),
                             "measures": {k: s[k][1] for k in MEASURES}}
    env = {k: max(p["measures"][k] for p in pairs.values()) for k in MEASURES}
    if any(not math.isfinite(v) for v in env.values()):
        sys.exit("an envelope pair has a NaN or infinity in a measure; it is not a noise envelope")
    e = {**key, "measures": env, "pairs": pairs}
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache + ".part", "w") as f:
            json.dump(e, f, indent=1)
        os.replace(cache + ".part", cache)
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline root (tag pre-refactor)")
    ap.add_argument("--b", help="candidate root (omitted: only measure and print the envelope)")
    ap.add_argument("--shas_file", required=True)
    ap.add_argument("--envelope", nargs="+", help="two or more runs of pre-refactor on the same GPU type")
    ap.add_argument("--envelope_shas", help="shapes the envelope is measured over (default: --shas_file)")
    ap.add_argument("--envelope_json", help="cache file for the envelope")
    ap.add_argument("--noise_a", help="same as --envelope <noise_a> <noise_b>")
    ap.add_argument("--noise_b")
    ap.add_argument("--csv", help="write the candidate's per-shape rows here")
    a = ap.parse_args(argv)
    runs = a.envelope or ([a.noise_a, a.noise_b] if a.noise_a and a.noise_b else None)
    if runs is not None and len(runs) < 2:
        sys.exit("--envelope needs two or more runs")
    shas = [l.strip() for l in open(a.shas_file) if l.strip()]
    if not shas:
        print(f"[numeric] n=0 verdict=EMPTY ({a.shas_file} lists no shape)")
        return 1
    if a.b is None:
        if runs is None:
            sys.exit("without --b, give --envelope")
        e = envelope(runs, shas, a.envelope_json)
        print(f"[envelope] {len(runs)} runs, {len(e['pairs'])} pairs ({sum(p['identical'] for p in e['pairs'].values())} "
              f"identical), {len(shas)} shapes: " + ", ".join(f"{k} {e['measures'][k]:.3g}" for k in MEASURES))
        return 0
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
    # Dongchen 2026-10-07: a shape whose four files differ by at most 1/255 per pixel, each on at most 0.5%
    # of its values, is noise and passes whatever the envelope says; every other shape meets the envelope.
    big = [r for r in rows if not within_tolerance(r)]
    by_share = sum(all(r.get(f"{fn[:-4]}_max", math.inf) <= TOLERANCE for fn in FILES) for r in big)
    print(f"[tolerance] {len(rows) - len(big)} of {len(rows)} shapes within {TOLERANCE}/255 on at most "
          f"{100 * TOLERANCE_MAX_FRACTION:g}% of the values in all four files (noise, pass); {len(big)} differ "
          f"by more" + (f" ({by_share} of them within {TOLERANCE}/255 but on more values)" if by_share else ""))
    if identical:
        verdict = "IDENTICAL"
    elif missing:
        verdict = "DIFFERENT"
    elif not big:
        verdict = "WITHIN_NOISE"
    elif runs is None:
        verdict = "DIFFERENT"
    else:
        s, _ = summarize(big)   # the envelope test covers only the shapes over the tolerance
        eshas = [l.strip() for l in open(a.envelope_shas)] if a.envelope_shas else shas
        eshas = [x for x in eshas if x]
        if set(shas) - set(eshas):
            sys.exit("--shas_file lists shapes the envelope is not measured over")
        e = envelope(runs, eshas, a.envelope_json)
        env = e["measures"]
        print(f"[envelope] {len(runs)} runs, {len(e['pairs'])} pairs ({sum(p['identical'] for p in e['pairs'].values())} "
              f"identical), {len(eshas)} shapes: " + ", ".join(f"{k} {env[k]:.3g}" for k in MEASURES))
        # `not (<=)`, so that a NaN in the candidate counts as over the envelope
        over = [k for k in MEASURES if not (s[k][1] <= env[k])]
        verdict = "WITHIN_NOISE" if not over else "OUTSIDE_NOISE (" + ", ".join(
            f"{k} {s[k][1]:.3g} > {env[k]:.3g}" for k in sorted(over)) + ")"
    print(f"[numeric] n={len(shas)} verdict={verdict}")
    return 0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1


if __name__ == "__main__":
    sys.exit(main())
