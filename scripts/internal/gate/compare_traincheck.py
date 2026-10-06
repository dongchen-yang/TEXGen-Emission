"""Compare two training-check records (train-check.sh's <out_dir>/record.jsonl).

  python compare_traincheck.py --a <baseline record> --b <candidate record>
      [--noise_a <run1 record> --noise_b <run2 record>]

First the structure must match: the same training steps in the same order with the same batch
scene ids, the same validation and test passes, and a saved checkpoint with the same keys, epoch,
step, scheduler state, state_dict size and pickled class paths. Then per quantity it reports the
largest absolute difference over all events:
  train   loss, lr, grad_norm (before clipping), param_sum (backbone weights after the step)
  val     param_sum, ema_sum and every val/* metric, per validation pass (sanity pass included)
  test    every metric of the test pass
  ckpt    state_sum (all floating state_dict entries)
A difference is NaN-safe: NaN on both sides counts as equal, NaN on one side or two unequal
infinities as an infinite difference, so a run that turns NaN can never pass.

Verdict (exit 0 for the first two):
  IDENTICAL     structure matches and every number is equal
  WITHIN_NOISE  structure matches, and every quantity's largest difference is <= the noise pair's
  DIFFERENT     structure differs, or numbers differ and no noise pair was given
  OUTSIDE_NOISE a quantity's largest difference exceeds the noise pair's
  INCOMPLETE    a record has no training step or no checkpoint summary (an empty or cut-off run)
"""
import argparse
import json
import math
import sys

TRAIN_KEYS = ("loss", "lr", "grad_norm", "param_sum")


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return {k: [r for r in recs if r["kind"] == k] for k in ("fit_start", "train", "val", "test", "ckpt")}


def structure(r):
    return {
        "train": [(x["gs"], x["epoch"], x["bi"], tuple(x["scene_ids"])) for x in r["train"]],
        "val": [(x["sanity"], x["gs"], x["epoch"], tuple(sorted(k for k in x if k.startswith("val/")))) for x in r["val"]],
        "test": [tuple(sorted(k for k in x if k not in ("kind", "gs"))) for x in r["test"]],
        "ckpt": [(x["keys"], x["epoch"], x["global_step"], x["scheduler"], x["lightning_scheduler"], x["n_state"],
                  x["n_optimizer_states"], x["classes"]) for x in r["ckpt"]],
        "fit_start": [(x["global_step"], x["epoch"], x["n_backbone_params"]) for x in r["fit_start"]],
    }


def absdiff(x, y):
    """|x - y| that never hides a NaN: both NaN -> 0, one NaN -> inf, equal infinities -> 0."""
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return 0.0 if math.isnan(x) and math.isnan(y) else math.inf
    return 0.0 if x == y else abs(x - y)


def incomplete(r, path):
    if not r["train"] or not r["ckpt"]:
        return f"{path}: {len(r['train'])} training steps, {len(r['ckpt'])} checkpoint summaries"
    return None


def diffs(a, b):
    """Quantity name -> largest |a - b| over all events (structure already checked equal)."""
    d = {}

    def put(name, x, y):
        d[name] = max(d.get(name, 0.0), absdiff(x, y))

    for x, y in zip(a["train"], b["train"]):
        for k in TRAIN_KEYS:
            put(f"train/{k}", x[k], y[k])
    for x, y in zip(a["val"], b["val"]):
        tag = "val_sanity" if x["sanity"] else "val"
        for k in x:
            if k.startswith("val/") or k in ("param_sum", "ema_sum"):
                put(f"{tag}:{k}", x[k], y[k])
    for x, y in zip(a["test"], b["test"]):
        for k in x:
            if k not in ("kind", "gs"):
                put(f"test:{k}", x[k], y[k])
    for x, y in zip(a["ckpt"], b["ckpt"]):
        put("ckpt:state_sum", x["state_sum"], y["state_sum"])
    return d


def first_mismatch(sa, sb):
    for k in sa:
        if sa[k] != sb[k]:
            n = min(len(sa[k]), len(sb[k]))
            i = next((i for i in range(n) if sa[k][i] != sb[k][i]), n)
            return f"{k}: event {i} ({len(sa[k])} vs {len(sb[k])} events)"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise_a")
    ap.add_argument("--noise_b")
    a = ap.parse_args(argv)
    ra, rb = load(a.a), load(a.b)
    empty = incomplete(ra, a.a) or incomplete(rb, a.b)
    if empty:
        print(f"[traincheck] {empty}")
        print("[traincheck] verdict=INCOMPLETE")
        return 1
    bad = first_mismatch(structure(ra), structure(rb))
    print(f"[traincheck] steps={len(ra['train'])} val_passes={len(ra['val'])} test={len(ra['test'])} "
          f"ckpts={len(ra['ckpt'])}  structure: {'same' if bad is None else 'DIFFERENT at ' + bad}")
    if bad is not None:
        print("[traincheck] verdict=DIFFERENT")
        return 1
    d = diffs(ra, rb)
    nd = None
    if a.noise_a and a.noise_b:
        na, nb = load(a.noise_a), load(a.noise_b)
        nempty = incomplete(na, a.noise_a) or incomplete(nb, a.noise_b)
        if nempty:
            sys.exit(f"the noise pair is incomplete: {nempty}")
        nbad = first_mismatch(structure(na), structure(nb))
        if nbad is not None:
            sys.exit(f"the noise pair's structures differ: {nbad}")
        if first_mismatch(structure(ra), structure(na)) is not None:
            sys.exit("the noise pair is not the same check as --a (different steps or passes)")
        nd = diffs(na, nb)
        if any(math.isinf(v) for v in nd.values()):
            sys.exit("the noise pair has a NaN or infinity on one side only; it is not a noise baseline")
    for k in sorted(d):
        print(f"[traincheck] {k:32s} max|d| {d[k]:.6g}" + (f"   noise {nd[k]:.6g}" if nd is not None else ""))
    if all(v == 0.0 for v in d.values()):
        verdict = "IDENTICAL"
    elif nd is None:
        verdict = "DIFFERENT"
    else:
        over = [k for k in d if d[k] > nd[k]]
        verdict = "WITHIN_NOISE" if not over else "OUTSIDE_NOISE (" + ", ".join(sorted(over)) + ")"
    print(f"[traincheck] verdict={verdict}")
    return 0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1


if __name__ == "__main__":
    sys.exit(main())
