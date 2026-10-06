"""Print one JSON line summarizing a saved checkpoint: what train-check.sh appends to the record.

  python summarize_ckpt.py <ckpt>        (run from the TEXGen root: the checkpoint pickles spuv classes)

Fields: the top-level keys, epoch, global_step, `_scheduler_states` (last_epoch, _last_lr) and
Lightning's own `lr_schedulers` last_epoch, the state_dict size and the float64 sum of its floating
entries, the optimizer state counts, and every non-torch class the file pickles (the class paths a
checkpoint written by this code depends on).
"""
import json
import os
import pickle
import sys
import types

import torch

sys.path.insert(0, os.getcwd())
classes = set()


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if not module.startswith(("torch", "collections", "builtins", "_codecs", "numpy")):
            classes.add(f"{module}.{name}")
        return super().find_class(module, name)


_pm = types.ModuleType("pickle")
_pm.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
_pm.Unpickler = _Unpickler

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, pickle_module=_pm)
sd = ck.get("state_dict", {})
floats = [v for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()]
print(json.dumps({
    "kind": "ckpt", "file": os.path.basename(sys.argv[1]), "keys": sorted(ck.keys()),
    "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
    "scheduler": [[s.get("last_epoch"), s.get("_last_lr")] for s in ck.get("_scheduler_states", [])],
    "lightning_scheduler": [s.get("last_epoch") for s in ck.get("lr_schedulers", [])],
    "n_state": len(sd),
    "state_sum": float(sum(v.double().sum().item() for v in floats)),
    "n_optimizer_states": [len(o.get("state", {})) for o in ck.get("optimizer_states", [])],
    "classes": sorted(classes),
}, sort_keys=True))
