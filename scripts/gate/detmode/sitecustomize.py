import os
if os.environ.get("TEXGEN_GATE_DETERMINISTIC") == "1":
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import sys; print("[detmode] torch.use_deterministic_algorithms(True, warn_only=True); cudnn.deterministic=True", file=sys.stderr, flush=True)
