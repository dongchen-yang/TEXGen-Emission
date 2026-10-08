#!/usr/bin/env python3
"""Fan-out driver for the refactor checks: one command runs every check for one TEXGen tree on the
GPUs that are free right now, compares each result with its own GPU type's baseline and prints one
verdict (REFACTOR_TODO.md step 7b; How we work: never wait for a GPU).

  python3 fanout.py --rev <commit>  [--checks paper,release,fresh,resume] [--paper_shas <file>]
  python3 fanout.py --tree <dir>    ...          a tree on the workstation (e.g. a git archive)
  python3 fanout.py ... --plan                   probe and print the plan, run nothing
  python3 fanout.py --rescore <C>/fanout/<label> compare a finished run's stored outputs again under the
                                                 current rule (no GPU); new summary_rescored_* files
  python3 fanout.py --stage <lab host>           stage a lab 4090: env, checkpoints, paper gate data
  python3 fanout.py --qualify <lab host>         subset20 of the paper gate on tag pre-refactor there,
                                                 against the 4090 envelope; a PASS admits the host

Run it on the workstation, from any directory; stdlib only (comparisons run in the texgen env).
  1. Probe, once, right before dispatch. Lab 4090s (workstation, sp4090, xg4090), over ssh: no compute
     process and enough free GPU memory, every 4090 lock of the host free (flock -n), enough free host
     RAM (another session can fill the workstation's 62 GB), the inputs staged, and, except for the
     workstation, admitted by --qualify. Solar: free L40S on our L40S nodes and theia-lab's
     (sinfo + scontrol, by the compute-access skill's hostprobe rules), with 6 CPUs and 64 GB per GPU
     free. Alliance clusters are
     not probed: no login socket is kept open for them and they have no baselines.
  2. Plan: the release gate and each training check go to a GPU whose type has that baseline, with an
     envelope of at least 4 pre-refactor runs (baselines.json, ruling R20), each where it finishes
     first; the paper gate's shapes are then split over every free GPU so that all finish together
     (per-type speeds below, from steps 7 and 7b).
  3. Solar first: one job per node, sized to its free L40S, largest first, with --deadline so that it
     starts within 2 minutes or Slurm cancels it; no -w on our nodes, -w on theia-lab's (SOLAR_OTHER,
     where we rank lower and cannot evict anyone); a job still pending after 30 s is
     cancelled. sacct then lists any preemption on the nodes used. The lab GPUs are planned again for
     the work no running job holds.
  4. Lab GPUs: one queue per GPU, all of it under that host's 4090 locks, each taken once (flock -n):
     a held lock skips the host. gate-det.sh and train-check.sh run with GATE_LOCK=none inside it.
  5. Outputs come back to <C>/fanout/<label>/out/; each task is compared with its own GPU type's
     baseline: gate images within 1/255 per pixel pass, larger differences meet the envelope; the
     training check meets its envelope (compare_numeric.py, compare_traincheck.py; the gate envelopes
     are cached in <C>/envelopes/). Its gpu.txt must name a card of the type it was planned for.
  6. Rerun once (R20): a task outside the envelope runs again, after a new probe, on another free GPU
     of the same type (the same GPU only when no other is free; on Solar, Slurm picks the GPU). It
     fails only if the rerun is outside too; the summary lists both runs.
Exit 0 PASS: every requested shape and check passed (IDENTICAL or WITHIN_NOISE, at once or on its
     rerun).  1 FAIL: a result outside the envelope twice, a structure or file difference, or the
     wrong card.  2 INCOMPLETE: something did not run, or a rerun found no free GPU (the summary names
     it and the command that finishes it).  3 NOTHING_FREE: no GPU free for any check.
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEXGEN_REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
L = "/local-scratch/localhome/dya78/lightgen_eval_local"
C = f"{L}/refactor_checks"
LIGHTGEN = "/local-scratch/localhome/dya78/code/lightgen"
PY = "/localhome/dya78/software/miniconda3/envs/texgen/bin/python"
ENV_DIR = "/localhome/dya78/software/miniconda3/envs/texgen"
HOSTPROBE = os.path.expanduser("~/.claude/skills/compute-access/scripts/hostprobe.py")
JUPITER = "/cs/3dlg-jupiter-project/lightgen/outputs"
PAPER_SHAS = f"{L}/test_agentic_clean.txt"
HF_ROOT = f"{C}/hf_root"
CKPTS = {"paper": ("texgen_alpha_74k_v2_agentic", "epoch=124-step=35500.ckpt",
                   "2e52305fd352a931f20d03e942b2b4e85665218d198d9e29cfe5c56ab61249f6"),
         "release": ("texgen_alpha_lightgenbench_v1", "epoch=124-step=35625.ckpt",
                     "6c42a311f5925f5bcfd87a1bb68592a76ae71d53ce0efe3c90e880bf8e3892aa")}
LOCK_GATE = "/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock"   # gate-det.sh's default
LOCK_SWEEPS = "/local-scratch/localhome/dya78/lightgen_4090.lock"       # the eval sweeps and hostprobe
LAB = {  # ckpts: the dir holding <run>/{ckpts,configs}; activate: "" = gate-det.sh's default env
    "local": {"ssh": None, "locks": [LOCK_GATE, LOCK_SWEEPS], "activate": "", "repo": LIGHTGEN,
              "ckpts": f"{LIGHTGEN}/TEXGen/outputs", "gate": HERE},
    "sp4090": {"ssh": "sp4090", "locks": [LOCK_SWEEPS], "activate": f"{C}/tooling/activate_sp4090.sh",
               "repo": f"{C}/tooling/repo", "ckpts": f"{C}/ckpts", "gate": f"{C}/tooling/gate"},
    "xg4090": {"ssh": "xg4090", "locks": [LOCK_SWEEPS], "activate": f"{C}/tooling/activate_xg4090.sh",
               "repo": f"{C}/tooling/repo", "ckpts": f"{C}/ckpts", "gate": f"{C}/tooling/gate"},
}
SOLAR_NODES = ["cs-venus-15", "cs-venus-16", "cs-venus-17"]   # our L40S nodes (compute-access node cheatsheet)
# Other labs' L40S nodes (theia-lab). Slurm does not route our unpinned jobs there, so they are pinned with
# -w: we rank below their owners, so a pinned job there can be preempted but never evicts anyone (on our
# own nodes -w would evict another lab's running job, so those are never pinned). --deadline still applies.
SOLAR_OTHER = ["cs-venus-18", "cs-venus-20"]
SOLAR_CPU, SOLAR_MEM_GB = 6, 64                                # per GPU; step 7's 4-GPU job peaked at 171 GB
# Seconds, measured in steps 7 and 7b: per paper shape, per gate start (model load), per fixed check.
SPEED = {"rtx4090": {"shape": 6.6, "gate_start": 40, "release": 180, "fresh": 120, "resume": 100},
         "l40s": {"shape": 18.5, "gate_start": 60, "release": 420, "fresh": 300, "resume": 300}}
SOLAR_START = 200          # job start, staging check, env probe; a node not staged yet adds ~2 min
VRAM_NEED_MIB = {"paper": 22000, "release": 22000, "resume": 22000, "fresh": 24000}  # free, on the card
# Free host RAM (MemAvailable) per check; step 7b measured drops of 13 (gate), 18 (fresh), 22 GB (resume).
RAM_NEED_GB = {"paper": 24, "release": 24, "fresh": 24, "resume": 32}
MIN_ENVELOPE = 4           # pre-refactor runs per GPU type and check (ruling R20)
MIN_SHARD = 4              # fewer paper shapes than this on a GPU cost more in model loading than they save
FIXED = ("release", "fresh", "resume")


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def say(*a):
    print(f"[fanout {now()}]", *a, flush=True)


def sh(cmd, timeout=60, check=False, input=None):
    """Run a shell command; returns (rc, stdout), stderr appended only on failure (star2 prints its
    login banner on stderr for every ssh). Never raises on timeout."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout,
                           input=input)
        out = p.stdout if p.returncode == 0 else p.stdout + p.stderr
        if check and p.returncode != 0:
            raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{out}")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"command timed out: {cmd}")
        return 124, ""


def on(host, cmd, timeout=60, check=False):
    """Run a shell command on a lab host (None = here)."""
    ssh = LAB[host]["ssh"]
    if ssh is None:
        return sh(["bash", "-c", cmd], timeout, check)
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh, cmd], timeout, check)


def star2(cmd, timeout=60, check=False, input=None):
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "star2", cmd], timeout, check, input)


def rsync(src, dst, *extra, timeout=1800):
    return sh(["rsync", "-a", *extra, src, dst], timeout, check=True)


def load_baselines():
    with open(os.path.join(HERE, "baselines.json")) as f:
        b = json.load(f)
    return b, b.pop("_gpu_names"), b.pop("_doc", None)


# ----------------------------------------------------------------------------------- probe
def probe_lab(host, need_staged=True):
    """One ssh: the card, its compute processes, the host's locks, free RAM and disk, staged inputs."""
    h = LAB[host]
    ck = h["ckpts"]
    files = [f"{ck}/{run}/ckpts/{f}" for run, f, _ in CKPTS.values()] + \
            [f"{ck}/{run}/configs/parsed.yaml" for run, _, _ in CKPTS.values()] + \
            [f"{h['repo']}/evaluation/newdata_eval/run_texgen_infer.py", PAPER_SHAS,
             f"{L}/atlas_root/eval_parquet_test204.parquet"] + ([h["activate"]] if h["activate"] else [])
    cmd = ("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits; echo ---; "
           "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; echo ---; "
           + "".join(f"(flock -n {shlex.quote(l)} true 2>/dev/null && echo free) || echo held; " for l in h["locks"])
           + "echo ---; awk '/MemAvailable/{print $2}' /proc/meminfo; echo ---; "
           "df -Pk /local-scratch | tail -1 | awk '{print $4}'; echo ---; "
           + "".join(f"[ -e {shlex.quote(p)} ] || echo {shlex.quote(p)}; " for p in files))
    rc, out = on(host, cmd, timeout=30)
    row = {"host": host, "kind": "lab", "free": False, "reason": "", "type": None}
    parts = out.split("---")
    if rc != 0 or len(parts) != 6:
        row["reason"] = f"unreachable (rc={rc})"
        return row
    gpus = [l for l in parts[0].strip().splitlines() if l.strip()]
    if len(gpus) != 1:
        row["reason"] = f"nvidia-smi: {len(gpus)} GPU lines"
        return row
    name, used, total = [x.strip() for x in gpus[0].split(",")]
    apps = [l for l in parts[1].strip().splitlines() if l.strip()]
    locks = parts[2].split()
    ram_gb = int(parts[3].strip() or 0) / 1048576
    disk_gb = int(parts[4].strip() or 0) / 1048576
    missing = [l for l in parts[5].strip().splitlines() if l.strip()]
    row.update(gpu=name, vram_free=int(total) - int(used), ram_free_gb=round(ram_gb, 1),
               disk_free_gb=round(disk_gb), apps=len(apps))
    reasons = []
    if apps:
        reasons.append(f"{len(apps)} compute process(es) on the GPU")
    if "held" in locks:
        reasons.append("a 4090 lock is held")
    if ram_gb < min(RAM_NEED_GB.values()):
        reasons.append(f"{ram_gb:.0f} GB free host RAM < {min(RAM_NEED_GB.values())}")
    if row["vram_free"] < min(VRAM_NEED_MIB.values()):
        reasons.append(f"{row['vram_free']} MiB free on the GPU")
    if disk_gb < 30:
        reasons.append(f"{disk_gb:.0f} GB free on /local-scratch")
    if need_staged and missing:
        reasons.append(f"not staged ({len(missing)} inputs missing, e.g. {missing[0]}): run --stage {host}")
    row["reason"] = "; ".join(reasons)
    row["free"] = not reasons
    return row


def load_hostprobe():
    spec = importlib.util.spec_from_file_location("hostprobe", HOSTPROBE)
    hp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp)
    return hp


def probe_solar(nodes):
    """Free L40S per node, by hostprobe's rules (free = CfgTRES - AllocTRES; node state usable)."""
    try:
        hp = load_hostprobe()
    except Exception as e:
        return [{"host": n, "kind": "solar", "free_gpus": 0, "reason": f"hostprobe unavailable: {e}"} for n in nodes]
    rc, si = hp.ssh("star2", 'sinfo -N -h -o "%N|%G|%m|%c|%t" | sort -u', timeout=30)
    rc2, ct = hp.ssh("star2", "scontrol -d show node " + ",".join(nodes), timeout=30)
    rows = []
    if rc != 0 or rc2 != 0:
        return [{"host": n, "kind": "solar", "free_gpus": 0, "reason": "star2 unreachable"} for n in nodes]
    si, ct = hp.parse_sinfo(si), hp.parse_scontrol(ct)
    for n in nodes:
        row = {"host": n, "kind": "solar", "free_gpus": 0, "reason": ""}
        if n not in ct or n not in si:
            row["reason"] = "no Slurm record"
        elif "l40s" not in si[n]["gres"]:
            row["reason"] = f"gres {si[n]['gres']}, not L40S"
        else:
            state, free, reason = hp.solar_state(si[n], ct[n])
            row["reason"] = f"{state}: {reason}"
            if state in ("idle", "partial"):
                cpus = ct[n]["cpu_total"] - ct[n]["cpu_alloc"]
                mem = (ct[n]["mem_total_mb"] - ct[n]["mem_alloc_mb"]) // 1024
                row["free_gpus"] = max(0, min(free, cpus // SOLAR_CPU, mem // SOLAR_MEM_GB))
        rows.append(row)
    return rows


def qualified(host):
    return host == "local" or os.path.isfile(f"{C}/qualified/{host}.txt")


# ------------------------------------------------------------------------------------ plan
def fits(w, check):
    return w["vram_free"] >= VRAM_NEED_MIB[check] and w["ram_free_gb"] >= RAM_NEED_GB[check]


def has_baseline(baselines, typ, check):
    """A check runs on a GPU type only with an envelope of at least MIN_ENVELOPE pre-refactor runs on disk."""
    b = baselines.get(typ, {}).get(check)
    runs = b.get("envelope", []) if b else []
    return len(runs) >= MIN_ENVELOPE and all(os.path.exists(p) for p in runs) and os.path.exists(b["run"])


def plan(workers, checks, shas, baselines):
    """{worker id: [(task name, check, shas or None)]}, {worker id: planned seconds}, [unplaced checks]."""
    load = {w["id"]: w["start"] for w in workers}
    tasks = {w["id"]: [] for w in workers}
    unplaced = []
    for c in sorted((c for c in checks if c in FIXED), key=lambda c: -SPEED["l40s"][c]):
        cand = [w for w in workers if has_baseline(baselines, w["type"], c) and fits(w, c)]
        if not cand:
            unplaced.append(c)
            continue
        w = min(cand, key=lambda w: load[w["id"]] + SPEED[w["type"]][c])
        tasks[w["id"]].append((c, c, None))
        load[w["id"]] += SPEED[w["type"]][c]
    if "paper" in checks and shas:
        active = [w for w in workers if has_baseline(baselines, w["type"], "paper") and fits(w, "paper")]
        while active:  # water-fill: the finish time T at which the shapes are all placed
            rate = {w["id"]: 1.0 / SPEED[w["type"]]["shape"] for w in active}
            off = {w["id"]: load[w["id"]] + SPEED[w["type"]]["gate_start"] for w in active}
            t_end = (len(shas) + sum(off[i] * rate[i] for i in rate)) / sum(rate.values())
            share = {i: (t_end - off[i]) * rate[i] for i in rate}
            small = [w for w in active if share[w["id"]] < MIN_SHARD]
            if not small or len(small) == len(active):
                break
            worst = min(small, key=lambda w: share[w["id"]])
            active = [w for w in active if w is not worst]
        if not active:
            unplaced.append("paper")
        else:
            n = {i: max(0, math.floor(s)) for i, s in share.items()}
            for i in sorted(share, key=lambda i: share[i] - math.floor(share[i]), reverse=True):
                if sum(n.values()) >= len(shas):
                    break
                n[i] += 1
            while sum(n.values()) > len(shas):          # only when a share was negative
                i = max(n, key=n.get)
                n[i] -= 1
            k = 0
            for w in active:
                cnt = n[w["id"]]
                if cnt:
                    tasks[w["id"]].append((f"paper_{w['id']}", "paper", shas[k:k + cnt]))
                    load[w["id"]] += SPEED[w["type"]]["gate_start"] + cnt * SPEED[w["type"]]["shape"]
                    k += cnt
    return tasks, {i: load[i] for i in load if tasks[i]}, unplaced


def show_plan(workers, tasks, secs):
    for w in workers:
        if tasks[w["id"]]:
            desc = ", ".join(c if s is None else f"paper {len(s)} shapes" for _, c, s in tasks[w["id"]])
            say(f"  {w['id']:<18} {w['type']:<8} {desc}  (~{secs[w['id']] / 60:.1f} min)")


# ------------------------------------------------------------------------------------- run
class Run:
    def __init__(self, a, baselines, names):
        self.a, self.baselines, self.names = a, baselines, names
        self.label = a.label
        self.fan = f"{C}/fanout/{self.label}"
        self.tree = a.tree_dir
        self.tree_name = os.path.basename(self.tree.rstrip("/"))
        self.results = {}       # task -> dict(worker, type, check, shas, out, status)
        self.pins = {}          # Solar job name -> node it is pinned to (other labs' nodes only)
        self.started = time.time()
        self.results_dir = f"{self.fan}/results"

    # -- lab ------------------------------------------------------------------------------
    def lab_tree(self, host):
        return self.tree if LAB[host]["ssh"] is None else f"{C}/trees/{self.tree_name}"

    def stage_lab_dispatch(self, host):
        """The tree, the tooling, the release check root and the fanout dir on a lab host."""
        h = LAB[host]
        if h["ssh"] is None:
            return
        on(host, f"mkdir -p {C}/trees {self.fan}/shas {self.fan}/workers {self.fan}/logs {self.fan}/out "
                 f"{h['gate']} {h['repo']}/evaluation/newdata_eval", check=True)
        rsync(self.tree.rstrip("/") + "/", f"{h['ssh']}:{self.lab_tree(host)}/", "--delete")
        rsync(HERE + "/", f"{h['ssh']}:{h['gate']}/", "--delete", "--exclude", "__pycache__")
        rsync(f"{LIGHTGEN}/evaluation/newdata_eval/run_texgen_infer.py",
              f"{h['ssh']}:{h['repo']}/evaluation/newdata_eval/")
        rsync(HF_ROOT + "/", f"{h['ssh']}:{HF_ROOT}/", "--delete")
        on(host, f"cd {HF_ROOT} && sha256sum -c --quiet MANIFEST.sha256", timeout=300, check=True)

    def worker_script(self, host, wid, tasks, rnd):
        h = LAB[host]
        gate, tree = h["gate"], self.lab_tree(host)
        pck = f"{h['ckpts']}/{CKPTS['paper'][0]}/ckpts/{CKPTS['paper'][1]}"
        rck = f"{h['ckpts']}/{CKPTS['release'][0]}/ckpts/{CKPTS['release'][1]}"
        R = HF_ROOT
        lines = ["#!/bin/bash", f"# fanout {self.label}: queue of {wid} on {host}", "set -uo pipefail",
                 f"echo $$ > {self.fan}/workers/{wid}.r{rnd}.pid   # setsid: also the process group"]
        for k, lock in enumerate(h["locks"]):
            fd = 8 - k
            lines.append(f"exec {fd}>{shlex.quote(lock)}; flock -n {fd} || {{ echo WORKER_LOCK_HELD {lock}; exit 7; }}")
        env = {"GATE_LOCK": "none", "GATE_REPO": h["repo"], "GATE_L": L, "TRAINCHECK_ROOT": R,
               "CUDA_VISIBLE_DEVICES": "0"}
        if h["activate"]:
            env["GATE_ACTIVATE"] = h["activate"]
        lines.append("export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()))
        lines.append(f'run() {{ local n=$1 s=$(date +%s) rc; shift; "$@" > {self.fan}/logs/$n.log 2>&1; rc=$?; '
                     f'echo "TASK $n exit=$rc secs=$(( $(date +%s) - s )) $(tail -1 {self.fan}/logs/$n.log)"; }}')
        lines.append(f"echo QUEUE_START host=$(hostname -s)")
        for name, check, shas in tasks:
            out = f"{self.fan}/out/{name}"
            if check == "paper":
                lines.append(f"run {name} bash {gate}/gate-det.sh {tree} {out} {pck} {self.fan}/shas/{name}.txt")
            elif check == "release":
                lines.append(f"run {name} bash {gate}/gate-det.sh {tree} {out} {rck} {R}/release_test20.txt "
                             f"{R} {R}/df_lightgenbench.parquet")
            else:
                lines.append(f"run {name} bash {gate}/train-check.sh {tree} {out} {check} {rck}")
        lines.append("echo WORKER_DONE")
        path = f"{self.fan}/workers/{wid}.r{rnd}.sh"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def launch_lab(self, host, wid, tasks, rnd):
        h = LAB[host]
        for name, check, shas in tasks:
            if shas is not None:
                with open(f"{self.fan}/shas/{name}.txt", "w") as f:
                    f.write("\n".join(shas) + "\n")
        script = self.worker_script(host, wid, tasks, rnd)
        if h["ssh"] is not None:
            rsync(f"{self.fan}/shas/", f"{h['ssh']}:{self.fan}/shas/")
            rsync(script, f"{h['ssh']}:{script}")
        log = open(f"{self.fan}/workers/{wid}.r{rnd}.log", "w")
        cmd = ["setsid", "-w", "bash", script]
        if h["ssh"] is not None:
            cmd = ["ssh", "-o", "BatchMode=yes", h["ssh"], " ".join(cmd)]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def collect_lab(self, host):
        h = LAB[host]
        if h["ssh"] is None:
            return
        for d in ("out", "logs"):
            sh(["rsync", "-a", "--exclude", "*.ckpt", "--exclude", "save", "--exclude", "tb_logs",
                f"{h['ssh']}:{self.fan}/{d}/", f"{self.fan}/{d}/"], 1800)

    # -- solar ----------------------------------------------------------------------------
    def stage_solar(self):
        star2("mkdir -p refactor_checks/trees refactor_checks/tooling/repo/evaluation/newdata_eval "
              f"refactor_checks/logs refactor_checks/fanout/{self.label}/shas", check=True)
        rsync(self.tree.rstrip("/") + "/", f"star2:refactor_checks/trees/{self.tree_name}/", "--delete")
        rsync(TEXGEN_REPO + "/scripts/", "star2:refactor_checks/tooling/scripts/", "--delete", "--exclude", "__pycache__")
        rsync(f"{LIGHTGEN}/evaluation/newdata_eval/run_texgen_infer.py",
              "star2:refactor_checks/tooling/repo/evaluation/newdata_eval/")
        rsync(HF_ROOT + "/", "star2:refactor_checks/hf_root/", "--delete")

    def submit_solar(self, jobs):
        """jobs: [(job name, [worker], {wid: tasks}, secs)] -> {job name: (job id, submit time)}."""
        sub = {}
        with open(os.path.join(HERE, "star2", "checks.sbatch")) as f:
            template = f.read()
        for jname, ws, tasks, secs in jobs:
            lines = []
            for g, w in enumerate(ws):
                for name, check, shas in tasks[w["id"]]:
                    rel = ""
                    if shas is not None:
                        rel = f"fanout/{self.label}/shas/{name}.txt"
                        with open(f"{self.fan}/shas/{name}.txt", "w") as f:
                            f.write("\n".join(shas) + "\n")
                    lines.append(f"{g} {name} {check} {rel}".rstrip())
            tf = f"{self.fan}/{jname}.tasks"
            with open(tf, "w") as f:
                f.write("\n".join(lines) + "\n")
            rsync(f"{self.fan}/shas/", f"star2:refactor_checks/fanout/{self.label}/shas/")
            rsync(tf, f"star2:refactor_checks/fanout/{self.label}/")
            limit = int(secs * 2 + 600) // 60 + 1
            k = len(ws)
            body = template
            for key, val in {"__TREE__": self.tree_name, "__LABEL__": f"{self.label}_{jname}", "__NGPU__": str(k),
                             "__TASKS__": f"fanout/{self.label}/{jname}.tasks", "__PAPER_RUNS__": "",
                             "__RELEASE_RUNS__": "", "__TRAIN_RUNS__": "", "__GRES__": f"l40s:{k}",
                             "__NCPU__": str(SOLAR_CPU * k), "__MEM__": f"{SOLAR_MEM_GB * k}G",
                             "__TIME__": f"{limit // 60:02d}:{limit % 60:02d}:00"}.items():
                body = body.replace(key, val)
            with open(f"{self.fan}/{jname}.sbatch", "w") as f:
                f.write(body)
            t0 = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            pin = f" -w {self.pins[jname]}" if self.pins.get(jname) else ""
            rc, out = star2(f"sbatch --deadline=now+{limit + 2}minutes{pin}", input=body)
            m = re.search(r"Submitted batch job (\d+)", out)
            if not m:
                say(f"solar {jname}: sbatch failed: {out.strip()}")
                continue
            sub[jname] = (m.group(1), t0)
            say(f"solar {jname}: job {m.group(1)}, {k} L40S, limit {limit} min, starts within 2 min or is cancelled")
        return sub

    def confirm_solar(self, sub):
        """After 30 s: RUNNING jobs stay; anything else is cancelled. {job name: node}."""
        if not sub:
            return {}
        time.sleep(30)    # step 7b: every job on free L40S was RUNNING within 20 s
        ids = ",".join(j for j, _ in sub.values())
        rc, out = star2(f"squeue -h -j {ids} -o '%i|%T|%N'")
        state = {l.split("|")[0]: l.split("|")[1:] for l in out.splitlines() if l.count("|") == 2}
        running = {}
        for jname, (jid, t0) in sub.items():
            st, node = state.get(jid, ["GONE", ""])
            if st == "RUNNING":
                running[jname] = node
                say(f"solar {jname}: job {jid} RUNNING on {node}")
            else:
                star2(f"scancel {jid}")
                say(f"solar {jname}: job {jid} {st}, cancelled; its work goes to the lab GPUs")
        return running

    def preemptions(self, sub, running):
        t0 = min(t for _, t in sub.values()) if sub else None
        if not running:
            return []
        rc, out = star2(f"sacct -N {','.join(sorted(set(running.values())))} -a -S {t0} -X -n -s PR "
                        "-o JobID,User,Account,Partition,State,NodeList")
        lines = [l for l in out.splitlines() if l.strip() and not l.startswith("Ubuntu")]
        return lines

    def solar_status(self, jobs):
        """{job name: status text or None}; one ssh to the head (reads files, no Slurm call). A status
        counts only once it has its last line (CHECKS_DONE, CHECKS_INCOMPLETE or CHECKS_FAIL)."""
        cmd = "; ".join(f"echo '=={j}'; cat refactor_checks/out/{self.label}_{j}/status.txt 2>/dev/null" for j in jobs)
        rc, out = star2(cmd + "; true")
        res, cur = {j: None for j in jobs}, None
        for line in out.splitlines():
            if line.startswith("=="):
                cur = line[2:]
            elif cur in res:
                res[cur] = (res[cur] or "") + line + "\n"
        return {j: t if t and re.search(r"^CHECKS_", t, re.M) else None for j, t in res.items()}

    def collect_solar(self, jname):
        sh(["rsync", "-a", f"star2:refactor_checks/out/{self.label}_{jname}/", f"{self.fan}/out/{jname}/"], 1800)

    # -- compare --------------------------------------------------------------------------
    def compare(self, task):
        r = self.results[task]
        out, check, typ = r["out"], r["check"], r["type"]
        base = self.baselines[typ][check]
        txt = f"{self.results_dir}/{task}.txt"
        gpu = ""
        if os.path.isfile(f"{out}/gpu.txt"):
            m = re.search(r"gpu=(.*)", open(f"{out}/gpu.txt").read())
            gpu = m.group(1).strip() if m else ""
        if self.names.get(gpu) != typ:
            r["verdict"] = f"MISROUTED (gpu.txt names {gpu!r}, planned {typ})"
            return
        if check in FIXED[1:]:
            cmd = [PY, "-I", f"{HERE}/compare_traincheck.py", "--a", base["run"], "--b", f"{out}/record.jsonl",
                   "--envelope", *base["envelope"]]
        else:
            full = PAPER_SHAS if check == "paper" else f"{HF_ROOT}/release_test20.txt"
            shas = f"{self.fan}/shas/{task}.txt" if check == "paper" else full
            cmd = [PY, f"{HERE}/compare_numeric.py", "--a", base["run"], "--b", out, "--shas_file", shas,
                   "--envelope", *base["envelope"], "--envelope_shas", full,
                   "--envelope_json", f"{C}/envelopes/{typ}_{check}.json", "--csv", txt[:-4] + ".csv"]
        rc, text = sh(cmd, 1800)
        with open(txt, "w") as f:
            f.write(text)
        m = re.findall(r"verdict=(\S+(?: \([^)]*\))?)", text)
        r["verdict"] = m[-1] if m else f"NO_VERDICT (rc={rc}, see {txt})"
        r["rc"] = rc


def tree_from_rev(rev):
    rc, sha = sh(["git", "-C", TEXGEN_REPO, "rev-parse", "--verify", f"{rev}^{{commit}}"], check=True)
    sha = sha.strip()
    d = f"{C}/trees/rev-{sha[:12]}"
    if not os.path.isdir(d):
        os.makedirs(d + ".part", exist_ok=True)
        arch = subprocess.Popen(["git", "-C", TEXGEN_REPO, "archive", "--format=tar", sha], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", d + ".part"], stdin=arch.stdout, check=True)
        if arch.wait() != 0:
            sys.exit(f"git archive {sha} failed")
        os.rename(d + ".part", d)
    return d, sha


# ---------------------------------------------------------------------------- the commands
def stage_host(host):
    """Stage a lab 4090 for the checks: the texgen env at the workstation's prefix, an activate
    script, the paper gate data, both checkpoints from jupiter (sha256-checked), the tooling."""
    h = LAB[host]
    if h["ssh"] is None:
        sys.exit("the workstation needs no staging")
    s = h["ssh"]
    say(f"stage {host}")
    on(host, f"mkdir -p {C}/trees {C}/tooling {h['repo']}/evaluation/newdata_eval {h['gate']} {L}", check=True)
    rc, _ = on(host, f"test -x {ENV_DIR}/bin/python")
    if rc != 0:
        say(f"  texgen env -> {host}:{ENV_DIR} (16 GB, same prefix as here)")
        on(host, f"mkdir -p {os.path.dirname(ENV_DIR)}", check=True)
        rsync(ENV_DIR + "/", f"{s}:{ENV_DIR}/")
    act = f"{C}/tooling/activate_{host}.sh"
    if h["activate"] == act:
        rc, _ = on(host, "test -f /localhome/dya78/miniconda3/etc/profile.d/conda.sh")
        if rc != 0:
            sys.exit(f"{host} has no /localhome/dya78/miniconda3 (conda.sh) to activate the env with")
        on(host, f"printf '%s\\n' '# {host}: its own conda install, the texgen env copied from the workstation "
                 f"at the same prefix' 'source /localhome/dya78/miniconda3/etc/profile.d/conda.sh' "
                 f"'conda activate {ENV_DIR}' > {act}", check=True)
    rsync(f"{LIGHTGEN}/evaluation/newdata_eval/run_texgen_infer.py", f"{s}:{h['repo']}/evaluation/newdata_eval/")
    rsync(HERE + "/", f"{s}:{h['gate']}/", "--delete", "--exclude", "__pycache__")
    say("  paper gate data (atlas_root keeps its absolute symlinks into out_eval_gt)")
    rsync(PAPER_SHAS, f"{s}:{L}/")
    rsync(f"{L}/out_eval_gt/", f"{s}:{L}/out_eval_gt/")
    rsync(f"{L}/atlas_root/", f"{s}:{L}/atlas_root/")
    rsync(f"{C}/trees/pre-refactor/", f"{s}:{C}/trees/pre-refactor/", "--delete")
    rsync(HF_ROOT + "/", f"{s}:{HF_ROOT}/", "--delete")
    for run, fname, digest in CKPTS.values():
        d = f"{h['ckpts']}/{run}"
        say(f"  {run}/{fname} from jupiter, on {host}")
        on(host, f"mkdir -p {d}/ckpts {d}/configs && rsync -a {JUPITER}/{run}/configs/parsed.yaml {d}/configs/ && "
                 f"rsync -a --inplace {JUPITER}/{run}/ckpts/{fname} {d}/ckpts/", timeout=3600, check=True)
        rc, out = on(host, f"sha256sum {d}/ckpts/{fname}", timeout=600)
        if not out.startswith(digest):
            sys.exit(f"sha256 mismatch on {host}: {d}/ckpts/{fname}: {out.strip()}")
    row = probe_lab(host)
    say(f"  staged; probe now: {'free' if row['free'] else row['reason']}")


def probe_all(a, names, tag, run=None):
    """Lab workers and Solar job groups free right now; Solar job names start with `tag`."""
    workers = []
    for host in [h for h in a.hosts.split(",") if h]:
        row = probe_lab(host)
        ok = row["free"] and (qualified(host) or host == a.admit)
        why = row["reason"] if not row["free"] else ("" if ok else f"not qualified yet: run --qualify {host}")
        say(f"probe {host:<12} {row.get('gpu', '')[:24]:<24} {'FREE' if ok else 'skip'}  {why}")
        if ok:
            typ = names.get(row["gpu"])
            if typ is None:
                say(f"  {host}: no baselines for {row['gpu']!r}; skipped")
                continue
            workers.append({"id": host, "host": host, "kind": "lab", "type": typ, "start": 0,
                            "vram_free": row["vram_free"], "ram_free_gb": row["ram_free_gb"]})
    solar = probe_solar(a.solar.split(",")) if a.solar else []
    for row in solar:
        say(f"probe {row['host']:<12} {'L40S':<24} {'FREE x' + str(row['free_gpus']) if row['free_gpus'] else 'skip'}"
            f"  {row['reason']}")
    jobs_ws = []
    for i, row in enumerate(sorted((r for r in solar if r["free_gpus"]), key=lambda r: -r["free_gpus"])):
        j = f"{tag}s{i + 1}"
        if run is not None and row["host"] in SOLAR_OTHER:
            run.pins[j] = row["host"]
        jobs_ws.append((j, [{"id": f"{j}g{g}", "host": "solar", "job": j, "kind": "solar", "type": "l40s",
                             "start": SOLAR_START, "vram_free": 46000, "ram_free_gb": SOLAR_MEM_GB}
                            for g in range(row["free_gpus"])]))
    return workers + [w for _, ws in jobs_ws for w in ws], jobs_ws, solar


def execute(run, workers, jobs_ws, tasks, secs, lab_plan, rnd):
    """One round: Solar jobs first, then the lab queues lab_plan(<task names running on Solar>) returns;
    waits for our own work only, collects it into run.results. Returns False if nothing started."""
    jobs = [(j, [w for w in ws if tasks.get(w["id"])], tasks, max(secs.get(w["id"], 0) for w in ws))
            for j, ws in jobs_ws if any(tasks.get(w["id"]) for w in ws)]
    running, sub = {}, {}
    if jobs:
        run.stage_solar()
        sub = run.submit_solar(jobs)
        running = run.confirm_solar(sub)
        for line in run.preemptions(sub, running):
            say(f"PREEMPTION on our nodes since submit (tell Dongchen): {line}")
    on_solar = {}
    for j, ws, t, s in jobs:
        if j in running:
            for g, w in enumerate(ws):
                for name, check, sh_ in tasks[w["id"]]:
                    on_solar[name] = (check, sh_)
                    run.results[name] = {"worker": f"{j}/{running[j]}/gpu{g}", "type": "l40s", "check": check,
                                         "shas": sh_, "out": f"{run.fan}/out/{j}/{name}", "status": "running",
                                         "round": rnd}
    ltasks, lsecs = lab_plan(on_solar)
    procs = {}
    for w in workers:
        if w["kind"] != "lab" or not ltasks.get(w["id"]):
            continue
        try:
            run.stage_lab_dispatch(w["host"])
        except RuntimeError as e:
            say(f"lab {w['host']}: staging failed, skipped: {str(e).splitlines()[0]}")
            continue
        for name, check, sh_ in ltasks[w["id"]]:
            run.results[name] = {"worker": w["host"], "type": w["type"], "check": check, "shas": sh_,
                                 "out": f"{run.fan}/out/{name}", "status": "running", "round": rnd}
        procs[w["id"]] = (run.launch_lab(w["host"], w["id"], ltasks[w["id"]], rnd), time.time(), lsecs[w["id"]])
        say(f"lab {w['host']}: queue started ({len(ltasks[w['id']])} tasks, ~{lsecs[w['id']] / 60:.1f} min)")
    if not procs and not running:
        return False

    pending_jobs = set(running)
    last_squeue = time.time()
    done_re = ("GATE_DONE", "TRAINCHECK_DONE")
    while procs or pending_jobs:
        time.sleep(30)
        for wid, (p, t0, plan_s) in list(procs.items()):
            rc = p.poll()
            if rc is None and time.time() - t0 > plan_s * 2 + 900:
                say(f"lab {wid}: over twice its planned time, killing its queue")
                on(wid, f"kill -TERM -- -$(cat {run.fan}/workers/{wid}.r{rnd}.pid)")
                p.wait()
                rc = p.returncode
            if rc is not None:
                del procs[wid]
                run.collect_lab(wid)
                log = open(f"{run.fan}/workers/{wid}.r{rnd}.log").read()
                for m in re.finditer(r"TASK (\S+) exit=(\d+) secs=(\d+) ?(.*)", log):
                    r = run.results[m.group(1)]
                    r["status"] = "done" if m.group(2) == "0" and m.group(4) in done_re else f"exit {m.group(2)}: {m.group(4)}"
                    r["secs"] = int(m.group(3))
                if "WORKER_LOCK_HELD" in log:
                    say(f"lab {wid}: a lock was taken between probe and start; its queue did not run")
                say(f"lab {wid}: queue ended (exit {rc}); " + ", ".join(
                    f"{n} {r['status']}" for n, r in run.results.items() if r["worker"] == wid and r["round"] == rnd))
        if pending_jobs:
            for j, text in run.solar_status(sorted(pending_jobs)).items():
                if text is None:
                    continue
                pending_jobs.discard(j)
                run.collect_solar(j)
                for line in text.splitlines():
                    m = re.match(r"(\S+) exit=(\d+) secs=(\d+) ?(.*)", line)
                    if m and m.group(1) in run.results:
                        r = run.results[m.group(1)]
                        r["status"] = "done" if m.group(2) == "0" and m.group(4) in done_re else f"exit {m.group(2)}: {m.group(4)}"
                        r["secs"] = int(m.group(3))
                say(f"solar {j}: {text.strip().splitlines()[-1]}")
            if pending_jobs and time.time() - last_squeue > 600:   # a job that died without a status file
                last_squeue = time.time()
                rc, out = star2(f"squeue -h -j {','.join(sub[j][0] for j in pending_jobs)} -o %i")
                alive = set(out.split())
                for j in list(pending_jobs):
                    if sub[j][0] not in alive and run.solar_status([j])[j] is None:
                        say(f"solar {j}: job {sub[j][0]} ended without a status file")
                        pending_jobs.discard(j)
    if running:
        for line in run.preemptions(sub, running):
            say(f"PREEMPTION on our nodes during the run (tell Dongchen): {line}")
    for name, r in run.results.items():
        if r["round"] == rnd:
            if r["status"] == "done":
                run.compare(name)
            else:
                r["verdict"] = f"NOT_RUN ({r['status']})"
    return True


def run_checks(a):
    baselines, names, _ = load_baselines()
    checks = [c for c in a.checks.split(",") if c]
    for c in checks:
        if c not in ("paper",) + FIXED:
            sys.exit(f"unknown check {c}")
    shas = [l.strip() for l in open(a.paper_shas) if l.strip()] if "paper" in checks else []
    run = Run(a, baselines, names)
    say(f"tree {run.tree} ({a.rev_sha or 'given dir'}), label {run.label}, checks {','.join(checks)}, "
        f"{len(shas)} paper shapes")

    # 1. probe, 2. plan
    workers, jobs_ws, solar = probe_all(a, names, "", run)
    if not workers:
        say("NOTHING_FREE: no GPU is free for any check right now; nothing was started")
        return 3
    tasks, secs, unplaced = plan(workers, checks, shas, baselines)
    say(f"plan: {sum(1 for w in workers if tasks[w['id']])} GPUs, ~{max(secs.values(), default=0) / 60:.1f} min"
        + (f"; not placed (no free GPU of a type with a complete envelope of {MIN_ENVELOPE}+ runs): "
           f"{', '.join(unplaced)}" if unplaced else ""))
    show_plan(workers, tasks, secs)
    if a.plan:
        return 0
    os.makedirs(f"{run.fan}/shas", exist_ok=False)
    for d in ("workers", "logs", "out", "results"):
        os.makedirs(f"{run.fan}/{d}")
    with open(f"{run.fan}/probe.json", "w") as f:
        json.dump({"workers": workers, "solar": solar}, f, indent=1)
    lab = [w for w in workers if w["kind"] == "lab"]

    def lab_plan(on_solar):   # 3, 4: the lab GPUs take the work no running Solar job holds
        left = [c for c in checks if c in FIXED and c not in {v[0] for v in on_solar.values()}]
        held = {s for v in on_solar.values() if v[0] == "paper" for s in v[1]}
        left_shas = [s for s in shas if s not in held]
        if "paper" in checks and left_shas:
            left.append("paper")
        if not (lab and left):
            return {}, {}
        lt, ls, lun = plan(lab, left, left_shas, baselines)
        if jobs_ws:
            say("lab plan for the rest:")
            show_plan(lab, lt, ls)
            if lun:
                say(f"not placed (no free GPU of a type with a complete envelope): {', '.join(lun)}")
        return lt, ls

    if not execute(run, workers, jobs_ws, tasks, secs, lab_plan, 1):
        say("NOTHING_FREE: no check could be started on any GPU right now")
        return 3

    # 5b. the rerun-once rule (R20): a task outside the envelope runs once more on another free GPU
    # of the same type (the same GPU only when no other is free) before it counts as failed
    outside = [n for n, r in run.results.items() if r["verdict"].startswith("OUTSIDE_NOISE")]
    if outside:
        say(f"outside the envelope, rerun once: {', '.join(outside)}")
        workers2, jobs_ws2, _ = probe_all(a, names, "r", run)
        load = {w["id"]: w["start"] for w in workers2}
        tasks2 = {w["id"]: [] for w in workers2}
        for n in outside:
            r = run.results[n]
            cand = [w for w in workers2 if w["type"] == r["type"] and fits(w, r["check"])]
            if not cand:
                say(f"  {n}: no free {r['type']} GPU now; it stays unresolved")
                continue
            w = min(cand, key=lambda w: (w["id"] == r["worker"], load[w["id"]]))
            dur = SPEED[w["type"]]["gate_start"] + len(r["shas"]) * SPEED[w["type"]]["shape"] \
                if r["check"] == "paper" else SPEED[w["type"]][r["check"]]
            tasks2[w["id"]].append((f"{n}_rerun", r["check"], r["shas"]))
            load[w["id"]] += dur
            r["rerun"] = f"{n}_rerun"
            say(f"  {n} -> {w['id']}" + (" (the same GPU: no other is free)" if w["id"] == r["worker"] else ""))
        secs2 = {i: load[i] for i in load if tasks2[i]}
        lab_ids = {w["id"] for w in workers2 if w["kind"] == "lab"}
        if secs2:
            execute(run, workers2, jobs_ws2, tasks2, secs2, lambda on_solar: (
                {i: t for i, t in tasks2.items() if t and i in lab_ids},
                {i: s for i, s in secs2.items() if i in lab_ids}), 2)
        for n in outside:
            rr = run.results.get(run.results[n].get("rerun", ""), {})
            rr["rerun_of"] = n
    return summarize(run, a, checks, shas)


def summarize(run, a, checks, shas, tag="", note=""):
    good = ("IDENTICAL", "WITHIN_NOISE")
    final = {}   # original task -> PASS | FAIL | UNRESOLVED | NOT_RUN
    for n, r in run.results.items():
        if r["round"] != 1:
            continue
        v = r["verdict"]
        if v in good:
            final[n] = "PASS"
        elif v.startswith("NOT_RUN"):
            final[n] = "NOT_RUN"
        elif v.startswith("OUTSIDE_NOISE"):
            rv = run.results.get(r.get("rerun", ""), {}).get("verdict", "")
            final[n] = "PASS" if rv in good else ("FAIL" if rv and not rv.startswith("NOT_RUN") else "UNRESOLVED")
        else:
            final[n] = "FAIL"
        r["final"] = final[n]
    ran_paper = {s for n, r in run.results.items() if final.get(n) == "PASS" and r["check"] == "paper" for s in r["shas"]}
    ran_fixed = {r["check"] for n, r in run.results.items() if final.get(n) == "PASS" and r["check"] in FIXED}
    missing_shas = [s for s in shas if s not in ran_paper]
    missing_checks = [c for c in checks if c in FIXED and c not in ran_fixed]
    lines = [f"fanout {run.label}: tree {run.tree} {a.rev_sha or ''}".rstrip()] + ([note] if note else []) + [
             f"{'task':<22}{'check':<9}{'type':<9}{'where':<26}{'n':>5}{'secs':>7}  verdict"]
    order = sorted(run.results.items(), key=lambda kv: (kv[1]["check"] != "paper", kv[1].get("rerun_of", kv[0]), kv[0]))
    for name, r in order:
        n = len(r["shas"]) if r["shas"] else ""
        tail = f"  -> {r['final']}" if "final" in r and r["final"] != "PASS" or r.get("rerun") else ""
        lines.append(f"{name:<22}{r['check']:<9}{r['type']:<9}{r['worker']:<26}{n:>5}{r.get('secs', ''):>7}  "
                     f"{r['verdict']}{tail}")
    wall = (time.time() - run.started) / 60
    if "FAIL" in final.values():
        verdict, code = "FAIL", 1
    elif missing_shas or missing_checks:
        verdict, code = "INCOMPLETE", 2
    else:
        verdict, code = "PASS", 0
    reruns = [n for n, r in run.results.items() if r.get("rerun")]
    lines.append(f"paper shapes passed {len(ran_paper)}/{len(shas)}; fixed checks passed {sorted(ran_fixed)} of "
                 f"{[c for c in checks if c in FIXED]}; reruns {len(reruns)}; wall {wall:.1f} min")
    if missing_shas:
        with open(f"{run.fan}/leftover_paper.txt", "w") as f:
            f.write("\n".join(missing_shas) + "\n")
    if verdict == "INCOMPLETE":
        rest = (["paper"] if missing_shas else []) + missing_checks
        lines.append(f"to finish: python3 {os.path.abspath(__file__)} --tree {run.tree} --checks {','.join(rest)}"
                     + (f" --paper_shas {run.fan}/leftover_paper.txt" if missing_shas else ""))
    lines.append(f"VERDICT {verdict}")
    with open(f"{run.fan}/summary{tag}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{run.fan}/summary{tag}.json", "w") as f:
        json.dump({"verdict": verdict, "tree": run.tree, "rev": a.rev_sha, "results": run.results,
                   "wall_min": wall}, f, indent=1)
    print("\n".join(lines), flush=True)
    say(f"summary: {run.fan}/summary{tag}.txt")
    return code


def rescore(fan):
    """Compare every stored output of a finished fanout run again under the current rule (no GPU):
    the round-1 tasks and the stored reruns. A task now outside the envelope whose rerun was never run
    stays unresolved. Writes summary_rescored_<time>.{txt,json} and results_rescored_<time>/ beside
    the originals, which stay as they were."""
    fan = os.path.abspath(fan)
    with open(f"{fan}/summary.json") as f:
        old = json.load(f)
    baselines, names, _ = load_baselines()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    a = argparse.Namespace(label=os.path.basename(fan), tree_dir=old["tree"], rev_sha=old.get("rev"))
    run = Run(a, baselines, names)
    run.fan, run.results, run.results_dir = fan, old["results"], f"{fan}/results_rescored_{stamp}"
    os.makedirs(run.results_dir)
    for name, r in run.results.items():
        r["verdict_before"] = r.get("verdict")
        r.pop("final", None)
        if r["status"] == "done":
            run.compare(name)
        else:
            r["verdict"] = f"NOT_RUN ({r['status']})"
    first = [r for r in run.results.values() if r.get("round", 1) == 1]
    checks = [c for c in ("paper",) + FIXED if any(r["check"] == c for r in first)]
    shas = [x for r in first if r["check"] == "paper" for x in r["shas"]]
    for r in run.results.values():
        r.setdefault("round", 1)
    return summarize(run, a, checks, shas, tag=f"_rescored_{stamp}",
                     note=f"re-scored {stamp} from the stored outputs under the current rule (no GPU run); "
                          f"the original verdicts stay in summary.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--rev", help="the tree under test is git archive of this commit (in this repo)")
    src.add_argument("--tree", help="the tree under test: a TEXGen tree on the workstation")
    src.add_argument("--stage", metavar="HOST", help="stage a lab 4090 (sp4090, xg4090) and exit")
    src.add_argument("--rescore", metavar="FANOUT_DIR", help="re-compare a finished run's stored outputs "
                                                            "under the current rule (no GPU)")
    src.add_argument("--qualify", metavar="HOST", help="admit a lab 4090: subset20 on tag pre-refactor there "
                                                       "must be IDENTICAL to the 4090 baseline")
    ap.add_argument("--checks", default="paper,release,fresh,resume")
    ap.add_argument("--paper_shas", default=PAPER_SHAS, help="paper gate shapes (default: all 204)")
    ap.add_argument("--label", help="run name (default: <tree>_<time>); outputs in <C>/fanout/<label>/")
    ap.add_argument("--hosts", default="local,sp4090,xg4090", help="lab 4090s to probe")
    ap.add_argument("--solar", default=",".join(SOLAR_NODES + SOLAR_OTHER),
                    help="Solar L40S nodes to probe ('' = none); SOLAR_OTHER ones are pinned")
    ap.add_argument("--plan", action="store_true", help="probe and print the plan, run nothing")
    a = ap.parse_args()
    a.admit, a.rev_sha = None, None
    if a.rescore:
        return rescore(a.rescore)
    if a.stage:
        stage_host(a.stage)
        return 0
    if a.qualify:
        a.tree, a.hosts, a.solar, a.admit = f"{C}/trees/pre-refactor", a.qualify, "", a.qualify
        a.checks, a.paper_shas = "paper", os.path.join(HERE, "subset20.txt")
    if a.rev:
        a.tree_dir, a.rev_sha = tree_from_rev(a.rev)
    elif a.tree:
        a.tree_dir = os.path.abspath(a.tree)
    else:
        ap.error("give --rev, --tree, --stage or --qualify")
    if not os.path.isfile(os.path.join(a.tree_dir, "launch.py")):
        sys.exit(f"{a.tree_dir} is not a TEXGen tree (no launch.py)")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    a.label = a.label or f"{'qualify-' + a.qualify if a.qualify else os.path.basename(a.tree_dir.rstrip('/'))}_{stamp}"
    code = run_checks(a)
    if a.qualify and code == 0:
        os.makedirs(f"{C}/qualified", exist_ok=True)
        with open(f"{C}/qualified/{a.qualify}.txt", "w") as f:
            f.write(f"{stamp} subset20 of the paper gate on tag pre-refactor IDENTICAL to the 4090 baseline "
                    f"({C}/fanout/{a.label}/summary.txt)\n")
        say(f"{a.qualify} admitted: {C}/qualified/{a.qualify}.txt")
    return code


if __name__ == "__main__":
    sys.exit(main())
