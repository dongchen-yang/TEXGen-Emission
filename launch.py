import argparse
import atexit
import contextlib
import logging
import os
import signal
import sys

# Disable HuggingFace Hub's chat template check for older models
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix for PyTorch 2.6 checkpoint loading
# PyTorch 2.6 changed weights_only default to True, but our checkpoints contain complex objects
# Since we trust our own checkpoints, we'll patch torch.load to always use weights_only=False
import torch
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that forces weights_only=False for checkpoint compatibility"""
    # Force weights_only=False since we trust our own checkpoints
    # This is necessary because PyTorch Lightning explicitly passes weights_only=True
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("spuv", "typeguard.typechecked")

    import spuv
    from spuv.systems.base import BaseSystem
    from spuv.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from spuv.utils.config import ExperimentConfig, load_config
    from spuv.utils.misc import get_rank, time_recorder
    from spuv.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.benchmark:
        time_recorder.enable(True)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    # Auto-resume: find latest checkpoint if enabled or if resume="last"/"latest"
    if (cfg.auto_resume or cfg.resume in ["last", "latest"]) and args.train:
        # Determine checkpoint directory
        if "dirpath" in cfg.checkpoint:
            ckpt_dir = cfg.checkpoint["dirpath"]
        else:
            ckpt_dir = os.path.join(cfg.trial_dir, "ckpts")
        
        if os.path.exists(ckpt_dir):
            # Find the most recently modified last*.ckpt (handles last.ckpt, last-v1.ckpt, etc.)
            import glob as _glob
            last_ckpts = _glob.glob(os.path.join(ckpt_dir, "last*.ckpt"))
            last_ckpt = max(last_ckpts, key=os.path.getmtime) if last_ckpts else None

            if last_ckpt:
                try:
                    # Verify it's a valid checkpoint
                    ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
                    step = ckpt.get('global_step', -1)
                    epoch = ckpt.get('epoch', -1)

                    cfg.resume = last_ckpt
                    spuv.info(f"Auto-resume: Found {os.path.basename(last_ckpt)}")
                    spuv.info(f"  Epoch: {epoch}, Global Step: {step}")
                except Exception as e:
                    spuv.warn(f"Failed to load {os.path.basename(last_ckpt)}: {e}")
                    cfg.resume = None
            else:
                spuv.info(f"Auto-resume enabled but no last*.ckpt found in {ckpt_dir}, starting fresh")
                cfg.resume = None
        else:
            cfg.resume = None
            spuv.info(f"Auto-resume enabled but checkpoint directory {ckpt_dir} does not exist, starting fresh")

    # Load checkpoint info if resuming (for wandb and proper step counting)
    wandb_run_id = None
    resume_step = None
    resume_epoch = None
    if cfg.resume is not None and args.train:
        try:
            # Load checkpoint with weights_only=False to allow OmegaConf objects
            # This is safe since we trust our own checkpoints
            ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            if args.wandb and 'wandb_run_id' in ckpt:
                wandb_run_id = ckpt['wandb_run_id']
                spuv.info(f"Resuming wandb run with ID: {wandb_run_id}")
            elif args.wandb:
                spuv.warn("No wandb run ID found in checkpoint, starting new wandb run")
            # Extract step and epoch for proper logging continuity
            resume_step = ckpt.get('global_step', None)
            resume_epoch = ckpt.get('epoch', None)
            if resume_step is not None:
                spuv.info(f"Resuming from epoch {resume_epoch}, global_step {resume_step}")
        except Exception as e:
            spuv.warn(f"Failed to load checkpoint info: {e}")

    dm = spuv.find(cfg.data_cls)(cfg.data)
    system: BaseSystem = spuv.find(cfg.system_cls)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        # Use custom checkpoint dirpath if specified, otherwise default to trial_dir/ckpts
        # This respects both cfg.checkpoint.dirpath and cfg.custom_output_dir
        if "dirpath" in cfg.checkpoint:
            ckpt_dirpath = cfg.checkpoint["dirpath"]
        else:
            # Default to trial_dir/ckpts (which already uses custom_output_dir if set)
            ckpt_dirpath = os.path.join(cfg.trial_dir, "ckpts")
        
        ckpt_config = {k: v for k, v in cfg.checkpoint.items() if k != "dirpath"}
        callbacks += [
            ModelCheckpoint(
                dirpath=ckpt_dirpath, **ckpt_config
            ),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        ]
        if args.wandb:
            # Default wandb configuration
            wandb_config = {
                "project": "LightGen",  # Default project name
                "name": f"{cfg.name}-{cfg.tag}",  # Default run name
                "config": dict(cfg),  # Log config
            }
            
            # Override with custom wandb settings from config if provided
            if hasattr(cfg, 'wandb') and cfg.wandb:
                # Update with custom settings, preserving defaults for unspecified fields
                if 'project' in cfg.wandb:
                    wandb_config['project'] = cfg.wandb['project']
                if 'name' in cfg.wandb:
                    wandb_config['name'] = cfg.wandb['name']
                if 'entity' in cfg.wandb:
                    wandb_config['entity'] = cfg.wandb['entity']
                if 'dir' in cfg.wandb:
                    # Custom directory for wandb logs
                    wandb_config['dir'] = cfg.wandb['dir']
                elif cfg.custom_output_dir:
                    # Use custom output directory for wandb if specified
                    wandb_config['dir'] = os.path.join(cfg.trial_dir, "wandb")

            # Handle wandb resumption if run ID is available
            if wandb_run_id is not None:
                wandb_config['id'] = wandb_run_id
                wandb_config['resume'] = 'allow'  # Resume if possible, otherwise start new run
                spuv.info(f"Configuring wandb to resume run ID: {wandb_run_id}")

                # CRITICAL: Clean up stale wandb local cache to prevent old un-synced
                # metrics from being flushed into the resumed run. When a previous session
                # was killed without wandb.finish(), its local cache contains un-synced data
                # that gets interleaved with new data at incorrect step numbers.
                # NOTE: wandb_config['dir'] must be resolved BEFORE this cleanup so we
                # search the same directory that the previous run wrote its cache to.
                #
                # TWO GUARDS, both learned the hard way on cs-venus-05 (run yylyobx0):
                #
                # 1. OFFLINE: in offline mode the local cache is not a stale artifact, it
                #    IS the run's only copy of the data. Deleting it on every resume throws
                #    away every metric recorded so far and leaves nothing to `wandb sync`.
                #    Online mode has already uploaded, so there the delete is harmless.
                #
                # 2. RANK: all four ranks execute main(), so without a rank guard ranks
                #    1..3 run this cleanup a moment after rank 0 has created the live
                #    directory — and since the glob keys on the run id, which they share,
                #    they delete the directory rank 0 is actively writing. Observed as a
                #    dangling wandb/latest-run symlink with no target and zero data on disk.
                _offline = os.environ.get("WANDB_MODE", "").lower() in ("offline", "dryrun")
                _lrank = int(os.environ.get("LOCAL_RANK",
                             os.environ.get("SLURM_LOCALID", "0")))
                if _offline:
                    spuv.info("wandb offline: keeping the local cache — it is the data, not a stale artifact")
                elif _lrank != 0:
                    pass  # only rank 0 touches wandb state
                else:
                    import glob as glob_module
                    import shutil
                    wandb_dir = wandb_config.get('dir', '.')
                    stale_dirs = glob_module.glob(os.path.join(wandb_dir, 'wandb', f'*-{wandb_run_id}'))
                    # Also search the default ./wandb/ in case dir was not set in a previous run
                    if wandb_dir != '.':
                        stale_dirs += glob_module.glob(os.path.join('.', 'wandb', f'*-{wandb_run_id}'))
                    for stale_dir in stale_dirs:
                        if os.path.isdir(stale_dir):
                            spuv.info(f"Cleaning stale wandb cache to prevent data oscillation: {stale_dir}")
                            shutil.rmtree(stale_dir, ignore_errors=True)

            # Pass through any remaining wandb settings (except id and resume which are handled above)
            if hasattr(cfg, 'wandb') and cfg.wandb:
                for key, value in cfg.wandb.items():
                    if key not in ['project', 'name', 'entity', 'dir', 'config', 'id', 'resume']:
                        wandb_config[key] = value

            wandb_logger = WandbLogger(**wandb_config)
            system._wandb_logger = wandb_logger
            # Set the wandb run ID in the system for future checkpoints
            if wandb_run_id is not None:
                system.set_wandb_run_id(wandb_run_id)
            loggers += [wandb_logger]
            
            import wandb
            
            # Configure x-axes: use trainer/global_step for everything.
            # CRITICAL: We NEVER pass step= to wandb.log(). Each wandb.log() call
            # auto-increments an internal counter. We include trainer/global_step as
            # a regular metric and use define_metric() to tell W&B to use it as
            # the chart x-axis. This avoids the step=epoch approach which silently
            # drops validation data (wandb.log(step=N) advances counter to N+1,
            # so the next call at step=N is rejected as going backwards).
            #
            # Rank-zero-only: wandb.init() runs on rank 0 only via WandbLogger.experiment.
            # On other ranks, the wandb module is the preinit stub and direct calls like
            # wandb.define_metric raise "must call wandb.init() before". Skip on non-zero ranks.
            #
            # NOTE: at this point in launch.py we're BEFORE trainer.fit().
            #
            # LOCAL_RANK is checked FIRST and that order is load-bearing. SLURM_LOCALID
            # is set per *task*: under Lightning's native DDP spawn (one task, four
            # spawned children) every child inherits the parent's SLURM_LOCALID=0, so
            # all four ranks believed they were rank 0, all four called define_metric(),
            # and the three that never ran wandb.init() died with "You must call
            # wandb.init() before wandb.define_metric()" — killing the job (vulcan
            # 217672, 2026-07-29). Lightning's subprocess launcher sets LOCAL_RANK
            # per child, so it is the truthful value when it exists.
            #
            # Under the srun path (ntasks == devices) LOCAL_RANK is unset here and this
            # falls back to SLURM_LOCALID, preserving the original fir behaviour exactly.
            _local_rank = int(os.environ.get("LOCAL_RANK",
                              os.environ.get("SLURM_LOCALID", "0")))
            if _local_rank == 0:
                _ = wandb_logger.experiment
                wandb.define_metric("trainer/global_step")
                wandb.define_metric("*", step_metric="trainer/global_step")
                spuv.info("Set wandb x-axes: all metrics use trainer/global_step (via define_metric)")

            # ── Monkey-patch WandbLogger.log_metrics ──────────────────────────
            # Include trainer/global_step and epoch as regular metrics in every
            # wandb.log() call. Do NOT pass step= to avoid counter conflicts.
            from pytorch_lightning.loggers.wandb import _add_prefix
            from pytorch_lightning.utilities.rank_zero import rank_zero_only as _rz

            def _make_consistent_step_logger(wl, sys_ref):
                @_rz
                def _log_metrics(metrics, step=None):
                    metrics = _add_prefix(metrics, wl._prefix, wl.LOGGER_JOIN_CHAR)
                    epoch = sys_ref.current_epoch
                    global_step = sys_ref.global_step
                    # ALWAYS use global_step for trainer/global_step, never the `step`
                    # arg from PL. PL calls log_metrics(metrics, step=current_epoch)
                    # for on_epoch=True metrics — if we used that epoch number as the
                    # trainer/global_step value, W&B would see e.g. step=3 after having
                    # already recorded step=17499 from per-step logs, triggering the
                    # "step must be monotonically increasing" warning and silently
                    # dropping validation data.
                    wl.experiment.log(
                        dict(metrics, **{"trainer/global_step": global_step, "epoch": epoch}),
                    )
                return _log_metrics

            wandb_logger.log_metrics = _make_consistent_step_logger(wandb_logger, system)
            spuv.info("Patched WandbLogger.log_metrics → trainer/global_step always = sys_ref.global_step")
            
            # Register robust shutdown handlers for wandb.
            # The try/finally block in trainer.fit() handles KeyboardInterrupt (SIGINT),
            # but SIGTERM (from schedulers, kill command) and other exits need coverage too.
            def _ensure_wandb_finish():
                """Ensure wandb run is finalized on ANY exit (atexit, SIGTERM, etc.)."""
                try:
                    import wandb as _wandb
                    if _wandb.run is not None:
                        # Block Ctrl+C during sync so the user can't interrupt it.
                        # This prevents the "Run data was not synced" error.
                        prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                        spuv.info(
                            "Finishing wandb run (syncing data, please wait -- "
                            "DO NOT press Ctrl+C again)..."
                        )
                        _wandb.finish()
                        signal.signal(signal.SIGINT, prev_handler)
                except Exception:
                    pass  # Best-effort; don't crash during shutdown
            
            atexit.register(_ensure_wandb_finish)
            
            # Handle SIGTERM gracefully (sent by SLURM, kill, scancel, etc.)
            _original_sigterm = signal.getsignal(signal.SIGTERM)
            def _sigterm_handler(signum, frame):
                spuv.info("Received SIGTERM, finishing wandb run before exit...")
                _ensure_wandb_finish()
                # Re-raise with original handler or default behavior
                if callable(_original_sigterm):
                    _original_sigterm(signum, frame)
                else:
                    sys.exit(128 + signum)
            signal.signal(signal.SIGTERM, _sigterm_handler)

            # Handle SIGUSR1 (used by some SLURM configs for preemption warnings)
            def _sigusr1_handler(signum, frame):
                spuv.info("Received SIGUSR1 (preemption warning), finishing wandb run...")
                _ensure_wandb_finish()
                sys.exit(128 + signum)
            signal.signal(signal.SIGUSR1, _sigusr1_handler)
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()
    print(f"Set up Trainer with {n_gpus} GPUs")
    
    trainer_kwargs = {
        "callbacks": callbacks,
        "logger": loggers,
        "inference_mode": False,
        "accelerator": "gpu",
        "devices": devices,
        **cfg.trainer,
    }
    
    trainer = Trainer(**trainer_kwargs)

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        #system = torch.compile(system)
        #spuv.info("compiled system")
        
        # CRITICAL FIX for checkpoint resume
        # PyTorch Lightning 2.x requires explicit checkpoint path handling
        if cfg.resume is not None:
            spuv.info(f"=" * 80)
            spuv.info(f"RESUMING TRAINING FROM CHECKPOINT: {cfg.resume}")
            spuv.info(f"Expected resume from epoch: {resume_epoch}, global_step: {resume_step}")
            spuv.info(f"=" * 80)
        
        try:
            trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
            trainer.test(system, datamodule=dm)
        except (KeyboardInterrupt, SystemExit):
            # Lightning intercepts KeyboardInterrupt and calls exit(1) → SystemExit.
            # We catch both to ensure wandb sync happens.
            spuv.info("Detected interrupt, attempting graceful shutdown ...")
        finally:
            # CRITICAL: Ensure wandb finishes syncing before exit.
            # Without this, interrupted runs leave un-synced metrics in the local
            # wandb cache. When the run is later resumed, those stale metrics get
            # flushed at new step numbers, causing oscillation in the epoch vs step
            # plot (old epoch values interleaved with new ones).
            if args.wandb:
                import wandb
                if wandb.run is not None:
                    # Block Ctrl+C during sync so user can't kill the sync process
                    prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
                    spuv.info(
                        "Syncing wandb data (please wait, do NOT press Ctrl+C)..."
                    )
                    try:
                        wandb.finish()
                    except Exception as e:
                        spuv.warn(f"wandb.finish() error: {e}")
                    finally:
                        signal.signal(signal.SIGINT, prev)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument("--wandb", action="store_true", help="if true, log to wandb")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="if true, set to benchmark mode to record running times"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
