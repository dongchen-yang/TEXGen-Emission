import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from einops import rearrange
from torch.autograd import profiler

import spuv
from .utils import parse_optimizer, parse_scheduler
from ..utils.base import (
    Updateable,
    update_end_if_possible,
    update_if_possible,
)
from ..utils.config import parse_structured
from ..utils.misc import C, cleanup, get_device, load_module_weights
from ..utils.saving import SaverMixin
from ..utils.typing import *
from spuv.utils.misc import time_recorder as tr


@dataclass
class BaseLossConfig:
    pass

def reset_ema_weights(state_dict):
    new_state_dict = {}
    ema_keys_set = []

    for key, value in state_dict.items():
        assert key.startswith("backbone.") or key.startswith("backbone_ema.")

        if key.startswith("backbone.") and not key.startswith("backbone_ema"):
            new_state_dict[key] = value  
            ema_key = "backbone_ema." + key[len("backbone."):].replace(".", "")
            assert ema_key in state_dict
            ema_keys_set.append(ema_key)
            new_state_dict[ema_key] = value

    for key, value in state_dict.items():
        if key.startswith("backbone_ema."):
            # assert key in ema_keys_set, key
            if key not in ema_keys_set:
                spuv.warn(f"key {key} not in ema_keys_set")

    spuv.warn("restart ema!!!")

    return new_state_dict


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loss: BaseLossConfig = field(default_factory=BaseLossConfig)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        weights_mapping: Optional[List[Dict[str, str]]] = None
        check_train_every_n_steps: int = 0
        check_val_limit_rank: int = 8
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False
        load_restart_ema: bool = False

    cfg: Config

    def __init__(self, cfg, resumed=False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}
        self._wandb_run_id: Optional[str] = None  # Store wandb run ID for resuming

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(
                self.cfg.weights,
                self.cfg.weights_ignore_modules,
                self.cfg.weights_mapping,
            )
        self.post_configure()

    def load_weights(
        self,
        weights: str,
        ignore_modules: Optional[List[str]] = None,
        mapping: Optional[List[Dict[str, str]]] = None,
    ):
        state_dict, epoch, global_step = load_module_weights(
            weights,
            ignore_modules=ignore_modules,
            mapping=mapping,
            map_location="cpu",
        )

        if self.cfg.load_restart_ema:
            state_dict = reset_ema_weights(state_dict)
        # state_dict = reset_ema_weights(state_dict)

        try:
            self.load_state_dict(state_dict, strict=False)
        except Exception as e:
            model_state_dict = self.state_dict()

            for key in list(state_dict.keys()):
                if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
                    spuv.warn(f"Skipping {key} due to size mismatch.")
                    del state_dict[key]
                elif key not in model_state_dict:
                    spuv.warn(f"Skipping {key} due to missing key.")
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)
            spuv.warn(f"Failed to load some keys from the checkpoint: {e}")
        # restore step-dependent states
        self.do_update_step(epoch, global_step, on_load_weights=True)


    def set_resume_status(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    def set_wandb_run_id(self, run_id: Optional[str]):
        """Set wandb run ID for resuming"""
        self._wandb_run_id = run_id
    
    def get_wandb_run_id(self) -> Optional[str]:
        """Get wandb run ID"""
        return self._wandb_run_id

    def on_save_checkpoint(self, checkpoint):
        """Save wandb run ID and critical training state to checkpoint"""
        if self._wandb_logger is not None and hasattr(self._wandb_logger.experiment, 'id'):
            checkpoint['wandb_run_id'] = self._wandb_logger.experiment.id
            spuv.info(f"Saved wandb run ID to checkpoint: {checkpoint['wandb_run_id']}")
        
        # CRITICAL: Ensure scheduler state is explicitly saved
        # PyTorch Lightning 2.x sometimes fails to restore scheduler state properly
        if hasattr(self, 'lr_schedulers'):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            
            checkpoint['_scheduler_states'] = []
            for i, scheduler in enumerate(schedulers):
                state = {
                    'last_epoch': scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else 0,
                    '_last_lr': scheduler._last_lr if hasattr(scheduler, '_last_lr') else None,
                    'state_dict': scheduler.state_dict(),
                }
                checkpoint['_scheduler_states'].append(state)
                spuv.info(f"Explicitly saved scheduler {i} state: last_epoch={state['last_epoch']}, last_lr={state['_last_lr']}")
    
    def on_load_checkpoint(self, checkpoint):
        """Load wandb run ID and restore scheduler state from checkpoint"""
        if 'wandb_run_id' in checkpoint:
            self._wandb_run_id = checkpoint['wandb_run_id']
            spuv.info(f"Loaded wandb run ID from checkpoint: {self._wandb_run_id}")
        
        # Log checkpoint state for debugging resume issues
        if 'epoch' in checkpoint and 'global_step' in checkpoint:
            spuv.info(f"Loading checkpoint from epoch {checkpoint['epoch']}, global_step {checkpoint['global_step']}")
        
        # Store scheduler state for restoration in on_fit_start
        # (can't restore here because schedulers aren't created yet)
        self._saved_scheduler_states = checkpoint.get('_scheduler_states', None)
        if self._saved_scheduler_states:
            spuv.info(f"Found explicitly saved scheduler states in checkpoint")
            for i, state in enumerate(self._saved_scheduler_states):
                spuv.info(f"  Scheduler {i}: last_epoch={state['last_epoch']}, last_lr={state['_last_lr']}")
        
        # Also check Lightning's built-in scheduler state
        if 'lr_schedulers' in checkpoint:
            spuv.info(f"Found {len(checkpoint['lr_schedulers'])} Lightning scheduler(s) in checkpoint")
            for i, sched_state in enumerate(checkpoint['lr_schedulers']):
                if 'last_epoch' in sched_state:
                    spuv.info(f"  Lightning Scheduler {i} last_epoch: {sched_state['last_epoch']}")
                if '_last_lr' in sched_state:
                    spuv.info(f"  Lightning Scheduler {i} last_lr: {sched_state['_last_lr']}")

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret

    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            spuv.info(f"Validation results will be saved to {self._save_dir}")
        else:
            spuv.warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )
        
        # NOTE: Do NOT restore scheduler state here!
        # on_fit_start() runs BEFORE Lightning restores the checkpoint state,
        # so self.global_step is still 0 at this point. Scheduler restoration
        # is handled in on_train_start() where global_step is correctly set.

    def training_step(self, batch, batch_idx):
        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None

        out = self.try_training_step(batch, batch_idx)

        return out

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()

        """
        tr.start("backward")
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)
        tr.end("backward")

    def try_training_step(self, batch, batch_idx):
        raise NotImplementedError

    def check_train(self, batch, outputs, **kwargs):
        if (
            self.global_rank == 0
            and self.cfg.check_train_every_n_steps > 0
            and self.true_global_step % self.cfg.check_train_every_n_steps == 0
        ):
            self.on_check_train(batch, outputs, **kwargs)

    def on_check_train(self, batch, outputs, **kwargs):
        pass

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_train_start(self):
        """Called at the start of training, AFTER checkpoint is loaded."""
        # CRITICAL: Verify checkpoint was actually loaded when resuming
        # This is called AFTER PyTorch Lightning loads the checkpoint
        if self._resumed:
            current_epoch = self.current_epoch
            current_step = self.global_step
            spuv.info(f"=" * 80)
            spuv.info(f"CHECKPOINT RESUME VERIFICATION:")
            spuv.info(f"  Current epoch: {current_epoch}")
            spuv.info(f"  Current global_step: {current_step}")
            
            if current_step == 0 and current_epoch == 0:
                spuv.warn(
                    "=" * 80 + "\n" +
                    "CRITICAL ERROR: Checkpoint resume FAILED!\n" +
                    "global_step and epoch are both 0, but resumed=True.\n" +
                    "This means PyTorch Lightning did not restore the checkpoint state.\n" +
                    "Training will start from scratch instead of resuming!\n" +
                    "=" * 80
                )
            else:
                spuv.info(f"  ✓ Checkpoint state successfully restored!")
                spuv.info(f"  Logging will continue from step {current_step}")
            
            spuv.info(f"=" * 80)
        
        # CRITICAL: Restore scheduler state when resuming training
        if self._resumed and hasattr(self, 'lr_schedulers'):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            
            # Check if we have explicitly saved scheduler states
            if hasattr(self, '_saved_scheduler_states') and self._saved_scheduler_states:
                spuv.info("Restoring scheduler states from explicit checkpoint save...")
                for i, (scheduler, saved_state) in enumerate(zip(schedulers, self._saved_scheduler_states)):
                    # Restore scheduler state
                    scheduler.load_state_dict(saved_state['state_dict'])
                    if saved_state['last_epoch'] is not None:
                        scheduler.last_epoch = saved_state['last_epoch']
                    if saved_state['_last_lr'] is not None:
                        scheduler._last_lr = saved_state['_last_lr']
                    spuv.info(f"Restored scheduler {i}: last_epoch={scheduler.last_epoch}, last_lr={scheduler._last_lr}")
            
            # Verify and correct scheduler state
            for i, scheduler in enumerate(schedulers):
                if hasattr(scheduler, 'last_epoch'):
                    expected_step = self.global_step
                    actual_step = scheduler.last_epoch
                    
                    # Allow small mismatch due to logging intervals
                    if abs(expected_step - actual_step) > 10:
                        spuv.warn(
                            f"Scheduler {i} last_epoch ({actual_step}) does not match global_step ({expected_step}). "
                            f"This may cause LR discontinuities. Correcting scheduler state..."
                        )
                        # Fix the scheduler's last_epoch to match global_step
                        scheduler.last_epoch = expected_step
                        # Step the scheduler to recalculate LR
                        scheduler.step()
                    
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 'unknown'
                    spuv.info(f"Scheduler {i} final state: step={scheduler.last_epoch}, LR={current_lr}")
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        import gc
        if hasattr(self, 'current_epoch'):
            # Only do expensive cleanup every 10 epochs to avoid slowdown
            if self.current_epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_epoch_end(self):
        pass

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            spuv.info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        pass

    def on_predict_end(self) -> None:
        if self._save_dir is not None:
            spuv.info(f"Export assets saved to {self._save_dir}")

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        tr.start("train_batch_start")
        self.preprocess_data(batch, "train")
        self.dataset = self.trainer.train_dataloader.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)
        tr.end("train_batch_start")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.dataset = self.trainer.val_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.dataset = self.trainer.test_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        tr.start("train_batch_end")
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        
        # Periodic garbage collection to prevent gradual memory accumulation
        if batch_idx % 10 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        tr.end("train_batch_end")

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.val_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.test_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        for name, p in self.named_parameters():
            if p.grad is None:
                spuv.info(f"{name} does not receive gradients!")
        """
        pass
