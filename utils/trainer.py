"""Pre-training trainers for BERT and DeBERTa.

Both trainers are thin subclasses over :class:`_PretrainTrainerBase`, sharing
the NSP + MLM training loop, checkpointing, and integration with
:class:`TrainingLogger` for structured logging.
"""

from pathlib import Path
from typing import Optional

import tqdm
import torch
import torch.nn as nn
from torch import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import BERTLM, BERT, DeBERTaLM, DeBERTa

from .common import parse_cuda_devices
from .logger import TrainingLogger
from .scheduler import get_warmup_linear_schedule


class _PretrainTrainerBase:
    """Shared training loop for BERT / DeBERTa pre-training.

    Subclasses set `mlm_label`, `ckpt_prefix`, `best_ckpt_name` and implement
    `_build_model(encoder, vocab_size)`.
    """

    mlm_label = "MLM"
    best_ckpt_name = "best_model.pt"

    def __init__(self, encoder, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: Optional[DataLoader] = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 warmup_steps: int = 10000, total_steps: int = 1000000,
                 grad_clip: float = 1.0, fp16: bool = True,
                 gradient_accumulation_steps: int = 1,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10,
                 training_logger: Optional[TrainingLogger] = None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.encoder = encoder
        self.model = self._build_model(encoder, vocab_size).to(self.device)

        self.fp16 = fp16
        self.grad_clip = grad_clip
        self.accum_steps = max(1, int(gradient_accumulation_steps))

        self.tlogger = training_logger

        if with_cuda and torch.cuda.device_count() > 1:
            self._log("Using %d GPUs", torch.cuda.device_count())
            # CUDA_VISIBLE_DEVICES already controls which GPUs are visible,
            # so DataParallel just uses all of them (device_ids=None).
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        no_decay = ('bias', 'LayerNorm.weight')
        named_params = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        # `fused=True` runs the AdamW update in a single CUDA kernel; on recent
        # PyTorch + CUDA it's 1.2–1.5× faster than the default foreach path.
        adamw_kwargs = dict(lr=lr, betas=betas, eps=1e-8)
        if torch.cuda.is_available() and cuda_condition:
            adamw_kwargs["fused"] = True
        self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        self.scheduler = get_warmup_linear_schedule(
            self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        )

        self.scaler = amp.GradScaler('cuda', enabled=fp16)

        # MLM head is now sparse — it outputs logits only at masked positions,
        # so the loss has no pad positions to ignore.
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.nsp_loss_fn = nn.CrossEntropyLoss()

        self.log_freq = log_freq
        self.step_count = 0
        self.best_loss = float('inf')

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log("Total Parameters: %.2fM", total / 1e6)
        self._log("Trainable Parameters: %.2fM", trainable / 1e6)

    # ----------------------------------------------------------- hooks

    def _build_model(self, encoder, vocab_size):
        raise NotImplementedError

    def _encoder_config(self):
        return {
            'hidden': self.encoder.hidden,
            'layers': self.encoder.n_layers,
            'attn_heads': self.encoder.attn_heads,
        }

    # ----------------------------------------------------------- helpers

    def _log(self, msg: str, *args) -> None:
        if self.tlogger is not None:
            self.tlogger.info(msg, *args)

    # ----------------------------------------------------------- public API

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        device = self.device
        # Keep metric accumulators on the GPU as float/long scalars so we
        # don't force a device sync (.item()) on every micro-batch — only
        # when we actually need to log a number.
        total_loss = torch.zeros((), device=device)
        total_nsp_loss = torch.zeros((), device=device)
        total_mlm_loss = torch.zeros((), device=device)
        total_nsp_correct = torch.zeros((), device=device, dtype=torch.long)
        total_mlm_correct = torch.zeros((), device=device, dtype=torch.long)
        total_mlm_element = torch.zeros((), device=device, dtype=torch.long)
        total_nsp_element = 0  # NSP element count is exactly batch_size — pure int

        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} Epoch {epoch:03d}",
            total=len(data_loader),
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )

        # Flush any stale grads before starting an accumulation window.
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        accum = self.accum_steps
        last_idx = len(data_loader) - 1

        for i, data in data_iter:
            data = {key: value.to(device, non_blocking=True) for key, value in data.items()}
            bert_label = data["bert_label"]
            is_next = data["is_next"]
            # MLM labels live on positions where bert_label != 0 (pad). Pass this
            # mask into the model so the MLM head projects only masked positions
            # (saves the per-position vocab matmul ~6x on 15% masking).
            mlm_mask = bert_label != 0
            mlm_labels_flat = bert_label[mlm_mask]  # [M]

            with amp.autocast('cuda', enabled=self.fp16):
                nsp_logits, mlm_logits = self.model(
                    data["bert_input"], data["segment_label"], mlm_mask=mlm_mask,
                )
                nsp_loss = self.nsp_loss_fn(nsp_logits, is_next)
                if mlm_logits.numel() == 0:
                    # No masked positions in this batch — rare but possible.
                    mlm_loss = torch.zeros((), device=device, dtype=nsp_loss.dtype)
                else:
                    mlm_loss = self.mlm_loss_fn(mlm_logits, mlm_labels_flat)
                loss = nsp_loss + mlm_loss

            if train:
                # Scale-down so accumulated grads across `accum` micro-batches
                # match what a single full-batch forward would produce.
                self.scaler.scale(loss / accum).backward()

                # Step only on accumulation boundaries (and flush the tail at
                # the end of the epoch so leftover grads don't leak to later).
                is_boundary = ((i + 1) % accum == 0) or (i == last_idx)
                if is_boundary:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step_count += 1

            # Accumulate metrics on the GPU — no .item() here.
            total_loss += loss.detach()
            total_mlm_loss += mlm_loss.detach()
            total_nsp_loss += nsp_loss.detach()

            nsp_preds = nsp_logits.argmax(dim=-1)
            total_nsp_correct += nsp_preds.eq(is_next).sum()
            total_nsp_element += is_next.numel()

            if mlm_logits.numel() > 0:
                mlm_preds = mlm_logits.argmax(dim=-1)
                total_mlm_correct += mlm_preds.eq(mlm_labels_flat).sum()
            total_mlm_element += mlm_labels_flat.numel()

            if i % self.log_freq == 0 or i == last_idx:
                # One device→host sync for the whole metric bundle (instead of
                # five per batch). Running means are safe on CPU after this.
                loss_cpu = total_loss.item()
                mlm_loss_cpu = total_mlm_loss.item()
                nsp_loss_cpu = total_nsp_loss.item()
                mlm_correct_cpu = total_mlm_correct.item()
                nsp_correct_cpu = total_nsp_correct.item()
                mlm_element_cpu = total_mlm_element.item()

                avg_loss = loss_cpu / (i + 1)
                mlm_acc = mlm_correct_cpu / max(1, mlm_element_cpu) * 100
                nsp_acc = nsp_correct_cpu / max(1, total_nsp_element) * 100
                current_lr = self.scheduler.get_last_lr()[0]

                data_iter.set_postfix({
                    "LR": f"{current_lr:.2e}",
                    "Loss": f"{avg_loss:.4f}",
                    f"{self.mlm_label} Acc": f"{mlm_acc:.2f}%",
                    "NSP Acc": f"{nsp_acc:.2f}%",
                })

                if train and self.tlogger is not None:
                    self.tlogger.log_step(
                        epoch=epoch, step=self.step_count,
                        lr=current_lr,
                        batch_loss=loss.item(),
                        batch_nsp_loss=nsp_loss.item(),
                        batch_mlm_loss=mlm_loss.item(),
                        running_loss=avg_loss,
                        running_mlm_acc=mlm_acc,
                        running_nsp_acc=nsp_acc,
                    )

        n_batches = max(1, len(data_loader))
        # One final sync to pull epoch-level aggregates off the GPU.
        avg_total_loss = total_loss.item() / n_batches
        avg_mlm_loss = total_mlm_loss.item() / n_batches
        avg_nsp_loss = total_nsp_loss.item() / n_batches
        mlm_acc = total_mlm_correct.item() / max(1, total_mlm_element.item()) * 100
        nsp_acc = total_nsp_correct.item() / max(1, total_nsp_element) * 100

        if self.tlogger is not None:
            self.tlogger.log_epoch(
                epoch=epoch, split=str_code,
                avg_loss=avg_total_loss,
                avg_mlm_loss=avg_mlm_loss,
                avg_nsp_loss=avg_nsp_loss,
                mlm_acc=mlm_acc,
                nsp_acc=nsp_acc,
            )
        else:
            # Fallback: still print a summary to stderr if no TrainingLogger attached.
            print(
                f"{str_code} Epoch {epoch} | Avg Loss: {avg_total_loss:.4f} "
                f"| {self.mlm_label} Loss: {avg_mlm_loss:.4f} | NSP Loss: {avg_nsp_loss:.4f} "
                f"| {self.mlm_label} Acc: {mlm_acc:.2f}% | NSP Acc: {nsp_acc:.2f}%"
            )

        return avg_total_loss, mlm_acc, nsp_acc

    # -------------------------------------------------------- checkpointing

    def _resolve_store_dir(self, store_path):
        if store_path is not None:
            return Path(store_path)
        if self.tlogger is not None:
            return self.tlogger.checkpoint_dir
        raise ValueError("store_path is required when no TrainingLogger is attached")

    def save_best(self, epoch, loss, store_path=None):
        """Save a checkpoint only if `loss` improves on `self.best_loss`.

        The checkpoint is written directly to ``best_model.pt`` under the
        resolved directory, overwriting any previous best. Returns
        ``(path, True)`` when saved, ``(None, False)`` otherwise.
        """
        if loss >= self.best_loss:
            return None, False

        self.best_loss = loss
        directory = self._resolve_store_dir(store_path)
        directory.mkdir(parents=True, exist_ok=True)

        model_state = (self.model.module.state_dict()
                       if isinstance(self.model, nn.DataParallel)
                       else self.model.state_dict())

        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': self._encoder_config(),
            'best_loss': self.best_loss,
        }

        save_path = directory / self.best_ckpt_name
        torch.save(checkpoint, save_path)
        self._log("New best loss %.4f at epoch %d — saved %s",
                  self.best_loss, epoch, save_path)
        return save_path, True

    def load_checkpoint(self, checkpoint_path, map_location=None):
        """Load model / optimizer / scheduler / scaler / step. Returns stored epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location or self.device)
        target = (self.model.module
                  if isinstance(self.model, nn.DataParallel)
                  else self.model)
        target.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.step_count = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('best_loss', self.best_loss)
        self._log("Resumed from %s (epoch=%d, step=%d, best_loss=%.4f)",
                  checkpoint_path, checkpoint['epoch'], self.step_count, self.best_loss)
        return checkpoint['epoch']


class BERTTrainer(_PretrainTrainerBase):
    """BERT pre-training trainer: NSP + MLM with CrossEntropyLoss on logits."""

    mlm_label = "MLM"
    best_ckpt_name = "best_model.pt"

    def __init__(self, bert: BERT, vocab_size: int, *args, **kwargs):
        super().__init__(bert, vocab_size, *args, **kwargs)

    def _build_model(self, encoder, vocab_size):
        return BERTLM(encoder, vocab_size)


class DeBERTaTrainer(_PretrainTrainerBase):
    """DeBERTa pre-training trainer: NSP + Enhanced Mask Decoder on logits."""

    mlm_label = "EMD"
    best_ckpt_name = "best_model.pt"

    def __init__(self, deberta: DeBERTa, vocab_size: int, *args, **kwargs):
        super().__init__(deberta, vocab_size, *args, **kwargs)

    def _build_model(self, encoder, vocab_size):
        return DeBERTaLM(encoder, vocab_size)

    def _encoder_config(self):
        cfg = super()._encoder_config()
        cfg['max_relative_positions'] = getattr(self.encoder, 'max_relative_positions', 512)
        return cfg
