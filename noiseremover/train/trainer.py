import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Config


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, cfg: Config, run_name: str = "run"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = next(model.parameters()).device

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, run_name))

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for noisy, clean, _ in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            mask = self.model(noisy)
            pred = _apply_mask(noisy, mask)
            loss = self.loss_fn(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
            self.global_step += 1

        return total_loss / len(self.train_loader)

    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i, (noisy, clean, _) in enumerate(tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False)):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                mask = self.model(noisy)
                pred = _apply_mask(noisy, mask)
                loss = self.loss_fn(pred, clean)
                total_loss += loss.item()

                if i == 0 and epoch % 5 == 0:
                    self.writer.add_image("val/noisy", _spec_to_image(noisy[0]), epoch)
                    self.writer.add_image("val/pred",  _spec_to_image(pred[0]),  epoch)
                    self.writer.add_image("val/clean", _spec_to_image(clean[0]), epoch)

        return total_loss / len(self.val_loader)

    def fit(self):
        cfg = self.cfg
        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            self.writer.add_scalar("val/loss_epoch", val_loss, epoch)
            self.writer.add_scalar("train/lr", lr, epoch)

            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e}")

            self._save_checkpoint(epoch, val_loss, is_last=True)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self.writer.close()

    def _save_checkpoint(self, epoch, val_loss, is_best=False, is_last=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if is_last:
            torch.save(state, os.path.join(self.cfg.checkpoint_dir, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(self.cfg.checkpoint_dir, "best.pt"))


def _apply_mask(noisy_db: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid mask in amplitude domain and return result in dB."""
    amplitude = 10 ** (noisy_db / 20.0)
    return 20.0 * torch.log10((amplitude * mask).clamp(min=1e-5))


def _spec_to_image(spec: torch.Tensor) -> torch.Tensor:
    # spec: (1, n_mels, T) — normalize to [0,1] for TensorBoard
    s = spec - spec.min()
    s = s / (s.max() + 1e-8)
    return s.flip(1)  # flip frequency axis so low freq is at bottom
