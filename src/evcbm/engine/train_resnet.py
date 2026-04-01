# -*- coding: utf-8 -*-
"""
Single-split trainer for BaselineResNet50.
- Model selection & early stopping by val_acc (higher is better).
- AMP uses torch.amp (no deprecated API).
- Returns (best_val_acc, best_state_dict).
"""
from typing import Tuple, Dict, Any
import time
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader


def make_amp(use_fp16: bool):
    """Tiny AMP helper using torch.amp (no deprecated API)."""
    if use_fp16 and torch.cuda.is_available():
        scaler = torch.amp.GradScaler(enabled=True)
        autocast = torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        scaler = None
        autocast = nullcontext()
    return scaler, autocast


def _train_one_epoch(model: torch.nn.Module,
                     dl: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     scaler,
                     autocast) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()

    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast:
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((preds == y).sum().item())
        total_count += bs

    denom = max(total_count, 1)
    return total_loss / denom, total_correct / denom


def _evaluate(model: torch.nn.Module,
              dl: DataLoader,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            preds = torch.argmax(logits, dim=1)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((preds == y).sum().item())
            total_count += bs

    denom = max(total_count, 1)
    return total_loss / denom, total_correct / denom


def train_one_split_baseline(model: torch.nn.Module,
                             dl_train: DataLoader,
                             dl_val: DataLoader,
                             cfg: Dict[str, Any],
                             device: torch.device = None) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Train on a single split (one train/val partition).

    cfg fields used:
      - learning_rate, weight_decay, epochs, patience, min_delta (optional), fp16 (optional)

    Returns:
      best_val_acc, best_state_dict
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimiser_name = cfg.get("optimiser_name", "AdamW")
    
    if optimiser_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=float(cfg["learning_rate"]),
                                weight_decay=float(cfg["weight_decay"]))
        scheduler_flag = False
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            momentum=0.9,
            weight_decay=float(cfg["weight_decay"]),
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=int(cfg.get("lr_patience", 5)),
        )
        scheduler_flag = True
        
    scaler, autocast = make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))

    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_cnt = 0
    min_delta = float(cfg.get("min_delta", 0.0))

    for epoch in range(1, int(cfg["label_epochs"]) + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(model, dl_train, optimizer, device, scaler, autocast)
        val_loss, val_acc = _evaluate(model, dl_val, device)
        dt = time.time() - t0
        
        if scheduler_flag:
            scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} ({dt:.1f}s)")
        
        if epoch <= 5:
            patience_cnt = 0
            continue
        
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            best_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif val_acc == best_acc and val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= int(cfg["patience"]):
                print("Early stopping.")
                break
            
    # restore best
    model.load_state_dict(best_state)
    return float(best_acc), best_state
