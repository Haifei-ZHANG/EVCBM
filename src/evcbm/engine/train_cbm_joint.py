# -*- coding: utf-8 -*-
"""
Single-split trainer for CBM (joint training).
- Loss = CE(label) + lambda_concepts * BCEWithLogits(concepts)
- Model selection by val label acc (higher is better)
- Reports both label acc and concept acc
"""
from typing import Tuple, Dict, Any
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import nullcontext


def _make_amp(use_fp16: bool):
    if use_fp16 and torch.cuda.is_available():
        scaler = torch.amp.GradScaler(enabled=True)
        autocast = torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        scaler = None
        autocast = nullcontext()
    return scaler, autocast


@torch.no_grad()
def _eval_concept_metrics(model: torch.nn.Module,
                          dl: DataLoader,
                          device: torch.device,
                          threshold: float = 0.5) -> Tuple[float, float]:
    """
    Returns:
      concept_bce (mean), concept_acc (element-wise mean accuracy over all samples × concepts)
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss, total_count = 0.0, 0
    right, total = 0, 0

    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        c_gt = batch["concepts"].float().to(device, non_blocking=True)  # (B, D)
        _, c_prob, c_logits = model(x, return_concepts=True)
        loss = bce(c_logits, c_gt)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        c_hat = (c_prob >= threshold).float()
        right += int((c_hat == c_gt).sum().item())
        total += int(c_gt.numel())
    concept_bce = total_loss / max(total_count, 1)
    concept_acc = right / max(total, 1)
    return concept_bce, concept_acc


def train_one_split_cbm_joint(model: torch.nn.Module,
                              dl_train: DataLoader,
                              dl_val: DataLoader,
                              cfg: Dict[str, Any],
                              device: torch.device = None) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """
    Train on a single split with joint CBM objective.

    cfg keys used:
      learning_rate, weight_decay, epochs, patience, min_delta, lambda_concepts,
      fp16(optional), concept_threshold(optional)

    Returns:
      best_val_label_acc, best_val_concept_acc_at_that_checkpoint, best_state_dict
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
        
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))

    ce = nn.CrossEntropyLoss()
    # Prefer model's concept BCE helper if provided (e.g., CEM), else use standard BCEWithLogits
    bce = getattr(model, "concept_bce_logits", None) or nn.BCEWithLogitsLoss()

    best_label_acc = -1.0
    best_concept_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_cnt = 0
    min_delta = float(cfg.get("min_delta", 0.0))
    lam = float(cfg.get("lambda_concepts", 1.0))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    for epoch in range(1, int(cfg["label_epochs"]) + 1):
        # ---- train ----
        model.train()
        tot_loss = tot_correct = tot_count = 0.0
        t0 = time.time()
        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            c_gt = batch["concepts"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast:
                    y_logits, c_prob, c_logits = model(x, return_concepts=True)
                    loss = ce(y_logits, y) + lam * bce(c_logits, c_gt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_logits, c_prob, c_logits = model(x, return_concepts=True)
                loss = ce(y_logits, y) + lam * bce(c_logits, c_gt)
                loss.backward()
                optimizer.step()

            preds = torch.argmax(y_logits, dim=1)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_correct += int((preds == y).sum().item())
            tot_count += bs
        tr_loss = tot_loss / max(tot_count, 1)
        tr_acc = tot_correct / max(tot_count, 1)

        # ---- val (label) ----
        model.eval()
        v_loss = v_correct = v_count = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                y_logits = model(x)  # label only
                loss = ce(y_logits, y)
                preds = torch.argmax(y_logits, dim=1)
                bs = y.size(0)
                v_loss += float(loss.item()) * bs
                v_correct += int((preds == y).sum().item())
                v_count += bs
        val_loss = v_loss / max(v_count, 1)
        val_label_acc = v_correct / max(v_count, 1)

        # ---- val (concept metrics) ----
        val_concept_bce, val_concept_acc = _eval_concept_metrics(model, dl_val, device, threshold=concept_threshold)
        dt = time.time() - t0
        
        if scheduler_flag:
            scheduler.step(val_loss)
            
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} label_acc={val_label_acc:.4f} "
              f"| concept_bce={val_concept_bce:.4f} concept_acc={val_concept_acc:.4f} ({dt:.1f}s)")

        # ---- model selection by label acc ----
        if epoch <= 5:
            patience_cnt = 0
            continue
        
        if val_label_acc > best_label_acc + min_delta:
            best_label_acc = val_label_acc
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_concept_acc
            patience_cnt = 0
        elif val_label_acc == best_label_acc and val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_concept_acc
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= int(cfg["patience"]):
                print("Phase B early stopping.")
                break

    model.load_state_dict(best_state)
    return float(best_label_acc), float(best_concept_acc), best_state
