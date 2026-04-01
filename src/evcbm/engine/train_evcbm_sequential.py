
# -*- coding: utf-8 -*-
"""
Sequential trainer for EVCBM that REQUIRES a concept backbone (encoder + concept_head)
pretrained & saved by CBM-sequential (per fold). No Phase A here.
Trains evidential fusion head (Phase B) with frozen backbone.
"""
from typing import Tuple, Dict, Any
import time, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evcbm.utils.concepts_io import load_concept_backbone
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
def _eval_concept_metrics_pred(model: torch.nn.Module,
                               dl,
                               device: torch.device,
                               threshold: float = 0.5):
    """Evaluate concept BCE/Acc using *predicted* concepts on dl."""
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss, total_count = 0.0, 0
    right, total = 0, 0
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        c_gt = batch["concepts"].float().to(device, non_blocking=True)
        _, c_prob, c_logits = model(x, return_concepts=True)
        loss = bce(c_logits, c_gt)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        c_hat = (c_prob >= threshold).float()
        right += int((c_hat == c_gt).sum().item())
        total += int(c_gt.numel())
    return total_loss / max(total_count, 1), right / max(total, 1)

def train_one_split_evcbm_sequential(model: torch.nn.Module,
                                      dl_train: DataLoader,
                                      dl_val: DataLoader,
                                      cfg: Dict[str, Any],
                                      device: torch.device = None) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))
    min_delta = float(cfg.get("min_delta", 0.0))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    ckpt_path = cfg.get("concept_ckpt_path", None)
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        raise RuntimeError("EVCBM sequential requires a backbone from CBM-sequential. "
                           "Missing cfg['concept_ckpt_path'] or file not found.")

    load_concept_backbone(model, ckpt_path, freeze=True)
    print(f"[Backbone] Loaded & frozen from {ckpt_path}")
    model.encoder.eval(); model.concept_head.eval()

    nll = nn.NLLLoss()
    # ce = nn.CrossEntropyLoss()
    
    optimiser_name = cfg.get("optimiser_name", "AdamW")
    lr=float(cfg.get("learning_rate", 1e-3))
    # lr = 1e-2
    if optimiser_name == "AdamW":
        optB = optim.AdamW([p for n,p in model.named_parameters()
                            if (not n.startswith("encoder.")) and (not n.startswith("concept_head.")) and p.requires_grad],
                            lr=lr, weight_decay=float(cfg["weight_decay"]))
        schedB_flag = False
    else:
        optB = optim.SGD(
            [p for n, p in model.named_parameters()
              if (not n.startswith("encoder.")) and (not n.startswith("concept_head.")) and p.requires_grad],
            lr=float(cfg["learning_rate"]),
            momentum=0.9,
            weight_decay=float(cfg["weight_decay"])
        )
        
        schedB = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optB,
            mode="min",
            factor=0.1,
            patience=int(cfg.get("lr_patience", 5)),
        )
        schedB_flag = True
        
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))
    
    best_label_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_concept_acc = 0.0
    patienceB = 0

    for epoch in range(1, int(cfg["label_epochs"]) + 1):
        model.train()
        
        model.encoder.eval(); model.concept_head.eval()
        tot_loss = tot_correct = tot_count = 0.0
        t0 = time.time()
        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optB.zero_grad(set_to_none=True)
            with autocast if scaler is not None else nullcontext():
                log_betp = model(x)
                loss_entropy = model.sparsity_regularizer()
                
                loss = nll(log_betp, y) + 0.001 * loss_entropy 
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optB); scaler.update()
            else:
                loss.backward(); optB.step()

            preds = torch.argmax(log_betp, dim=1)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_correct += int((preds == y).sum().item())
            tot_count += bs

        tr_loss = tot_loss / max(tot_count, 1)
        tr_acc = tot_correct / max(tot_count, 1)

        # validation
        model.eval()
        v_loss = v_correct = v_count = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                log_betp = model(x)
                loss = nll(log_betp, y)
                preds = torch.argmax(log_betp, dim=1)
                bs = y.size(0)
                v_loss += float(loss.item()) * bs
                v_correct += int((preds == y).sum().item())
                v_count += bs
        val_loss = v_loss / max(v_count, 1)
        val_label_acc = v_correct / max(v_count, 1)

        val_c_bce, val_c_acc = _eval_concept_metrics_pred(model, dl_val, device, threshold=concept_threshold)
        dt = time.time() - t0
        
        if schedB_flag:
            schedB.step(val_loss)
            
        print(f"[B] Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} label_acc={val_label_acc:.4f} "
              f"| concept_bce={val_c_bce:.4f} concept_acc={val_c_acc:.4f} ({dt:.1f}s)")
        
        if epoch <= 5:
            patienceB = 0
            continue
        
        if val_label_acc > best_label_acc + min_delta:
            best_label_acc = val_label_acc
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_c_acc
            patienceB = 0
        elif val_label_acc == best_label_acc and val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_c_acc
            patienceB = 0
        else:
            patienceB += 1
            if patienceB >= int(cfg["patience"]):
                print("Phase B early stopping.")
                break

    model.load_state_dict(best_state)
    return float(best_label_acc), float(best_concept_acc), best_state
