# -*- coding: utf-8 -*-
"""
Single-split trainer for ProbCBM (sequential).
Phase A: train (encoder + concept_head) with BCEWithLogits on c_logits; select by val concept loss (min).
Phase B: freeze A, train label_head with CE on GT concepts; select by val label acc (max).
Reports label acc and concept acc at the Phase-B-best checkpoint.
"""
from typing import Tuple, Dict, Any
import time, math
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
def _eval_concept_metrics_pred(model: torch.nn.Module,
                               dl: DataLoader,
                               device: torch.device,
                               threshold: float = 0.5):
    """
    Evaluate concept loss/ACC using model's outputs.
    Loss: BCEWithLogits on c_logits (AMP-safe).
    ACC: threshold on probabilities (c_prob >= threshold).
    """
    model.eval()
    bce_logits = nn.BCEWithLogitsLoss()
    total_loss, total_count = 0.0, 0
    right, total = 0, 0
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        c_gt = batch["concepts"].float().to(device, non_blocking=True)
        # forward to get prob and logits
        _, c_prob, c_logits = model(x, return_concepts=True)
        loss = bce_logits(c_logits, c_gt)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        c_hat = (c_prob >= threshold).float()
        right += int((c_hat == c_gt).sum().item())
        total += int(c_gt.numel())
    return total_loss / max(total_count, 1), right / max(total, 1)


def train_one_split_probcbm(model: torch.nn.Module,
                                    dl_train: DataLoader,
                                    dl_val: DataLoader,
                                    cfg: Dict[str, Any],
                                    device: torch.device = None) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """
    Returns:
      best_val_label_acc (Phase B), concept_acc_at_B_best, best_state_dict
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))
    min_delta = float(cfg.get("min_delta", 0.0))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    # ---------- Phase A: concept learning (LOGITS loss) ----------
    bce_logits = nn.BCEWithLogitsLoss()  # AMP-safe
    optimiser_name = cfg.get("optimiser_name", "AdamW")
    
    if optimiser_name == "AdamW":
        optA = optim.AdamW(list(model.encoder.parameters()) + list(model.concept_head.parameters()),
                            lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        schedA_flag = False
    else:
        optA = optim.SGD(
            list(model.encoder.parameters()) + list(model.concept_head.parameters()),
            lr=float(cfg["learning_rate"]),
            momentum=0.9,
            weight_decay=float(cfg["weight_decay"]),
        )
        schedA = optim.lr_scheduler.ReduceLROnPlateau(
            optA,
            mode="min",
            factor=0.1,
            patience=int(cfg.get("lr_patience", 5)),
        )
        schedA_flag = True
        
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))
    
    best_c_loss = math.inf
    bestA_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patienceA = 0

    for epoch in range(1, int(cfg["concept_epochs"]) + 1):
        model.train()
        tot_loss = tot_count = 0.0
        t0 = time.time()

        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            c_gt = batch["concepts"].float().to(device, non_blocking=True)
            bs = x.size(0)

            optA.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast:
                    # fast path: only encoder+concept_head
                    c_prob, c_logits = model.concepts_only(x)
                    loss = bce_logits(c_logits, c_gt)
                scaler.scale(loss).backward()
                scaler.step(optA)
                scaler.update()
            else:
                c_prob, c_logits = model.concepts_only(x)
                loss = bce_logits(c_logits, c_gt)
                loss.backward()
                optA.step()

            tot_loss += float(loss.item()) * bs
            tot_count += bs

        tr_c_loss = tot_loss / max(tot_count, 1)

        # val concept loss
        model.eval()
        v_tot = v_cnt = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["image"].to(device, non_blocking=True)
                c_gt = batch["concepts"].float().to(device, non_blocking=True)
                c_prob, c_logits = model.concepts_only(x)
                loss = bce_logits(c_logits, c_gt)
                bs = x.size(0)
                v_tot += float(loss.item()) * bs
                v_cnt += bs
        val_c_loss = v_tot / max(v_cnt, 1)
        
        dt = time.time() - t0
        
        if schedA_flag:
            schedA.step(val_c_loss)
            
        print(f"[A] Epoch {epoch:02d}: train_c_loss={tr_c_loss:.4f} | val_c_loss={val_c_loss:.4f} ({dt:.1f}s)")

        if val_c_loss < best_c_loss - min_delta:
            best_c_loss = val_c_loss
            patienceA = 0
            bestA_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patienceA += 1
            if patienceA >= int(cfg["patience"]):
                print("Phase A early stopping.")
                break

    # load best A
    model.load_state_dict(bestA_state)

    # ---------- Phase B: label head on GT concepts ----------
    # freeze encoder + concept_head
    for p in list(model.encoder.parameters()) + list(model.concept_head.parameters()):
        p.requires_grad = False
    model.encoder.eval()
    model.concept_head.eval()

    ce = nn.CrossEntropyLoss()

    optimiser_name = cfg.get("optimiser_name", "AdamW")
    
    if optimiser_name == "AdamW":
        optB = optim.AdamW([p for n,p in model.named_parameters()
                            if (not n.startswith("encoder.")) and (not n.startswith("concept_head.")) and p.requires_grad],
                            lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        schedB_flag = False
    else:
        optB = optim.SGD(
            [p for n, p in model.named_parameters()
              if (not n.startswith("encoder.")) and (not n.startswith("concept_head.")) and p.requires_grad],
            lr=float(cfg["learning_rate"]),
            momentum=0.9,
            weight_decay=float(cfg["weight_decay"]),
        )
        schedB = optim.lr_scheduler.ReduceLROnPlateau(
            optB,
            mode="min",
            factor=0.1,
            patience=int(cfg.get("lr_patience", 5)),
        )
        schedB_flag = True
        
    scaler, autocast = _make_amp(bool(cfg.get("fp16", False)) and (device.type == "cuda"))
    
    best_label_acc = -1.0
    best_state_B = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_concept_acc = 0.0
    patienceB = 0

    for epoch in range(1, int(cfg["label_epochs"]) + 1):
        # train on GT concepts
        model.train()
        model.encoder.eval()
        model.concept_head.eval()
        tot_loss = tot_correct = tot_count = 0.0
        t0 = time.time()
        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optB.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast:
                    y_logits = model(x)
                    loss = ce(y_logits, y)
                scaler.scale(loss).backward()
                scaler.step(optB)
                scaler.update()
            else:
                y_logits = model(x)
                loss = ce(y_logits, y)
                loss.backward()
                optB.step()

            preds = torch.argmax(y_logits, dim=1)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_correct += int((preds == y).sum().item())
            tot_count += bs
        tr_loss = tot_loss / max(tot_count, 1)
        tr_acc = tot_correct / max(tot_count, 1)

        # val using predicted concept means (end-to-end)
        model.eval()
        v_loss = v_correct = v_count = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                y_logits = model(x)  # forward uses predicted mean probs internally
                loss = ce(y_logits, y)
                preds = torch.argmax(y_logits, dim=1)
                bs = y.size(0)
                v_loss += float(loss.item()) * bs
                v_correct += int((preds == y).sum().item())
                v_count += bs
        val_loss = v_loss / max(v_count, 1)
        val_label_acc = v_correct / max(v_count, 1)

        # recompute concept metrics for current (frozen) A
        val_c_loss_ep, val_c_acc_ep = _eval_concept_metrics_pred(
            model, dl_val, device, threshold=concept_threshold
        )
        dt = time.time() - t0
        
        if schedB_flag:
            schedB.step(val_loss)
            
        print(f"[B] Epoch {epoch:02d}: train_y_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_y_loss={val_loss:.4f} label_acc={val_label_acc:.4f} "
              f"| concept_bce={val_c_loss_ep:.4f} concept_acc={val_c_acc_ep:.4f} ({dt:.1f}s)")
        
        if epoch <= 5:
            patienceB = 0
            continue
        
        if val_label_acc > best_label_acc + min_delta:
            best_label_acc = val_label_acc
            best_loss = val_loss
            best_state_B = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_c_acc_ep
            patienceB = 0
        elif val_label_acc == best_label_acc and val_loss < best_loss:
            best_loss = val_loss
            best_state_B = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_concept_acc = val_c_acc_ep
            patienceB = 0
        else:
            patienceB += 1
            if patienceB >= int(cfg["patience"]):
                print("Phase B early stopping.")
                break

    model.load_state_dict(best_state_B)
    return float(best_label_acc), float(best_concept_acc), best_state_B
