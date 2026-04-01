# -*- coding: utf-8 -*-
"""
Single-split trainer for EVCBM (joint training, stabilized).
- Joint objective with decoupled gradients:
    * Label loss (NLL over log-betp) updates ONLY fusion-side params
      by feeding detached concept probabilities into fusion.
    * Concept loss (BCEWithLogits on c_logits) updates encoder + concept_head.
- Optional warmup epochs that train ONLY the concept loss first.
- Model selection by val label acc (higher is better).
- Reports both label acc and concept acc at the best checkpoint.
"""

from typing import Tuple, Dict, Any
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext

# ---- tiny AMP helper ----
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
    Evaluate concept BCE-with-logits and element-wise concept accuracy.
    Returns: (mean_bce, mean_acc)
    """
    model.eval()
    bce = getattr(model, "concept_bce_logits", None) or nn.BCEWithLogitsLoss()
    total_loss, total_count = 0.0, 0
    right, total = 0, 0

    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        c_gt = batch["concepts"].float().to(device, non_blocking=True)  # (B, D)
        # EVCBM / CBM / CEM share the same API:
        #   model(x, return_concepts=True) -> (label_out, c_prob, c_logits)
        _, c_prob, c_logits = model(x, return_concepts=True)
        loss = bce(c_logits, c_gt)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

        c_hat = (c_prob >= threshold).float()
        right += int((c_hat == c_gt).sum().item())
        total += int(c_gt.numel())

    mean_bce = total_loss / max(total_count, 1)
    mean_acc = right / max(total, 1)
    return mean_bce, mean_acc


def train_one_split_evcbm(model: torch.nn.Module,
                                dl_train: DataLoader,
                                dl_val: DataLoader,
                                cfg: Dict[str, Any],
                                device: torch.device = None) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """
    Train EVCBM on a single split with stabilized joint objective.

    cfg keys used:
      - learning_rate, weight_decay, epochs, patience, min_delta
      - lambda_concepts
      - fp16 (optional), concept_threshold (optional)
      - joint_warmup_epochs (optional, default 0)
      - max_grad_norm (optional, default 5.0)

    Returns:
      best_val_label_acc, best_val_concept_acc_at_best_label, best_state_dict
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # lr = float(cfg["learning_rate"])
    # lr = 1e-3
    # weight_decay = float(cfg["weight_decay"])
    # lam = float(cfg.get("lambda_concepts", 1.0))
    # min_delta = float(cfg.get("min_delta", 0.0))
    # patience = int(cfg.get("patience", 5))
    # concept_threshold = float(cfg.get("concept_threshold", 0.5))
    # warmup_epochs = int(cfg.get("joint_warmup_epochs", 0))
    # max_grad_norm = float(cfg.get("max_grad_norm", 5.0))
   

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    base_lr = float(cfg.get("learning_rate", 1e-3))
    backbone_lr_ratio = float(cfg.get("backbone_lr_ratio", 0.1))  # backbone lr 比 fusion 小一些

    lr_fusion = base_lr
    lr_backbone = base_lr * backbone_lr_ratio

    weight_decay = float(cfg["weight_decay"])
    lam = float(cfg.get("lambda_concepts", 1.0))
    min_delta = float(cfg.get("min_delta", 0.0))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))
    warmup_epochs = int(cfg.get("joint_warmup_epochs", 0))
    max_grad_norm = float(cfg.get("max_grad_norm", 5.0))

    # 拆分参数：encoder+concept_head vs 其余（fusion: base_logits, discount_net, ctx）
    backbone_params, fusion_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder.") or n.startswith("concept_head."):
            backbone_params.append(p)
        else:
            fusion_params.append(p)
            
    optimiser_name = cfg.get("optimiser_name", "AdamW")
    
    if optimiser_name == "AdamW":
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": fusion_params,  "lr": lr_fusion},
            ],
            weight_decay=weight_decay,
        )
        scheduler_flag = False
    else:
        optimizer = optim.SGD(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": fusion_params,  "lr": lr_fusion},
            ],
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

    nll = nn.NLLLoss()
    # ce = nn.CrossEntropyLoss()
    # Prefer model's helper if present
    bce = getattr(model, "concept_bce_logits", None) or nn.BCEWithLogitsLoss()

    best_label_acc = -1.0
    best_concept_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_cnt = 0

    for epoch in range(1, int(cfg["label_epochs"]) + 1):
        model.train()
        tot_loss = tot_correct = tot_count = 0.0
        t0 = time.time()

        joint_phase = epoch > warmup_epochs  # warmup: only concept loss

        for batch in dl_train:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            c_gt = batch["concepts"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast:
                    # ---- 1) concept branch (encoder + concept_head) ----
                    z = model.encoder(x)
                    c_logits = model.concept_head(z)       # (B, D)
                    c_prob = torch.sigmoid(c_logits)       # (B, D)

                    # ---- 2) label branch uses DETACHED concept probs to update only fusion ----
                    bel, mtheta = model.forward_fusion_only(c_prob.detach())
                    betp = model._betp_from_bel(bel, mtheta)
                    log_betp = torch.log(torch.clamp(betp, min=1e-12))
                    
                    loss_entropy = model.sparsity_regularizer()

                    # ---- 3) losses ----
                    loss = (nll(log_betp, y) if joint_phase else 0.0) + lam * bce(c_logits, c_gt) + 0.001 * loss_entropy
                    # loss = (ce(betp, y) if joint_phase else 0.0) + lam * bce(c_logits, c_gt)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # no-AMP path
                z = model.encoder(x)
                c_logits = model.concept_head(z)
                c_prob = torch.sigmoid(c_logits)

                bel, mtheta = model.forward_fusion_only(c_prob.detach())
                betp = model._betp_from_bel(bel, mtheta)
                log_betp = torch.log(torch.clamp(betp, min=1e-12))
                
                loss_entropy = model.sparsity_regularizer()
      

                loss = (nll(log_betp, y) if joint_phase else 0.0) + lam * bce(c_logits, c_gt) + 0.001 * loss_entropy
                # loss = (ce(betp, y) if joint_phase else 0.0) + lam * bce(c_logits, c_gt)
                loss.backward()
                if max_grad_norm and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # train stats (use label-side preds)
            preds = torch.argmax(log_betp, dim=1)
            bs = y.size(0)
            tot_loss += float(loss.item()) * bs
            tot_correct += int((preds == y).sum().item())
            tot_count += bs

        tr_loss = tot_loss / max(tot_count, 1)
        tr_acc = tot_correct / max(tot_count, 1)

        # ---- validation: label metrics ----
        model.eval()
        v_loss = v_correct = v_count = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                # forward uses full pipeline (no gradient anyway)
                log_betp = model(x)  # (B, C) log-betp
                loss = nll(log_betp, y)
                preds = torch.argmax(log_betp, dim=1)
                bs = y.size(0)
                v_loss += float(loss.item()) * bs
                v_correct += int((preds == y).sum().item())
                v_count += bs
        val_loss = v_loss / max(v_count, 1)
        val_label_acc = v_correct / max(v_count, 1)

        # ---- validation: concept metrics ----
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
