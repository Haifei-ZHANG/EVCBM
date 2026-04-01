# -*- coding: utf-8 -*-
# utils/concepts_eval.py
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()
def eval_concept_metrics(model: torch.nn.Module,
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

        # 假设模型签名: model(x, return_concepts=True) -> (y_prob, c_prob, c_logits)
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
