# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Sequence
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

__all__ = [
    "compute_confusion_matrix",
    "collect_preds",
    "save_fold_summary",
]

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

@torch.no_grad()
def collect_preds(model: torch.nn.Module,
                  dataloader: DataLoader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model on a dataloader and return (y_true, y_pred) as numpy arrays.
    Assumes batch dict has keys: 'image' (tensor) and 'y' (class indices).
    """
    model.eval()
    ys, yh = [], []
    for batch in dataloader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        yh.append(pred.cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(yh, axis=0)

def save_fold_summary(out_base: str,
                      fold: int,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sep: str = ";") -> str:
    """
    Save confusion matrix for a fold using numeric labels [0..K-1] on both rows and columns.
    The number of classes K is inferred from y_true/y_pred.
    """
    os.makedirs(out_base, exist_ok=True)

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    labels = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)

    path = os.path.join(out_base, f"fold{fold}_confusion.csv")
    df.to_csv(path, index=True, sep=sep)
    return path