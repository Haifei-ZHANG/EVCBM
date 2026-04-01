# -*- coding: utf-8 -*-
"""
K-fold CV runner for BaselineResNet50.
- StratifiedKFold on labels.
- Reports per-fold val_acc and final mean±std.
- Uses your project's dataset/transform utilities.
"""
import os
from typing import Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from evcbm.utils.mem import cleanup_torch
from evcbm.data.transforms import get_transforms            
from evcbm.data.datasets import ConceptDataset, build_dataloaders  
from evcbm.utils.seed import set_seed                 
from .train_resnet import train_one_split_baseline
from evcbm.eval.metrics import collect_preds, save_fold_summary
from evcbm.utils.concepts_io import save_concept_backbone


def run_kfold_resnet(cfg: Dict[str, Any],
                     model_factory: Callable[[int], torch.nn.Module],
                     get_transforms_fn: Callable[[bool], Tuple]=get_transforms):
    """
    cfg needs:
      root_dir, dataset, num_folds, seed,
      batch_size, num_workers, augment,
      learning_rate, weight_decay, epochs, patience, out_dir,
      fp16(optional), min_delta(optional),
      inner_val_ratio(optional, default 0.2)
    """
    set_seed(int(cfg["seed"]))
    out_base = os.path.join(cfg["out_dir"], cfg["dataset"], "resnet50")
    os.makedirs(out_base, exist_ok=True)

    # 用非增强的 transform 构建 full_ds（deterministic features）
    _, val_tfms = get_transforms_fn(augment=False)
    full_ds = ConceptDataset(root_dir=cfg["root_dir"],
                             dataset=cfg["dataset"],
                             transform=val_tfms)
    # labels 用于 StratifiedKFold 和 inner split
    labels = full_ds.Y_onehot.argmax(axis=1)

    skf = StratifiedKFold(
        n_splits=int(cfg["num_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inner_val_ratio = float(cfg.get("inner_val_ratio", 0.2))

    results = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n===== Fold {fold}/{cfg['num_folds']} =====")

        # 这一折使用的 train / val transforms
        train_tfms, valid_tfms = get_transforms_fn(augment=bool(cfg["augment"]))

        # 这里的 idx_tr 是外层 train；idx_va 是外层 test（原来的“val”）
        dl_train, dl_val, dl_test = build_dataloaders(
            ds=full_ds,
            idx_train=idx_tr,
            idx_test=idx_va,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            train_tfms=train_tfms,
            val_tfms=valid_tfms,
            inner_val_ratio=inner_val_ratio,
            seed=int(cfg["seed"])  # 固定 seed，保证不同方法用同一划分
        )

        model = model_factory(full_ds.num_classes)

        # dl_train / dl_val 用于 early stopping，返回 best_val_acc & best_state
        best_val_acc, best_state = train_one_split_baseline(
            model, dl_train, dl_val, cfg, device=device
        )
        print(f"Fold {fold} (VAL) best val_acc={best_val_acc:.4f}")

        # 用 best checkpoint 在 test 集上评估
        model.load_state_dict(best_state)
        full_path = os.path.join(out_base, f"model-full-{fold}.pt")
        torch.save({"model_state": model.state_dict()}, full_path)

        y_true, y_pred = collect_preds(model, dl_test, device)

        # 计算这一折的 test accuracy
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_true_np = np.asarray(y_true)

        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = np.asarray(y_pred)

        test_acc = float((y_true_np == y_pred_np).mean())
        print(f"Fold {fold} test_acc={test_acc:.4f}")

        # 保存这一折的混淆矩阵 / 报告等（现在基于 test 集）
        save_fold_summary(out_base, fold, y_true, y_pred)

        # 记录结果（同时存 val_acc 和 test_acc，方便之后分析）
        results.append({
            "fold": fold,
            "val_acc": float(best_val_acc),
            "test_acc": float(test_acc)
        })

        # free GPU & RAM for the next fold
        del dl_train, dl_val, dl_test, model, best_state
        cleanup_torch()

    df = pd.DataFrame(results)
    mean_acc = df["test_acc"].mean()
    std_acc = df["test_acc"].std(ddof=1) if len(df) > 1 else 0.0

    out_csv = os.path.join(out_base, "cv_results_resnet.csv")
    df.to_csv(out_csv, index=False)

    print("\nCV summary:\n", df)
    print(f"Saved to: {out_csv}")
    print(f"\nFinal Test Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    return float(mean_acc), float(std_acc)
