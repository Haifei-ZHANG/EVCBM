# -*- coding: utf-8 -*-
"""
K-fold CV for ProbCBM (sequential).
REQUIRES a per-fold concept backbone previously exported (e.g., by CBM-sequential).
"""
import os
from typing import Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from evcbm.data.transforms import get_transforms
from evcbm.data.datasets import ConceptDataset, build_dataloaders
from evcbm.utils.seed import set_seed
from evcbm.eval.metrics import collect_preds, save_fold_summary
from evcbm.utils.mem import cleanup_torch
from .train_probcbm import train_one_split_probcbm
from evcbm.utils.concepts_io import save_concept_backbone
from evcbm.utils.concepts_eval import eval_concept_metrics


def _concept_ckpt_path_for_fold(cfg, fold: int) -> str:
    """
    Build a stable per-fold path for the concept backbone checkpoint.
    Default: {out_dir}/{dataset}/concept_bank/fold_{fold}.pt
    """
    base = os.path.join(cfg["out_dir"], cfg["dataset"], "concept_bank")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"fold_{fold}.pt")


def run_kfold_probcbm(cfg: Dict[str, Any],
                              model_factory: Callable[[int, int], torch.nn.Module],
                              get_transforms_fn: Callable[[bool], Tuple] = get_transforms):
    set_seed(int(cfg["seed"]))
    out_base = os.path.join(cfg["out_dir"], cfg["dataset"], "probcbm")
    os.makedirs(out_base, exist_ok=True)

    # 用非增强 transform 构建完整数据集
    _, val_tfms = get_transforms_fn(augment=False)
    full_ds = ConceptDataset(
        root_dir=cfg["root_dir"],
        dataset=cfg["dataset"],
        transform=val_tfms
    )
    labels = full_ds.Y_onehot.argmax(axis=1)

    skf = StratifiedKFold(
        n_splits=int(cfg["num_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inner_val_ratio = float(cfg.get("inner_val_ratio", 0.2))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    rows = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n===== ProbCBM Fold {fold}/{cfg['num_folds']} =====")

        tr_tfms, va_tfms = get_transforms_fn(augment=bool(cfg["augment"]))

        # idx_tr: 外层 train（再切 inner train/val）
        # idx_va: 外层 test
        dl_train, dl_val, dl_test = build_dataloaders(
            ds=full_ds,
            idx_train=idx_tr,
            idx_test=idx_va,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            train_tfms=tr_tfms,
            val_tfms=va_tfms,
            inner_val_ratio=inner_val_ratio,
            seed=int(cfg["seed"])
        )

        # 读取该 fold 预先训练好的概念 backbone（来自 CBM-sequential）
        ckpt_path = _concept_ckpt_path_for_fold(cfg, fold)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[ProbCBM] Expected concept backbone not found for fold {fold}: {ckpt_path}. "
                f"Run CBM-sequential first to export it."
            )

        model = model_factory(full_ds.num_concepts, full_ds.num_classes)
        cfg_local = dict(cfg)
        cfg_local["concept_ckpt_path"] = ckpt_path
        cfg_local["save_concept_ckpt"] = False

        # 在 inner train/val 上训练 + early stopping（返回的是验证集上的最佳指标）
        best_label_acc, best_concept_acc, best_state = train_one_split_probcbm(
            model, dl_train, dl_val, cfg_local, device=device
        )
        print(
            f"Fold {fold} (VAL) best label_acc={best_label_acc:.4f} | "
            f"concept_acc={best_concept_acc:.4f}"
        )

        # 用 best checkpoint 在 outer test 上评估
        model.load_state_dict(best_state)
        model_path = os.path.join(out_base, f"concept-model-{fold}.pt")
        save_concept_backbone(model, model_path)
        full_path = os.path.join(out_base, f"model-full-{fold}.pt")
        torch.save({"model_state": model.state_dict()}, full_path)

        # 1) TEST 上的 label acc
        y_true, y_pred = collect_preds(model, dl_test, device)
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_true_np = np.asarray(y_true)

        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = np.asarray(y_pred)

        test_label_acc = float((y_true_np == y_pred_np).mean())

        # 2) TEST 上的 concept acc
        _, test_concept_acc = eval_concept_metrics(model, dl_test, device, threshold=concept_threshold)

        print(
            f"Fold {fold} (TEST) label_acc={test_label_acc:.4f} | "
            f"concept_acc={test_concept_acc:.4f}"
        )

        # confusion matrix / summary 基于 test
        save_fold_summary(out_base, fold, y_true, y_pred)

        rows.append({
            "fold": fold,
            "val_label_acc": float(best_label_acc),
            "val_concept_acc": float(best_concept_acc),
            "test_label_acc": float(test_label_acc),
            "test_concept_acc": float(test_concept_acc),
        })

        del dl_train, dl_val, dl_test, model, best_state
        cleanup_torch()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_base, "cv_results_pcbm_sequential.csv")
    df.to_csv(out_csv, index=False)
    print("\nCV summary:\n", df)
    print(f"Saved to: {out_csv}")

    # 最终报告：都用 TEST 指标
    lbl_mean = df["test_label_acc"].mean()
    lbl_std  = df["test_label_acc"].std(ddof=1) if len(df) > 1 else 0.0
    cpt_mean = df["test_concept_acc"].mean()
    cpt_std  = df["test_concept_acc"].std(ddof=1) if len(df) > 1 else 0.0

    print(f"\nLabel Acc (TEST): {lbl_mean:.4f} ± {lbl_std:.4f}")
    print(f"Concept Acc (TEST): {cpt_mean:.4f} ± {cpt_std:.4f}")

    return float(lbl_mean), float(lbl_std), float(cpt_mean), float(cpt_std)
