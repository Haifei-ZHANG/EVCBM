# -*- coding: utf-8 -*-
"""
K-fold CV for CBM (joint).
Reports mean±std for both label acc and concept acc.
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
from .train_cbm_joint import train_one_split_cbm_joint
from evcbm.eval.metrics import collect_preds, save_fold_summary
from evcbm.utils.concepts_io import save_concept_backbone
from evcbm.utils.concepts_eval import eval_concept_metrics


def run_kfold_cbm_joint(cfg: Dict[str, Any],
                        model_factory: Callable[[int, int], torch.nn.Module],
                        get_transforms_fn: Callable[[bool], Tuple] = get_transforms):

    set_seed(int(cfg["seed"]))
    out_base = os.path.join(cfg["out_dir"], cfg["dataset"], "cbm_joint")
    os.makedirs(out_base, exist_ok=True)

    # Build full dataset once（固定的非增强 transform）
    _, val_tfms = get_transforms_fn(augment=False)
    full_ds = ConceptDataset(root_dir=cfg["root_dir"],
                             dataset=cfg["dataset"],
                             transform=val_tfms)

    # labels 用于外层 StratifiedKFold 和 inner split
    labels = full_ds.Y_onehot.argmax(axis=1)

    skf = StratifiedKFold(
        n_splits=int(cfg["num_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inner_val_ratio = float(cfg.get("inner_val_ratio", 0.2))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    results = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n===== CBM-Joint Fold {fold}/{cfg['num_folds']} =====")

        train_tfms, valid_tfms = get_transforms_fn(augment=bool(cfg["augment"]))

        # idx_tr: 外层 train（再切 inner train/val）
        # idx_va: 外层 test
        dl_train, dl_val, dl_test = build_dataloaders(
            ds=full_ds,
            idx_train=idx_tr,
            idx_test=idx_va,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            train_tfms=train_tfms,
            val_tfms=valid_tfms,
            inner_val_ratio=inner_val_ratio,
            seed=int(cfg["seed"])
        )

        model = model_factory(full_ds.num_concepts, full_ds.num_classes)

        # 在 inner train/val 上训练 + early stopping
        best_label_acc, best_concept_acc, best_state = train_one_split_cbm_joint(
            model, dl_train, dl_val, cfg, device=device
        )
        print(
            f"Fold {fold} (VAL) best label_acc={best_label_acc:.4f} | "
            f"concept_acc={best_concept_acc:.4f}"
        )

        # 用 best checkpoint 在外层 test 集上评估
        model.load_state_dict(best_state)
        model_path = os.path.join(out_base, f"concept-model-{fold}.pt")
        save_concept_backbone(model, model_path)
        full_path = os.path.join(out_base, f"model-full-{fold}.pt")
        torch.save({"model_state": model.state_dict()}, full_path)

        # 1) 先算标签的 test acc
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

        # 2) 再在 test 上算概念的 acc
        _, test_concept_acc = eval_concept_metrics(
            model, dl_test, device, threshold=concept_threshold
        )

        print(
            f"Fold {fold} (TEST) label_acc={test_label_acc:.4f} | "
            f"concept_acc={test_concept_acc:.4f}"
        )

        # 报告/混淆矩阵基于 test 集
        save_fold_summary(out_base, fold, y_true, y_pred)

        # 同时保留 val 和 test 的指标，方便后续分析
        results.append({
            "fold": fold,
            "val_label_acc": float(best_label_acc),
            "val_concept_acc": float(best_concept_acc),
            "test_label_acc": float(test_label_acc),
            "test_concept_acc": float(test_concept_acc),
        })

        # free GPU & RAM for the next fold
        del dl_train, dl_val, dl_test, model, best_state
        cleanup_torch()

    df = pd.DataFrame(results)

    # 最终报告：Label 和 Concept 都用 TEST 的指标
    lbl_mean = df["test_label_acc"].mean()
    lbl_std  = df["test_label_acc"].std(ddof=1) if len(df) > 1 else 0.0
    cpt_mean = df["test_concept_acc"].mean()
    cpt_std  = df["test_concept_acc"].std(ddof=1) if len(df) > 1 else 0.0

    out_csv = os.path.join(out_base, "cv_results_cbm_joint.csv")
    df.to_csv(out_csv, index=False)

    print("\nCV summary:\n", df)
    print(f"Saved to: {out_csv}")
    print(f"\nLabel Acc (TEST): {lbl_mean:.4f} ± {lbl_std:.4f}")
    print(f"Concept Acc (TEST): {cpt_mean:.4f} ± {cpt_std:.4f}")

    return float(lbl_mean), float(lbl_std), float(cpt_mean), float(cpt_std)


