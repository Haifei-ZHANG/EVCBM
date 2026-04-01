# -*- coding: utf-8 -*-
"""
Sweep EVCBM top-k (number of fused concepts) and evaluate via outer K-fold CV.

Outputs saved under:
  outputs/topk/<dataset>/evcbm_topk<k>/

Based on:
- eval_accuracy.py config/factory style  :contentReference[oaicite:2]{index=2}
- cv_evcbm.py K-fold training/eval flow :contentReference[oaicite:3]{index=3}
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Callable, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# make 'src' importable (same pattern)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evcbm.utils.seed import set_seed
from evcbm.utils.mem import cleanup_torch
from evcbm.data.transforms import get_transforms
from evcbm.data.datasets import ConceptDataset, build_dataloaders

from evcbm.models.evcbm import EVCBM

# training / evaluation utilities (mirrors cv_evcbm.py but imported from package)
from evcbm.engine.train_evcbm import train_one_split_evcbm
from evcbm.eval.metrics import collect_preds, save_fold_summary
from evcbm.utils.concepts_eval import eval_concept_metrics
from evcbm.utils.concepts_io import save_concept_backbone


# =========================
# CONFIG (match your training choices)
# =========================
BASE_CONFIG: Dict[str, Any] = {
    "root_dir": "data_ready",
    "dataset": "Derm7pt",
    "num_folds": 5,
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,
    "concept_epochs": 50,
    "label_epochs": 50,
    "learning_rate": 1e-3,          # EVCBM default in your eval_accuracy.py path
    "weight_decay": 1e-4,
    "patience": 30,
    "augment": True,
    "fp16": True,
    "out_dir": os.path.join("outputs", "topk"),
    "min_delta": 5e-4,
    "optimiser_name": "AdamW",
    "concept_threshold": 0.5,
    "inner_val_ratio": 0.2,

    # ---- EVCBM ----
    "evcbm_topk": 32,
    "evcbm_lambda_yager_blend": 0.5,
    "evcbm_discount_hidden": 128,
    "evcbm_discount_dropout": 0.0,
    "evcbm_use_context": True,
    "evcbm_context_hidden": 128,
    "evcbm_context_dropout": 0.0,
    "backbone_lr_ratio": 0.1,

    # trainer extras used in your eval_accuracy.py branch for evcbm
    "joint_warmup_epochs": 3,
    "max_grad_norm": 1.0,
}

DATASETS: List[str] = [ "Derm7pt", "CUB", "AwA2"]

DATASETS: List[str] = ["AwA2"]

TOPK_LIST: List[int] = [4, 8, 16, 32, 64]

# TOPK_LIST: List[int] = [96]


# =========================
# Factory
# =========================
def make_evcbm_factory(cfg: Dict[str, Any]) -> Callable[[int, int], torch.nn.Module]:
    def _factory(num_concepts: int, num_classes: int) -> torch.nn.Module:
        return EVCBM(
            num_concepts=num_concepts,
            num_classes=num_classes,
            pretrained=True,
            topk=int(cfg.get("evcbm_topk", 32)),
            lambda_yager_blend=float(cfg.get("evcbm_lambda_yager_blend", 0.0)),
            eps=1e-6,
            discount_hidden=int(cfg.get("evcbm_discount_hidden", 128)),
            discount_dropout=float(cfg.get("evcbm_discount_dropout", 0.0)),
            use_context=bool(cfg.get("evcbm_use_context", True)),
            context_hidden=int(cfg.get("evcbm_context_hidden", 128)),
            context_dropout=float(cfg.get("evcbm_context_dropout", 0.0)),
        )
    return _factory


# =========================
# EVCBM-specific stats on TEST: momega and max(betp)
# =========================
@torch.no_grad()
def compute_momega_and_maxbetp(model: torch.nn.Module,
                              dl_test: torch.utils.data.DataLoader,
                              device: torch.device) -> Tuple[float, float]:
    """
    Returns:
      momega_mean: mean of omega/Theta mass over test samples
      max_betp_mean: mean over samples of max pignistic probability
    """
    model.eval()
    momega_list = []
    maxbetp_list = []

    for batch in dl_test:
        x = batch["image"].to(device, non_blocking=True)

        # get concept probabilities
        out = model(x, return_concepts=True)
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError("EVCBM forward(return_concepts=True) must return (y_logits, c_prob, extra).")
        _, c_prob, _ = out

        belief, momega = model.forward_fusion_only(c_prob)
        betp = model._betp_from_bel(belief, momega)

        momega_np = momega.detach().cpu().numpy().reshape(-1)
        betp_np = betp.detach().cpu().numpy()
        maxbetp_np = betp_np.max(axis=1)

        momega_list.append(momega_np)
        maxbetp_list.append(maxbetp_np)

    momega_all = np.concatenate(momega_list, axis=0)
    maxbetp_all = np.concatenate(maxbetp_list, axis=0)
    return float(momega_all.mean()), float(maxbetp_all.mean())


# =========================
# K-fold CV runner (EVCBM only) with topk-aware output path
# =========================
def run_kfold_evcbm_topk(cfg: Dict[str, Any],
                         model_factory: Callable[[int, int], torch.nn.Module]) -> Dict[str, Any]:
    """
    Outer K-fold: train/test
    Inner split inside train for early stopping
    Saves fold artifacts into outputs/topk/<dataset>/evcbm_topk<k>/
    Returns a dict summary with per-fold records and cross-fold mean/std.
    """
    set_seed(int(cfg["seed"]))

    ds = cfg["dataset"]
    topk = int(cfg["evcbm_topk"])
    out_base = os.path.join(cfg["out_dir"], ds, f"evcbm_topk{topk}")
    os.makedirs(out_base, exist_ok=True)

    # Build full dataset once (no augmentation transform)
    _, val_tfms = get_transforms(augment=False)
    full_ds = ConceptDataset(root_dir=cfg["root_dir"], dataset=ds, transform=val_tfms)

    labels = full_ds.Y_onehot.argmax(axis=1)

    skf = StratifiedKFold(
        n_splits=int(cfg["num_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner_val_ratio = float(cfg.get("inner_val_ratio", 0.2))
    concept_threshold = float(cfg.get("concept_threshold", 0.5))

    fold_rows: List[Dict[str, Any]] = []

    for fold, (idx_tr, idx_te) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n===== Dataset={ds} | topk={topk} | Fold {fold}/{cfg['num_folds']} =====")

        train_tfms, valid_tfms = get_transforms(augment=bool(cfg.get("augment", True)))

        dl_train, dl_val, dl_test = build_dataloaders(
            ds=full_ds,
            idx_train=idx_tr,      # outer train (will be split internally to inner train/val)
            idx_test=idx_te,       # outer test
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["num_workers"]),
            train_tfms=train_tfms,
            val_tfms=valid_tfms,
            inner_val_ratio=inner_val_ratio,
            seed=int(cfg["seed"]),
        )

        model = model_factory(full_ds.num_concepts, full_ds.num_classes)

        # train with early stopping on inner val
        best_val_label_acc, best_val_concept_acc, best_state = train_one_split_evcbm(
            model, dl_train, dl_val, cfg, device=device
        )
        print(
            f"Fold {fold} (VAL best) label_acc={best_val_label_acc:.4f} | "
            f"concept_acc={best_val_concept_acc:.4f}"
        )

        # load best state and evaluate on outer test
        model.load_state_dict(best_state)

        y_true, y_pred = collect_preds(model, dl_test, device)

        y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
        y_pred_np = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)
        test_label_acc = float((y_true_np == y_pred_np).mean())

        test_concept_bce, test_concept_acc = eval_concept_metrics(
            model, dl_test, device, threshold=concept_threshold
        )

        # extra uncertainty stats
        momega_mean, maxbetp_mean = compute_momega_and_maxbetp(model, dl_test, device)

        print(
            f"Fold {fold} (TEST) label_acc={test_label_acc:.4f} | "
            f"concept_acc={test_concept_acc:.4f} | "
            f"momega_mean={momega_mean:.4f} | max_betp_mean={maxbetp_mean:.4f}"
        )

        # save fold summary artifacts
        save_fold_summary(out_base, fold, y_true, y_pred)

        # save model checkpoints (so you can inspect later)
        concept_path = os.path.join(out_base, f"concept-model-{fold}.pt")
        save_concept_backbone(model, concept_path)
        full_path = os.path.join(out_base, f"model-full-{fold}.pt")
        torch.save({"model_state": model.state_dict()}, full_path)

        fold_rows.append({
            "dataset": ds,
            "topk": topk,
            "fold": fold,
            "val_label_acc_best": float(best_val_label_acc),
            "val_concept_acc_best": float(best_val_concept_acc),
            "test_label_acc": float(test_label_acc),
            "test_concept_acc": float(test_concept_acc),
            "test_concept_bce": float(test_concept_bce),
            "test_momega_mean": float(momega_mean),
            "test_max_betp_mean": float(maxbetp_mean),
        })

        del dl_train, dl_val, dl_test, model, best_state
        cleanup_torch()

    df = pd.DataFrame(fold_rows)

    # cross-fold stats (TEST)
    def _mean_std(x: pd.Series) -> Tuple[float, float]:
        m = float(x.mean())
        s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        return m, s

    lbl_m, lbl_s = _mean_std(df["test_label_acc"])
    cpt_m, cpt_s = _mean_std(df["test_concept_acc"])
    omg_m, omg_s = _mean_std(df["test_momega_mean"])
    mbp_m, mbp_s = _mean_std(df["test_max_betp_mean"])

    # save per-topk CSV + JSON summary
    out_csv = os.path.join(out_base, f"cv_results_evcbm_topk{topk}.csv")
    df.to_csv(out_csv, index=False, sep=";")

    summary = {
        "dataset": ds,
        "topk": topk,
        "num_folds": int(cfg["num_folds"]),
        "seed": int(cfg["seed"]),
        "test_label_acc_mean": lbl_m,
        "test_label_acc_std": lbl_s,
        "test_concept_acc_mean": cpt_m,
        "test_concept_acc_std": cpt_s,
        "test_momega_mean_mean": omg_m,
        "test_momega_mean_std": omg_s,
        "test_max_betp_mean_mean": mbp_m,
        "test_max_betp_mean_std": mbp_s,
        "results_csv": out_csv,
    }

    out_json = os.path.join(out_base, f"cv_summary_evcbm_topk{topk}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    print(
        f"\n[Summary TEST] label_acc={lbl_m:.4f}±{lbl_s:.4f} | "
        f"concept_acc={cpt_m:.4f}±{cpt_s:.4f} | "
        f"momega_mean={omg_m:.4f}±{omg_s:.4f} | "
        f"max_betp_mean={mbp_m:.4f}±{mbp_s:.4f}"
    )

    return {"df": df, "summary": summary}


# =========================
# Main: sweep datasets and topk values
# =========================
def apply_dataset_defaults(cfg: Dict[str, Any], ds: str) -> None:
    """
    Mirrors your dataset-specific epoch settings in eval_accuracy.py :contentReference[oaicite:4]{index=4}
    Adjust if you have updated these elsewhere.
    """
    if ds == "Derm7pt":
        cfg["concept_epochs"] = 50
        cfg["label_epochs"] = 50
        cfg["patience"] = 25
    elif ds == "CUB":
        cfg["concept_epochs"] = 40
        cfg["label_epochs"] = 40
        cfg["patience"] = 20
    else:
        cfg["concept_epochs"] = 20
        cfg["label_epochs"] = 20
        cfg["patience"] = 10

    # your eval_accuracy.py sets evcbm learning_rate=1e-3 with AdamW
    cfg["optimiser_name"] = "AdamW"
    cfg["learning_rate"] = 1e-3


def main():
    os.makedirs(BASE_CONFIG["out_dir"], exist_ok=True)
    set_seed(int(BASE_CONFIG["seed"]))

    all_rows = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_csv = os.path.join(BASE_CONFIG["out_dir"], f"summary_topk_sweep_{ts}.csv")

    for ds in DATASETS:
        for k in TOPK_LIST:
            if ds=="Derm7pt" and k > 16:
                continue
            
            cfg = dict(BASE_CONFIG)
            cfg["dataset"] = ds
            cfg["evcbm_topk"] = int(k)

            apply_dataset_defaults(cfg, ds)

            print("\n" + "=" * 60)
            print(f"TOPK SWEEP | dataset={ds} | topk={k}")
            print("=" * 60)

            # fresh seed per (dataset, topk)
            set_seed(int(cfg["seed"]))

            try:
                factory = make_evcbm_factory(cfg)
                out = run_kfold_evcbm_topk(cfg, model_factory=factory)
                summary = out["summary"]
                all_rows.append(summary)

            except Exception as e:
                print("!! Error for (dataset, topk) =", ds, k)
                traceback.print_exc()
                all_rows.append({
                    "dataset": ds,
                    "topk": k,
                    "error": str(e),
                })

            df_all = pd.DataFrame(all_rows)
            df_all.to_csv(sweep_csv, index=False, sep=";")
            print("\nAll sweeps done. Saved:", sweep_csv)
            print(df_all)


if __name__ == "__main__":
    main()

