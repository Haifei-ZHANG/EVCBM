# -*- coding: utf-8 -*-
"""
Sweep EVCBM lambda_yager_blend and evaluate via outer K-fold CV.
(Adapted from sweep_topk_evcbm.py) :contentReference[oaicite:1]{index=1}

We ONLY train and evaluate EVCBM; trained models are NOT saved.
We ONLY save evaluation metric tables (CSV) and summaries (JSON).

Outputs saved under:
  outputs/lambda_yager_blend/<dataset>/evcbm_topk<k>_lambda<lam>/
and a global summary csv:
  outputs/lambda_yager_blend/summary_lambda_sweep_<timestamp>.csv
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

# make 'src' importable (same pattern as your scripts)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evcbm.utils.seed import set_seed
from evcbm.utils.mem import cleanup_torch
from evcbm.data.transforms import get_transforms
from evcbm.data.datasets import ConceptDataset, build_dataloaders

from evcbm.models.evcbm import EVCBM

# training / evaluation utilities
from evcbm.engine.train_evcbm import train_one_split_evcbm
from evcbm.eval.metrics import collect_preds
from evcbm.utils.concepts_eval import eval_concept_metrics


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

    # training schedule (dataset-specific defaults applied later)
    "concept_epochs": 50,
    "label_epochs": 50,
    "patience": 30,

    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "augment": True,
    "fp16": True,
    "min_delta": 5e-4,
    "optimiser_name": "AdamW",
    "max_grad_norm": 1.0,

    "concept_threshold": 0.5,
    "inner_val_ratio": 0.2,

    # ---- EVCBM ----
    "evcbm_topk": 32,                  # will be fixed per dataset below
    "evcbm_lambda_yager_blend": 0.5,   # will be swept
    "evcbm_discount_hidden": 128,
    "evcbm_discount_dropout": 0.0,
    "evcbm_use_context": True,
    "evcbm_context_hidden": 128,
    "evcbm_context_dropout": 0.0,
    "backbone_lr_ratio": 0.1,

    # trainer extras used in your evcbm training branch
    "joint_warmup_epochs": 3,

    # output root
    "out_dir": os.path.join("outputs", "lambda_yager_blend"),
}

DATASETS: List[str] = ["Derm7pt", "CUB", "AwA2"]

# fixed topk per dataset (as requested)
TOPK_BY_DATASET: Dict[str, int] = {
    "Derm7pt": 16,
    "CUB": 32,
    "AwA2": 32,
}

LAMBDA_LIST: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


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
      maxbetp_mean: mean over samples of max pignistic probability
    """
    model.eval()
    momega_list = []
    maxbetp_list = []

    for batch in dl_test:
        x = batch["image"].to(device, non_blocking=True)

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
# K-fold CV runner (EVCBM only) for a fixed (topk, lambda)
# =========================
def run_kfold_evcbm_lambda(cfg: Dict[str, Any],
                           model_factory: Callable[[int, int], torch.nn.Module]) -> Dict[str, Any]:
    """
    Outer K-fold: train/test
    Inner split inside train for early stopping

    Does NOT save trained model checkpoints.
    Only saves metric tables (CSV) + summary (JSON).

    Returns: {"df": df, "summary": summary}
    """
    set_seed(int(cfg["seed"]))

    ds = cfg["dataset"]
    topk = int(cfg["evcbm_topk"])
    lam = float(cfg["evcbm_lambda_yager_blend"])

    out_base = os.path.join(cfg["out_dir"], ds, f"evcbm_topk{topk}_lambda{lam:.2f}")
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
        print(f"\n===== Dataset={ds} | topk={topk} | lambda={lam:.2f} | Fold {fold}/{cfg['num_folds']} =====")

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

        # uncertainty stats
        momega_mean, maxbetp_mean = compute_momega_and_maxbetp(model, dl_test, device)

        print(
            f"Fold {fold} (TEST) label_acc={test_label_acc:.4f} | "
            f"concept_acc={test_concept_acc:.4f} | "
            f"momega_mean={momega_mean:.4f} | max_betp_mean={maxbetp_mean:.4f}"
        )

        fold_rows.append({
            "dataset": ds,
            "topk": topk,
            "lambda_yager_blend": lam,
            "fold": fold,
            "val_label_acc_best": float(best_val_label_acc),
            "val_concept_acc_best": float(best_val_concept_acc),
            "test_label_acc": float(test_label_acc),
            "test_concept_acc": float(test_concept_acc),
            "test_concept_bce": float(test_concept_bce),
            "test_momega_mean": float(momega_mean),
            "test_max_betp_mean": float(maxbetp_mean),
        })

        # free memory
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

    out_csv = os.path.join(out_base, f"cv_results_evcbm_lambda{lam:.2f}.csv")
    df.to_csv(out_csv, index=False, sep=";")

    summary = {
        "dataset": ds,
        "topk": topk,
        "lambda_yager_blend": lam,
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

    out_json = os.path.join(out_base, f"cv_summary_evcbm_lambda{lam:.2f}.json")
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
# Dataset-specific defaults (keep consistent with your training recipes)
# =========================
def apply_dataset_defaults(cfg: Dict[str, Any], ds: str) -> None:
    """
    Mirrors the schedule used in your sweep_topk_evcbm.py :contentReference[oaicite:2]{index=2}
    Adjust here if you changed training elsewhere.
    """
    if ds == "Derm7pt":
        cfg["concept_epochs"] = 50
        cfg["label_epochs"] = 50
        cfg["patience"] = 50
    elif ds == "CUB":
        cfg["concept_epochs"] = 40
        cfg["label_epochs"] = 40
        cfg["patience"] = 40
    else:
        cfg["concept_epochs"] = 20
        cfg["label_epochs"] = 20
        cfg["patience"] = 20

    cfg["optimiser_name"] = "AdamW"
    cfg["learning_rate"] = 1e-3


def main():
    os.makedirs(BASE_CONFIG["out_dir"], exist_ok=True)
    set_seed(int(BASE_CONFIG["seed"]))

    all_rows: List[Dict[str, Any]] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_csv = os.path.join(BASE_CONFIG["out_dir"], f"summary_lambda_sweep_{ts}.csv")

    for ds in DATASETS:
        fixed_topk = int(TOPK_BY_DATASET[ds])

        for lam in LAMBDA_LIST:
            cfg = dict(BASE_CONFIG)
            cfg["dataset"] = ds
            cfg["evcbm_topk"] = fixed_topk
            cfg["evcbm_lambda_yager_blend"] = float(lam)

            apply_dataset_defaults(cfg, ds)

            print("\n" + "=" * 70)
            print(f"LAMBDA SWEEP | dataset={ds} | topk={fixed_topk} | lambda_yager_blend={lam:.2f}")
            print("=" * 70)

            set_seed(int(cfg["seed"]))

            try:
                factory = make_evcbm_factory(cfg)
                out = run_kfold_evcbm_lambda(cfg, model_factory=factory)
                summary = out["summary"]
                all_rows.append(summary)

            except Exception as e:
                print("!! Error for (dataset, lambda) =", ds, lam)
                traceback.print_exc()
                all_rows.append({
                    "dataset": ds,
                    "topk": fixed_topk,
                    "lambda_yager_blend": float(lam),
                    "error": str(e),
                })

            # incremental save (safer for long sweeps)
            df_all = pd.DataFrame(all_rows)
            df_all.to_csv(sweep_csv, index=False, sep=";")
            print("\n[Progress Saved]:", sweep_csv)

    print("\nAll sweeps done. Saved:", sweep_csv)
    print(pd.DataFrame(all_rows))


if __name__ == "__main__":
    main()
