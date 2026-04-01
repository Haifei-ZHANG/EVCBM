# -*- coding: utf-8 -*-
"""
Ablation: ContextHead ON/OFF for EVCBM.
Compare label prediction and concept prediction differences.

This script is consistent with:
- EVCBM model definition (use_context toggles ContextHead) :contentReference[oaicite:2]{index=2}
- CV logic / metrics utilities in your project :contentReference[oaicite:3]{index=3}
- Training routine train_one_split_evcbm :contentReference[oaicite:4]{index=4}

Key requirements:
- For each dataset, use the provided best (topk, lambda_yager_blend).
- Run 2 settings: use_context=True vs False.
- Do NOT save trained models.
- Save only results (csv/json) with resume to (dataset, setting, fold).

Output per dataset:
  outputs/context_ablation/<dataset>/
    cv_results_evcbm_ctx<0|1>_topk<k>_lambda<lam>.csv        (incremental per fold)
    cv_summary_evcbm_ctx<0|1>_topk<k>_lambda<lam>.json       (after all folds)
    summary.csv                                              (rolling; updated after each fold)
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, Callable, Tuple, List, Set

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# Make 'src' importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evcbm.utils.seed import set_seed
from evcbm.utils.mem import cleanup_torch
from evcbm.data.transforms import get_transforms
from evcbm.data.datasets import ConceptDataset, build_dataloaders
from evcbm.eval.metrics import collect_preds
from evcbm.utils.concepts_eval import eval_concept_metrics
from evcbm.models.evcbm import EVCBM
from evcbm.engine.train_evcbm import train_one_split_evcbm


# =========================
# USER: fill in best params
# =========================
# You said you will specify best topk & lambda per dataset.
# Fill here before running.
BEST_PARAMS: Dict[str, Dict[str, float]] = {
    # example placeholders: replace with your best values
    "Derm7pt": {"topk": 8, "lambda_yager_blend": 0.2},
    "CUB":     {"topk": 32, "lambda_yager_blend": 0.1},
    "AwA2":    {"topk": 32, "lambda_yager_blend": 0.1},
}


# =========================
# BASE CONFIG
# =========================
BASE_CONFIG: Dict[str, Any] = {
    "root_dir": "data_ready",
    "num_folds": 5,
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,

    # training schedule (dataset-specific defaults applied later)
    "concept_epochs": 50,
    "label_epochs": 50,
    "patience": 25,

    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "augment": True,
    "fp16": True,
    "min_delta": 5e-4,
    "optimiser_name": "AdamW",
    "max_grad_norm": 1.0,
    "backbone_lr_ratio": 0.1,
    "joint_warmup_epochs": 3,

    "concept_threshold": 0.5,
    "inner_val_ratio": 0.2,

    # EVCBM structure defaults (kept same; only use_context is ablated)
    "evcbm_discount_hidden": 128,
    "evcbm_discount_dropout": 0.0,
    "evcbm_use_context": True,         # toggled in sweep
    "evcbm_context_hidden": 128,
    "evcbm_context_dropout": 0.0,

    # output root
    "out_dir": os.path.join("outputs", "context_ablation"),
}

DATASETS: List[str] = ["Derm7pt", "CUB", "AwA2"]


def apply_dataset_defaults(cfg: Dict[str, Any], ds: str) -> None:
    # Keep aligned with your existing recipes.
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


# =========================
# Filenames / atomic writers / resume helpers
# =========================
def _lam_str(lam: float) -> str:
    # stable filename
    return f"{lam:.3f}".rstrip("0").rstrip(".")  # e.g. 0.2, 0.5, 0.125


def results_csv_path(out_dir_ds: str, use_ctx: bool, topk: int, lam: float) -> str:
    ctx = 1 if use_ctx else 0
    return os.path.join(out_dir_ds, f"cv_results_evcbm_ctx{ctx}_topk{topk}_lambda{_lam_str(lam)}.csv")


def summary_json_path(out_dir_ds: str, use_ctx: bool, topk: int, lam: float) -> str:
    ctx = 1 if use_ctx else 0
    return os.path.join(out_dir_ds, f"cv_summary_evcbm_ctx{ctx}_topk{topk}_lambda{_lam_str(lam)}.json")


def dataset_summary_csv_path(out_dir_ds: str) -> str:
    return os.path.join(out_dir_ds, "summary.csv")


def atomic_json_dump(obj: Dict[str, Any], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def atomic_csv_dump(df: pd.DataFrame, path: str, sep: str = ";") -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, sep=sep)
    os.replace(tmp, path)


def is_setting_done(out_dir_ds: str, use_ctx: bool, topk: int, lam: float) -> bool:
    p = summary_json_path(out_dir_ds, use_ctx, topk, lam)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            _ = json.load(f)
        return True
    except Exception:
        return False


def load_completed_folds(results_csv: str) -> Tuple[pd.DataFrame, Set[int]]:
    if not os.path.exists(results_csv):
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(results_csv, sep=";")
        if "fold" not in df.columns:
            return pd.DataFrame(), set()
        done = {int(x) for x in df["fold"].dropna().tolist()}
        return df, done
    except Exception:
        # corrupted -> restart this setting
        return pd.DataFrame(), set()


def upsert_dataset_summary(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    # key by (use_context, topk, lambda)
    key = (int(row.get("use_context", -1)), int(row.get("topk", -1)), float(row.get("lambda_yager_blend", -1.0)))
    out = []
    replaced = False
    for r in rows:
        k2 = (int(r.get("use_context", -1)), int(r.get("topk", -1)), float(r.get("lambda_yager_blend", -1.0)))
        if k2 == key:
            out.append(row)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(row)
    return out


def save_dataset_summary(rows: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(rows)
    if len(df) > 0:
        # sort: use_context first? user didn't specify; keep deterministic:
        # topk, lambda, use_context
        sort_cols = [c for c in ["topk", "lambda_yager_blend", "use_context"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")
    atomic_csv_dump(df, path, sep=";")


# =========================
# Model factory
# =========================
def make_evcbm_factory(cfg: Dict[str, Any]) -> Callable[[int, int], torch.nn.Module]:
    def _factory(num_concepts: int, num_classes: int) -> torch.nn.Module:
        return EVCBM(
            num_concepts=num_concepts,
            num_classes=num_classes,
            pretrained=True,
            topk=int(cfg["evcbm_topk"]),
            lambda_yager_blend=float(cfg["evcbm_lambda_yager_blend"]),
            eps=1e-6,
            discount_hidden=int(cfg.get("evcbm_discount_hidden", 128)),
            discount_dropout=float(cfg.get("evcbm_discount_dropout", 0.0)),
            use_context=bool(cfg.get("evcbm_use_context", True)),
            context_hidden=int(cfg.get("evcbm_context_hidden", 128)),
            context_dropout=float(cfg.get("evcbm_context_dropout", 0.0)),
        )
    return _factory


# =========================
# One fold runner
# =========================
def run_one_fold(
    cfg: Dict[str, Any],
    full_ds: ConceptDataset,
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    fold: int,
    model_factory: Callable[[int, int], torch.nn.Module],
    device: torch.device
) -> Dict[str, Any]:
    train_tfms, valid_tfms = get_transforms(augment=bool(cfg.get("augment", True)))

    dl_train, dl_val, dl_test = build_dataloaders(
        ds=full_ds,
        idx_train=idx_tr,
        idx_test=idx_te,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        train_tfms=train_tfms,
        val_tfms=valid_tfms,
        inner_val_ratio=float(cfg.get("inner_val_ratio", 0.2)),
        seed=int(cfg["seed"])
    )

    model = model_factory(full_ds.num_concepts, full_ds.num_classes)

    best_val_label_acc, best_val_concept_acc, best_state = train_one_split_evcbm(
        model, dl_train, dl_val, cfg, device=device
    )

    model.load_state_dict(best_state)

    # label acc on outer test
    y_true, y_pred = collect_preds(model, dl_test, device)
    y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_pred_np = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)
    test_label_acc = float((y_true_np == y_pred_np).mean())

    # concept metrics on outer test (aligned with your eval_concept_metrics usage) :contentReference[oaicite:5]{index=5}
    thr = float(cfg.get("concept_threshold", 0.5))
    test_concept_bce, test_concept_acc = eval_concept_metrics(model, dl_test, device, threshold=thr)

    row = {
        "dataset": cfg["dataset"],
        "fold": int(fold),
        "use_context": int(bool(cfg["evcbm_use_context"])),
        "topk": int(cfg["evcbm_topk"]),
        "lambda_yager_blend": float(cfg["evcbm_lambda_yager_blend"]),

        "val_label_acc_best": float(best_val_label_acc),
        "val_concept_acc_best": float(best_val_concept_acc),

        "test_label_acc": float(test_label_acc),
        "test_concept_acc": float(test_concept_acc),
        "test_concept_bce": float(test_concept_bce),
    }

    # cleanup
    del dl_train, dl_val, dl_test, model, best_state
    cleanup_torch()
    return row


def finalize_setting_summary(df_folds: pd.DataFrame, cfg: Dict[str, Any], out_dir_ds: str) -> Dict[str, Any]:
    def _mean_std(x: pd.Series) -> Tuple[float, float]:
        m = float(x.mean())
        s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        return m, s

    lbl_m, lbl_s = _mean_std(df_folds["test_label_acc"])
    cpt_m, cpt_s = _mean_std(df_folds["test_concept_acc"])
    bce_m, bce_s = _mean_std(df_folds["test_concept_bce"])

    summary = {
        "dataset": cfg["dataset"],
        "use_context": int(bool(cfg["evcbm_use_context"])),
        "topk": int(cfg["evcbm_topk"]),
        "lambda_yager_blend": float(cfg["evcbm_lambda_yager_blend"]),
        "num_folds": int(cfg["num_folds"]),
        "seed": int(cfg["seed"]),

        "test_label_acc_mean": lbl_m,
        "test_label_acc_std": lbl_s,

        "test_concept_acc_mean": cpt_m,
        "test_concept_acc_std": cpt_s,

        "test_concept_bce_mean": bce_m,
        "test_concept_bce_std": bce_s,

        "results_csv": results_csv_path(out_dir_ds, bool(cfg["evcbm_use_context"]), int(cfg["evcbm_topk"]),
                                        float(cfg["evcbm_lambda_yager_blend"])),
    }

    out_json = summary_json_path(out_dir_ds, bool(cfg["evcbm_use_context"]), int(cfg["evcbm_topk"]),
                                 float(cfg["evcbm_lambda_yager_blend"]))
    atomic_json_dump(summary, out_json)
    print(f"[Saved] {out_json}")
    return summary


# =========================
# Main
# =========================
def main():
    os.makedirs(BASE_CONFIG["out_dir"], exist_ok=True)
    set_seed(int(BASE_CONFIG["seed"]))

    for ds in DATASETS:
        if ds not in BEST_PARAMS:
            raise ValueError(f"Missing BEST_PARAMS for dataset='{ds}'. Please fill it first.")

        topk = int(BEST_PARAMS[ds]["topk"])
        lam = float(BEST_PARAMS[ds]["lambda_yager_blend"])

        out_dir_ds = os.path.join(BASE_CONFIG["out_dir"], ds)
        os.makedirs(out_dir_ds, exist_ok=True)

        # rolling per-dataset summary.csv
        ds_sum_path = dataset_summary_csv_path(out_dir_ds)
        if os.path.exists(ds_sum_path):
            try:
                ds_rows = pd.read_csv(ds_sum_path, sep=";").to_dict("records")
            except Exception:
                ds_rows = []
        else:
            ds_rows = []

        print("\n" + "=" * 90)
        print(f"DATASET: {ds} | best(topk={topk}, lambda={_lam_str(lam)}) | Ablation: use_context ON/OFF")
        print(f"Output dir: {out_dir_ds}")
        print("=" * 90)

        # Prepare dataset & folds once (stable fold IDs for resume)
        _, val_tfms = get_transforms(augment=False)
        full_ds = ConceptDataset(root_dir=BASE_CONFIG["root_dir"], dataset=ds, transform=val_tfms)
        labels = full_ds.Y_onehot.argmax(axis=1)

        skf = StratifiedKFold(
            n_splits=int(BASE_CONFIG["num_folds"]),
            shuffle=True,
            random_state=int(BASE_CONFIG["seed"]),
        )
        fold_splits = list(skf.split(np.zeros(len(labels)), labels))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for use_ctx in [True, False]:
            cfg = dict(BASE_CONFIG)
            cfg["dataset"] = ds
            cfg["evcbm_topk"] = topk
            cfg["evcbm_lambda_yager_blend"] = lam
            cfg["evcbm_use_context"] = bool(use_ctx)
            apply_dataset_defaults(cfg, ds)

            # If summary exists -> fully done
            if is_setting_done(out_dir_ds, use_ctx, topk, lam):
                print(f"[SKIP SETTING] DONE: dataset={ds} use_context={int(use_ctx)}")
                try:
                    with open(summary_json_path(out_dir_ds, use_ctx, topk, lam), "r", encoding="utf-8") as f:
                        done_sum = json.load(f)
                    done_row = dict(done_sum)
                    done_row["status"] = "done"
                    done_row["completed_folds"] = int(cfg["num_folds"])
                    ds_rows = upsert_dataset_summary(ds_rows, done_row)
                    save_dataset_summary(ds_rows, ds_sum_path)
                except Exception:
                    pass
                continue

            res_csv = results_csv_path(out_dir_ds, use_ctx, topk, lam)
            df_existing, done_folds = load_completed_folds(res_csv)

            print("\n" + "-" * 90)
            print(f"SETTING: dataset={ds} use_context={int(use_ctx)} topk={topk} lambda={_lam_str(lam)}")
            print(f"Completed folds: {sorted(list(done_folds))}")
            print("-" * 90)

            model_factory = make_evcbm_factory(cfg)

            try:
                fold_rows = df_existing.to_dict("records") if len(df_existing) > 0 else []

                for fold_id, (idx_tr, idx_te) in enumerate(fold_splits, start=1):
                    if fold_id in done_folds:
                        print(f"[SKIP FOLD] fold={fold_id}")
                        continue

                    print(f"\n===== RUN FOLD {fold_id}/{cfg['num_folds']} (use_context={int(use_ctx)}) =====")
                    row = run_one_fold(cfg, full_ds, idx_tr, idx_te, fold_id, model_factory, device)
                    fold_rows.append(row)

                    # incremental fold results
                    df_now = pd.DataFrame(fold_rows).sort_values(["fold"], kind="mergesort")
                    atomic_csv_dump(df_now, res_csv, sep=";")
                    print(f"[Saved fold results] {res_csv}")

                    # update dataset summary after each fold (running)
                    running_row = {
                        "dataset": ds,
                        "use_context": int(use_ctx),
                        "topk": topk,
                        "lambda_yager_blend": lam,
                        "status": "running",
                        "completed_folds": int(df_now["fold"].nunique()),
                        "num_folds": int(cfg["num_folds"]),
                        "results_csv": res_csv,
                    }
                    ds_rows = upsert_dataset_summary(ds_rows, running_row)
                    save_dataset_summary(ds_rows, ds_sum_path)
                    print(f"[Progress Saved] {ds_sum_path}")

                # finalize summary json when all folds complete
                df_final, done_final = load_completed_folds(res_csv)
                if len(done_final) == int(cfg["num_folds"]):
                    df_final = df_final.sort_values(["fold"], kind="mergesort")
                    setting_sum = finalize_setting_summary(df_final, cfg, out_dir_ds)

                    done_row = dict(setting_sum)
                    done_row["status"] = "done"
                    done_row["completed_folds"] = int(cfg["num_folds"])
                    ds_rows = upsert_dataset_summary(ds_rows, done_row)
                    save_dataset_summary(ds_rows, ds_sum_path)
                    print(f"[Progress Saved] {ds_sum_path}")
                else:
                    print(f"[WARN] Setting not complete: done_folds={len(done_final)}/{cfg['num_folds']}")

            except Exception as e:
                print(f"!! Error: dataset={ds} use_context={int(use_ctx)}")
                traceback.print_exc()
                err_row = {
                    "dataset": ds,
                    "use_context": int(use_ctx),
                    "topk": topk,
                    "lambda_yager_blend": lam,
                    "status": "error",
                    "completed_folds": len(done_folds),
                    "num_folds": int(cfg["num_folds"]),
                    "error": str(e),
                    "results_csv": res_csv,
                }
                ds_rows = upsert_dataset_summary(ds_rows, err_row)
                save_dataset_summary(ds_rows, ds_sum_path)
                print(f"[Progress Saved] {ds_sum_path}")

        print(f"\n[DONE DATASET] {ds} -> {ds_sum_path}")


if __name__ == "__main__":
    main()
