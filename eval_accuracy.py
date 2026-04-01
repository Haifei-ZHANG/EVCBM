# -*- coding: utf-8 -*-
"""
Run K-fold CV across datasets and models in Spyder.
- ResNet50: label acc only
- CBM (joint / sequential): label acc + concept acc
- CEM (joint): label acc + concept acc
Saves a summary CSV across all (dataset, model) runs.
"""

import os, sys, json, traceback
from datetime import datetime
import pandas as pd



# make 'src' importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ====== Imports from your project ======
from evcbm.utils.seed import set_seed
from evcbm.data.transforms import get_transforms

from evcbm.models.resnet import BaselineResNet50
from evcbm.models.cbm import CBMSequential, CBMJoint
from evcbm.models.cem import CEM
from evcbm.models.evcbm import EVCBM
from evcbm.models.probcbm import ProbCBM

from evcbm.engine.cv_resnet import run_kfold_resnet
from evcbm.engine.cv_cbm_joint import run_kfold_cbm_joint
from evcbm.engine.cv_cbm_sequential import run_kfold_cbm_sequential
from evcbm.engine.cv_cem import run_kfold_cem
from evcbm.engine.cv_probcbm import run_kfold_probcbm
from evcbm.engine.cv_evcbm import run_kfold_evcbm
from evcbm.engine.cv_evcbm_sequential import run_kfold_evcbm_sequential



# ====== Base CONFIG (can be edited) ======
CONFIG = {
    "root_dir": "data_ready",
    "dataset": "Derm7pt",                   # default; will be overwritten in the loop
    "num_folds": 5,
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,                       # On Windows, set to 0 if you meet DataLoader issues
    "concept_epochs": 50,
    "label_epochs": 50,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "patience": 30,
    "augment": True,
    "fp16": True,
    "out_dir": "outputs",
    "min_delta": 5e-4,
    "optimiser_name": "AdamW",

    # joint objectives
    "lambda_concepts": 0.1,
    "concept_threshold": 0.5,

    # CEM extras
    "cem_emb_size": 16,
    "cem_shared_prob_gen": True,
    "cem_c2y_layers": [256, 128],           # [] for linear
    "training_intervention_prob": 0.25,
    
    # ---- EVCBM ----
    "evcbm_topk": 32,                  
    "evcbm_lambda_yager_blend": 0.2,
    "evcbm_discount_hidden": 128,
    "evcbm_discount_dropout": 0.0,
    "evcbm_use_context": True,
    "evcbm_context_hidden": 128,
    "evcbm_context_dropout": 0.0,
    "backbone_lr_ratio": 0.1,
}

# ====== Factories ======
def _factory_resnet(num_classes: int):
    return BaselineResNet50(num_classes=num_classes, pretrained=True)

def _factory_cbm_seq(num_concepts: int, num_classes: int):
    return CBMSequential(num_concepts=num_concepts, num_classes=num_classes, pretrained=True)

def _factory_cbm_joint(num_concepts: int, num_classes: int):
    return CBMJoint(num_concepts=num_concepts, num_classes=num_classes, pretrained=True)

def _factory_cem(num_concepts: int, num_classes: int):
    return CEM(
        num_concepts=num_concepts,
        num_classes=num_classes,
        pretrained=True,
        emb_size=CONFIG.get("cem_emb_size", 16),
        embedding_activation="leakyrelu",
        shared_prob_gen=CONFIG.get("cem_shared_prob_gen", True),
        c2y_layers=CONFIG.get("cem_c2y_layers", []),
        )

def _factory_pcbm(num_concepts: int, num_classes: int):
    return ProbCBM(num_concepts=num_concepts, num_classes=num_classes, pretrained=True)

def _factory_evcbm(num_concepts: int, num_classes: int):
    return EVCBM(
        num_concepts=num_concepts,
        num_classes=num_classes,
        pretrained=True,
        topk=CONFIG.get("evcbm_topk", 32),
        lambda_yager_blend=CONFIG.get("evcbm_lambda_yager_blend", 0.0),
        eps=1e-6,
        discount_hidden=CONFIG.get("evcbm_discount_hidden", 128),
        discount_dropout=CONFIG.get("evcbm_discount_dropout", 0.0),
        use_context=CONFIG.get("evcbm_use_context", True),
        context_hidden=CONFIG.get("evcbm_context_hidden", 128),
        context_dropout=CONFIG.get("evcbm_context_dropout", 0.0),
        )


# ====== What to run ======
DATASETS = ["AwA2", "CUB", "Derm7pt"]

MODELS = ["resnet50", "cbm_joint", "cbm_sequential", "cem", "probcbm", "evcbm", "evcbm_sequential"]
MODELS = ["evcbm"]

def _print_header(cfg):
    print("\n=== CONFIG ===")
    view = dict(cfg)
    view["cem_c2y_layers"] = list(cfg.get("cem_c2y_layers", []))
    print(json.dumps(view, indent=2, ensure_ascii=False))

def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    summary_rows = []

    for ds in DATASETS:
        CONFIG["dataset"] = ds
        print("\n" + "=" * 30)
        print(f"Dataset: {ds}")
        print("=" * 30)
        
        if ds == "Derm7pt":
            CONFIG["concept_epochs"] = 50
            CONFIG["label_epochs"] = 50
            CONFIG["patience"] = 50
            CONFIG["cem_emb_size"] = 16
            CONFIG["evcbm_topk"] = 16
            CONFIG["evcbm_lambda_yager_blend"] = 0.1
        elif ds == "CUB":
            CONFIG["concept_epochs"] = 40
            CONFIG["label_epochs"] = 40
            CONFIG["patience"] = 40
            CONFIG["cem_emb_size"] = 64
            CONFIG["evcbm_topk"] = 32
            CONFIG["evcbm_lambda_yager_blend"] = 0.1
        else:
            CONFIG["concept_epochs"] = 20
            CONFIG["label_epochs"] = 20
            CONFIG["patience"] = 20
            CONFIG["cem_emb_size"] = 64
            CONFIG["evcbm_topk"] = 32
            CONFIG["evcbm_lambda_yager_blend"] = 0.1
            
        for model_name in MODELS:
            if len(summary_rows) > 0:
                print(summary_rows)
            print("\n" + "-" * 20)
            print(f"Model: {model_name}")
            print("-" * 20)
            
            if model_name in ["evcbm", "evcbm_sequential"]:
                CONFIG["optimiser_name"] = "AdamW"
                CONFIG["learning_rate"] = 1e-3
            else:
                CONFIG["optimiser_name"] = "SGD"
                CONFIG["learning_rate"] = 1e-2
            
            # fresh seed per (dataset, model) combo to keep comparability
            set_seed(CONFIG["seed"])

            try:
                if model_name == "resnet50":
                    def factory(num_classes): return _factory_resnet(num_classes)
                    lbl_m, lbl_s = run_kfold_resnet(CONFIG, model_factory=factory, get_transforms_fn=get_transforms)

                elif model_name == "cbm_joint":
                    def factory(D, C): return _factory_cbm_joint(D, C)
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_cbm_joint(CONFIG, model_factory=factory, get_transforms_fn=get_transforms)

                elif model_name == "cbm_sequential":
                    def factory(D, C): return _factory_cbm_seq(D, C)
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_cbm_sequential(CONFIG, model_factory=factory, get_transforms_fn=get_transforms)

                elif model_name == "cem":
                    def factory(D, C): return _factory_cem(D, C)
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_cem(CONFIG, model_factory=factory, get_transforms_fn=get_transforms)
                
                elif model_name == "probcbm":
                    def factory(D, C): return _factory_pcbm(D, C)
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_probcbm(CONFIG, model_factory=factory, get_transforms_fn=get_transforms)
        
                elif model_name == "evcbm":
                    def model_factory(num_concepts, num_classes):
                        return _factory_evcbm(num_concepts, num_classes)
                    CONFIG.update({"joint_warmup_epochs": 3, "max_grad_norm": 1.0})
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_evcbm(CONFIG, model_factory=model_factory, get_transforms_fn=get_transforms)

                elif model_name == "evcbm_sequential":
                    def model_factory(num_concepts, num_classes):
                        return _factory_evcbm(num_concepts, num_classes)
                    lbl_m, lbl_s, cpt_m, cpt_s = run_kfold_evcbm_sequential(CONFIG, model_factory=model_factory, get_transforms_fn=get_transforms)
                    
                else:
                    raise ValueError(f"Unknown model_name={model_name}")
                    
                if model_name == "resnet50":
                    summary_rows.append({
                        "dataset": ds,
                        "model": model_name,
                        "label_mean": round(lbl_m, 4), "label_std": round(lbl_s, 4),
                        "concept_mean": None, "concept_std": None
                    })
                    print(f"[Summary] ResNet50 Label Acc: {lbl_m:.4f} ± {lbl_s:.4f}")
                else:
                    summary_rows.append({
                        "dataset": ds,
                        "model": model_name,
                        "label_mean": round(lbl_m, 4), "label_std": round(lbl_s, 4),
                        "concept_mean": round(cpt_m, 4), "concept_std": round(cpt_s, 4)
                    })
                    print(f"[Summary] {model_name} Label Acc: {lbl_m:.4f} ± {lbl_s:.4f} | Concept Acc: {cpt_m:.4f} ± {cpt_s:.4f}")

            except Exception as e:
                # Keep going even if one combo fails; record the error
                print("!! Error occurred for (dataset, model) =", ds, model_name)
                traceback.print_exc()
                summary_rows.append({
                    "dataset": ds,
                    "model": model_name,
                    "label_mean": None, "label_std": None,
                    "concept_mean": None, "concept_std": None,
                    "error": str(e)
                })

        # Save combined summary
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv = os.path.join(CONFIG["out_dir"], f"summary_all_{ts}.csv")
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_csv, index=False,sep=';')
        print("\nAll runs done. Summary saved to:", summary_csv)
        print(df)

if __name__ == "__main__":
    main()
