# -*- coding: utf-8 -*-

from typing import Tuple, List
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import os, csv
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# --------- Dataset ----------
class ConceptDataset(Dataset):
    """
    读取 data_ready/<DATASET> 下的数据。
    兼容 samples.csv 的 rel_path 两种写法：
      1) "xxx.jpg"（不带 images/）
      2) "images/xxx.jpg" 或 "/images/xxx.jpg"
    """
    def __init__(self, root_dir: str, dataset: str, transform=None,
                 return_onehot: bool = True, dtype_concepts=np.float32):
        self.root = os.path.join(root_dir, dataset)
        self.img_dir = os.path.join(self.root, "images")
        self.ann_dir = os.path.join(self.root, "annotations")
        self.meta_dir = os.path.join(self.root, "metadata")
        self.transform = transform
        self.return_onehot = return_onehot

        # 读 samples.csv
        samples_csv = os.path.join(self.meta_dir, "samples.csv")
        self.samples = []
        with open(samples_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "sample_id": row["sample_id"],
                    "rel_path": row["rel_path"],
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                })

        # 读 npy
        self.Y_onehot = np.load(os.path.join(self.ann_dir, "labels_onehot.npy"))
        self.C_per_img = np.load(os.path.join(self.ann_dir, "concepts_per_image.npy"))
        assert len(self.samples) == self.Y_onehot.shape[0] == self.C_per_img.shape[0], \
            "Row count mismatch between samples.csv and npy files."

        # 校验 class_id 与 onehot 一致
        argmax_ids = self.Y_onehot.argmax(axis=1)
        csv_ids = np.array([s["class_id"] for s in self.samples])
        if not np.all(argmax_ids == csv_ids):
            raise ValueError("class_id in samples.csv does not match labels_onehot argmax.")

        self.num_samples = len(self.samples)
        self.num_classes = self.Y_onehot.shape[1]
        self.num_concepts = self.C_per_img.shape[1]
        self.C_per_img = self.C_per_img.astype(dtype_concepts)

    def _resolve_img_path(self, rel_path: str) -> str:
        p = rel_path.lstrip("/\\")
        # 若已经以 images/ 开头，则直接和数据集根目录拼
        if p.startswith("images/") or p.startswith("images\\"):
            return os.path.join(self.root, p)
        # 否则拼到 images 目录下
        return os.path.join(self.img_dir, p)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = self._resolve_img_path(rec["rel_path"])
        with Image.open(img_path).convert("RGB") as im:
            img = self.transform(im) if self.transform is not None else im

        y_onehot = self.Y_onehot[idx]
        y = int(y_onehot.argmax())
        concepts = self.C_per_img[idx]

        out = {
            "image": img,
            "y": y,
            "concepts": concepts,
            "sample_id": rec["sample_id"],
            "class_name": rec["class_name"],
            "rel_path": rec["rel_path"],
        }
        if self.return_onehot:
            out["y_onehot"] = y_onehot
        return out

# 用于在 K 折中切换不同 transform
class TransformedDataset(Dataset):
    def __init__(self, base: ConceptDataset, tfm):
        self.base = base
        self.tfm = tfm

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        # 复用 base 的路径解析，重新应用 tfm
        rec = self.base.samples[idx]
        img_path = self.base._resolve_img_path(rec["rel_path"])
        with Image.open(img_path).convert("RGB") as im:
            img = self.tfm(im)
        # 其余字段直接从 base 取
        item = {
            "image": img,
            "y": int(self.base.Y_onehot[idx].argmax()),
            "concepts": self.base.C_per_img[idx],
            "sample_id": rec["sample_id"],
            "class_name": rec["class_name"],
            "rel_path": rec["rel_path"],
            "y_onehot": self.base.Y_onehot[idx],
        }
        return item

# def build_dataloaders(ds: ConceptDataset,
#                       idx_train, idx_val,
#                       batch_size: int, num_workers: int,
#                       train_tfms, val_tfms) -> Tuple[DataLoader, DataLoader]:
#     ds_train = Subset(TransformedDataset(ds, train_tfms), indices=list(idx_train))
#     ds_val   = Subset(TransformedDataset(ds, val_tfms),   indices=list(idx_val))
#     pin = torch.cuda.is_available()
#     dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
#                           num_workers=num_workers, pin_memory=pin)
#     dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=pin)
#     return dl_train, dl_val

def build_dataloaders(
    ds: "ConceptDataset",
    idx_train: List[int],
    idx_test: List[int],
    batch_size: int,
    num_workers: int,
    train_tfms,
    val_tfms,
    inner_val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    idx_train = np.array(list(idx_train))
    labels = ds.Y_onehot.argmax(axis=1)

    y_train = labels[idx_train]

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=inner_val_ratio,
        random_state=seed  # 固定 seed，保证可重复
    )
    idx_tr_rel, idx_va_rel = next(sss.split(np.zeros(len(idx_train)), y_train))

    idx_tr_inner = idx_train[idx_tr_rel].tolist()
    idx_va_inner = idx_train[idx_va_rel].tolist()

    # ---- 2) 构建三个 Subset（train / val / test）----
    ds_train = Subset(TransformedDataset(ds, train_tfms), indices=idx_tr_inner)
    ds_val   = Subset(TransformedDataset(ds, val_tfms),   indices=idx_va_inner)
    ds_test  = Subset(TransformedDataset(ds, val_tfms),   indices=list(idx_test))

    # ---- 3) DataLoader ----
    pin = torch.cuda.is_available()

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    return dl_train, dl_val, dl_test