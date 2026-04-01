import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import csv
import numpy as np
from PIL import Image, ImageFile, ImageOps
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated images



def preprocess_pil(img: Image.Image, target_size: int = 224, shortest: int = 256) -> Image.Image:
    """
    TorchVision-style preprocessing:
    - EXIF-aware transpose
    - Convert to RGB
    - Resize so the shortest side equals `shortest` with antialiasing
    - Center-crop to `target_size`
    Returns a PIL.Image.
    """
    # Fix EXIF orientation (important for some JPEGs)
    img = ImageOps.exif_transpose(img)

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize shortest side to `shortest`
    img = F.resize(img, shortest, interpolation=InterpolationMode.BILINEAR, antialias=True)

    # Center-crop to `target_size` x `target_size`
    img = F.center_crop(img, [target_size, target_size])

    return img



def scan_raw_folders(raw_root: Path, class_names: List[str], class_dir_name: Optional[str] = None) -> List[Tuple[str, Path]]:
    """
    Find images under <base>/<class_name>/*.
    If class_dir_name is provided, base = raw_root/class_dir_name; otherwise,
    prefer raw_root/images if it exists, else raw_root.
    Returns a list of (class_name, image_path).
    """
    if class_dir_name is not None:
        base = raw_root / class_dir_name
        if not base.is_dir():
            raise FileNotFoundError(f"Class base dir not found: {base}")
    else:
        base = raw_root / "images" if (raw_root / "images").is_dir() else raw_root

    results: List[Tuple[str, Path]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls in class_names:  # preserve the exact order from `classes`
        cls_dir = base / cls
        if not cls_dir.is_dir():
            # silently skip missing class folder, or raise if you prefer strict checking
            continue
        files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        # deterministic ordering within the class
        files.sort(key=lambda p: str(p).lower())
        for p in files:
            results.append((cls, p))

    return results



def run_preprocess(
    classes: np.ndarray,                 # shape (C,), dtype=str
    concepts: np.ndarray,                # shape (D,), dtype=str  (concept names)
    class_level_concepts: np.ndarray,    # shape (C, D), values in {0,1}
    raw_root: str,
    out_root: str,
    image_size: int = 224,
    shortest_side: int = 256,
    jpeg_quality: int = 90,
    save_npy: bool = True,
    class_dir_name: Optional[str] = None,
):
    """
    Process class-foldered raw images into:
      - <out_root>/images/*.jpg  (RGB, shortest=256, center-crop 224)
      - <out_root>/metadata/{samples.csv, classes.csv, concepts.csv, class_level_concepts.csv, stats.json}
      - Optional: <out_root>/annotations/{labels_onehot.npy (N x C), concepts_per_image.npy (N x D)}
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    images_out = out_root / "images"
    meta_out = out_root / "metadata"
    annot_out = out_root / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)
    annot_out.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    assert classes.ndim == 1, "classes must be a 1D array of class names"
    assert concepts.ndim == 1, "concepts must be a 1D array of concept names"
    assert class_level_concepts.ndim == 2, "class_level_concepts must be 2D (C, D)"
    assert len(classes) == class_level_concepts.shape[0], "mismatch: len(classes) != class_level_concepts.shape[0]"
    assert len(concepts) == class_level_concepts.shape[1], "mismatch: len(concepts) != class_level_concepts.shape[1]"

    C = int(len(classes))
    D = int(len(concepts))
    class_names = [str(x) for x in classes.tolist()]
    concept_names = [str(x) for x in concepts.tolist()]
    class_name2id = {name: idx for idx, name in enumerate(class_names)}

    # Scan raw images (your scan_raw_folders must support class_dir_name)
    pairs = scan_raw_folders(raw_root, class_names, class_dir_name=class_dir_name)
    if not pairs:
        raise RuntimeError(f"No images found under {raw_root}. Check paths and class folders.")

    # Process images
    sample_rows: List[Tuple[str, str, int, str]] = []  # (sample_id, rel_path, class_id, class_name)
    labels_onehot: List[np.ndarray] = []
    concepts_per_image: List[np.ndarray] = []

    counter = 0  # global counter across all images

    for cls_name, img_path in tqdm(pairs, desc="Preprocessing images"):
        class_id = class_name2id[cls_name]
        try:
            with Image.open(img_path) as img:
                img = preprocess_pil(img, target_size=image_size, shortest=shortest_side)
        except Exception as e:
            print(f"[WARN] Skipping corrupted/invalid image: {img_path} ({e})")
            continue

        counter += 1
        cls_code = f"{class_id:03d}"   # 01..C (assuming C <= 99)
        global_code = f"{counter:05d}"     # 000001..999999
        sid = f"{cls_code}_{global_code}"

        out_file = images_out / f"{sid}.jpg"
        img.save(out_file, format="JPEG", quality=jpeg_quality, optimize=False)

        rel_path = f"images/{sid}.jpg"
        sample_rows.append((sid, rel_path, class_id, cls_name))

        y = np.zeros((C,), dtype=np.float32)
        y[class_id] = 1.0
        labels_onehot.append(y)

        cvec = class_level_concepts[class_id].astype(np.float32)
        concepts_per_image.append(cvec)

    if not sample_rows:
        raise RuntimeError("All images failed during preprocessing. Check raw data integrity.")

    labels_onehot = np.stack(labels_onehot, axis=0)           # (N, C)
    concepts_per_image = np.stack(concepts_per_image, axis=0) # (N, D)

    # Write metadata CSVs
    samples_csv = meta_out / "samples.csv"
    with samples_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "rel_path", "class_id", "class_name"])
        w.writerows(sample_rows)

    classes_csv = meta_out / "classes.csv"
    with classes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name"])
        for i, name in enumerate(class_names):
            w.writerow([i, name])

    concepts_csv = meta_out / "concepts.csv"
    with concepts_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "concept_name"])
        for i, name in enumerate(concept_names):
            w.writerow([i, name])

    clc_csv = meta_out / "class_level_concepts.csv"
    with clc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class_id"] + [f"c{i+1}" for i in range(D)]
        w.writerow(header)
        for i in range(C):
            w.writerow([i] + list(map(int, class_level_concepts[i])))

    stats = {
        "num_images": int(labels_onehot.shape[0]),
        "num_classes": int(C),
        "num_concepts": int(D),
        "class_hist": {
            f"{i}-{class_names[i]}": int(sum(1 for row in sample_rows if row[2] == i)) for i in range(C)
        },
        "image_size": image_size,
        "shortest_side": shortest_side,
    }
    with (meta_out / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Wrote {labels_onehot.shape[0]} images.")
    print(f" - images:   {images_out}")
    print(f" - metadata: {meta_out}")

    # Optional NPY dumps
    if save_npy:
        np.save(annot_out / "labels_onehot.npy", labels_onehot)            # (N, C)
        np.save(annot_out / "concepts_per_image.npy", concepts_per_image)  # (N, D)
        print(f" - annotations: {annot_out} (labels_onehot.npy, concepts_per_image.npy)")
        print(labels_onehot.shape, concepts_per_image.shape)