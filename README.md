# Evidential Concept Bottleneck Models (EVCBM)

Implementation of **Evidential Concept Bottleneck Models (EVCBM)** for interpretable image classification with uncertainty-aware concept-to-label reasoning.

## Overview

Concept Bottleneck Models (CBMs) improve interpretability by predicting human-understandable concepts before predicting class labels. However, standard CBMs usually:

- do not explicitly propagate concept uncertainty to label prediction,
- and may lose concept fidelity under joint training.

To address this, **EVCBM** introduces an evidential concept-to-label module based on **Dempster–Shafer theory**. Each predicted concept is converted into class evidence, discounted by its calibrated reliability, and then fused to produce:

- class probabilities,
- and an explicit uncertainty score.

## Key Features

- **Uncertainty-aware classification**: uncertain concepts contribute more to ignorance instead of forcing overconfident predictions.
- **Concept reliability calibration**: a lightweight calibrator estimates how trustworthy each concept prediction is.
- **Evidence fusion**: concept evidences are aggregated with a tunable rule between **Dempster's rule** and **Yager's rule**.
- **Better concept-label balance**: helps preserve concept faithfulness while maintaining competitive label accuracy.

## Method Summary

Given an input image:

1. A CNN predicts concept scores.
2. A calibrator estimates concept reliability scores.
3. Each concept is converted into a class-supporting mass function.
4. The mass is discounted according to concept reliability.
5. The top-\(M\) most reliable concept evidences are fused.
6. The final evidential belief is transformed into class probabilities.

In addition to label prediction, the model also outputs an uncertainty signal through the mass assigned to ignorance.

## Datasets

The experiments in the paper are conducted on:

- **Derm7pt**
- **CUB**
- **AwA2**

The processed datasets are available on Zenodo: [10.5281/zenodo.19368745](https://doi.org/10.5281/zenodo.19368746)


## Repository Structure

```text
EVCBM/
├── preprocessing/
│   ├── derm7pt_preprocessing.ipynb
│   ├── cub_preprocessing.ipynb
│   ├── awa2_preprocessing.ipynb
│   └── utils.py
├── src/evcbm/
│   ├── data/
│   ├── engine/
│   ├── eval/
│   ├── models/
│   └── utils/
├── eval_accuracy.py
├── sweep_topk_evcbm.py
├── sweep_lambda_yager_blend_evcbm.py
├── context_ablation.py
├── requirements.txt
└── README.md
```

## Data Preparation

The code expects processed datasets under the directory data_ready, with one subfolder per dataset. Expected folder structure is:

```
data_ready/
├── Derm7pt/
│   ├── images/
│   ├── annotations/
│   │   ├── labels_onehot.npy
│   │   └── concepts_per_image.npy
│   └── metadata/
│       └── samples.csv
├── CUB/
│   ├── images/
│   ├── annotations/
│   │   ├── labels_onehot.npy
│   │   └── concepts_per_image.npy
│   └── metadata/
│       └── samples.csv
└── AwA2/
    ├── images/
    ├── annotations/
    │   ├── labels_onehot.npy
    │   └── concepts_per_image.npy
    └── metadata/
        └── samples.csv
```

The dataset loader reads the following fields from ```metadata/samples.csv```:

- ```sample_id```
- ```rel_path```
- ```class_id```
- ```class_name```


## Training and Evaluation

The main entry point is:

```python eval_accuracy.py```

This script runs cross-validation experiments.