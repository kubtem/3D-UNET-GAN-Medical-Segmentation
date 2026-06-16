# Dynamic Amyloid PET → Synthetic FDG-PET Generation

3D U-Net + GAN framework for cross-modal volumetric PET image synthesis

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/kubtem/3d-u-net-gan-pipeline)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end deep learning pipeline that synthesises **[18F]FDG-PET** volumes from **dynamic amyloid PET** scans using deep learning approaches for cross-modal medical image synthesis.

---

## Overview

| | |
|---|---|
| **Task** | Cross-modal PET image synthesis (amyloid → FDG) |
| **Architecture** | 3D U-Net generator + patch-based GAN discriminator |
| **Input** | Dynamic amyloid PET data |
| **Output** | Synthetic FDG-PET volume |
| **Framework** | PyTorch |
| **Platform** | Kaggle GPU |

---

## Pipeline

Dynamic Amyloid PET
│
▼
┌─────────────────────┐
│  Data Preparation   │
│  Anonymization      │
│  Preprocessing      │
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│  3D U-Net Generator │
│  + GAN Discriminator│
│  Adversarial Training│
└──────────┬──────────┘
│
▼
Synthetic FDG-PET Volume

---

## Repository Structure
├── notebooks/
│   └── 3d-unet-gan-pet-synthesis.ipynb
├── requirements.txt
└── README.md

---

## Notebook Contents

| Step | Description |
|------|-------------|
| 1 | Environment setup & GPU configuration |
| 2 | PET data loading and preprocessing |
| 3 | DICOM anonymization workflow |
| 4 | Dataset preparation and DataLoaders |
| 5 | 3D U-Net generator architecture |
| 6 | Training pipeline and loss monitoring |
| 7 | GAN-based adversarial training |
| 8 | Synthetic PET visualization and evaluation |

---

## Quickstart

```bash
git clone https://github.com/kubtem/3D-UNET-GAN-Medical-Segmentation.git

cd 3D-UNET-GAN-Medical-Segmentation

pip install -r requirements.txt

jupyter notebook notebooks/3d-unet-gan-pet-synthesis.ipynb

```

Key Design Decisions

* 3D deep learning approach to preserve volumetric spatial information.
* Image-to-image translation framework for mapping amyloid PET representations to FDG-like synthetic images.
* GAN-based training strategy to improve generated image characteristics.
* Automated preprocessing workflow for reproducible medical imaging experiments.

⸻

Requirements

Core dependencies:

* torch
* pydicom
* nibabel
* numpy
* scikit-learn
* tqdm
* matplotlib

⸻

Background

This project explores deep learning methods for synthetic PET image generation, combining medical imaging workflows with modern AI techniques.

⸻

Part of [kubtem](https://github.com/kubtem)’s AI/ML portfolio
