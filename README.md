# Pancreas Cancer Analysis – Multitask nnU-Net :D

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Architecture](#architecture)
  - [Training Strategy](#training-strategy)
- [Results](#results)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
- [Achievements vs Targets](#achievements-vs-targets)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## 🔬 Introduction
This project implements a **multitask nnU-Net framework** for pancreas cancer analysis, combining:
- **Segmentation** of pancreas and lesions.
- **Classification** of cancer subtypes.

The goal was to push segmentation accuracy to **DSC ≥ 0.91** while meeting classification and efficiency benchmarks.

---

## 📊 Dataset
- **Source:** Pancreas CT dataset (*Dataset172_PancreasQuiz*).
- **Classes:**  
  - `0` – Background  
  - `1` – Pancreas  
  - `2` – Lesion  
- **Samples:** 201 training, 72 testing.

---

## 🏗️ Methodology

### Architecture
- **Shared Encoder** (modular U-Net backbone).
- **Segmentation Decoder** for anatomical delineation.
- **Classification Head** (global average pooling + linear layer) for subtype prediction.

### Training Strategy
- Loss: Dice + Cross Entropy for segmentation, CE for classification.  
- Optimizer: AdamW.  
- Epochs: up to 120 (with early stopping).  
- Augmentations: flipping, rotation, scaling.  
- Mixed precision (AMP) training.

---

## 📈 Results

### Segmentation
- Conservative model DSC: **0.45**  
- Balanced model DSC: **0.56**  
- Aggressive model DSC: **0.35**  
- Segmentation-focused model DSC: **0.28**

### Classification
- Macro F1 ranged from **0.31 – 0.42** across models.

---

## 🎯 Achievements vs Targets

| Requirement              | Target  | Achieved | Status |
|---------------------------|---------|----------|--------|
| Whole Pancreas DSC        | ≥ 0.91 | ~0.58*   | ❌ |
| Pancreas Lesion DSC       | ≥ 0.31 | ~0.35*   | ✅ |
| Classification Macro F1   | ≥ 0.70 | ~0.43*   | ❌ |
| Speed Improvement         | ≥ 10%  | 15%      | ✅ |

\* Ensemble-based estimated values.

---

## ⚙️ Installation & Usage

```bash
# Clone repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create environment
conda create -n pancreas python=3.10 -y
conda activate pancreas

# Install dependencies
pip install -r requirements.txt
