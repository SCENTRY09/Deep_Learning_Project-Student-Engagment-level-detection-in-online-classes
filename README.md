# Automated Student Engagement Detection in E-Learning

> Comparative Evaluation of Deep Learning Architectures with Bagging and Stacking Ensemble Strategies on the DAiSEE Dataset

---

## Authors

| Name | Roll No. | Email |
|---|---|---|
| Aditi Namde | 202301040034 | 202301040034@mitaoe.ac.in |
| Manoj Ghadge | 202301040056 | 202301040056@mitaoe.ac.in |
| Shivam Khedkar | 202301040063 | 202301040063@mitaoe.ac.in |
| Sujal Sonawane | 202301040131 | 202301040131@mitaoe.ac.in |

**Guided by:** Mrs. Sunita Barve & Mrs. Diptee Chikmurge
**Institution:** School of Computer Engineering, MIT Academy of Engineering, Pune

---

## Overview

This project presents a multi-architecture ensemble deep learning framework for **four-class student engagement detection** using the [DAiSEE dataset](https://people.iith.ac.in/vineethnb/resources/daisee/index.html). It builds upon and extends the bagging ensemble baseline proposed by Santoni et al. (2022) by:

- Replicating baseline 1D CNN and 1D ResNet architectures under controlled conditions
- Introducing **six novel architectures** across CNN variant and encoder-decoder families
- Applying advanced training techniques: focal loss, Mixup augmentation, cosine annealing, and Test-Time Augmentation (TTA)
- Evaluating **three ensemble strategies**: bagging, hybrid aggregation, and stacking meta-learning

---

## Dataset

**DAiSEE** (Dataset for Affective States in E-Environments)

- 9,068 ten-second video clips from 112 students during e-learning sessions
- Four engagement levels: `Very Low (0)`, `Low (1)`, `High (2)`, `Very High (3)`
- Official split: **64% Train | 16% Validation | 20% Test**
- Severe class imbalance skewed toward High and Very High classes

---

## Pipeline

```
DAiSEE Videos
     │
     ▼
OpenFace 2.0 Feature Extraction
(709-dim per frame: landmarks, gaze, head pose, action units)
     │
     ▼
Temporal Aggregation → mean + std + min + max → 2836-dim descriptor
     │
     ▼
Preprocessing: StandardScaler → SMOTE → TruncatedSVD (n=35) → (N, 35, 1)
     │
     ▼
┌────────────────────────────────────────────────────────┐
│         8 Parallel Deep Learning Architectures          │
│  (each trained on 5 bootstrap bags independently)       │
├──────────────┬──────────────────┬──────────────────────┤
│  Baselines   │  Novel CNN       │  Encoder-Decoder     │
│  1D CNN      │  DepthwiseCNN    │  Autoencoder Clf.    │
│  1D ResNet   │  DenseNet        │  U-Net Classifier    │
│              │  InceptionNet    │                      │
│              │  AttentionResNet │                      │
└──────────────┴──────────────────┴──────────────────────┘
     │
     ▼
Ensemble Strategies: BEA | BEM | Hybrid | Stacking (LR / MLP)
     │
     ▼
Final 4-Class Prediction + Evaluation
```

---

## Architectures

### Baseline Models
| Model | Description |
|---|---|
| **1D CNN** | Pyramidal Conv1D with filter counts 128→512, kernel size 7 stem, 3 double-conv blocks, GAP → Dense(512, 256) |
| **1D ResNet** | 3 residual blocks, Conv1D(64)×2 per block, identity skip connections, GAP output |

### Novel CNN Variants
| Model | Key Idea |
|---|---|
| **1D DepthwiseCNN** | Depthwise separable convolutions (EfficientNet-style) — fewer parameters |
| **1D DenseNet** | Dense connectivity — each layer receives all preceding feature maps |
| **1D InceptionNet** | Multi-scale parallel branches with kernel sizes 1, 3, 5 |
| **AttentionResNet** | ResNet + Squeeze-and-Excitation (SE) channel attention modules |

### Encoder-Decoder Models
| Model | Key Idea |
|---|---|
| **Autoencoder Classifier** | Joint classification + input reconstruction; auxiliary reconstruction loss regularises encoder |
| **U-Net Classifier** | Skip connections between encoder and decoder; bottleneck classification head |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 5×10⁻⁴ (max), 1×10⁻⁷ (min) |
| LR Schedule | Cosine Annealing with Warm Restarts (4 cycles) |
| Epochs | Up to 3000 |
| Batch Size | 32 |
| Early Stopping Patience | 120 epochs |
| Loss Function | Focal Loss (γ=2.0) + Label Smoothing (ε=0.05) |
| Augmentation | Mixup (α=0.2, Beta distribution) |
| Bagging | 5 bootstrap subsets per architecture |
| TTA | 5 passes with Gaussian noise σ=0.01 |
| SVD Components | 35 (selected via logistic regression probe sweep: {20, 30, 35, 45, 55}) |
| Class Imbalance | SMOTE oversampling on training set |

---

## Results

### Individual Model Performance

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| 1D CNN | 47.98 | 44.44 | 47.98 | 36.06 |
| 1D ResNet | **50.56** | 55.55 | 50.56 | 38.25 |
| 1D DepthwiseCNN | 45.07 | 45.20 | 45.07 | 43.13 |
| 1D DenseNet | 48.88 | 48.29 | 48.88 | 41.00 |
| 1D InceptionNet | 46.86 | 47.24 | 46.86 | 46.90 |
| AttentionResNet | 50.22 | 50.60 | 50.22 | 40.33 |
| Autoencoder Clf. | 43.89 | 46.56 | 43.89 | 40.93 |
| U-Net Clf. | 48.60 | 48.45 | 48.60 | **48.45** |

### Ensemble Performance (Top Configurations)

| Configuration | Accuracy (%) | F1 (%) |
|---|---|---|
| **Stacking LR** | **50.06** | 37.19 |
| 1D DenseNet BEA | 50.00 | — |
| AEClassifier BEA | 49.83 | **49.01** |
| 1D CNN BEA | 49.50 | 33.16 |
| 1D ResNet BEA | 49.33 | 33.00 |
| Hybrid BEA (CNN + ResNet) | 49.44 | 32.81 |

> **Reference:** Santoni et al. reported Hybrid BEA accuracy of **94.25%** — the gap is attributed to use of official dataset splits and richer 4-statistic temporal aggregation vs. mean-only.

---

## Repository Structure

```
DAiSEE-Engagement-Detection/
│
├── DAiSEE_Complete_Architecture_(1)_(5).ipynb   # Main experiment notebook
│
├── features/                        # Preprocessed OpenFace features (Drive)
│   ├── Train/
│   │   ├── X.npy
│   │   └── clip_ids.npy
│   ├── Validation/
│   ├── Test/
│   └── Labels/
│       ├── TrainLabels.csv
│       ├── ValidationLabels.csv
│       └── TestLabels.csv
│
├── output/
│   ├── models/                      # Saved model weights (.weights.h5)
│   ├── plots/                       # Generated visualisations
│   └── logs/                        # Training CSV logs
│
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install tensorflow scikit-learn imbalanced-learn \
            seaborn matplotlib pandas numpy scipy tqdm
```

- Python 3.10+
- TensorFlow 2.x
- GPU with CUDA acceleration recommended (trained on NVIDIA GPU, 16 GB RAM)

### Running on Google Colab

1. Upload the notebook to Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place your OpenFace feature `.npy` files and label CSVs in the paths defined in **Cell 2**
4. Run all cells sequentially (A → J)

### Notebook Sections

| Section | Content |
|---|---|
| **A** | Setup, Data Loading, EDA, SMOTE, SVD |
| **B** | Baseline 1D CNN & 1D ResNet (Individual + Bagging) |
| **C** | Novel CNN 1 — 1D DepthwiseCNN |
| **D** | Novel CNN 2 — 1D DenseNet |
| **E** | Novel CNN 3 — 1D InceptionNet |
| **F** | Novel CNN 4 — AttentionResNet |
| **G** | Encoder-Decoder 1 — Autoencoder Classifier |
| **H** | Encoder-Decoder 2 — U-Net Style |
| **I** | Stacking Ensemble (Meta-Learner) |
| **J** | Grand Comparative Analysis |

### Checkpoint Resume

The notebook includes smart checkpoint resumption — if weights already exist for a model+bag combination, they are loaded instead of retraining:

```python
# Automatically skips training if weights file exists
model, history = load_or_train(build_fn, name='1D_CNN', ..., bag=1)
```

---

## Key Findings

- **Stacking LR** achieves the highest overall accuracy at **50.06%**
- **AEClassifier BEA** leads all ensembles in weighted F1 at **49.01%**, attributed to its dual-objective reconstruction regularisation
- Bootstrap aggregation improves 1D CNN by +1.52pp but reduces 1D ResNet by −1.23pp, revealing architecture-specific sensitivity to subsampling
- No proposed model achieves statistically significant improvement over 1D CNN BEA at α=0.05 (except U-Net BEA which is significantly *worse*, p=0.0089)
- U-Net Classifier posts the strongest individual-model F1 at **48.45%**

---

## Limitations

- DAiSEE covers only 112 students — limits generalisability
- Class imbalance in minority classes persists despite SMOTE + focal loss
- Pipeline depends on pre-extracted OpenFace features (not end-to-end)
- ~44 percentage point gap vs. Santoni et al. reference, due to differing data partitioning strategy

---

## Future Work

- End-to-end transformer architectures operating directly on raw video frames
- Cross-dataset evaluation on EmotiW and MAHNOB-HCI
- Lightweight model distillation for edge deployment
- Multimodal fusion with audio or physiological signals

---

## References

1. Gupta et al. (2016) — DAiSEE dataset
2. Santoni et al. (2022) — Bagging ensemble baseline, *Applied Sciences*
3. Baltrusaitis et al. (2018) — OpenFace 2.0
4. He et al. (2016) — ResNet
5. Huang et al. (2017) — DenseNet
6. Szegedy et al. (2015) — Inception
7. Hu et al. (2018) — Squeeze-and-Excitation Networks
8. Lin et al. (2017) — Focal Loss
9. Zhang et al. (2018) — Mixup
10. Chawla et al. (2002) — SMOTE
11. Chollet (2017) — Depthwise Separable Convolutions
12. Ronneberger et al. (2015) — U-Net

---

*MIT Academy of Engineering, Pune | School of Computer Engineering | 2024–25*
