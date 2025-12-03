# AI-based-Automated-Detection-of-Cytoplasmic-Strings-in-Microscopic-Images

This repository contains the full pipeline developed for the detection of cytoplasmic strings (CS) in human embryo images. The project combines unsupervised anomaly detection, region-based proposal generation, and self-supervised visual embeddings to localize and classify subtle CS structures under limited annotation conditions.

Our approach consists of two complementary stages:

**1. Autoencoder-based Classification Filter.**  
A lightweight convolutional autoencoder (AE) is trained exclusively on string-free embryo images to model the appearance of normal cytoplasm. During inference, reconstruction errors,measured inside the smooth cytoplasmic interior, serve as anomaly scores. This stage acts as a *coarse filter* that flag frames likely to contain CS, significantly reducing downstream workload and improving robustness under class imbalance.

**2. Region-based Weakly Supervised Localization.**  
On top of the filtered frames, CS detection is formulated as a region-based, weakly supervised object localization problem. Dense region proposals are generated using Selective Search, encoded using a self-supervised DINOv3 Vision Transformer, and scored using a dual-head multilayer perceptron that jointly predicts the probability of CS presence and the expected localization quality. This allows the system to refine detections and produce spatially resolved bounding-box predictions even with minimal manual annotations.

Together, these two components form an efficient, scalable pipeline for CS screening and localization under real-world clinical constraints, where ground truth is scarce, CS appearance is highly variable, and processing speed is critical.



The repository includes:
- code for training and evaluating both **weighted and unweighted autoencoders**,  
- scripts for **region proposal generation and embedding extraction**,  
- the weakly supervised **region-scoring network**,  
- inference utilities and visualization tools for qualitative analysis.

This README provides a full guide on installation, dataset setup, training, evaluation, and visualization.

---

For the convolutional autoencoder (AE)–based anomaly detector for **cytoplasmic string (CS)** classification in human embryo images.

The key idea is:
- Train the AE **only on normal (string-free) embryos**.
- At inference, measure **reconstruction error** inside the cytoplasmic cavity.
- Use this error as an **anomaly score**: high error ⇒ likely CS present.

Two variants are provided:

- **Unweighted AE** – 
- **Weighted AE (v2)** – reconstruction loss is **spatially weighted** using a GLCM-derived low-texture mask so the model focuses on the cytoplasmic interior where strings occur.

---


## 📂 Dataset Layout

Your dataset should be organized as follows:

```
cs_strings_dataset/
├── train/
│   └── normal/
├── val/
│   ├── normal/
│   └── abnormal/
└── test/
    ├── normal/
    └── abnormal/
```

## Tested with:

- Python 3.9+  
- PyTorch (GPU preferred)





