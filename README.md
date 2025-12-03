# AI-based-Automated-Detection-of-Cytoplasmic-Strings-in-Microscopic-Images

This repository contains the full pipeline developed for the detection of cytoplasmic strings (CS) in human embryo images. The project combines unsupervised anomaly detection, region-based proposal generation, and self-supervised visual embeddings to localize and classify subtle CS structures under limited annotation conditions.

Our approach consists of two complementary stages:

**1. Autoencoder-based Classification Filter.**  
A lightweight convolutional autoencoder (AE) is trained exclusively on string-free embryo images to model the appearance of normal cytoplasm. During inference, reconstruction errors—measured inside the smooth cytoplasmic interior—serve as anomaly scores. This stage acts as a *coarse filter* that flag frames likely to contain CS, significantly reducing downstream workload and improving robustness under class imbalance.
