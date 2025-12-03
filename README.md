# AI-based-Automated-Detection-of-Cytoplasmic-Strings-in-Microscopic-Images

This repository contains the full pipeline developed for the detection of cytoplasmic strings (CS) in human embryo images. The project combines unsupervised anomaly detection, region-based proposal generation, and self-supervised visual embeddings to localize and classify subtle CS structures under limited annotation conditions.

Our approach consists of two complementary stages:

**1. Autoencoder-based Classification Filter.**  
A lightweight convolutional autoencoder (AE) is trained exclusively on string-free embryo images to model the appearance of normal cytoplasm. During inference, reconstruction errors,measured inside the smooth cytoplasmic interior, serve as anomaly scores. This stage acts as a _coarse filter_ that flag frames likely to contain CS, significantly reducing downstream workload and improving robustness under class imbalance.

**2. Region-based Weakly Supervised Localization.**  
On top of the filtered frames, CS detection is formulated as a region-based, weakly supervised object localization problem. Dense region proposals are generated using Selective Search, encoded using a self-supervised DINOv3 Vision Transformer, and scored using a dual-head multilayer perceptron that jointly predicts the probability of CS presence and the expected localization quality. This allows the system to refine detections and produce spatially resolved bounding-box predictions even with minimal manual annotations.

Together, these two components form an efficient, scalable pipeline for CS screening and localization under real-world clinical constraints, where ground truth is scarce, CS appearance is highly variable, and processing speed is critical.

👉 [Check our Demo Video!](https://youtu.be/0OcU0-dwSpE)

[![Watch the video](https://img.youtube.com/vi/0OcU0-dwSpE/maxresdefault.jpg)](https://youtu.be/0OcU0-dwSpE)

The repository includes:

-   Code for training and evaluating both **weighted and unweighted autoencoders**,
-   Scripts for **region proposal generation and embedding extraction**,
-   The weakly supervised **region-scoring network**,
-   Inference utilities and visualization tools for qualitative analysis.
-   GUI for the framework along with a demo video

This README provides a full guide on installation, dataset setup, training, evaluation, and visualization.

You can read the full details in our paper:

👉 [Coming Soon]

---

# Classification AutoEncoder for Anomaly Detection

For the convolutional autoencoder (AE)–based anomaly detector for **cytoplasmic string (CS)** classification in human embryo images.

The key idea is:

-   Train the AE **only on normal (string-free) embryos**.
-   At inference, measure **reconstruction error** inside the cytoplasmic cavity.
-   Use this error as an **anomaly score**: high error ⇒ likely CS present.

Two variants are provided:

-   **Unweighted AE** – unweighted reconstruction loss over the whole image.
-   **Weighted AE (v2)** – reconstruction loss is **spatially weighted** using a GLCM-derived low-texture mask so the model focuses on the cytoplasmic interior where strings occur.

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

-   Python 3.9+
-   PyTorch (GPU preferred)

## Running the Autoencoder

### 0. Environment

Install the main dependencies (example, adapt to your setup):

```bash
pip install -r requirements.txt
```

### 1. Training the AutoEncoder

Go to the AutoEncoder folder:

```
cd Classification_AutoEncoder_Anomaly/
# Make sure the dataset is saved in this folder
```

To run the unweighted autoencoder script, simply execute:

```
python Training/unweighted_AE.py --data-root "./cs_strings_dataset/" --img-size 256 --epochs 30 --batch-size 16 --lr 1e-3 --device "cuda" --seed 1337 --ckpt-out ae_unweighted_best.pt --num-workers 2
```

To run the weighted autoencoder script, simply execute:

```
python Training/weighted_AE.py --data-root "./cs_strings_dataset/" --img-size 256 --epochs 30 --batch-size 16 --lr 1e-3 --device "cuda" --seed 1337 --ckpt-out ae_unweighted_best.pt --num-workers 2
```

### 2. Inference on Anomaly Detection

Once trained you can run CS identification on new images using our inference script:

```
python Inference/infer_view_weighted.py
# Make sure to change: dataset and checkpoint location
```

---

# Region-based Weakly Supervised Object Localization

We detect CS regions using a weakly supervised, region-based localization approach:

### 1. Region Proposals

-   Generate up to 2000 candidate regions per frame using Selective Search.
-   Discard invalid or very small boxes and clamp proposals to image boundaries.

### 2. Region Embeddings

-   Crop each proposal from the image.
-   Extract a 768-D global embedding using a DINOv3 ViT-B/16 model.

### 3. Scoring Network

-   A lightweight MLP processes each region embedding.
-   It outputs:
    -   A probability that the region contains CS.
    -   A predicted localization quality score.

### 4. Weak Supervision

-   Only the image’s bounding-box annotation is used.
-   A region is considered positive if its overlap with the ground truth exceeds a small threshold.
-   The network jointly learns classification and localization from these weak labels.

### 5. Inference

-   Combine each region’s classification probability and localization score into a final confidence.
-   Select the region with the highest confidence as the CS prediction.

## 📂 Dataset Layout

The code allows for the independence of folder locations. All the scripts just need the path to a folder to execute. However it is recommended to separate the dataset in training/validation/testing set from the begging to avoid leakage. Bare in mind that the data split should be done at the embryo level.

## Tested with:

-   Python 3.9+
-   PyTorch (GPU preferred)

## Running the Region-based approach

The folder is organized as a **four-stage training pipeline** plus an **inference script**:

-   `Training/Stage_1_Selective_Search/`
-   `Training/Stage_2_DINOV3/`
-   `Training/Stage_3_Weak_labeling/`
-   `Training/Stage_4_FCN/`
-   `Inference/inference_single_img.py`

Below is the recommended order of execution.

---

### 0. Environment

Install the main dependencies (example, adapt to your setup):

```bash
pip install -r requirements.txt
```

### 1. Set paths to inputs and outputs folders/files

-   IMAGE_DIR → folder with original images
-   OUT_DIR → output folder for proposals / visualizations
-   LABEL_FOLDER → OUT_DIR/yolo from Stage 1.1
-   B0_DIR → YOLO GT labels
-   B1_DIR → proposal labels from Stage 1.1 (OUT_DIR/yolo)
-   OUT_CSV → path to the IoU CSV

### 2. Run the training scripts in order

```
python Training/Stage_1_Selective_Search/1_create_regions_ss.py
python Training/Stage_1_Selective_Search/2_yolo_crops_to_imgs.py
python Training/Stage_2_DINOV3/1_extract_features.py
python Training/Stage_3_Weak_labeling/1_calculate_iou_aligned_bb.py
python Training/Stage_3_Weak_labeling/2_create_classification_labels.py
python Training/Stage_3_Weak_labeling/3_balance_class_labels.py
python Training/Stage_4_FCN/1_train_fcn_multihead.py
```

### 3. Once the pipeline has been trained, we can run inference on single images

Set the path to the model checkpoint and the path to the image to be processed.

```
python Inference/inference_single_img.py
```

---

# Streamlit Demo: Interactive CS Detection Explorer

This Streamlit app provides an **interactive interface** for exploring your trained CS-detection model.  
It lets you:

-   Upload a TLM image
-   Generate Selective Search region proposals
-   Extract DINOv3 features
-   Score each region with your trained two-head MLP
-   Visualize the best detection, top-k candidates, and the embedding heatmap

It’s designed to be friendly, visual, and perfect for demonstrating your model to collaborators.

### Edit the configuration fields in the script:

-   CHECKPOINT → path to your trained model (e.g. best_model.pt)
-   DINO_MODEL_NAME → typically facebook/dinov3-vitb16-pretrain-lvd1689m

### Run the App

Launch Streamlit:

```
streamlit run inference_app.py
```
