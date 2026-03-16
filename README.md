# Research-on-Cross-Domain-Robust-Recognition-and-Lightweight-Model-of-Chinese-Sign-Language

## 1. Project Overview

This project implements a **Continuous Sign Language Recognition
(CSLR)** system using deep learning.\
The goal is to recognize **gloss sequences from sign language videos**
using multi-modal learning.

The system is evaluated on the **CE-CSL Chinese Continuous Sign Language
dataset**.

The project explores progressively more advanced architectures:

  Model       Description
  ----------- ------------------------------------------
  **M0**      RGB baseline (MobileNetV3 + TCN + CTC)
  **M1**      RGB + Skeleton fusion
  **M2**      RGB + Skeleton with Perception Alignment
  **M3**      RGB + Skeleton + CLIP Semantic Alignment
  **M3+KF**   M3 with Keyframe Sampling

Each stage evaluates the contribution of additional modalities or
alignment strategies.

------------------------------------------------------------------------

# 2. Submission Contents

The submitted **ZIP folder** contains the following:

### 1. Complete Implementation Code

All source code required to reproduce the experiments:

    train_m0_rgb_ctc_autodl_earlystop.py
    train_m1_rgb_skeleton_ctc_autodl_stgcn.py
    train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py
    train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py
    train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py

### 2. Data Processing Scripts

    manifest.py
    generate_final_manifest_with_valid_skeleton.py
    gloss_map.py
    extract_skeleton_mediapipe.py
    sanity_check.py

### 3. Dataset

Due to dataset size limitations, the dataset is **not included in the
zip file**.

Instructions for accessing the dataset are provided below.

### 4. Experimental Results

Screenshots documenting key results including:

-   training logs
-   model comparison tables
-   WER results
-   GPU usage

These screenshots are stored in:

    results/

------------------------------------------------------------------------

# 3. Project Structure

    project/
    │
    ├── training scripts
    │   ├── train_m0_rgb_ctc_autodl_earlystop.py
    │   ├── train_m1_rgb_skeleton_ctc_autodl_stgcn.py
    │   ├── train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py
    │   ├── train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py
    │   └── train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py
    │
    ├── preprocessing
    │   ├── manifest.py
    │   ├── generate_final_manifest_with_valid_skeleton.py
    │   ├── gloss_map.py
    │   ├── extract_skeleton_mediapipe.py
    │   └── sanity_check.py
    │
    ├── manifests/
    │   ├── train.jsonl
    │   ├── dev.jsonl
    │   └── test.jsonl
    │
    ├── skeleton_tasks75/
    │   └── skeleton keypoints
    │
    ├── experiments/
    │   └── saved checkpoints
    │
    ├── results/
    │   └── experiment screenshots
    │
    ├── github_link.txt
    │
    └── README.md

------------------------------------------------------------------------

# 4. Environment Setup

## Python Version

Python **3.9+** is recommended.

------------------------------------------------------------------------

## Install Dependencies

    pip install torch torchvision torchaudio
    pip install numpy
    pip install opencv-python
    pip install mediapipe
    pip install openai-clip

Recommended environment:

    CUDA >= 11
    PyTorch GPU version

------------------------------------------------------------------------

# 5. Dataset

## CE-CSL Dataset

Experiments are conducted using the **Chinese Continuous Sign Language
Dataset (CE-CSL)**.

Dataset repository:

https://github.com/woshisad159/TFNet

Dataset statistics:

  Split   Samples
  ------- ---------
  Train   4973
  Dev     515
  Test    500

Video properties:

-   Format: MP4
-   Frame length: 30--530 frames
-   Diverse real-world backgrounds

Each sample contains:

    video
    gloss sequence
    Chinese sentence
    signer ID

Example gloss:

    10 / 年 / 鱼 / 禁止1 / 区 / 时间 / 长 / 不

------------------------------------------------------------------------

# 6. Data Preprocessing

## Step 1 --- Generate Dataset Manifest

    python manifest.py

Outputs:

    manifests/train.jsonl
    manifests/dev.jsonl
    manifests/test.jsonl

------------------------------------------------------------------------

## Step 2 --- Dataset Sanity Check

    python sanity_check.py

Checks:

-   corrupted videos
-   frame distribution
-   gloss distribution

------------------------------------------------------------------------

## Step 3 --- Build Gloss Vocabulary

    python gloss_map.py

Example output:

    {
    "<blank>":0,
    "禁止":1,
    "鱼":2,
    "区域":3
    }

------------------------------------------------------------------------

## Step 4 --- Extract Skeleton Features

    python extract_skeleton_mediapipe.py

Each frame produces:

    75 keypoints
    (33 pose + 21 left hand + 21 right hand)

Saved format:

    [F, 75, 3]

------------------------------------------------------------------------
## Step 5 --- Filter Invalid Skeleton Samples

    python generate_final_manifest_with_valid_skeleton.py

Delete the samples where skeleton extraction failed
Only retain the valid skeleton data

output:

    train_final.jsonl
    dev_final.jsonl
    test_final.jsonl

# 7. Model Training

Common architecture:

    MobileNetV3
       ↓
    Temporal Convolution Network
       ↓
    CTC Loss

Training configuration:

    Epochs: 55
    Batch size: 2–10
    Learning rate: 1e-4
    Optimizer: AdamW

------------------------------------------------------------------------

# 8. Training Commands

### Train M0

    python train_m0_rgb_ctc_autodl_earlystop.py

### Train M1

    python train_m1_rgb_skeleton_ctc_autodl_stgcn.py

### Train M2

    python train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py

### Train M3

    python train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py

### Train M3 + Keyframe Sampling

    python train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offline.py

------------------------------------------------------------------------

# 9. Evaluation

Primary evaluation metric:

### Word Error Rate (WER)

    WER = (Substitution + Deletion + Insertion) / Reference Words

Evaluation is conducted on:

    Dev set
    Test set

------------------------------------------------------------------------

# 10. Hardware

Experiments were conducted using:

    GPU: NVIDIA RTX 5090
    CUDA: 12.x
    RAM: 32GB
    Framework: PyTorch

Training time may vary depending on GPU performance.

------------------------------------------------------------------------

# 11. Reproducibility

To reproduce the experiments:

    1 Install dependencies
    2 Download CE-CSL dataset
    3 Run manifest.py
    4 Run gloss_map.py
    5 Extract skeleton features
    6 Train models (M0 → M3 → M3+KF)
    7 Evaluate using WER
