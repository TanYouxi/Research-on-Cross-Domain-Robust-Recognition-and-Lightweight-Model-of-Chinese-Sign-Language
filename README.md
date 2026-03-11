# Research-on-Cross-Domain-Robust-Recognition-and-Lightweight-Model-of-Chinese-Sign-Language

## 1. Project Overview

This project implements a continuous sign language recognition system
based on the CE‑CSL dataset. The goal is to recognize gloss sequences
from sign language videos using deep learning models.

The project explores multiple model architectures with increasing
complexity:

-   **M0:** RGB baseline (MobileNetV3 + TCN + CTC)
-   **M1:** RGB + Skeleton fusion
-   **M2:** RGB + Skeleton with Cross‑Modal Perception Alignment
-   **M3:** RGB + Skeleton + CLIP Semantic Alignment

These models progressively improve recognition performance by
incorporating multi‑modal features and semantic alignment.

------------------------------------------------------------------------

## 2. Project Structure

    project/
    │
    ├── M0.py                     # RGB baseline model
    ├── M1.py                     # RGB + Skeleton fusion model
    ├── M2.py                     # Cross‑modal alignment model
    ├── M3.py                     # CLIP semantic alignment model
    │
    ├── extract_skeleton.py       # MediaPipe skeleton extraction
    ├── gloss_map.py              # Gloss vocabulary construction
    ├── manifest.py               # Dataset manifest generation
    ├── sanity_check.py           # Dataset validation
    │
    ├── manifests/                # train/dev/test jsonl files
    │
    ├── skeleton_tasks75/         # extracted skeleton data
    │
    └── README.md

------------------------------------------------------------------------

## 3. Environment Setup

### Python Version

Python 3.9 or above is recommended.

### Required Libraries

Install dependencies using pip:

    pip install torch torchvision torchaudio
    pip install opencv-python
    pip install numpy
    pip install mediapipe
    pip install openai-clip

Recommended environment:

-   CUDA 11+
-   PyTorch GPU version

------------------------------------------------------------------------

## 4. Dataset

### Dataset Used

Experiments are conducted using the **CE‑CSL Chinese Continuous Sign
Language dataset**. https://github.com/woshisad159/TFNet

Dataset statistics:

-   Train: 4973 videos
-   Dev: 515 videos
-   Test: 500 videos

Video characteristics:

-   Format: MP4
-   Frame length: approximately 30 -- 530 frames
-   Diverse real-world backgrounds

Each sample contains:

-   sign language video
-   gloss sequence
-   Chinese sentence
-   signer ID

Example gloss format:

    10 / 年 / 鱼 / 禁止1 / 区 / 时间 / 长 / 不

------------------------------------------------------------------------

## 5. Data Preprocessing

### Step 1 -- Generate Dataset Manifest

Create JSONL manifest files:

    python manifest.py

This generates:

    manifests/train.jsonl
    manifests/dev.jsonl
    manifests/test.jsonl

Example entry:

    {
    "id": "train-00001",
    "video": "path/to/video.mp4",
    "gloss": ["鱼","禁止"],
    "sentence": "...",
    "signer": "A"
    }

------------------------------------------------------------------------

### Step 2 -- Dataset Sanity Check

Verify dataset integrity:

    python sanity_check.py

This script checks:

-   unreadable videos
-   gloss length distribution
-   frame count distribution
-   fps information

------------------------------------------------------------------------

### Step 3 -- Build Gloss Vocabulary

Generate vocabulary for CTC training:

    python gloss_map.py

Output example:

    {
    "<blank>":0,
    "禁止":1,
    "鱼":2,
    "区域":3
    }

------------------------------------------------------------------------

### Step 4 -- Extract Skeleton Features

Skeleton features are extracted using **MediaPipe Tasks**.

    python extract_skeleton.py

Each frame produces:

-   75 keypoints\
    (pose 33 + left hand 21 + right hand 21)

Saved format:

    [F, 75, 3]

Output directory:

    skeleton_tasks75/

------------------------------------------------------------------------

## 6. Model Training

All models share a similar backbone structure:

-   MobileNetV3 frame encoder
-   Temporal Convolution Network (TCN)
-   CTC Loss for sequence prediction

Training configuration:

-   Epochs: 55
-   Batch size: 2
-   Learning rate: 1e‑4
-   Optimizer: AdamW

------------------------------------------------------------------------

### Train Baseline Model (M0)

RGB only.

    python M0.py

Architecture:

RGB → MobileNetV3 → TCN → CTC

------------------------------------------------------------------------

### Train RGB + Skeleton Model (M1)

    python M1.py

Architecture:

RGB encoder + Skeleton encoder → Fusion → TCN → CTC

------------------------------------------------------------------------

### Train Cross‑Modal Alignment Model (M2)

    python M2.py

Adds cross‑modal attention to align RGB and skeleton features.

------------------------------------------------------------------------

### Train CLIP‑Aligned Model (M3)

    python M3.py

Additional supervision using CLIP semantic embeddings.

Loss function:

    Total Loss = CTC Loss + λ × CLIP Alignment Loss

------------------------------------------------------------------------

## 7. Evaluation

Primary evaluation metric:

**Word Error Rate (WER)**

    WER = (Substitution + Deletion + Insertion) / Reference Words

Additional metrics (in some models):

-   Accuracy
-   Precision
-   Recall
-   F1 Score

------------------------------------------------------------------------

## 8. Experimental Results

Training logs and example outputs are provided in:

    results/

These include:

-   training curves
-   WER evaluation results
-   model checkpoints

------------------------------------------------------------------------

## 9. Reproducibility

To reproduce the experiments:

1.  Install dependencies
2.  Download the dataset
3.  Run `manifest.py`
4.  Run `gloss_map.py`
5.  Run `extract_skeleton.py`
6.  Train models (`M0 → M3`)

------------------------------------------------------------------------

## 10. Hardware

Experiments were conducted using:

-   GPU: NVIDIA RTX 3090Ti
-   Memory: 24GB
-   Framework: PyTorch

Training time may vary depending on GPU performance.
