# Research on Lightweight Chinese Continuous Sign Language Recognition with Semantic Alignment

## 1. Project Overview

This repository contains the implementation code for the thesis:

**Research on Lightweight Chinese Continuous Sign Language Recognition with Semantic Alignment**

The project implements a **Chinese Continuous Sign Language Recognition (CSLR)** system that predicts gloss sequences from sign language videos. The main purpose of the study is to explore a lightweight and deployment-oriented CSLR framework by combining:

- a lightweight RGB visual front-end,
- RGB and skeleton multimodal fusion,
- perception alignment between RGB and skeleton features,
- CLIP-derived semantic alignment between visual features and gloss-level text semantics,
- and keyframe sampling as an efficiency-oriented extension.

The experiments are conducted on the **CE-CSL / CE-CNSL Chinese Continuous Sign Language dataset**, which contains continuous sign language videos collected under complex real-world environments.

The project follows a staged experimental design. Each model variant adds one major component so that the contribution of each module can be evaluated clearly.

| Model Variant | Description | Main Purpose |
|---|---|---|
| **RGB-baseline** | RGB-only recognition model based on MobileNetV3, TCN, and CTC | Establish the lightweight visual baseline |
| **RGB-Skeleton Fusion** | RGB branch combined with ST-GCN skeleton branch | Verify the contribution of skeleton modality |
| **Perception-Aligned Fusion** | RGB-Skeleton Fusion with cross-modal perception alignment | Verify the role of perception-level alignment between RGB and skeleton features |
| **Semantic-Aligned Fusion** | Perception-Aligned Fusion with CLIP-derived semantic alignment | Final semantic-aligned model for gloss sequence recognition |
| **Semantic-Aligned Fusion+KF** | Semantic-Aligned Fusion with skeleton-motion keyframe sampling | Analyze the efficiency-performance trade-off of keyframe sampling |

> Note: Some script filenames still contain internal stage identifiers such as `m0`, `m1`, `m2`, and `m3`. These are retained only for code organization and experiment tracking. In the thesis and this README, the model variants are referred to by their formal names: **RGB-baseline**, **RGB-Skeleton Fusion**, **Perception-Aligned Fusion**, **Semantic-Aligned Fusion**, and **Semantic-Aligned Fusion+KF**.

---

## 2. Submission Contents

The submitted project folder contains the following categories of code.

### 2.1 Model Training Scripts

These scripts train the staged CSLR model variants used in the thesis experiments.

```text
train_m0_rgb_ctc_autodl_earlystop.py                         # RGB-baseline
train_m1_rgb_skeleton_ctc_autodl_stgcn.py                    # RGB-Skeleton Fusion
train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py              # Perception-Aligned Fusion
train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py  # Semantic-Aligned Fusion
train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py # Semantic-Aligned Fusion+KF
```

### 2.2 Data Preparation Scripts

These scripts are used to build manifests, generate the gloss vocabulary, extract skeleton keypoints, and filter invalid samples.

```text
manifest.py
gloss_map.py
sanity_check.py
extract_skeleton_tasks75_final.py
generate_final_manifest_with_valid_skeleton.py
```

### 2.3 CLIP Semantic Cache Script

This script generates the offline CLIP text embedding cache required by **Semantic-Aligned Fusion** and **Semantic-Aligned Fusion+KF**.

```text
build_clip_text_cache.py
```

Default output:

```text
/root/autodl-tmp/CE-CSL/clip_text_cache/clip_text_cache_all.pt
```

### 2.4 Evaluation and Analysis Scripts

These scripts are used for accuracy-efficiency profiling and qualitative failure case analysis.

```text
profile_accuracy_efficiency_final_vs_kf_batch10.py
export_kf_failure_cases.py
```

The profiling script generates the accuracy-efficiency comparison table for **Semantic-Aligned Fusion** and **Semantic-Aligned Fusion+KF**. It reports information such as parameters, FLOPs, inference time, GPU memory, and training time.

The failure-case script exports sample-level predictions from **Semantic-Aligned Fusion** and **Semantic-Aligned Fusion+KF** and identifies cases where keyframe sampling produces worse predictions.

### 2.5 Dataset and Checkpoints

Due to file size limitations, the dataset and trained checkpoints are **not included** in the submitted ZIP file. The code assumes that the CE-CSL dataset and checkpoints are placed in the expected AutoDL directory structure, or that the paths in the script configuration are manually updated.

---

## 3. Recommended Project Structure

A recommended folder structure is shown below.

```text
project/
│
├── training/
│   ├── train_m0_rgb_ctc_autodl_earlystop.py
│   ├── train_m1_rgb_skeleton_ctc_autodl_stgcn.py
│   ├── train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py
│   ├── train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
│   └── train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
│
├── preprocessing/
│   ├── manifest.py
│   ├── gloss_map.py
│   ├── sanity_check.py
│   ├── extract_skeleton_tasks75_final.py
│   └── generate_final_manifest_with_valid_skeleton.py
│
├── semantic_cache/
│   └── build_clip_text_cache.py
│
├── evaluation/
│   ├── profile_accuracy_efficiency_final_vs_kf_batch10.py
│   └── export_kf_failure_cases.py
│
├── manifests/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── train_final.jsonl
│   ├── dev_final.jsonl
│   └── test_final.jsonl
│
├── skeleton_tasks75/
│   ├── train/
│   ├── dev/
│   └── test/
│
├── clip_text_cache/
│   └── clip_text_cache_all.pt
│
├── experiments/
│   └── saved checkpoints
│
├── results/
│   └── result screenshots, CSV files, and LaTeX tables
│
└── README.md
```

The scripts can also be placed in the same AutoDL working directory. In that case, make sure the paths in each script are consistent.

---

## 4. Environment Setup

### 4.1 Python Version

Python **3.9+** is recommended.

### 4.2 Main Dependencies

Recommended installation commands:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas opencv-python mediapipe
pip install thop
```

For CLIP text embedding generation, install at least one of the following options depending on the backend used in the semantic-aligned scripts:

```bash
pip install open_clip_torch
```

or

```bash
pip install transformers
```

Recommended hardware environment:

```text
CUDA-enabled GPU
PyTorch GPU version
AutoDL or another Linux-based training environment
```

The default dataset root used in the scripts is:

```text
/root/autodl-tmp/CE-CSL
```

It can also be overridden by setting the environment variable:

```bash
export CECSL_ROOT=/path/to/CE-CSL
```

---

## 5. Dataset

The experiments are conducted on the **CE-CSL / CE-CNSL Chinese Continuous Sign Language dataset**.

Dataset source:

```text
https://github.com/woshisad159/TFNet
```

Dataset split used in this project:

| Split | Samples |
|---|---:|
| Train | 4,973 |
| Dev | 515 |
| Test | 500 |

Each sample contains:

```text
video path
gloss sequence
Chinese sentence
signer information
other annotation information
```

The gloss sequence is used as the target output for CSLR. The primary evaluation metric is **Word Error Rate (WER)**.

---

## 6. Data Preprocessing Pipeline

### Step 1: Generate Dataset Manifest

```bash
python manifest.py
```

Expected outputs:

```text
manifests/train.jsonl
manifests/dev.jsonl
manifests/test.jsonl
```

The manifest records the sample ID, split, video path, signer ID, gloss sequence, Chinese sentence, translator, and notes.

### Step 2: Run Sanity Check

```bash
python sanity_check.py
```

This script checks basic dataset information, including:

- gloss length distribution,
- vocabulary statistics,
- sampled video readability,
- frame count distribution,
- possible abnormal samples.

### Step 3: Build Gloss Vocabulary

```bash
python gloss_map.py
```

This script builds the gloss vocabulary from the training split and saves it as:

```text
gloss_vocab.json
```

The vocabulary is used by the CTC-based recognition models.

### Step 4: Extract Skeleton Keypoints

```bash
python extract_skeleton_tasks75_final.py
```

This script extracts 75 skeleton keypoints per frame using MediaPipe Tasks:

```text
33 pose keypoints + 21 left-hand keypoints + 21 right-hand keypoints = 75 keypoints
```

Each skeleton file is saved as a NumPy array:

```text
[F, 75, 3]
```

where `F` is the number of sampled frames, and the three channels represent keypoint coordinates and confidence/presence information.

### Step 5: Filter Invalid Samples

```bash
python generate_final_manifest_with_valid_skeleton.py
```

This script removes samples with problems such as:

- empty gloss target,
- unreadable video,
- invalid frame count,
- CTC length mismatch,
- missing or invalid skeleton files,
- NaN or infinite skeleton values.

Expected outputs:

```text
manifests/train_final.jsonl
manifests/dev_final.jsonl
manifests/test_final.jsonl
```

### Step 6: Build Offline CLIP Text Cache

Before training **Semantic-Aligned Fusion** or **Semantic-Aligned Fusion+KF**, build the offline CLIP text embedding cache:

```bash
python build_clip_text_cache.py
```

Default output:

```text
/root/autodl-tmp/CE-CSL/clip_text_cache/clip_text_cache_all.pt
```

This script reads the train/dev/test manifests, converts each gloss sequence into the same `gloss_text` format used during training, encodes the unique gloss texts with CLIP, and saves the resulting text embeddings.

This step is required for:

```text
train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
```

---

## 7. Model Training

### 7.1 Common Settings

The main training settings used in the experiments are:

```text
Epochs: 55
Batch size: 10
Learning rate: 1e-4
Optimizer: AdamW
Image size: 224 × 224
RGB frame encoder: MobileNetV3-Small
Skeleton encoder: ST-GCN
Temporal modeling: TCN
Sequence loss: CTC loss
Decoding: CTC greedy decoding
```

### 7.2 Training Commands

Train **RGB-baseline**:

```bash
python train_m0_rgb_ctc_autodl_earlystop.py
```

Train **RGB-Skeleton Fusion**:

```bash
python train_m1_rgb_skeleton_ctc_autodl_stgcn.py
```

Train **Perception-Aligned Fusion**:

```bash
python train_m2_rgb_skeleton_align_ctc_autodl_stgcn.py
```

Build offline CLIP semantic cache before training the semantic-aligned variants:

```bash
python build_clip_text_cache.py
```

Train **Semantic-Aligned Fusion**:

```bash
python train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
```

Train **Semantic-Aligned Fusion+KF**:

```bash
python train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
```

---

## 8. Evaluation and Analysis

### 8.1 Recognition Metric

The primary evaluation metric is **Word Error Rate (WER)**:

```text
WER = (Substitutions + Deletions + Insertions) / Number of Reference Glosses
```

The experiments evaluate model performance on the dev and test sets.

### 8.2 Accuracy-Efficiency Profiling

To generate the accuracy-efficiency comparison between **Semantic-Aligned Fusion** and **Semantic-Aligned Fusion+KF**, run:

```bash
python profile_accuracy_efficiency_final_vs_kf_batch10.py
```

This script profiles:

- parameter count,
- trainable parameter count,
- FLOPs per sample,
- checkpoint size,
- GPU memory usage,
- training time per epoch,
- inference time per sample,
- inference throughput.

Expected outputs:

```text
accuracy_efficiency_final_vs_kf_batch10.csv
accuracy_efficiency_final_vs_kf_batch10.tex
```

Before running this script, check the following fields inside `MODEL_JOBS`:

```text
display_name
checkpoint_path
test_wer_percent
train_gpu_memory_mib
training_time_epoch_min
```

The display names should be set as:

```text
Semantic-Aligned Fusion
Semantic-Aligned Fusion+KF
```

### 8.3 Keyframe Failure Case Export

To export qualitative failure cases for **Semantic-Aligned Fusion** and **Semantic-Aligned Fusion+KF**, run:

```bash
python export_kf_failure_cases.py
```

This script compares sample-level predictions between the final semantic-aligned model and the keyframe-sampling variant. It helps identify cases where keyframe sampling removes useful motion transitions, gloss boundaries, or long-range temporal cues.

Example outputs may include:

```text
kf_predictions.json
kf_failure_candidates_top.csv
suggested_failure_cases_for_paper.csv
suggested_failure_cases_for_paper.json
```

---

## 9. Reported Experimental Results

The following values summarize the main reported results in the thesis. The exact values may vary if the models are retrained with different hardware, checkpoints, or random seeds.

| Model Variant | Main Component | Test WER (%) |
|---|---|---:|
| RGB-baseline | RGB visual baseline | 75.31 |
| RGB-Skeleton Fusion | RGB + skeleton multimodal fusion | 74.64 |
| Perception-Aligned Fusion | RGB-skeleton perception alignment | 73.97 |
| Semantic-Aligned Fusion | CLIP-derived semantic alignment | 53.30 |
| Semantic-Aligned Fusion+KF | Keyframe sampling extension | 55.53 |

Efficiency comparison between the final model and the keyframe-sampling variant:

| Model Variant | Frame Strategy | Max Frames | Test WER (%) | Training Time / Epoch | GPU Memory |
|---|---|---:|---:|---:|---:|
| Semantic-Aligned Fusion | Stride + frame cap | 96 | 53.30 | 19.71 min | 21,350 MiB |
| Semantic-Aligned Fusion+KF | Skeleton-motion keyframe sampling | 48 | 55.53 | 16.24 min | 11,316 MiB |

The results show that semantic alignment significantly improves recognition performance, while keyframe sampling reduces computational and memory costs at the expense of a small decrease in test-set recognition accuracy.

---

## 10. Reproducibility Steps

To reproduce the full experimental pipeline:

```text
1. Install dependencies.
2. Download and place the CE-CSL dataset under the expected dataset root.
3. Run manifest.py to generate initial manifests.
4. Run sanity_check.py to inspect dataset quality.
5. Run gloss_map.py to build gloss_vocab.json.
6. Run extract_skeleton_tasks75_final.py to extract skeleton sequences.
7. Run generate_final_manifest_with_valid_skeleton.py to create final valid manifests.
8. Run build_clip_text_cache.py to generate offline CLIP text embeddings.
9. Train RGB-baseline, RGB-Skeleton Fusion, Perception-Aligned Fusion, Semantic-Aligned Fusion, and Semantic-Aligned Fusion+KF using the corresponding training scripts.
10. Run profile_accuracy_efficiency_final_vs_kf_batch10.py for efficiency profiling.
11. Run export_kf_failure_cases.py for qualitative keyframe failure analysis.
```

---

## 11. Important Notes

1. The dataset and trained checkpoints are not included due to size limitations.
2. The default code paths are designed for the AutoDL environment. If the code is run locally or on another server, update `CECSL_ROOT` or modify the paths in the `CONFIG` dictionaries.
3. The semantic-aligned scripts require the offline CLIP cache file. If `clip_text_cache_all.pt` is missing, run `build_clip_text_cache.py` first.
4. The profiling script depends on valid trained checkpoints. If the checkpoint paths are different, update `MODEL_JOBS` before running.
5. This project focuses on CSLR gloss recognition rather than end-to-end sign language translation.
6. Keyframe sampling is used as an efficiency-oriented extension. It is not claimed to always improve recognition accuracy.
