# -*- coding: utf-8 -*-
# Converted from profile_accuracy_efficiency_final_vs_kf.ipynb
# Usage: edit the paths/model names/checkpoint paths near MODEL_JOBS, then run:
#     python profile_accuracy_efficiency_final_vs_kf.py


# ==============================================================================
# # Accuracy–Efficiency Profiling Notebook
#
# This notebook profiles the final model and its keyframe-sampling variant for the thesis accuracy–efficiency comparison table.
#
# It is designed for your current CE-CSL pipeline with RGB + ST-GCN skeleton + perception alignment + CLIP semantic alignment. The recommended scripts are the offline-cache pair:
#
# - `train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py`
# - `train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py`
#
# Please update the paths, display names, checkpoint paths, and existing WER/training-memory/training-time values before running.
# ==============================================================================


# ---- Cell 1 ----
# Cell 1: Basic setup
import os
import sys
import time
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Project/script folder on AutoDL. Put this file in the same folder as your training scripts.
SCRIPT_DIR = Path.cwd()

# Make imported training scripts use the current folder as CECSL_ROOT unless you have already set it manually.
os.environ.setdefault("CECSL_ROOT", str(SCRIPT_DIR))

FINAL_SCRIPT_PATH = SCRIPT_DIR / "train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py"
KF_SCRIPT_PATH = SCRIPT_DIR / "train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py"

print("Final script exists:", FINAL_SCRIPT_PATH.exists(), FINAL_SCRIPT_PATH)
print("KF script exists:", KF_SCRIPT_PATH.exists(), KF_SCRIPT_PATH)

# ---- Cell 2 ----
# Cell 2: Import training scripts safely without running main()
def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

final_mod = import_module_from_path("final_model_script", FINAL_SCRIPT_PATH)
kf_mod = import_module_from_path("kf_model_script", KF_SCRIPT_PATH)

print("Imported final experiment:", final_mod.CONFIG.get("experiment_name"))
print("Imported KF experiment:", kf_mod.CONFIG.get("experiment_name"))

# ---- Cell 3 ----
# Cell 3: Fill in model display names and checkpoint paths
# Replace these display names with the model names used in your revised thesis.

MODEL_JOBS = [
    {
        "display_name": "Final Model Name",
        "module": final_mod,
        "checkpoint_path": Path(final_mod.CONFIG["output_dir"]) / "checkpoints" / "best.pt",
        "frame_strategy": "Stride + frame cap",
        "max_frames": 96,
        "test_wer_percent": 53.30,
        "train_gpu_memory_mib": 21350,
        "training_time_epoch_min": 19.71,
        "train_batch_size": 10,  # actual training batch size
    },
    {
        "display_name": "Final Model Name + Keyframe Sampling",
        "module": kf_mod,
        "checkpoint_path": Path(kf_mod.CONFIG["output_dir"]) / "checkpoints" / "best.pt",
        "frame_strategy": "Skeleton-motion keyframe sampling",
        "max_frames": 48,
        "test_wer_percent": 55.53,
        "train_gpu_memory_mib": 11316,
        "training_time_epoch_min": 16.24,
        "train_batch_size": 10,  # actual training batch size
    },
]

for job in MODEL_JOBS:
    print(job["display_name"], "checkpoint exists:", job["checkpoint_path"].exists(), job["checkpoint_path"])

# ---- Cell 4 ----
# Cell 4: Helper functions
def clone_cfg(module, max_frames=None):
    cfg = dict(module.CONFIG)
    cfg["device"] = DEVICE
    cfg["batch_size"] = 1  # profiling inference uses batch size 1
    cfg["num_workers"] = 0
    cfg["pin_memory"] = False
    if max_frames is not None:
        cfg["max_frames"] = int(max_frames)
    return cfg

def build_model(module, cfg, checkpoint_path: Path):
    vocab = module.load_existing_vocab(cfg["vocab_path"])
    model = module.M3RGBSkeletonCLIPCTC(vocab.size, cfg).to(DEVICE)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.eval()
    return model, vocab

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def checkpoint_size_mb(path):
    return os.path.getsize(str(path)) / (1024 ** 2) if Path(path).exists() else np.nan

# ---- Cell 5 ----
# Cell 5: Build a real batch from test split
def make_loader(module, cfg):
    vocab = module.load_existing_vocab(cfg["vocab_path"])
    test_manifest = cfg.get("test_manifest")
    ds = module.CECSLDataset(test_manifest, "test", vocab, cfg)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=module.cecsl_collate_fn,
    )
    return loader

def get_first_batch(module, cfg):
    loader = make_loader(module, cfg)
    batch = next(iter(loader))
    batch["frames"] = batch["frames"].to(DEVICE)
    batch["skeleton"] = batch["skeleton"].to(DEVICE)
    batch["input_lengths"] = batch["input_lengths"].to(DEVICE)
    return batch

# This cell checks that one batch can be loaded from each model setting.
for job in MODEL_JOBS:
    cfg = clone_cfg(job["module"], max_frames=job["max_frames"])
    batch = get_first_batch(job["module"], cfg)
    print(job["display_name"])
    print(" frames:", tuple(batch["frames"].shape))
    print(" skeleton:", tuple(batch["skeleton"].shape))
    print(" input_lengths:", batch["input_lengths"].tolist())
    print(" gloss_text example:", batch["gloss_texts"][0][:80])

# ---- Cell 6 ----
# Cell 6: Inference-time measurement on actual test samples
@torch.inference_mode()
def measure_inference_time_on_loader(model, module, cfg, max_batches=100, warmup_batches=10):
    loader = make_loader(module, cfg)

    # warmup
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        frames = batch["frames"].to(DEVICE)
        skeleton = batch["skeleton"].to(DEVICE)
        input_lengths = batch["input_lengths"].to(DEVICE)
        gloss_texts = batch["gloss_texts"]
        _ = model(frames, skeleton, input_lengths, gloss_texts=gloss_texts)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # timed forward only; data loading is outside timer
    times = []
    n_samples = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch["frames"].to(DEVICE)
        skeleton = batch["skeleton"].to(DEVICE)
        input_lengths = batch["input_lengths"].to(DEVICE)
        gloss_texts = batch["gloss_texts"]

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(frames, skeleton, input_lengths, gloss_texts=gloss_texts)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        n_samples += frames.shape[0]

    avg_time = float(np.mean(times)) if times else np.nan
    std_time = float(np.std(times)) if times else np.nan
    throughput = 1.0 / avg_time if avg_time and avg_time > 0 else np.nan
    return avg_time, std_time, throughput

# ---- Cell 7 ----
# Cell 7: FLOPs calculation with THOP
# If THOP is not installed, run this once:
# !pip install thop

try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception as e:
    print("THOP is not available:", e)
    THOP_AVAILABLE = False

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model, gloss_texts):
        super().__init__()
        self.model = model
        self.gloss_texts = gloss_texts
    def forward(self, frames, skeleton, input_lengths):
        out = self.model(frames, skeleton, input_lengths, gloss_texts=self.gloss_texts)
        return out["log_probs"]

def compute_gflops(model, module, cfg):
    if not THOP_AVAILABLE:
        return np.nan
    batch = get_first_batch(module, cfg)
    frames = batch["frames"].to(DEVICE)
    skeleton = batch["skeleton"].to(DEVICE)
    input_lengths = batch["input_lengths"].to(DEVICE)
    gloss_texts = batch["gloss_texts"]

    wrapper = ForwardWrapper(model, gloss_texts).to(DEVICE).eval()
    with torch.no_grad():
        flops, _params = profile(
            wrapper,
            inputs=(frames, skeleton, input_lengths),
            verbose=False,
        )
    return float(flops) / 1e9

# ---- Cell 8 ----
# Cell 8: Run profiling and generate comparison table

def resolve_checkpoint_path(module, configured_path: Path):
    """
    Resolve checkpoint path robustly.

    It first uses the configured path. If that does not exist, it also checks
    common AutoDL folder layouts:
    1) module.CONFIG["output_dir"]/checkpoints/best.pt or latest.pt
    2) SCRIPT_DIR / experiment_name / checkpoints / best.pt or latest.pt
    3) SCRIPT_DIR / experiments / experiment_name / checkpoints / best.pt or latest.pt
    4) SCRIPT_DIR / output_dir folder name / checkpoints / best.pt or latest.pt
    """
    configured_path = Path(configured_path)
    candidates = [
        configured_path,
        configured_path.with_name("latest.pt"),
    ]

    exp_name = module.CONFIG.get("experiment_name", "")
    output_dir = Path(module.CONFIG.get("output_dir", ""))

    candidates += [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "checkpoints" / "latest.pt",
        SCRIPT_DIR / exp_name / "checkpoints" / "best.pt",
        SCRIPT_DIR / exp_name / "checkpoints" / "latest.pt",
        SCRIPT_DIR / "experiments" / exp_name / "checkpoints" / "best.pt",
        SCRIPT_DIR / "experiments" / exp_name / "checkpoints" / "latest.pt",
    ]

    if output_dir.name:
        candidates += [
            SCRIPT_DIR / output_dir.name / "checkpoints" / "best.pt",
            SCRIPT_DIR / output_dir.name / "checkpoints" / "latest.pt",
        ]

    seen = set()
    unique_candidates = []
    for p in candidates:
        p = Path(p)
        if str(p) not in seen:
            unique_candidates.append(p)
            seen.add(str(p))

    for p in unique_candidates:
        if p.exists():
            print("Resolved checkpoint:", p)
            return p

    print("Checkpoint candidates tried:")
    for p in unique_candidates:
        print(" -", p)
    raise FileNotFoundError("No checkpoint found. Please edit MODEL_JOBS checkpoint_path manually.")


results = []

for job in MODEL_JOBS:
    print("=" * 80)
    print("Profiling:", job["display_name"])

    module = job["module"]
    cfg = clone_cfg(module, max_frames=job["max_frames"])
    ckpt_path = resolve_checkpoint_path(module, Path(job["checkpoint_path"]))

    model, vocab = build_model(module, cfg, ckpt_path)

    total_params_m = count_total_params(model) / 1e6
    trainable_params_m = count_trainable_params(model) / 1e6
    ckpt_mb = checkpoint_size_mb(ckpt_path)

    try:
        gflops = compute_gflops(model, module, cfg)
    except Exception as e:
        print("FLOPs calculation failed:", repr(e))
        gflops = np.nan

    try:
        avg_t, std_t, throughput = measure_inference_time_on_loader(
            model, module, cfg, max_batches=100, warmup_batches=10
        )
    except Exception as e:
        print("Inference timing failed:", repr(e))
        avg_t, std_t, throughput = np.nan, np.nan, np.nan

    results.append({
        "Model": job["display_name"],
        "Frame Strategy": job["frame_strategy"],
        "Max Frames": job["max_frames"],
        "Test WER (%)": job["test_wer_percent"],
        "Params (M)": total_params_m,
        "Trainable Params (M)": trainable_params_m,
        "FLOPs / Sample (G)": gflops,
        "Checkpoint Size (MB)": ckpt_mb,
        f"Train GPU Memory (MiB, Batch={job['train_batch_size']})": job["train_gpu_memory_mib"],
        f"Training Time / Epoch (min, Batch={job['train_batch_size']})": job["training_time_epoch_min"],
        "Inference Time / Sample (s)": avg_t,
        "Inference Throughput (sample/s)": throughput,
    })

df = pd.DataFrame(results)
df

# ---- Cell 9 ----
# Cell 9: Save CSV and LaTeX table
output_csv = "accuracy_efficiency_final_vs_kf_batch10.csv"
output_tex = "accuracy_efficiency_final_vs_kf_batch10.tex"

df.to_csv(output_csv, index=False, encoding="utf-8-sig")

latex = df.to_latex(index=False, float_format=lambda x: f"{x:.2f}")
Path(output_tex).write_text(latex, encoding="utf-8")

print("Saved:", output_csv)
print("Saved:", output_tex)
print(latex)

# ==============================================================================
# ## Suggested thesis wording
#
# Use this table as the accuracy–efficiency comparison between the final model and the keyframe-sampling variant. Earlier staged models can remain in the ablation table for WER-only component analysis, while this profiling table focuses on the two deployment-oriented candidates.
# ==============================================================================

