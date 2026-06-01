# -*- coding: utf-8 -*-
"""
M3+KF-offline-cache: RGB + ST-GCN Skeleton + Perception Alignment + Offline-CLIP Semantic Alignment + Keyframe Sampling for CE-CSL
AutoDL-ready version

Design goals:
- Keep the M2 training/inference pipeline stable
- Add CLIP-based text semantic alignment and skeleton-motion-based keyframe sampling
- Preserve AutoDL/Linux path defaults and robust checkpoint / early-stop logic
- Support either open_clip or HuggingFace transformers backends for CLIP text encoding

Recommended package options (one of them is enough):
1) pip install open_clip_torch
2) pip install transformers
"""
import ast
import os
import json
import time
import random

# Avoid invalid or empty OMP setting on AutoDL
if not str(os.environ.get("OMP_NUM_THREADS", "")).isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


DATA_ROOT = os.environ.get("CECSL_ROOT", "/root/autodl-tmp/CE-CSL")

CONFIG = {
    "base_dir": DATA_ROOT,
    "train_manifest": f"{DATA_ROOT}/manifests/train_final.jsonl",
    "dev_manifest": f"{DATA_ROOT}/manifests/dev_final.jsonl",
    "test_manifest": f"{DATA_ROOT}/manifests/test.jsonl",
    "skeleton_root": f"{DATA_ROOT}/skeleton_tasks75",
    "gloss_key": "gloss",
    "video_key": "video",
    "id_key": "id",
    "vocab_path": f"{DATA_ROOT}/gloss_vocab.json",
    "experiment_name": "m3_kf_rgb_skel_align_clip_stgcn_offcache",
    "output_dir": f"{DATA_ROOT}/experiments/m3_kf_rgb_skel_align_clip_stgcn_offcache",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 55,
    "batch_size": 10,
    "num_workers": 8,
    "pin_memory": True,
    "resume_path": f"{DATA_ROOT}/experiments/m3_kf_v2_rgb_skel_align_clip_stgcn_offcache/checkpoints/latest.pt",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "use_scheduler": True,
    "scheduler_type": "multistep",
    "milestones": [12, 18],
    "gamma": 0.5,
    "image_size": 224,
    "sample_stride": 2,
    "max_frames": 48,
    "use_keyframe_sampling": True,
    "keyframe_candidate_max_frames": 96,
    "keyframe_num_segments": 48,
    "keyframe_keep_first": True,
    "keyframe_keep_last": True,
    "keyframe_score_smoothing": 3,
    "keyframe_use_visibility_mask": False,
    "min_frames": 8,
    "normalize_with_imagenet": True,
    "skeleton_num_joints": 75,
    "skeleton_channels": 3,
    "skeleton_input_dim": 225,
    "skeleton_hidden_dim": 256,
    "skeleton_num_layers": 2,
    "skeleton_dropout": 0.2,
    "skeleton_encoder_type": "stgcn",
    "stgcn_num_blocks": 4,
    "stgcn_temporal_kernel": 9,
    "stgcn_residual": True,
    "frame_feature_dim": 576,
    "fusion_dim": 512,
    "tcn_hidden_dim": 512,
    "tcn_num_layers": 4,
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.2,
    "use_perception_alignment": True,
    "align_num_heads": 4,
    "align_dropout": 0.1,
    "align_loss_weight": 0.1,
    "use_clip_semantic_alignment": True,
    "clip_backend": "auto",  # auto | open_clip | transformers
    "clip_model_name": "ViT-B-32",  # open_clip: ViT-B-32 ; transformers fallback ignores this
    "clip_pretrained": "openai",    # open_clip pretrained tag
    "clip_local_weight_path": "/root/autodl-tmp/clip_weights/open_clip_pytorch_model.bin",
    "clip_hf_model_name": "openai/clip-vit-base-patch32",
    "clip_local_files_only": True,
    "clip_cache_text_embeddings": True,
    "clip_text_trainable": False,
    "clip_semantic_dim": 512,
    "clip_num_heads": 4,
    "clip_dropout": 0.1,
    "semantic_loss_weight": 0.1,
    "semantic_loss_type": "cosine",
    "use_offline_clip_text_cache": True,
    "offline_clip_cache_path": f"{DATA_ROOT}/clip_text_cache/clip_text_cache_all.pt",
    "log_interval": 20,
    "test_after_training": True,
    "early_stopping": True,
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 0.001,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Dict[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_video_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def sample_frame_indices(num_frames: int, stride: int, max_frames: Optional[int]) -> List[int]:
    idxs = list(range(0, num_frames, max(1, stride)))
    if max_frames is not None and len(idxs) > max_frames:
        base = idxs
        pick = np.linspace(0, len(base) - 1, max_frames).round().astype(int).tolist()
        idxs = [base[i] for i in pick]
    return idxs


def temporal_resize_indices(old_len: int, new_len: int) -> List[int]:
    if old_len <= 0:
        return [0] * new_len
    if old_len == new_len:
        return list(range(old_len))
    return np.linspace(0, old_len - 1, new_len).round().astype(int).tolist()


def smooth_1d_scores(scores: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1 or scores.size <= 1:
        return scores
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(padded, kernel, mode="valid")


def compute_skeleton_motion_scores(skeleton: np.ndarray, use_visibility_mask: bool = False, smoothing: int = 3) -> np.ndarray:
    # skeleton: [T, J, C], C>=3 (x,y,z[,vis])
    t = skeleton.shape[0]
    if t <= 1:
        return np.zeros((t,), dtype=np.float32)
    xyz = skeleton[..., :3].astype(np.float32)
    diffs = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)  # [T-1, J]
    if use_visibility_mask and skeleton.shape[-1] >= 4:
        vis = skeleton[..., 3].astype(np.float32)
        vis_pair = np.minimum(vis[1:], vis[:-1])
        diffs = diffs * vis_pair
    scores = np.zeros((t,), dtype=np.float32)
    scores[1:] = diffs.mean(axis=1)
    scores = smooth_1d_scores(scores, kernel_size=smoothing)
    return scores


def select_keyframe_indices_from_scores(scores: np.ndarray, target_len: int, keep_first: bool = True, keep_last: bool = True) -> List[int]:
    n = int(scores.shape[0])
    if n <= 0:
        return []
    if target_len is None or target_len <= 0 or n <= target_len:
        return list(range(n))

    reserved = []
    if keep_first:
        reserved.append(0)
    if keep_last and n > 1:
        reserved.append(n - 1)
    reserved = sorted(set(reserved))
    remaining = max(0, target_len - len(reserved))
    if remaining <= 0:
        return reserved[:target_len]

    candidate_mask = np.ones((n,), dtype=bool)
    if reserved:
        candidate_mask[reserved] = False
    candidate_positions = np.nonzero(candidate_mask)[0]
    if candidate_positions.size == 0:
        return reserved

    # temporal segmentation + local peak frame selection
    seg_edges = np.linspace(0, candidate_positions.size, remaining + 1).round().astype(int)
    picked = []
    for s, e in zip(seg_edges[:-1], seg_edges[1:]):
        if e <= s:
            continue
        seg_pos = candidate_positions[s:e]
        local_scores = scores[seg_pos]
        picked.append(int(seg_pos[int(np.argmax(local_scores))]))

    selected = sorted(set(reserved + picked))
    if len(selected) < target_len:
        extras = [int(i) for i in np.argsort(scores)[::-1] if int(i) not in selected]
        selected.extend(extras[: target_len - len(selected)])
        selected = sorted(set(selected))
    if len(selected) > target_len:
        selected = selected[:target_len]
    return selected


def ensure_gloss_tokens(gloss: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    if gloss is None:
        return []
    if isinstance(gloss, list):
        return [str(x).strip() for x in gloss if str(x).strip()]
    if isinstance(gloss, tuple):
        return [str(x).strip() for x in gloss if str(x).strip()]
    text = str(gloss).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            val = ast.literal_eval(text)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
    return [t for t in text.split() if t]


def edit_distance(ref: List[int], hyp: List[int]) -> Tuple[int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "E"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(choices, key=lambda x: x[0])
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "E":
            i -= 1
            j -= 1
        elif cur == "S":
            s += 1
            i -= 1
            j -= 1
        elif cur == "D":
            d += 1
            i -= 1
        elif cur == "I":
            ins += 1
            j -= 1
        else:
            break
    return s, d, ins


def compute_wer(refs: List[List[int]], hyps: List[List[int]]) -> float:
    total_s = total_d = total_i = total_ref = 0
    for ref, hyp in zip(refs, hyps):
        s, d, i = edit_distance(ref, hyp)
        total_s += s
        total_d += d
        total_i += i
        total_ref += len(ref)
    return (total_s + total_d + total_i) / total_ref if total_ref > 0 else 0.0


def compute_token_metrics(refs: List[List[int]], hyps: List[List[int]]) -> Dict[str, float]:
    total_ref = total_pred = total_correct = 0
    for ref, hyp in zip(refs, hyps):
        s, d, i = edit_distance(ref, hyp)
        total_ref += len(ref)
        total_pred += len(hyp)
        total_correct += max(0, len(ref) - s - d)
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = total_correct / total_ref if total_ref > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class CTCVocab:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.blank_id = token_to_id["<BLANK>"]
        self.unk_id = token_to_id["<UNK>"]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, gloss: Union[str, List[str], Tuple[str, ...], None]) -> List[int]:
        tokens = ensure_gloss_tokens(gloss)
        unk_id = self.token_to_id["<UNK>"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]


def load_existing_vocab(vocab_path: str) -> CTCVocab:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    return CTCVocab(token_to_id)


class CECSLDataset(Dataset):
    def __init__(self, manifest_path: str, split_name: str, vocab: CTCVocab, cfg: Dict[str, Any]):
        self.items = read_jsonl(manifest_path)
        self.split_name = split_name
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_video_path(self, video_path: str) -> str:
        if os.path.exists(video_path):
            return video_path
        vp = str(video_path).replace("\\", "/")
        old_prefix = "E:/CE-CSL/CE-CSL"
        if vp.startswith(old_prefix):
            candidate = vp.replace(old_prefix, self.cfg["base_dir"], 1)
            if os.path.exists(candidate):
                return candidate
        marker = "/video/"
        if marker in vp:
            tail = vp.split(marker, 1)[1]
            candidate = os.path.join(self.cfg["base_dir"], "video", tail)
            if os.path.exists(candidate):
                return candidate
        return video_path

    def _read_rgb_frames(self, video_path: str, candidate_max_frames: Optional[int] = None) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = candidate_max_frames if candidate_max_frames is not None else self.cfg["max_frames"]
        idxs = set(sample_frame_indices(num_frames if num_frames > 0 else 999999, self.cfg["sample_stride"], max_frames))
        selected = []
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if num_frames <= 0 or fi in idxs:
                selected.append(frame)
            fi += 1
        cap.release()
        if len(selected) == 0:
            raise RuntimeError(f"No selected RGB frames: {video_path}")
        return np.stack(selected, axis=0)

    def _read_skeleton(self, sample_id: str, target_len: int) -> np.ndarray:
        sk_path = os.path.join(self.cfg["skeleton_root"], self.split_name, f"{sample_id}.npy")
        if not os.path.exists(sk_path):
            raise FileNotFoundError(f"Skeleton not found: {sk_path}")
        arr = np.load(sk_path)
        if arr.ndim != 3 or arr.shape[1] != self.cfg["skeleton_num_joints"] or arr.shape[2] < self.cfg["skeleton_channels"]:
            raise RuntimeError(f"Unexpected skeleton shape for {sk_path}: {arr.shape}")
        idxs = temporal_resize_indices(arr.shape[0], target_len)
        arr = arr[idxs]
        return arr.astype(np.float32)

    def _apply_keyframe_sampling(self, rgb_frames_np: np.ndarray, skeleton_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        candidate_len = rgb_frames_np.shape[0]
        target_len = int(self.cfg.get("keyframe_num_segments", self.cfg["max_frames"]))
        scores = compute_skeleton_motion_scores(
            skeleton_np,
            use_visibility_mask=self.cfg.get("keyframe_use_visibility_mask", False),
            smoothing=self.cfg.get("keyframe_score_smoothing", 3),
        )
        selected = select_keyframe_indices_from_scores(
            scores,
            target_len=target_len,
            keep_first=self.cfg.get("keyframe_keep_first", True),
            keep_last=self.cfg.get("keyframe_keep_last", True),
        )
        rgb_frames_np = rgb_frames_np[selected]
        skeleton_np = skeleton_np[selected]
        return rgb_frames_np, skeleton_np, selected

    def _postprocess_frames_and_skeleton(self, rgb_frames_np: np.ndarray, skeleton_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        while rgb_frames_np.shape[0] < self.cfg["min_frames"]:
            rgb_frames_np = np.concatenate([rgb_frames_np, rgb_frames_np[-1:]], axis=0)
            skeleton_np = np.concatenate([skeleton_np, skeleton_np[-1:]], axis=0)
        frames = []
        for frame in rgb_frames_np:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.cfg["image_size"], self.cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        rgb_frames = torch.from_numpy(np.stack(frames, axis=0))
        skeleton = torch.from_numpy(skeleton_np[..., : self.cfg["skeleton_channels"]].astype(np.float32))
        return rgb_frames, skeleton

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        sample_id = rec[self.cfg["id_key"]]
        video_path = self._resolve_video_path(rec[self.cfg["video_key"]])
        gloss = rec.get(self.cfg["gloss_key"], [])
        gloss_tokens = ensure_gloss_tokens(gloss)

        candidate_max_frames = self.cfg.get("keyframe_candidate_max_frames", self.cfg["max_frames"]) if self.cfg.get("use_keyframe_sampling", False) else self.cfg["max_frames"]
        rgb_frames_np = self._read_rgb_frames(video_path, candidate_max_frames=candidate_max_frames)
        skeleton_np = self._read_skeleton(sample_id, rgb_frames_np.shape[0])

        if self.cfg.get("use_keyframe_sampling", False):
            rgb_frames_np, skeleton_np, selected_idx = self._apply_keyframe_sampling(rgb_frames_np, skeleton_np)
        else:
            selected_idx = list(range(rgb_frames_np.shape[0]))

        rgb_frames, skeleton = self._postprocess_frames_and_skeleton(rgb_frames_np, skeleton_np)
        if self.cfg["normalize_with_imagenet"]:
            rgb_frames = normalize_video_tensor(rgb_frames)
        t = rgb_frames.shape[0]
        target_ids = self.vocab.encode(gloss_tokens)
        return {
            "id": sample_id,
            "video": video_path,
            "frames": rgb_frames,
            "skeleton": skeleton,
            "input_length": t,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_length": len(target_ids),
            "gloss_text": " ".join(gloss_tokens),
            "selected_keyframes": selected_idx,
        }


def cecsl_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = sorted(batch, key=lambda x: x["input_length"], reverse=True)
    b = len(batch)
    t_max = max(x["input_length"] for x in batch)
    c, h, w = batch[0]["frames"].shape[1:]
    j, k = batch[0]["skeleton"].shape[1:]
    frames = torch.zeros((b, t_max, c, h, w), dtype=batch[0]["frames"].dtype)
    skels = torch.zeros((b, t_max, j, k), dtype=batch[0]["skeleton"].dtype)
    input_lengths = torch.zeros((b,), dtype=torch.long)
    targets = []
    target_lengths = torch.zeros((b,), dtype=torch.long)
    ids, videos, gloss_texts, selected_keyframes = [], [], [], []
    for i, x in enumerate(batch):
        t = x["input_length"]
        frames[i, :t] = x["frames"]
        skels[i, :t] = x["skeleton"]
        input_lengths[i] = t
        target_lengths[i] = x["target_length"]
        if x["target_length"] > 0:
            targets.append(x["target_ids"])
        ids.append(x["id"])
        videos.append(x["video"])
        gloss_texts.append(x["gloss_text"])
        selected_keyframes.append(x.get("selected_keyframes", list(range(t))))
    targets_concat = torch.cat(targets, dim=0) if targets else torch.empty((0,), dtype=torch.long)
    return {
        "ids": ids,
        "videos": videos,
        "gloss_texts": gloss_texts,
        "selected_keyframes": selected_keyframes,
        "frames": frames,
        "skeleton": skels,
        "input_lengths": input_lengths,
        "targets": targets_concat,
        "target_lengths": target_lengths,
    }


class FrameEncoderMobileNetV3Small(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.avgpool = net.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return feat.view(b, t, -1)


def build_mediapipe_75_adjacency() -> torch.Tensor:
    num_joints = 75
    pose_edges = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
        (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
        (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]
    hand_local_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    left_offset = 33
    right_offset = 54
    left_hand_edges = [(a + left_offset, b + left_offset) for a, b in hand_local_edges]
    right_hand_edges = [(a + right_offset, b + right_offset) for a, b in hand_local_edges]
    cross_edges = [(15, left_offset), (16, right_offset)]
    undirected_edges = pose_edges + left_hand_edges + right_hand_edges + cross_edges
    a_self = torch.eye(num_joints, dtype=torch.float32)
    a_neigh = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in undirected_edges:
        a_neigh[i, j] = 1.0
        a_neigh[j, i] = 1.0
    deg = a_neigh.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)
    a_neigh = d_inv_sqrt @ a_neigh @ d_inv_sqrt
    return torch.stack([a_self, a_neigh], dim=0)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_subsets: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        x = self.conv(x)
        x = x.view(n, self.num_subsets, self.out_channels, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, a)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_subsets: int, temporal_kernel: int, dropout: float, residual: bool = True):
        super().__init__()
        pad = (temporal_kernel - 1) // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, a)
        x = self.tcn(x) + res
        return F.relu(x, inplace=True)


class SkeletonEncoderSTGCN(nn.Module):
    def __init__(self, num_joints: int, in_channels: int, hidden_dim: int, num_blocks: int, temporal_kernel: int, dropout: float, use_residual: bool = True):
        super().__init__()
        a = build_mediapipe_75_adjacency()
        if a.shape[-1] != num_joints:
            raise ValueError(f"Adjacency joint count mismatch: expected {num_joints}, got {a.shape[-1]}")
        self.register_buffer("A", a)
        self.data_bn = nn.BatchNorm1d(num_joints * in_channels)
        widths = [64]
        while len(widths) < max(1, num_blocks - 1):
            widths.append(min(hidden_dim, widths[-1] * 2))
        widths.append(hidden_dim)
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(widths):
            layers.append(STGCNBlock(in_ch, out_ch, a.shape[0], temporal_kernel, dropout, residual=(use_residual and i > 0)))
            in_ch = out_ch
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, j * c, t)
        x = self.data_bn(x)
        x = x.view(b, j, c, t).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=-1)
        return x.transpose(1, 2).contiguous()


class SkeletonEncoder1D(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        x = x.reshape(b, t, j * c)
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.relu(self.bn1(self.conv1(x)), inplace=True))
        out = self.dropout(F.relu(self.bn2(self.conv2(out)), inplace=True))
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res, inplace=True)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(TemporalBlock(in_dim, hidden_dim, kernel_size, 2 ** i, dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class PerceptionAlignment(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.skel_norm = nn.LayerNorm(dim)
        self.rgb_proj = nn.Linear(dim, dim)
        self.skel_proj = nn.Linear(dim, dim)
        self.rgb_to_skel_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.skel_to_rgb_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(dim * 4, dim)

    def forward(self, rgb_feat: torch.Tensor, sk_feat: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        r = self.rgb_proj(self.rgb_norm(rgb_feat))
        s = self.skel_proj(self.skel_norm(sk_feat))
        s_from_r, _ = self.rgb_to_skel_attn(query=s, key=r, value=r, key_padding_mask=padding_mask)
        r_from_s, _ = self.skel_to_rgb_attn(query=r, key=s, value=s, key_padding_mask=padding_mask)
        r_aligned = r + r_from_s
        s_aligned = s + s_from_r
        gate_in = torch.cat([r_aligned, s_aligned, torch.abs(r_aligned - s_aligned), r_aligned * s_aligned], dim=-1)
        g = self.gate(gate_in)
        fused = torch.cat([r_aligned, s_aligned, g * r_aligned, (1.0 - g) * s_aligned], dim=-1)
        fused = self.out_proj(fused)
        return r_aligned, s_aligned, fused


def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    denom = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * valid_mask.unsqueeze(-1).float()).sum(dim=1) / denom


def build_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def cosine_alignment_loss(x: torch.Tensor, y: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    loss = 1.0 - (x * y).sum(dim=-1)
    if valid_mask is not None:
        loss = loss * valid_mask.float()
        denom = valid_mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def cosine_global_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


class CLIPTextSemanticEncoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: str):
        super().__init__()
        self.cfg = cfg
        self.runtime_device = device
        self.backend = None
        self.cache_enabled = cfg.get("clip_cache_text_embeddings", True)
        self.cache: Dict[str, torch.Tensor] = {}
        self.out_dim = None
        backend = cfg.get("clip_backend", "auto")
        self._init_backend(backend)
        if self.out_dim is None:
            raise RuntimeError("CLIP text encoder initialization failed: output dim is unknown.")
        if not cfg.get("clip_text_trainable", False):
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def _init_backend(self, backend: str) -> None:
        backend = (backend or "auto").lower()
        errors = []
        if backend in ("auto", "open_clip"):
            try:
                import open_clip
                model_name = self.cfg.get("clip_model_name", "ViT-B-32")
                local_weight = self.cfg.get("clip_local_weight_path", None)
                if local_weight:
                    if not os.path.exists(local_weight):
                        raise FileNotFoundError(f"clip_local_weight_path not found: {local_weight}")
                    model = open_clip.create_model(model_name, pretrained=None)
                    state = torch.load(local_weight, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing:
                        print(f"[clip] open_clip missing keys: {len(missing)}")
                    if unexpected:
                        print(f"[clip] open_clip unexpected keys: {len(unexpected)}")
                else:
                    pretrained = self.cfg.get("clip_pretrained", "openai")
                    model = open_clip.create_model(model_name, pretrained=pretrained)

                tokenizer = open_clip.get_tokenizer(model_name)
                self.open_clip_model = model
                self.open_clip_tokenizer = tokenizer
                self.backend = "open_clip"
                self.out_dim = int(getattr(model, "text_projection").shape[1]) if getattr(model, "text_projection", None) is not None else 512
                self.open_clip_model.to(self.runtime_device)
                return
            except Exception as e:  # pragma: no cover - backend availability depends on env
                errors.append(f"open_clip unavailable: {e}")
        if backend in ("auto", "transformers"):
            try:
                from transformers import CLIPTokenizer, CLIPTextModel
                model_name = self.cfg.get("clip_hf_model_name", "openai/clip-vit-base-patch32")
                local_only = bool(self.cfg.get("clip_local_files_only", True))
                self.hf_tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_only)
                self.hf_text_model = CLIPTextModel.from_pretrained(model_name, local_files_only=local_only)
                self.backend = "transformers"
                self.out_dim = int(self.hf_text_model.config.hidden_size)
                self.hf_text_model.to(self.runtime_device)
                return
            except Exception as e:  # pragma: no cover
                errors.append(f"transformers unavailable: {e}")
        raise ImportError(
            "No CLIP text backend is available. Install either open_clip_torch or transformers, and provide local CLIP weights if offline. "
            + " | ".join(errors)
        )

    @torch.no_grad()
    def _encode_open_clip(self, texts: List[str]) -> torch.Tensor:
        tokens = self.open_clip_tokenizer(texts).to(self.runtime_device)
        feats = self.open_clip_model.encode_text(tokens)
        return feats.float()

    @torch.no_grad()
    def _encode_transformers(self, texts: List[str]) -> torch.Tensor:
        batch = self.hf_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.runtime_device) for k, v in batch.items()}
        outputs = self.hf_text_model(**batch)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]
        return feats.float()

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty((0, self.out_dim), device=device)
        uncached_texts = []
        uncached_idx = []
        out_cpu: List[Optional[torch.Tensor]] = [None] * len(texts)
        if self.cache_enabled:
            for i, text in enumerate(texts):
                if text in self.cache:
                    out_cpu[i] = self.cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_idx.append(i)
        else:
            uncached_texts = list(texts)
            uncached_idx = list(range(len(texts)))
        if uncached_texts:
            if self.backend == "open_clip":
                feats = self._encode_open_clip(uncached_texts)
            elif self.backend == "transformers":
                feats = self._encode_transformers(uncached_texts)
            else:
                raise RuntimeError(f"Unsupported CLIP backend: {self.backend}")
            feats = feats.detach().cpu()
            for idx, feat, text in zip(uncached_idx, feats, uncached_texts):
                out_cpu[idx] = feat
                if self.cache_enabled:
                    self.cache[text] = feat
        stacked = torch.stack([x for x in out_cpu], dim=0)
        return stacked.to(device=device, dtype=torch.float32)



class OfflineCLIPTextFeatureProvider(nn.Module):
    def __init__(self, cache_path: str):
        super().__init__()
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Offline CLIP cache not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        self.text_to_embedding = payload["text_to_embedding"]
        self.out_dim = int(payload["embedding_dim"])

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        feats = []
        missing = []
        for text in texts:
            if text not in self.text_to_embedding:
                missing.append(text)
            else:
                feats.append(self.text_to_embedding[text])
        if missing:
            preview = missing[:3]
            raise KeyError(f"{len(missing)} gloss_text entries missing from offline CLIP cache. Examples: {preview}")
        return torch.stack(feats, dim=0).to(device=device, dtype=torch.float32)


class CLIPSemanticAlignment(nn.Module):
    def __init__(self, visual_dim: int, semantic_in_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.semantic_norm = nn.LayerNorm(semantic_in_dim)
        self.semantic_proj = nn.Linear(semantic_in_dim, visual_dim)
        self.visual_to_text_attn = nn.MultiheadAttention(visual_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(visual_dim * 4, visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(visual_dim * 4, visual_dim)

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # visual_feat: [B, T, D], text_feat: [B, D_text]
        v = self.visual_norm(visual_feat)
        t = self.semantic_proj(self.semantic_norm(text_feat)).unsqueeze(1)  # [B, 1, D]
        semantic_context, _ = self.visual_to_text_attn(query=v, key=t, value=t)
        v_aligned = v + semantic_context
        t_expand = t.expand(-1, v_aligned.shape[1], -1)
        gate_in = torch.cat([v_aligned, t_expand, torch.abs(v_aligned - t_expand), v_aligned * t_expand], dim=-1)
        g = self.gate(gate_in)
        fused = torch.cat([v_aligned, t_expand, g * v_aligned, (1.0 - g) * t_expand], dim=-1)
        fused = self.out_proj(fused)
        return v_aligned, t.squeeze(1), fused


class M3RGBSkeletonCLIPCTC(nn.Module):
    def __init__(self, vocab_size: int, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.use_perception_alignment = cfg.get("use_perception_alignment", True)
        self.use_clip_semantic_alignment = cfg.get("use_clip_semantic_alignment", True)
        self.align_loss_weight = cfg.get("align_loss_weight", 0.1)
        self.semantic_loss_weight = cfg.get("semantic_loss_weight", 0.1)

        self.rgb_encoder = FrameEncoderMobileNetV3Small()
        if cfg.get("skeleton_encoder_type", "stgcn").lower() == "stgcn":
            self.skel_encoder = SkeletonEncoderSTGCN(
                cfg["skeleton_num_joints"],
                cfg["skeleton_channels"],
                cfg["skeleton_hidden_dim"],
                cfg.get("stgcn_num_blocks", 4),
                cfg.get("stgcn_temporal_kernel", 9),
                cfg["skeleton_dropout"],
                use_residual=cfg.get("stgcn_residual", True),
            )
        else:
            self.skel_encoder = SkeletonEncoder1D(
                cfg["skeleton_input_dim"],
                cfg["skeleton_hidden_dim"],
                cfg["skeleton_num_layers"],
                cfg["skeleton_dropout"],
            )
        self.rgb_proj = nn.Linear(cfg["frame_feature_dim"], cfg["fusion_dim"])
        self.skel_proj = nn.Linear(cfg["skeleton_hidden_dim"], cfg["fusion_dim"])

        if self.use_perception_alignment:
            self.perception_alignment = PerceptionAlignment(
                dim=cfg["fusion_dim"],
                num_heads=cfg.get("align_num_heads", 4),
                dropout=cfg.get("align_dropout", 0.1),
            )
        else:
            self.fusion = nn.Linear(cfg["fusion_dim"] * 2, cfg["fusion_dim"])

        if self.use_clip_semantic_alignment:
            if cfg.get("use_offline_clip_text_cache", False):
                self.text_encoder = OfflineCLIPTextFeatureProvider(cfg["offline_clip_cache_path"])
            else:
                self.text_encoder = CLIPTextSemanticEncoder(cfg, cfg["device"])
            self.semantic_alignment = CLIPSemanticAlignment(
                visual_dim=cfg["fusion_dim"],
                semantic_in_dim=self.text_encoder.out_dim,
                num_heads=cfg.get("clip_num_heads", 4),
                dropout=cfg.get("clip_dropout", 0.1),
            )
        else:
            self.text_encoder = None
        self.temporal = TemporalConvNet(
            cfg["fusion_dim"],
            cfg["tcn_hidden_dim"],
            cfg["tcn_num_layers"],
            cfg["tcn_kernel_size"],
            cfg["tcn_dropout"],
        )
        self.classifier = nn.Linear(cfg["tcn_hidden_dim"], vocab_size)

    def forward(self, frames: torch.Tensor, skeleton: torch.Tensor, input_lengths: torch.Tensor, gloss_texts: Optional[List[str]] = None):
        rgb_feat = self.rgb_proj(self.rgb_encoder(frames))
        sk_feat = self.skel_proj(self.skel_encoder(skeleton))
        padding_mask = build_padding_mask(input_lengths, max_len=rgb_feat.shape[1])
        valid_mask = ~padding_mask

        if self.use_perception_alignment:
            rgb_feat, sk_feat, fused = self.perception_alignment(rgb_feat, sk_feat, padding_mask=padding_mask)
            align_loss = cosine_alignment_loss(rgb_feat, sk_feat, valid_mask=valid_mask)
        else:
            fused = self.fusion(torch.cat([rgb_feat, sk_feat], dim=-1))
            align_loss = fused.new_zeros(())

        if self.use_clip_semantic_alignment:
            if gloss_texts is None:
                raise ValueError("gloss_texts must be provided when use_clip_semantic_alignment=True")
            text_feat = self.text_encoder.encode_texts(gloss_texts, fused.device)
            semantic_visual_feat, projected_text_feat, fused = self.semantic_alignment(fused, text_feat, padding_mask=padding_mask)
            pooled_visual = masked_mean(semantic_visual_feat, valid_mask)
            semantic_loss = cosine_global_loss(pooled_visual, projected_text_feat)
        else:
            text_feat = None
            semantic_visual_feat = fused
            projected_text_feat = None
            semantic_loss = fused.new_zeros(())

        temporal_feat = self.temporal(fused)
        logits = self.classifier(temporal_feat)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return {
            "log_probs": log_probs,
            "output_lengths": input_lengths,
            "align_loss": align_loss,
            "semantic_loss": semantic_loss,
            "rgb_feat": rgb_feat,
            "sk_feat": sk_feat,
            "fused_feat": fused,
            "semantic_visual_feat": semantic_visual_feat,
            "projected_text_feat": projected_text_feat,
            "raw_text_feat": text_feat,
        }


def ctc_greedy_decode(log_probs: torch.Tensor, output_lengths: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    pred = log_probs.argmax(dim=-1)
    t, b = pred.shape
    hyps = []
    for bi in range(b):
        raw = pred[: int(output_lengths[bi].item()), bi].tolist()
        out, prev = [], None
        for p in raw:
            if p != blank_id and p != prev:
                out.append(p)
            prev = p
        hyps.append(out)
    return hyps


def unpack_targets_concat(targets_concat: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
    out, st = [], 0
    for l in target_lengths.tolist():
        out.append(targets_concat[st: st + l].tolist())
        st += l
    return out


def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer, scheduler, best_dev_wer: float, cfg: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_dev_wer": best_dev_wer,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", -1)), float(ckpt.get("best_dev_wer", float("inf")))


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.bad_epochs = 0

    def step(self, current: float, best: float) -> bool:
        if current < (best - self.min_delta):
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass
class EpochResult:
    loss: float
    wer: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    seconds: float
    ctc_loss: float = 0.0
    align_loss: float = 0.0
    semantic_loss: float = 0.0


def train_one_epoch(model, loader, criterion, optimizer, device, blank_id, grad_clip, align_loss_weight: float = 0.0, semantic_loss_weight: float = 0.0, log_interval: int = 20) -> EpochResult:
    model.train()
    if getattr(model, "text_encoder", None) is not None and not model.cfg.get("clip_text_trainable", False):
        model.text_encoder.eval()
    t0 = time.time()
    total_loss = 0.0
    total_batches = 0
    total_ctc_loss = 0.0
    total_align_loss = 0.0
    total_semantic_loss = 0.0
    all_refs, all_hyps = [], []

    for bi, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        gloss_texts = batch["gloss_texts"]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames, skeleton, input_lengths, gloss_texts=gloss_texts)
        log_probs = outputs["log_probs"]
        output_lengths = outputs["output_lengths"]
        ctc_loss = criterion(log_probs, targets, output_lengths, target_lengths)
        align_loss = outputs.get("align_loss", ctc_loss.new_zeros(()))
        semantic_loss = outputs.get("semantic_loss", ctc_loss.new_zeros(()))
        loss = ctc_loss + align_loss_weight * align_loss + semantic_loss_weight * semantic_loss
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_ctc_loss += float(ctc_loss.item())
        total_align_loss += float(align_loss.item())
        total_semantic_loss += float(semantic_loss.item())
        total_batches += 1

        with torch.no_grad():
            all_hyps.extend(ctc_greedy_decode(log_probs.detach(), output_lengths.detach(), blank_id))
            all_refs.extend(unpack_targets_concat(batch["targets"], batch["target_lengths"]))

        if (bi + 1) % log_interval == 0:
            print(
                f"[train] batch {bi + 1}/{len(loader)} total_loss={loss.item():.4f} "
                f"ctc_loss={ctc_loss.item():.4f} align_loss={align_loss.item():.4f} semantic_loss={semantic_loss.item():.4f}"
            )

    wer = compute_wer(all_refs, all_hyps)
    metrics = compute_token_metrics(all_refs, all_hyps)
    return EpochResult(
        loss=total_loss / max(total_batches, 1),
        ctc_loss=total_ctc_loss / max(total_batches, 1),
        align_loss=total_align_loss / max(total_batches, 1),
        semantic_loss=total_semantic_loss / max(total_batches, 1),
        wer=wer,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        seconds=time.time() - t0,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, blank_id, align_loss_weight: float = 0.0, semantic_loss_weight: float = 0.0) -> EpochResult:
    model.eval()
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.eval()
    t0 = time.time()
    total_loss = 0.0
    total_batches = 0
    total_ctc_loss = 0.0
    total_align_loss = 0.0
    total_semantic_loss = 0.0
    all_refs, all_hyps = [], []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        gloss_texts = batch["gloss_texts"]

        outputs = model(frames, skeleton, input_lengths, gloss_texts=gloss_texts)
        log_probs = outputs["log_probs"]
        output_lengths = outputs["output_lengths"]
        ctc_loss = criterion(log_probs, targets, output_lengths, target_lengths)
        align_loss = outputs.get("align_loss", ctc_loss.new_zeros(()))
        semantic_loss = outputs.get("semantic_loss", ctc_loss.new_zeros(()))
        loss = ctc_loss + align_loss_weight * align_loss + semantic_loss_weight * semantic_loss

        total_loss += float(loss.item())
        total_ctc_loss += float(ctc_loss.item())
        total_align_loss += float(align_loss.item())
        total_semantic_loss += float(semantic_loss.item())
        total_batches += 1

        all_hyps.extend(ctc_greedy_decode(log_probs, output_lengths, blank_id))
        all_refs.extend(unpack_targets_concat(batch["targets"], batch["target_lengths"]))

    wer = compute_wer(all_refs, all_hyps)
    metrics = compute_token_metrics(all_refs, all_hyps)
    return EpochResult(
        loss=total_loss / max(total_batches, 1),
        ctc_loss=total_ctc_loss / max(total_batches, 1),
        align_loss=total_align_loss / max(total_batches, 1),
        semantic_loss=total_semantic_loss / max(total_batches, 1),
        wer=wer,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        seconds=time.time() - t0,
    )


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = cfg["device"]
    out_dir = cfg["output_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[paths] output_dir={out_dir}")
    print(f"[paths] ckpt_dir={ckpt_dir}")
    print(f"[paths] log_dir={log_dir}")
    save_json(cfg, os.path.join(out_dir, "config.json"))

    vocab = load_existing_vocab(cfg["vocab_path"])
    print("=" * 80)
    print("Experiment:", cfg["experiment_name"])
    print("Device    :", device)
    print("Vocab size:", vocab.size)
    print("=" * 80)

    train_ds = CECSLDataset(cfg["train_manifest"], "train", vocab, cfg)
    dev_ds = CECSLDataset(cfg["dev_manifest"], "dev", vocab, cfg)
    test_ds = CECSLDataset(cfg["test_manifest"], "test", vocab, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_collate_fn)

    model = M3RGBSkeletonCLIPCTC(vocab.size, cfg).to(device)
    if getattr(model, "text_encoder", None) is not None:
        print(f"[clip] text_provider={type(model.text_encoder).__name__}, text_dim={model.text_encoder.out_dim}")
        if cfg.get("use_offline_clip_text_cache", False):
            print(f"[clip] offline_cache={cfg.get('offline_clip_cache_path')}")
        elif cfg.get("clip_local_weight_path"):
            print(f"[clip] local_weight={cfg.get('clip_local_weight_path')}")
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = None
    if cfg["use_scheduler"] and cfg["scheduler_type"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])

    start_epoch = 0
    best_dev_wer = float("inf")
    early_stopper = EarlyStopping(
        patience=cfg.get("early_stopping_patience", 8),
        min_delta=cfg.get("early_stopping_min_delta", 0.0),
    )

    if cfg["resume_path"] is not None and os.path.exists(cfg["resume_path"]):
        last_epoch, best_dev_wer = load_checkpoint(cfg["resume_path"], model, optimizer, scheduler, map_location=device)
        start_epoch = last_epoch + 1
        print(f"[resume] start_epoch={start_epoch}, best_dev_wer={best_dev_wer:.4f}")

    history_path = os.path.join(log_dir, "history.json")
    history = []
    if os.path.exists(history_path):
        try:
            old = load_json(history_path)
            history = old.get("history", [])
            history = [x for x in history if x.get("epoch", -1) < start_epoch]
        except Exception:
            history = []

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n{'=' * 30} Epoch {epoch + 1}/{cfg['epochs']} {'=' * 30}")
        train_res = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            vocab.blank_id,
            cfg["grad_clip"],
            cfg.get("align_loss_weight", 0.0),
            cfg.get("semantic_loss_weight", 0.0),
            cfg["log_interval"],
        )
        dev_res = evaluate(
            model,
            dev_loader,
            criterion,
            device,
            vocab.blank_id,
            cfg.get("align_loss_weight", 0.0),
            cfg.get("semantic_loss_weight", 0.0),
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_res.loss,
            "train_ctc_loss": train_res.ctc_loss,
            "train_align_loss": train_res.align_loss,
            "train_semantic_loss": train_res.semantic_loss,
            "train_wer": train_res.wer,
            "train_accuracy": train_res.accuracy,
            "train_precision": train_res.precision,
            "train_recall": train_res.recall,
            "train_f1": train_res.f1,
            "dev_loss": dev_res.loss,
            "dev_ctc_loss": dev_res.ctc_loss,
            "dev_align_loss": dev_res.align_loss,
            "dev_semantic_loss": dev_res.semantic_loss,
            "dev_wer": dev_res.wer,
            "dev_accuracy": dev_res.accuracy,
            "dev_precision": dev_res.precision,
            "dev_recall": dev_res.recall,
            "dev_f1": dev_res.f1,
        }
        history.append(row)
        print(
            f"[epoch {epoch + 1}] train_loss={train_res.loss:.4f} "
            f"(ctc={train_res.ctc_loss:.4f}, align={train_res.align_loss:.4f}, semantic={train_res.semantic_loss:.4f}) "
            f"train_WER={train_res.wer:.4f} | dev_loss={dev_res.loss:.4f} "
            f"(ctc={dev_res.ctc_loss:.4f}, align={dev_res.align_loss:.4f}, semantic={dev_res.semantic_loss:.4f}) "
            f"dev_WER={dev_res.wer:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        prev_best_dev_wer = best_dev_wer
        improved = dev_res.wer < (prev_best_dev_wer - cfg.get("early_stopping_min_delta", 0.0))
        if improved:
            best_dev_wer = dev_res.wer
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), epoch, model, optimizer, scheduler, best_dev_wer, cfg)
            print(f"[best] saved best checkpoint with dev_WER={best_dev_wer:.4f}")

        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), epoch, model, optimizer, scheduler, best_dev_wer, cfg)
        save_json({"history": history}, history_path)

        if cfg.get("early_stopping", False):
            should_stop = early_stopper.step(dev_res.wer, prev_best_dev_wer)
            print(f"[early-stop] bad_epochs={early_stopper.bad_epochs}/{early_stopper.patience}")
            if should_stop:
                print(f"[early-stop] triggered at epoch {epoch + 1}, best_dev_WER={best_dev_wer:.4f}")
                break

    if cfg["test_after_training"]:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            load_checkpoint(best_path, model, map_location=device)
        test_res = evaluate(
            model,
            test_loader,
            criterion,
            device,
            vocab.blank_id,
            cfg.get("align_loss_weight", 0.0),
            cfg.get("semantic_loss_weight", 0.0),
        )
        print("\n" + "=" * 80)
        print("[TEST RESULT]")
        print(
            f"loss={test_res.loss:.4f} | WER={test_res.wer:.4f} | Acc={test_res.accuracy:.4f} | "
            f"P={test_res.precision:.4f} | R={test_res.recall:.4f} | F1={test_res.f1:.4f}"
        )
        print("=" * 80)
        save_json(
            {
                "test_loss": test_res.loss,
                "test_ctc_loss": test_res.ctc_loss,
                "test_align_loss": test_res.align_loss,
                "test_semantic_loss": test_res.semantic_loss,
                "test_wer": test_res.wer,
                "test_accuracy": test_res.accuracy,
                "test_precision": test_res.precision,
                "test_recall": test_res.recall,
                "test_f1": test_res.f1,
            },
            os.path.join(log_dir, "test_result.json"),
        )


if __name__ == "__main__":
    main()
