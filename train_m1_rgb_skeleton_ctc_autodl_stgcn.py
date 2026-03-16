# -*- coding: utf-8 -*-
"""
M1: RGB + Skeleton baseline for CE-CSL
AutoDL-ready version

Main updates:
- compatible with gloss stored as either list or string
- Linux/AutoDL paths by default
- server-friendly dataloader defaults
- optional environment-variable override for dataset root
"""
import ast
import os
import json
import time
import random
from pathlib import Path
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
    "test_manifest": f"{DATA_ROOT}/manifests/test_final.jsonl",
    "skeleton_root": f"{DATA_ROOT}/skeleton_tasks75",
    "gloss_key": "gloss",
    "video_key": "video",
    "id_key": "id",
    "vocab_path": f"{DATA_ROOT}/gloss_vocab.json",
    "experiment_name": "m1_rgb_skel",
    "output_dir": f"{DATA_ROOT}/experiments/m1_rgb_skel_stgcn",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 55,
    "batch_size": 10,
    "num_workers": 8,
    "pin_memory": True,
    "resume_path": f"{DATA_ROOT}/experiments/m1_rgb_skel_stgcn/checkpoints/latest.pt",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "use_scheduler": True,
    "scheduler_type": "multistep",
    "milestones": [12, 18],
    "gamma": 0.5,
    "image_size": 224,
    "sample_stride": 2,
    "max_frames": 96,
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
    "log_interval": 20,
    "test_after_training": True,
    "early_stopping": True,
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 0.0,
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

    # 支持字符串化 list
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
                choices = [(dp[i - 1][j - 1] + 1, "S"), (dp[i - 1][j] + 1, "D"), (dp[i][j - 1] + 1, "I")]
                dp[i][j], op[i][j] = min(choices, key=lambda x: x[0])
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "E":
            i -= 1
            j -= 1
        elif cur == "S":
            S += 1
            i -= 1
            j -= 1
        elif cur == "D":
            D += 1
            i -= 1
        elif cur == "I":
            I += 1
            j -= 1
        else:
            break
    return S, D, I


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


class CECSLM1Dataset(Dataset):
    def __init__(self, manifest_path: str, split_name: str, vocab: CTCVocab, cfg: Dict[str, Any]):
        self.items = read_jsonl(manifest_path)
        self.split_name = split_name
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_video_path(self, video_path: str) -> str:
    # 1) 原路径本来就能用
        if os.path.exists(video_path):
            return video_path
    
        # 2) 把 Windows 反斜杠统一成 Linux 风格
        vp = str(video_path).replace("\\", "/")
    
        # 3) 如果 manifest 里是旧的 Windows 前缀，替换成当前 AutoDL base_dir
        old_prefix = "E:/CE-CSL/CE-CSL"
        if vp.startswith(old_prefix):
            candidate = vp.replace(old_prefix, self.cfg["base_dir"], 1)
            if os.path.exists(candidate):
                return candidate
    
        # 4) 更通用一点：只要路径里含有 /video/，就截出后半段重新拼接
        marker = "/video/"
        if marker in vp:
            tail = vp.split(marker, 1)[1]   # 例如 train/A/train-00001.mp4
            candidate = os.path.join(self.cfg["base_dir"], "video", tail)
            if os.path.exists(candidate):
                return candidate
    
        # 5) 兜底：返回原值，后面报错时还能看到原路径
        return video_path

    def _read_rgb_frames(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = set(sample_frame_indices(num_frames if num_frames > 0 else 999999, self.cfg["sample_stride"], self.cfg["max_frames"]))
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
        while len(selected) < self.cfg["min_frames"]:
            selected.append(selected[-1].copy())
        frames = []
        for frame in selected:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.cfg["image_size"], self.cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        return torch.from_numpy(np.stack(frames, axis=0))

    def _read_skeleton(self, sample_id: str, target_len: int) -> torch.Tensor:
        sk_path = os.path.join(self.cfg["skeleton_root"], self.split_name, f"{sample_id}.npy")
        if not os.path.exists(sk_path):
            raise FileNotFoundError(f"Skeleton not found: {sk_path}")
        arr = np.load(sk_path)
        if arr.ndim != 3 or arr.shape[1] != self.cfg["skeleton_num_joints"] or arr.shape[2] != self.cfg["skeleton_channels"]:
            raise RuntimeError(f"Unexpected skeleton shape for {sk_path}: {arr.shape}")
        idxs = temporal_resize_indices(arr.shape[0], target_len)
        arr = arr[idxs]
        while arr.shape[0] < self.cfg["min_frames"]:
            arr = np.concatenate([arr, arr[-1:]], axis=0)
        return torch.from_numpy(arr.astype(np.float32))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        sample_id = rec[self.cfg["id_key"]]
        video_path = self._resolve_video_path(rec[self.cfg["video_key"]])
        gloss = rec.get(self.cfg["gloss_key"], [])
        gloss_tokens = ensure_gloss_tokens(gloss)
        rgb_frames = self._read_rgb_frames(video_path)
        if self.cfg["normalize_with_imagenet"]:
            rgb_frames = normalize_video_tensor(rgb_frames)
        T = rgb_frames.shape[0]
        skeleton = self._read_skeleton(sample_id, T)
        target_ids = self.vocab.encode(gloss_tokens)
        return {
            "id": sample_id,
            "video": video_path,
            "frames": rgb_frames,
            "skeleton": skeleton,
            "input_length": T,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_length": len(target_ids),
            "gloss_text": " ".join(gloss_tokens),
        }


def cecsl_m1_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = sorted(batch, key=lambda x: x["input_length"], reverse=True)
    B = len(batch)
    T_max = max(x["input_length"] for x in batch)
    C, H, W = batch[0]["frames"].shape[1:]
    J, K = batch[0]["skeleton"].shape[1:]
    frames = torch.zeros((B, T_max, C, H, W), dtype=batch[0]["frames"].dtype)
    skels = torch.zeros((B, T_max, J, K), dtype=batch[0]["skeleton"].dtype)
    input_lengths = torch.zeros((B,), dtype=torch.long)
    targets = []
    target_lengths = torch.zeros((B,), dtype=torch.long)
    ids, videos, gloss_texts = [], [], []
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
    targets_concat = torch.cat(targets, dim=0) if targets else torch.empty((0,), dtype=torch.long)
    return {
        "ids": ids,
        "videos": videos,
        "gloss_texts": gloss_texts,
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
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return feat.view(B, T, -1)


def build_mediapipe_75_adjacency() -> torch.Tensor:
    num_joints = 75
    self_links = [(i, i) for i in range(num_joints)]

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

    cross_edges = [
        (15, left_offset),
        (16, right_offset),
    ]

    undirected_edges = pose_edges + left_hand_edges + right_hand_edges + cross_edges

    A_self = torch.eye(num_joints, dtype=torch.float32)
    A_neigh = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in undirected_edges:
        A_neigh[i, j] = 1.0
        A_neigh[j, i] = 1.0

    deg = A_neigh.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_neigh = D_inv_sqrt @ A_neigh @ D_inv_sqrt

    return torch.stack([A_self, A_neigh], dim=0)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_subsets: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        x = self.conv(x)
        x = x.view(n, self.num_subsets, self.out_channels, t, v)
        return torch.einsum('nkctv,kvw->nctw', x, A)


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

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return F.relu(x, inplace=True)


class SkeletonEncoderSTGCN(nn.Module):
    def __init__(self, num_joints: int, in_channels: int, hidden_dim: int, num_blocks: int, temporal_kernel: int, dropout: float, use_residual: bool = True):
        super().__init__()
        A = build_mediapipe_75_adjacency()
        if A.shape[-1] != num_joints:
            raise ValueError(f"Adjacency joint count mismatch: expected {num_joints}, got {A.shape[-1]}")
        self.register_buffer('A', A)
        self.data_bn = nn.BatchNorm1d(num_joints * in_channels)

        widths = [64]
        while len(widths) < max(1, num_blocks - 1):
            widths.append(min(hidden_dim, widths[-1] * 2))
        widths.append(hidden_dim)

        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(widths):
            layers.append(STGCNBlock(in_ch, out_ch, A.shape[0], temporal_kernel, dropout, residual=(use_residual and i > 0)))
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


class M1RGBSkeletonCTC(nn.Module):
    def __init__(self, vocab_size: int, cfg: Dict[str, Any]):
        super().__init__()
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
        self.fusion = nn.Linear(cfg["fusion_dim"] * 2, cfg["fusion_dim"])
        self.temporal = TemporalConvNet(
            cfg["fusion_dim"],
            cfg["tcn_hidden_dim"],
            cfg["tcn_num_layers"],
            cfg["tcn_kernel_size"],
            cfg["tcn_dropout"],
        )
        self.classifier = nn.Linear(cfg["tcn_hidden_dim"], vocab_size)

    def forward(self, frames: torch.Tensor, skeleton: torch.Tensor, input_lengths: torch.Tensor):
        rgb_feat = self.rgb_proj(self.rgb_encoder(frames))
        sk_feat = self.skel_proj(self.skel_encoder(skeleton))
        fused = self.fusion(torch.cat([rgb_feat, sk_feat], dim=-1))
        temporal_feat = self.temporal(fused)
        logits = self.classifier(temporal_feat)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, input_lengths


def ctc_greedy_decode(log_probs: torch.Tensor, output_lengths: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    pred = log_probs.argmax(dim=-1)
    T, B = pred.shape
    hyps = []
    for b in range(B):
        raw = pred[: int(output_lengths[b].item()), b].tolist()
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
        out.append(targets_concat[st : st + l].tolist())
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


def train_one_epoch(model, loader, criterion, optimizer, device, blank_id, grad_clip, log_interval=20) -> EpochResult:
    model.train()
    t0 = time.time()
    total_loss = 0.0
    total_batches = 0
    all_refs, all_hyps = [], []
    for bi, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        log_probs, output_lengths = model(frames, skeleton, input_lengths)
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        with torch.no_grad():
            all_hyps.extend(ctc_greedy_decode(log_probs.detach(), output_lengths.detach(), blank_id))
            all_refs.extend(unpack_targets_concat(batch["targets"], batch["target_lengths"]))
        if (bi + 1) % log_interval == 0:
            print(f"[train] batch {bi + 1}/{len(loader)} loss={loss.item():.4f}")
    wer = compute_wer(all_refs, all_hyps)
    metrics = compute_token_metrics(all_refs, all_hyps)
    return EpochResult(total_loss / max(total_batches, 1), wer, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], time.time() - t0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, blank_id) -> EpochResult:
    model.eval()
    t0 = time.time()
    total_loss = 0.0
    total_batches = 0
    all_refs, all_hyps = [], []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        log_probs, output_lengths = model(frames, skeleton, input_lengths)
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        total_loss += float(loss.item())
        total_batches += 1
        all_hyps.extend(ctc_greedy_decode(log_probs, output_lengths, blank_id))
        all_refs.extend(unpack_targets_concat(batch["targets"], batch["target_lengths"]))
    wer = compute_wer(all_refs, all_hyps)
    metrics = compute_token_metrics(all_refs, all_hyps)
    return EpochResult(total_loss / max(total_batches, 1), wer, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], time.time() - t0)


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

    train_ds = CECSLM1Dataset(cfg["train_manifest"], "train", vocab, cfg)
    dev_ds = CECSLM1Dataset(cfg["dev_manifest"], "dev", vocab, cfg)
    test_ds = CECSLM1Dataset(cfg["test_manifest"], "test", vocab, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m1_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m1_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m1_collate_fn)

    model = M1RGBSkeletonCTC(vocab.size, cfg).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
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
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device, vocab.blank_id, cfg["grad_clip"], cfg["log_interval"])
        dev_res = evaluate(model, dev_loader, criterion, device, vocab.blank_id)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_res.loss,
            "train_wer": train_res.wer,
            "train_accuracy": train_res.accuracy,
            "train_precision": train_res.precision,
            "train_recall": train_res.recall,
            "train_f1": train_res.f1,
            "dev_loss": dev_res.loss,
            "dev_wer": dev_res.wer,
            "dev_accuracy": dev_res.accuracy,
            "dev_precision": dev_res.precision,
            "dev_recall": dev_res.recall,
            "dev_f1": dev_res.f1,
        }
        history.append(row)
        print(f"[epoch {epoch + 1}] train_loss={train_res.loss:.4f} train_WER={train_res.wer:.4f} | dev_loss={dev_res.loss:.4f} dev_WER={dev_res.wer:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")
        prev_best_dev_wer = best_dev_wer
        improved = dev_res.wer < (prev_best_dev_wer - cfg.get("early_stopping_min_delta", 0.0))
        
        if improved:
            best_dev_wer = dev_res.wer
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"),
                epoch,
                model,
                optimizer,
                scheduler,
                best_dev_wer,
                cfg,
            )
            print(f"[best] saved best checkpoint with dev_WER={best_dev_wer:.4f}")
        
        save_checkpoint(
            os.path.join(ckpt_dir, "latest.pt"),
            epoch,
            model,
            optimizer,
            scheduler,
            best_dev_wer,
            cfg,
        )
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
        test_res = evaluate(model, test_loader, criterion, device, vocab.blank_id)
        print("\n" + "=" * 80)
        print("[TEST RESULT]")
        print(f"loss={test_res.loss:.4f} | WER={test_res.wer:.4f} | Acc={test_res.accuracy:.4f} | P={test_res.precision:.4f} | R={test_res.recall:.4f} | F1={test_res.f1:.4f}")
        print("=" * 80)
        save_json(
            {
                "test_loss": test_res.loss,
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
