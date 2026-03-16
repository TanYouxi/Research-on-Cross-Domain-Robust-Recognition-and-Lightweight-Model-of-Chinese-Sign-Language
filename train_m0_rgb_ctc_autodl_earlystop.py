# -*- coding: utf-8 -*-
"""
M0: RGB baseline for CE-CSL
AutoDL-ready version

Main updates:
- compatible with gloss stored as either list or string
- Linux/AutoDL paths by default
- server-friendly dataloader defaults
- optional environment-variable override for dataset root
- early stopping support
- safer resume defaults for new experiments
- fix invalid OMP_NUM_THREADS on some AutoDL images
"""
import ast
import os
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

# Avoid invalid or empty OMP setting on AutoDL
if not str(os.environ.get("OMP_NUM_THREADS", "")).isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

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
    "gloss_key": "gloss",
    "video_key": "video",
    "id_key": "id",
    "vocab_path": f"{DATA_ROOT}/gloss_vocab.json",
    "experiment_name": "m0_rgb",
    "output_dir": f"{DATA_ROOT}/experiments/m0_rgb",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 55,
    "batch_size": 10,
    "num_workers": 8,
    "pin_memory": True,
    "resume_path": None,
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


class CECSLM0Dataset(Dataset):
    def __init__(self, manifest_path: str, vocab: CTCVocab, cfg: Dict[str, Any]):
        self.items = read_jsonl(manifest_path)
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        sample_id = str(item[self.cfg["id_key"]])
        video_path = self._resolve_video_path(str(item[self.cfg["video_key"]]))
        frames = self._read_rgb_frames(video_path)
        if self.cfg["normalize_with_imagenet"]:
            frames = normalize_video_tensor(frames)
        target_ids = torch.tensor(self.vocab.encode(item.get(self.cfg["gloss_key"])), dtype=torch.long)
        return {
            "id": sample_id,
            "video": video_path,
            "gloss_text": ensure_gloss_tokens(item.get(self.cfg["gloss_key"])),
            "frames": frames,
            "input_length": int(frames.shape[0]),
            "target_ids": target_ids,
            "target_length": int(target_ids.numel()),
        }


def cecsl_m0_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = sorted(batch, key=lambda x: x["input_length"], reverse=True)
    B = len(batch)
    T_max = max(x["input_length"] for x in batch)
    C, H, W = batch[0]["frames"].shape[1:]
    frames = torch.zeros((B, T_max, C, H, W), dtype=batch[0]["frames"].dtype)
    input_lengths = torch.zeros((B,), dtype=torch.long)
    targets = []
    target_lengths = torch.zeros((B,), dtype=torch.long)
    ids, videos, gloss_texts = [], [], []
    for i, x in enumerate(batch):
        t = x["input_length"]
        frames[i, :t] = x["frames"]
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


class M0RGBCTC(nn.Module):
    def __init__(self, vocab_size: int, cfg: Dict[str, Any]):
        super().__init__()
        self.rgb_encoder = FrameEncoderMobileNetV3Small()
        self.rgb_proj = nn.Linear(cfg["frame_feature_dim"], cfg["fusion_dim"])
        self.temporal = TemporalConvNet(
            cfg["fusion_dim"],
            cfg["tcn_hidden_dim"],
            cfg["tcn_num_layers"],
            cfg["tcn_kernel_size"],
            cfg["tcn_dropout"],
        )
        self.classifier = nn.Linear(cfg["tcn_hidden_dim"], vocab_size)

    def forward(self, frames: torch.Tensor, input_lengths: torch.Tensor):
        rgb_feat = self.rgb_encoder(frames)
        rgb_feat = self.rgb_proj(rgb_feat)
        temporal_feat = self.temporal(rgb_feat)
        logits = self.classifier(temporal_feat)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, input_lengths


def ctc_greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_id: int) -> List[List[int]]:
    pred = log_probs.argmax(dim=-1).transpose(0, 1)
    hyps = []
    for seq, L in zip(pred, input_lengths.tolist()):
        seq = seq[:L].tolist()
        out = []
        prev = None
        for x in seq:
            if x != blank_id and x != prev:
                out.append(x)
            prev = x
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
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        log_probs, output_lengths = model(frames, input_lengths)
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
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        log_probs, output_lengths = model(frames, input_lengths)
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

    train_ds = CECSLM0Dataset(cfg["train_manifest"], vocab, cfg)
    dev_ds = CECSLM0Dataset(cfg["dev_manifest"], vocab, cfg)
    test_ds = CECSLM0Dataset(cfg["test_manifest"], vocab, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m0_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m0_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"], collate_fn=cecsl_m0_collate_fn)

    model = M0RGBCTC(vocab.size, cfg).to(device)
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
