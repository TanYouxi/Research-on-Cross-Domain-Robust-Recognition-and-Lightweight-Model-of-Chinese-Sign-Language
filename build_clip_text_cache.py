# -*- coding: utf-8 -*-
"""
Build offline CLIP text embedding cache for CE-CSL gloss sequences.

This script is designed for the offline-cache M3 / M3+KF training scripts:
  - train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py
  - train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py

It reads train/dev/test manifests, converts each sample's gloss sequence into the
same `gloss_text` format used during training, encodes the unique texts with the
same CLIP text encoder configuration, and saves a cache file that can be loaded by
`OfflineCLIPTextFeatureProvider`.

Recommended AutoDL usage:
  python build_clip_text_cache.py

Common optional usage:
  python build_clip_text_cache.py \
    --m3-script /root/autodl-tmp/train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py \
    --output-path /root/autodl-tmp/CE-CSL/clip_text_cache/clip_text_cache_all.pt \
    --device cuda \
    --batch-size 64
"""

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_SCRIPT_NAME = "train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py"
DEFAULT_DATA_ROOT = os.environ.get("CECSL_ROOT", "/root/autodl-tmp/CE-CSL")


def import_module_from_path(module_name: str, file_path: Path):
    """Import a Python module from an explicit file path."""
    file_path = file_path.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"M3 script not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def fallback_ensure_gloss_tokens(gloss: Any) -> List[str]:
    """Fallback parser used only if the imported M3 script has no parser."""
    if gloss is None:
        return []
    if isinstance(gloss, (list, tuple)):
        return [str(x).strip() for x in gloss if str(x).strip()]

    text = str(gloss).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            value = ast.literal_eval(text)
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
        except Exception:
            pass

    return [t for t in text.split() if t]


def record_to_gloss_text(record: Dict[str, Any], gloss_key: str, parser) -> str:
    """
    Reproduce the exact text key used by the training dataset:
      gloss_tokens = ensure_gloss_tokens(record[gloss_key])
      gloss_text = " ".join(gloss_tokens)
    """
    if gloss_key in record:
        tokens = parser(record.get(gloss_key))
        return " ".join(tokens)

    # Safety fallback for custom manifests that already store gloss_text.
    if "gloss_text" in record:
        return str(record["gloss_text"]).strip()

    return ""


def collect_manifest_paths(cfg: Dict[str, Any], explicit_manifests: Optional[Sequence[str]]) -> List[Path]:
    """Collect manifest paths. Explicit CLI paths take priority."""
    if explicit_manifests:
        paths = [Path(p).expanduser() for p in explicit_manifests]
    else:
        paths = []
        for key in ("train_manifest", "dev_manifest", "test_manifest"):
            value = cfg.get(key)
            if value:
                paths.append(Path(value).expanduser())

        # Helpful fallback: include both test.jsonl and test_final.jsonl if they exist.
        base_dir = Path(cfg.get("base_dir", DEFAULT_DATA_ROOT)).expanduser()
        manifest_dir = base_dir / "manifests"
        for split in ("train", "dev", "test"):
            for suffix in ("", "_clean", "_final"):
                candidate = manifest_dir / f"{split}{suffix}.jsonl"
                if candidate.exists():
                    paths.append(candidate)

    # Deduplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved_key = str(path)
        if resolved_key not in seen:
            seen.add(resolved_key)
            unique.append(path)
    return unique


def collect_unique_texts(manifest_paths: Sequence[Path], gloss_key: str, parser) -> Tuple[List[str], Dict[str, List[str]]]:
    """Read manifests and return unique non-empty gloss_text values."""
    text_set = set()
    source_map: Dict[str, List[str]] = {}

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            print(f"[WARN] Manifest not found, skipped: {manifest_path}")
            continue

        records = read_jsonl(manifest_path)
        added = 0
        for record in records:
            text = record_to_gloss_text(record, gloss_key=gloss_key, parser=parser)
            if not text:
                continue
            if text not in text_set:
                added += 1
            text_set.add(text)
            source_map.setdefault(text, []).append(str(manifest_path))

        print(f"[manifest] {manifest_path} | samples={len(records)} | new_unique_texts={added}")

    texts = sorted(text_set)
    return texts, source_map


@torch.no_grad()
def encode_texts_in_batches(text_encoder, texts: Sequence[str], device: torch.device, batch_size: int) -> Dict[str, torch.Tensor]:
    """Encode text list and return {text: embedding_cpu_tensor}."""
    text_to_embedding: Dict[str, torch.Tensor] = {}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch_texts = list(texts[start:start + batch_size])
        feats = text_encoder.encode_texts(batch_texts, device=device)
        feats = feats.detach().cpu().float()

        for text, feat in zip(batch_texts, feats):
            text_to_embedding[text] = feat.clone()

        end = min(start + batch_size, total)
        print(f"[encode] {end}/{total}")

    return text_to_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline CLIP text embedding cache for CE-CSL.")
    parser.add_argument(
        "--m3-script",
        type=str,
        default=DEFAULT_SCRIPT_NAME,
        help="Path to train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py. "
             "Default assumes it is in the current working directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output .pt cache path. Default uses CONFIG['offline_clip_cache_path'] from the M3 script.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit manifest jsonl paths. If omitted, train/dev/test paths from CONFIG are used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu. Default follows CONFIG['device'], then falls back to cuda if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Text encoding batch size.",
    )
    parser.add_argument(
        "--clip-local-weight-path",
        type=str,
        default=None,
        help="Optional override for CONFIG['clip_local_weight_path'].",
    )
    parser.add_argument(
        "--clip-backend",
        type=str,
        default=None,
        choices=["auto", "open_clip", "transformers"],
        help="Optional override for CONFIG['clip_backend'].",
    )
    parser.add_argument(
        "--clip-local-files-only",
        action="store_true",
        help="Force HuggingFace CLIP loading with local_files_only=True.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Set local_files_only=False for HuggingFace fallback. Use only if the environment has internet access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    m3_script = Path(args.m3_script)
    if not m3_script.exists() and not m3_script.is_absolute():
        # Useful fallback when this script is launched from another directory.
        candidate = Path(__file__).resolve().parent / args.m3_script
        if candidate.exists():
            m3_script = candidate

    module = import_module_from_path("m3_for_clip_cache", m3_script)
    cfg = dict(module.CONFIG)

    if args.clip_local_weight_path is not None:
        cfg["clip_local_weight_path"] = args.clip_local_weight_path
    if args.clip_backend is not None:
        cfg["clip_backend"] = args.clip_backend
    if args.allow_download:
        cfg["clip_local_files_only"] = False
    elif args.clip_local_files_only:
        cfg["clip_local_files_only"] = True

    device_str = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device_str = "cpu"
    cfg["device"] = device_str
    device = torch.device(device_str)

    output_path = Path(args.output_path or cfg.get("offline_clip_cache_path") or (Path(DEFAULT_DATA_ROOT) / "clip_text_cache" / "clip_text_cache_all.pt"))
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parser = getattr(module, "ensure_gloss_tokens", fallback_ensure_gloss_tokens)
    gloss_key = cfg.get("gloss_key", "gloss")
    manifest_paths = collect_manifest_paths(cfg, args.manifest)

    print("=" * 90)
    print(f"M3 script       : {m3_script.resolve()}")
    print(f"Device          : {device}")
    print(f"CLIP backend    : {cfg.get('clip_backend')}")
    print(f"CLIP local file : {cfg.get('clip_local_weight_path')}")
    print(f"Output cache    : {output_path}")
    print("=" * 90)

    texts, source_map = collect_unique_texts(manifest_paths, gloss_key=gloss_key, parser=parser)
    if not texts:
        raise RuntimeError("No gloss_text entries were collected. Please check manifest paths and gloss field names.")

    print(f"[summary] unique gloss_text entries: {len(texts)}")

    # Use exactly the same encoder class as the training script.
    text_encoder = module.CLIPTextSemanticEncoder(cfg, device_str).to(device)
    text_encoder.eval()

    text_to_embedding = encode_texts_in_batches(
        text_encoder=text_encoder,
        texts=texts,
        device=device,
        batch_size=max(1, int(args.batch_size)),
    )

    embedding_dim = int(next(iter(text_to_embedding.values())).numel())
    payload = {
        "text_to_embedding": text_to_embedding,
        "embedding_dim": embedding_dim,
        "num_texts": len(text_to_embedding),
        "texts": texts,
        "source_map": source_map,
        "metadata": {
            "m3_script": str(m3_script.resolve()),
            "clip_backend": cfg.get("clip_backend"),
            "clip_model_name": cfg.get("clip_model_name"),
            "clip_pretrained": cfg.get("clip_pretrained"),
            "clip_hf_model_name": cfg.get("clip_hf_model_name"),
            "clip_local_weight_path": cfg.get("clip_local_weight_path"),
            "gloss_key": gloss_key,
            "manifest_paths": [str(p) for p in manifest_paths],
        },
    }

    torch.save(payload, output_path)
    print("=" * 90)
    print(f"Saved offline CLIP cache: {output_path}")
    print(f"Embedding dim           : {embedding_dim}")
    print(f"Total cached texts      : {len(text_to_embedding)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
