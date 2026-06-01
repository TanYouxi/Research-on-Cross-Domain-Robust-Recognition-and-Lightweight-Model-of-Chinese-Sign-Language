# -*- coding: utf-8 -*-
"""
Export sample-level predictions for the final model and the keyframe-sampling model,
then generate candidate failure cases for qualitative analysis.

Recommended use on AutoDL:
python export_kf_failure_cases.py \
  --script-dir /root/autodl-tmp \
  --output-dir /root/autodl-tmp/CE-CSL/failure_analysis \
  --eval-batch-size 1

If checkpoints are not found automatically, pass them manually:
python export_kf_failure_cases.py \
  --script-dir /root/autodl-tmp \
  --final-checkpoint /path/to/final/best.pt \
  --kf-checkpoint /path/to/kf/best.pt \
  --output-dir /root/autodl-tmp/CE-CSL/failure_analysis
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

FINAL_SCRIPT_NAME = "train_m3_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py"
KF_SCRIPT_NAME = "train_m3_kf_rgb_skeleton_clip_ctc_autodl_stgcn_offlinecache.py"


def import_module_from_path(module_name: str, file_path: Path):
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Script not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(module, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    output_dir = Path(module.CONFIG["output_dir"])
    candidates = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "checkpoints" / "latest.pt",
        output_dir / "best.pt",
        output_dir / "latest.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found automatically. Tried:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nPlease pass --final-checkpoint or --kf-checkpoint manually."
    )


def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def ids_to_tokens(ids: List[int], id_to_token: Dict[int, str]) -> List[str]:
    return [id_to_token.get(int(x), "<UNK>") for x in ids]


def build_test_loader(module, cfg: Dict[str, Any], vocab, batch_size: int, num_workers: int):
    dataset = module.CECSLDataset(cfg["test_manifest"], "test", vocab, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=module.cecsl_collate_fn,
    )
    return dataset, loader


@torch.no_grad()
def export_predictions(
    module,
    model_label: str,
    checkpoint_path: Path,
    output_json: Path,
    eval_batch_size: int,
    eval_num_workers: int,
    test_manifest_override: Optional[str],
    device_override: Optional[str],
):
    cfg = dict(module.CONFIG)
    if test_manifest_override:
        cfg["test_manifest"] = test_manifest_override

    device = device_override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    cfg["device"] = device
    cfg["batch_size"] = eval_batch_size

    print("=" * 90)
    print(f"Model label     : {model_label}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Test manifest   : {cfg['test_manifest']}")
    print(f"Max frames      : {cfg.get('max_frames')}")
    print(f"Use keyframes   : {cfg.get('use_keyframe_sampling', False)}")
    print(f"Eval batch size : {eval_batch_size}")
    print(f"Device          : {device}")

    vocab = module.load_existing_vocab(cfg["vocab_path"])
    _, loader = build_test_loader(module, cfg, vocab, eval_batch_size, eval_num_workers)

    model = module.M3RGBSkeletonCLIPCTC(vocab.size, cfg).to(device)
    module.load_checkpoint(str(checkpoint_path), model, optimizer=None, scheduler=None, map_location=device)
    model.eval()

    id_to_token = vocab.id_to_token
    blank_id = vocab.blank_id
    results = []
    all_refs, all_hyps = [], []

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        gloss_texts = batch["gloss_texts"]

        outputs = model(frames=frames, skeleton=skeleton, input_lengths=input_lengths, gloss_texts=gloss_texts)
        hyps = module.ctc_greedy_decode(outputs["log_probs"], outputs["output_lengths"], blank_id=blank_id)
        refs = module.unpack_targets_concat(targets.detach().cpu(), target_lengths.detach().cpu())
        selected_keyframes = batch.get("selected_keyframes", [None] * len(refs))

        all_refs.extend(refs)
        all_hyps.extend(hyps)

        for sample_id, video_path, gt_ids, pred_ids, selected_idx in zip(
            batch["ids"], batch["videos"], refs, hyps, selected_keyframes
        ):
            gt_tokens = ids_to_tokens(gt_ids, id_to_token)
            pred_tokens = ids_to_tokens(pred_ids, id_to_token)
            s, d, ins = module.edit_distance(gt_ids, pred_ids)
            sample_wer = (s + d + ins) / max(1, len(gt_ids))
            results.append({
                "id": sample_id,
                "video": video_path,
                "ground_truth_tokens": gt_tokens,
                "ground_truth_text": " ".join(gt_tokens),
                "prediction_tokens": pred_tokens,
                "prediction_text": " ".join(pred_tokens),
                "sample_wer": float(sample_wer),
                "substitutions": int(s),
                "deletions": int(d),
                "insertions": int(ins),
                "target_length": int(len(gt_ids)),
                "prediction_length": int(len(pred_ids)),
                "selected_keyframes": make_json_safe(selected_idx),
            })

        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx + 1}/{len(loader)} batches")

    overall_wer = module.compute_wer(all_refs, all_hyps)
    metrics = module.compute_token_metrics(all_refs, all_hyps)
    payload = {
        "model_label": model_label,
        "checkpoint": str(checkpoint_path),
        "test_manifest": cfg["test_manifest"],
        "max_frames": cfg.get("max_frames"),
        "use_keyframe_sampling": cfg.get("use_keyframe_sampling", False),
        "num_samples": len(results),
        "overall_wer": float(overall_wer),
        "token_metrics": {k: float(v) for k, v in metrics.items()},
        "samples": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions: {output_json}")
    print(f"Overall WER      : {overall_wer:.4f}")
    return payload


def error_type_from_counts(s: int, d: int, i: int) -> str:
    parts = []
    if d > 0:
        parts.append("Deletion")
    if s > 0:
        parts.append("Substitution")
    if i > 0:
        parts.append("Insertion")
    return " + ".join(parts) if parts else "Correct / Minor"


def possible_cause(row: Dict[str, Any]) -> str:
    d = int(row["kf_deletions"])
    s = int(row["kf_substitutions"])
    target_len = int(row["target_length"])
    if d > 0 and target_len >= 6:
        return "Long-range temporal dependency may be weakened by temporal compression."
    if d > 0:
        return "Gloss boundary cues or transition frames may be lost."
    if s > 0:
        return "Subtle hand-shape transitions may be removed or under-represented."
    return "Temporal continuity may be weakened after keyframe sampling."


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    fieldnames = [
        "id", "ground_truth", "final_prediction", "kf_prediction",
        "final_wer", "kf_wer", "wer_gap", "target_length",
        "kf_error_type", "possible_cause", "kf_substitutions",
        "kf_deletions", "kf_insertions", "kf_selected_keyframes",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row = dict(r)
            if isinstance(row.get("kf_selected_keyframes"), list):
                row["kf_selected_keyframes"] = " ".join(str(x) for x in row["kf_selected_keyframes"])
            writer.writerow(row)


def select_three_representative_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected, used = [], set()

    def pick(condition):
        for r in rows:
            if r["id"] not in used and condition(r):
                selected.append(r)
                used.add(r["id"])
                return

    pick(lambda r: r["kf_deletions"] > 0 and r["target_length"] < 6)
    pick(lambda r: r["kf_substitutions"] > 0)
    pick(lambda r: r["kf_deletions"] > 0 and r["target_length"] >= 6)

    for r in rows:
        if len(selected) >= 3:
            break
        if r["id"] not in used:
            selected.append(r)
            used.add(r["id"])
    return selected[:3]


def merge_and_select(final_json: Path, kf_json: Path, output_dir: Path, top_k: int):
    with open(final_json, "r", encoding="utf-8") as f:
        final_payload = json.load(f)
    with open(kf_json, "r", encoding="utf-8") as f:
        kf_payload = json.load(f)

    final_map = {x["id"]: x for x in final_payload["samples"]}
    kf_map = {x["id"]: x for x in kf_payload["samples"]}
    rows = []

    for sid, f_item in final_map.items():
        if sid not in kf_map:
            continue
        k_item = kf_map[sid]
        row = {
            "id": sid,
            "ground_truth": f_item["ground_truth_text"],
            "final_prediction": f_item["prediction_text"],
            "kf_prediction": k_item["prediction_text"],
            "final_wer": float(f_item["sample_wer"]),
            "kf_wer": float(k_item["sample_wer"]),
            "wer_gap": float(k_item["sample_wer"]) - float(f_item["sample_wer"]),
            "target_length": int(f_item["target_length"]),
            "kf_substitutions": int(k_item["substitutions"]),
            "kf_deletions": int(k_item["deletions"]),
            "kf_insertions": int(k_item["insertions"]),
            "kf_selected_keyframes": k_item.get("selected_keyframes"),
        }
        row["kf_error_type"] = error_type_from_counts(row["kf_substitutions"], row["kf_deletions"], row["kf_insertions"])
        row["possible_cause"] = possible_cause(row)
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x["wer_gap"], x["kf_wer"]), reverse=True)
    worse_rows = [r for r in rows if r["wer_gap"] > 0]
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = output_dir / "kf_failure_candidates_top.csv"
    suggested_csv = output_dir / "suggested_failure_cases_for_paper.csv"
    suggested_json = output_dir / "suggested_failure_cases_for_paper.json"

    write_csv(candidates_csv, worse_rows[:top_k])
    suggested = select_three_representative_cases(worse_rows)
    write_csv(suggested_csv, suggested)
    with open(suggested_json, "w", encoding="utf-8") as f:
        json.dump(suggested, f, ensure_ascii=False, indent=2)

    print(f"Saved top candidates : {candidates_csv}")
    print(f"Saved suggested cases: {suggested_csv}")
    print(f"Saved suggested JSON : {suggested_json}")
    return worse_rows, suggested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-dir", type=str, default=".")
    parser.add_argument("--final-script", type=str, default=FINAL_SCRIPT_NAME)
    parser.add_argument("--kf-script", type=str, default=KF_SCRIPT_NAME)
    parser.add_argument("--final-checkpoint", type=str, default=None)
    parser.add_argument("--kf-checkpoint", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./failure_analysis_outputs")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    script_dir = Path(args.script_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    final_module = import_module_from_path("final_model_script", script_dir / args.final_script)
    kf_module = import_module_from_path("kf_model_script", script_dir / args.kf_script)

    final_ckpt = resolve_checkpoint(final_module, args.final_checkpoint)
    kf_ckpt = resolve_checkpoint(kf_module, args.kf_checkpoint)

    final_json = output_dir / "final_predictions.json"
    kf_json = output_dir / "kf_predictions.json"

    export_predictions(final_module, "Final model", final_ckpt, final_json, args.eval_batch_size, args.eval_num_workers, args.test_manifest, args.device)
    export_predictions(kf_module, "Keyframe sampling model", kf_ckpt, kf_json, args.eval_batch_size, args.eval_num_workers, args.test_manifest, args.device)

    worse_rows, suggested = merge_and_select(final_json, kf_json, output_dir, args.top_k)

    print("=" * 90)
    print(f"Total samples where KF is worse: {len(worse_rows)}")
    print("Suggested cases for the paper:")
    for idx, r in enumerate(suggested, 1):
        print("-" * 90)
        print(f"Case {idx}: {r['id']}")
        print(f"GT    : {r['ground_truth']}")
        print(f"Final : {r['final_prediction']}")
        print(f"KF    : {r['kf_prediction']}")
        print(f"Type  : {r['kf_error_type']}")
        print(f"Cause : {r['possible_cause']}")


if __name__ == "__main__":
    main()
