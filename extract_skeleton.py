import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

# MediaPipe Tasks (recommended; avoids mp.solutions.holistic binarypb issues)
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ============================
# Skeleton layout (Tasks 75)
# ============================
# 75 = pose(33) + left hand(21) + right hand(21)
POSE_J = 33
HAND_J = 21
J_TOTAL = POSE_J + HAND_J * 2
C = 3  # x, y, conf(presence)


def read_manifest_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _presence_to_conf(lm) -> float:
    """MediaPipe Tasks landmark presence can be None; map None->0.0."""
    try:
        v = getattr(lm, "presence", None)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def _lm_to_xyc(lm) -> Tuple[float, float, float]:
    return float(lm.x), float(lm.y), _presence_to_conf(lm)


class Tasks75Extractor:
    """
    Load landmarker models ONCE, then reuse for all frames/videos.
    This is much faster than reloading per video.
    """

    def __init__(self, pose_task: str, hand_task: str):
        for p in [pose_task, hand_task]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Model file not found: {p}")

        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_task),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
            )
        )
        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=hand_task),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
            )
        )

    def close(self) -> None:
        self.pose.close()
        self.hand.close()

    def detect_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        frame_rgb: HxWx3 uint8 RGB
        return: (75,3) float32
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # --- Pose (33) ---
        pose_xyc = np.zeros((POSE_J, 3), dtype=np.float32)
        pose_res = self.pose.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            lms = pose_res.pose_landmarks[0]
            n = min(POSE_J, len(lms))
            for j in range(n):
                pose_xyc[j] = _lm_to_xyc(lms[j])

        # --- Hands: fixed order [Left 21][Right 21] ---
        left_xyc = np.zeros((HAND_J, 3), dtype=np.float32)
        right_xyc = np.zeros((HAND_J, 3), dtype=np.float32)

        hand_res = self.hand.detect(mp_image)
        if hand_res.hand_landmarks:
            for idx, lms in enumerate(hand_res.hand_landmarks):
                label = None
                if hand_res.handedness and idx < len(hand_res.handedness) and hand_res.handedness[idx]:
                    label = hand_res.handedness[idx][0].category_name  # "Left"/"Right"
                target = left_xyc if label == "Left" else right_xyc if label == "Right" else None
                if target is None:
                    continue
                n = min(HAND_J, len(lms))
                for j in range(n):
                    target[j] = _lm_to_xyc(lms[j])

        return np.concatenate([pose_xyc, left_xyc, right_xyc], axis=0)  # (75,3)


def extract_one_video_tasks75(
    extractor: Tasks75Extractor,
    video_path: str,
    out_path: str,
    resize_hw: Optional[Tuple[int, int]] = (640, 360),  # (W,H) for cv2.resize
    max_frames: Optional[int] = None,
    frame_stride: int = 1,  # 1=every frame; 2=every 2 frames, etc.
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Extract skeleton (Tasks 75) for frames.
    Save as .npy: [F, 75, 3]
    """
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    n_read = 0
    n_used = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        n_read += 1

        if frame_stride > 1 and ((n_read - 1) % frame_stride != 0):
            continue

        if max_frames is not None and n_used >= max_frames:
            break

        if resize_hw is not None:
            frame_bgr = cv2.resize(frame_bgr, resize_hw, interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sk75 = extractor.detect_frame(frame_rgb)  # float32 (75,3)
        frames.append(sk75)
        n_used += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded: {video_path}")

    arr = np.stack(frames, axis=0)  # [F,75,3]
    if dtype == "float16":
        arr = arr.astype(np.float16)
    elif dtype == "float32":
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)

    dt = time.time() - t0
    return {
        "video": video_path,
        "out": out_path,
        "frames_read": n_read,
        "frames_used": int(arr.shape[0]),
        "shape": list(arr.shape),
        "seconds": float(dt),
    }


def run_split(
    extractor: Tasks75Extractor,
    manifest_jsonl: str,
    out_dir: str,
    resume: bool = True,
    resize_hw: Optional[Tuple[int, int]] = (640, 360),
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    dtype: str = "float16",
    limit: Optional[int] = 10,
) -> Dict[str, Any]:
    """
    Read manifest jsonl with fields:
      - id: sample id, e.g. "train-00001"
      - video: absolute mp4 path

    Output:
      out_dir/{id}.npy
      out_dir/failures.txt
      out_dir/failures_traceback.txt
      out_dir/meta.json
    """
    t0 = time.time()
    out_dir_p = Path(out_dir)
    ensure_dir(out_dir_p)

    items = read_manifest_jsonl(manifest_jsonl)
    total_all = len(items)
    if limit is not None:
        items = items[:limit]

    failures: List[Tuple[str, str, str]] = []
    processed = 0
    skipped = 0
    total_frames = 0

    tb_path = out_dir_p / "failures_traceback.txt"
    fail_path = out_dir_p / "failures.txt"

    # clear old logs for this run
    if tb_path.exists():
        tb_path.unlink()
    if fail_path.exists():
        fail_path.unlink()

    for rec in items:
        sid = rec.get("id", "")
        vpath = rec.get("video", "")
        out_path = str(out_dir_p / f"{sid}.npy")

        if resume and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            rep = extract_one_video_tasks75(
                extractor=extractor,
                video_path=vpath,
                out_path=out_path,
                resize_hw=resize_hw,
                max_frames=max_frames,
                frame_stride=frame_stride,
                dtype=dtype,
            )
            processed += 1
            total_frames += int(rep["frames_used"])
        except Exception as e:
            failures.append((sid, vpath, f"{type(e).__name__}: {e}"))
            # append traceback
            with open(tb_path, "a", encoding="utf-8") as f:
                f.write("--\n")
                f.write(f"ID: {sid}\n")
                f.write(f"VIDEO: {vpath}\n")
                f.write(traceback.format_exc())
                f.write("\n")

    dt = time.time() - t0
    meta = {
        "manifest": manifest_jsonl,
        "out_dir": out_dir,
        "J": J_TOTAL,
        "C": C,
        "dtype": dtype,
        "resize_hw": list(resize_hw) if resize_hw is not None else None,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "limit": limit,
        "total_in_manifest": total_all,
        "total_considered": len(items),
        "skipped": skipped,
        "processed": processed,
        "failures": len(failures),
        "total_frames_processed": total_frames,
        "total_seconds": float(dt),
        "avg_fps_effective": float(total_frames / dt) if dt > 0 else 0.0,
        "notes": "MediaPipe Tasks: pose(33)+hands(21*2) => [F,75,3].",
    }

    with open(out_dir_p / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if failures:
        with open(fail_path, "w", encoding="utf-8") as fw:
            for sid, vpath, err in failures:
                fw.write(f"{sid}\t{vpath}\t{err}\n")

    return meta


if __name__ == "__main__":
    # -----------------------------
    # You only need to edit ROOT / MODEL_DIR if needed
    # -----------------------------
    ROOT = Path(r"E:\CE-CSL\CE-CSL")  # <<< change to your dataset root

    # Where you put .task models:
    #   pose_landmarker_full.task
    #   hand_landmarker.task
    MODEL_DIR = ROOT / "mp_tasks_models"
    POSE_TASK = str(MODEL_DIR / "pose_landmarker_full.task")
    HAND_TASK = str(MODEL_DIR / "hand_landmarker.task")

    # Output directory: [F,75,3]
    OUT_ROOT = ROOT / "skeleton_tasks75"

    # First run a small test to confirm everything is OK
    TEST_LIMIT = None  # set None for full run

    print("mediapipe:", mp.__version__)
    print("POSE_TASK:", POSE_TASK)
    print("HAND_TASK:", HAND_TASK)

    extractor = Tasks75Extractor(POSE_TASK, HAND_TASK)
    try:
        for split in ["train", "dev", "test"]:
            manifest = ROOT / "manifests" / f"{split}.jsonl"
            out_dir = OUT_ROOT / split

            print(f"\n=== Split: {split} ===")
            print(f"Manifest: {manifest}")
            print(f"Out dir : {out_dir}")

            meta = run_split(
                extractor=extractor,
                manifest_jsonl=str(manifest),
                out_dir=str(out_dir),
                resume=True,
                resize_hw=(640, 360),
                max_frames=None,        # None = all frames (A strategy)
                frame_stride=1,         # 1 = every frame; increase to speed up
                dtype="float16",
                limit=TEST_LIMIT,
            )
            print("Meta:", json.dumps(meta, ensure_ascii=False, indent=2))
    finally:
        extractor.close()

    # After the test passes:
    # 1) set TEST_LIMIT=None
    # 2) (optional) set frame_stride=2 to speed up (at the cost of temporal resolution)
