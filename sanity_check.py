import json
import random
from pathlib import Path
from collections import Counter

import cv2


# ====== 你只需要改这里 ======
DATA_ROOT = Path(r"E:\CE-CSL\CE-CSL")
MANIFEST_DIR = DATA_ROOT / "manifests"
SPLITS = ["train", "dev", "test"]
SAMPLE_PER_SPLIT = 50          # 随机抽样做“实际读取帧”检查（读太多会慢）
EXPORT_SUSPECTS = True         # 是否输出疑似异常样本列表
# ===========================


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def probe_video(video_path: str):
    """
    读取视频基本信息：是否可打开、帧数、fps、时长（秒）
    注意：某些编码 fps 可能为0，这种情况用 None 表示
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "frames": 0, "fps": None, "duration": None}

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = None

    duration = None
    if fps and frames > 0:
        duration = frames / fps

    # 额外读一帧，确认能解码（有些视频能 open 但读帧失败）
    ok_read, _ = cap.read()
    cap.release()

    if not ok_read:
        return {"ok": False, "frames": frames, "fps": fps, "duration": duration, "reason": "cannot decode first frame"}

    return {"ok": True, "frames": frames, "fps": fps, "duration": duration}


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = int(round((p / 100) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(k, len(sorted_vals) - 1))]


def summarize_numeric(name, values):
    values = [v for v in values if v is not None]
    values_sorted = sorted(values)
    if not values_sorted:
        print(f"{name}: (no data)")
        return
    print(f"{name}: n={len(values_sorted)} min={values_sorted[0]} p5={percentile(values_sorted,5)} "
          f"p50={percentile(values_sorted,50)} p95={percentile(values_sorted,95)} max={values_sorted[-1]}")


def main():
    random.seed(42)

    for split in SPLITS:
        path = MANIFEST_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        records = list(read_jsonl(path))
        n = len(records)
        print("\n" + "=" * 80)
        print(f"[{split}] samples: {n}")

        # gloss 基本统计
        gloss_lens = []
        empty_gloss = 0
        token_counter = Counter()

        for r in records:
            gloss = r.get("gloss", [])
            if not gloss:
                empty_gloss += 1
            gloss_lens.append(len(gloss))
            token_counter.update(gloss)

        print(f"Empty gloss: {empty_gloss}")
        summarize_numeric("Gloss length", gloss_lens)
        print(f"Gloss vocab size (in this split): {len(token_counter)}")

        # 视频检查：只抽样做真实打开/读帧（避免太慢）
        sample_n = min(SAMPLE_PER_SPLIT, n)
        sampled = random.sample(records, sample_n)

        video_frames = []
        video_fps = []
        video_duration = []
        bad_videos = []
        suspects = []

        for r in sampled:
            vid = r["video"]
            pid = r["id"]
            g_len = len(r.get("gloss", []))

            info = probe_video(vid)
            if not info["ok"]:
                bad_videos.append({"id": pid, "video": vid, "reason": info.get("reason", "open failed")})
                continue

            f = info["frames"]
            fps = info["fps"]
            dur = info["duration"]

            video_frames.append(f)
            if fps is not None:
                video_fps.append(fps)
            if dur is not None:
                video_duration.append(dur)

            # 简单“极端不匹配”启发式：视频很短但 gloss 很长，或 gloss 为空
            if g_len == 0 or (f > 0 and g_len > 0 and f / max(g_len, 1) < 2):
                suspects.append({
                    "id": pid,
                    "video": vid,
                    "frames": f,
                    "fps": fps,
                    "duration": dur,
                    "gloss_len": g_len,
                    "sentence": r.get("sentence", "")
                })

        print(f"\nVideo probe sample size: {sample_n}")
        print(f"Unreadable videos in sample: {len(bad_videos)}")
        if bad_videos:
            print("First 10 bad videos:")
            for b in bad_videos[:10]:
                print(f"  - {b['id']} | {b['reason']} | {b['video']}")

        summarize_numeric("Video frames (sampled)", video_frames)
        summarize_numeric("Video fps (sampled)", video_fps)
        summarize_numeric("Video duration seconds (sampled)", video_duration)

        # 导出疑似异常样本，供你人工检查
        if EXPORT_SUSPECTS and suspects:
            out = MANIFEST_DIR / f"{split}_suspects.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for s in suspects:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"\nExported suspects: {len(suspects)} -> {out}")

        # 可选：导出一个人工 spot-check 列表（随机10条）
        spot = random.sample(records, min(10, n))
        out_spot = MANIFEST_DIR / f"{split}_spotcheck.txt"
        with open(out_spot, "w", encoding="utf-8") as f:
            for r in spot:
                f.write(f"ID: {r['id']}\n")
                f.write(f"Video: {r['video']}\n")
                f.write(f"Gloss({len(r.get('gloss', []))}): {' '.join(r.get('gloss', []))}\n")
                f.write(f"Sentence: {r.get('sentence','')}\n")
                f.write("-" * 60 + "\n")
        print(f"Spot-check file -> {out_spot}")


if __name__ == "__main__":
    main()
