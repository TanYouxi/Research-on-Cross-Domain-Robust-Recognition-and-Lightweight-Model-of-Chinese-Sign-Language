import os, json, ast, cv2
import numpy as np

DATA_ROOT = "/root/autodl-tmp/CE-CSL"

def ensure_gloss_tokens(gloss):
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

def sample_frame_indices(num_frames, stride=2, max_frames=96):
    idxs = list(range(0, num_frames, max(1, stride)))
    if max_frames is not None and len(idxs) > max_frames:
        base = idxs
        pick = np.linspace(0, len(base) - 1, max_frames).round().astype(int).tolist()
        idxs = [base[i] for i in pick]
    return idxs

def resolve_video_path(video_path):
    vp = str(video_path).replace("\\", "/")
    old_prefix = "E:/CE-CSL/CE-CSL"
    if vp.startswith(old_prefix):
        vp = vp.replace(old_prefix, DATA_ROOT, 1)
    return vp

bad_ids = set()
bad_records = []

for split in ["train", "dev", "test"]:
    manifest = os.path.join(DATA_ROOT, "manifests", f"{split}.jsonl")
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["id"]

            # 1. gloss检查
            gloss_tokens = ensure_gloss_tokens(rec.get("gloss", []))
            if len(gloss_tokens) == 0:
                bad_ids.add(sid)
                bad_records.append((sid, split, "empty_target"))
                continue

            # 2. video检查
            video_path = resolve_video_path(rec["video"])
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                bad_ids.add(sid)
                bad_records.append((sid, split, "video_open_fail"))
                continue

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            if num_frames <= 0:
                bad_ids.add(sid)
                bad_records.append((sid, split, "bad_video_frames"))
                continue

            # 3. CTC长度检查
            input_len = len(sample_frame_indices(num_frames, stride=2, max_frames=96))
            input_len = max(input_len, 8)

            if input_len < len(gloss_tokens):
                bad_ids.add(sid)
                bad_records.append((sid, split, f"ctc_len_invalid input={input_len} target={len(gloss_tokens)}"))
                continue

            # 4. skeleton检查
            sk_path = os.path.join(DATA_ROOT, "skeleton_tasks75", split, f"{sid}.npy")
            try:
                arr = np.load(sk_path, allow_pickle=True).astype(np.float32)
            except Exception as e:
                bad_ids.add(sid)
                bad_records.append((sid, split, f"skeleton_load_fail {repr(e)}"))
                continue

            if arr.ndim != 3 or arr.shape[1:] != (75, 3):
                bad_ids.add(sid)
                bad_records.append((sid, split, f"bad_skeleton_shape {arr.shape}"))
                continue

            if not np.isfinite(arr).all():
                bad_ids.add(sid)
                bad_records.append((sid, split, "skeleton_nan_inf"))
                continue

# 保存坏样本报告
report_path = os.path.join(DATA_ROOT, "manifests", "bad_samples_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    for sid, split, reason in bad_records:
        f.write(f"{split}\t{sid}\t{reason}\n")

print("total bad ids:", len(bad_ids))
print("report saved to:", report_path)

# 生成 final manifest
for split in ["train", "dev", "test"]:
    in_path = os.path.join(DATA_ROOT, "manifests", f"{split}_clean.jsonl")
    out_path = os.path.join(DATA_ROOT, "manifests", f"{split}_final.jsonl")

    kept = 0
    removed = 0

    with open(in_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as g:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in bad_ids:
                removed += 1
                continue
            g.write(line)
            kept += 1

    print(f"{split}: kept={kept}, removed={removed}, out={out_path}")