import csv
import json
from pathlib import Path

# ====== 你只需要改这里 ======
DATA_ROOT = Path(r"E:\CE-CSL\CE-CSL")  # 改成你的数据集根目录
# 期望目录：
# DATA_ROOT / "video" / split / signer / "{id}.mp4"
# DATA_ROOT / "label" / "{split}.csv"
OUT_DIR = DATA_ROOT / "manifests"
# ===========================


def index_videos(video_root: Path):
    """
    扫描 video_root 下所有 mp4，建立：basename -> (fullpath, signer)
    basename: dev-00001 （不含 .mp4）
    """
    mp4s = list(video_root.rglob("*.mp4"))
    idx = {}
    dup = {}

    for p in mp4s:
        base = p.stem  # dev-00001
        signer = p.parent.name  # A/B/C...
        if base in idx:
            dup.setdefault(base, []).append(str(p))
        else:
            idx[base] = (str(p), signer)

    return idx, dup


def parse_gloss(gloss_str: str):
    """
    你的 gloss 格式是用 / 分割，例如：10/年/鱼/禁止1/区/时间/长/不
    """
    if gloss_str is None:
        return []
    gloss_str = gloss_str.strip()
    if not gloss_str:
        return []
    # 去掉可能出现的首尾分隔符
    gloss_str = gloss_str.strip("/")
    tokens = [t.strip() for t in gloss_str.split("/") if t.strip()]
    return tokens


def build_split(split: str):
    label_csv = DATA_ROOT / "label" / f"{split}.csv"
    video_root = DATA_ROOT / "video" / split

    if not label_csv.exists():
        raise FileNotFoundError(f"Label file not found: {label_csv}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video folder not found: {video_root}")

    video_idx, dup = index_videos(video_root)

    if dup:
        print(f"[WARN] Duplicate video ids in {split}: {len(dup)}")
        # 打印前几个重复
        for k in list(dup.keys())[:5]:
            print(f"  - {k}: {dup[k]}")

    out_path = OUT_DIR / f"{split}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, ok, miss = 0, 0, 0
    missing_ids = []

    with open(label_csv, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        # 你截图里字段名类似：Number, Translator, Chinese Sentences, Gloss, Note
        # 为兼容可能的列名差异，这里做个安全读取
        for row in reader:
            total += 1

            sample_id = (row.get("Number") or row.get("number") or "").strip()
            translator = (row.get("Translator") or row.get("translator") or "").strip()
            sentence = (row.get("Chinese Sentences") or row.get("Chinese") or row.get("Sentence") or "").strip()
            gloss_raw = (row.get("Gloss") or row.get("gloss") or "").strip()
            note = (row.get("Note") or row.get("note") or "").strip()

            if not sample_id:
                # 跳过空行或异常行
                continue

            # video file id 可能是 sample_id 本身（如 dev-00001）
            # 也可能 row 里带 .mp4，这里统一去掉后缀
            sid = sample_id.replace(".mp4", "")

            if sid not in video_idx:
                miss += 1
                missing_ids.append(sid)
                continue

            video_path, signer = video_idx[sid]

            record = {
                "id": sid,
                "split": split,
                "video": video_path,
                "signer": signer,
                "gloss": parse_gloss(gloss_raw),
                "sentence": sentence,
                "translator": translator,
                "note": note,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            ok += 1

    print(f"[{split}] total rows: {total}, written: {ok}, missing video: {miss}")
    if missing_ids:
        print(f"[{split}] First 20 missing ids: {missing_ids[:20]}")
    print(f"[{split}] Output: {out_path}")


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        build_split(split)
