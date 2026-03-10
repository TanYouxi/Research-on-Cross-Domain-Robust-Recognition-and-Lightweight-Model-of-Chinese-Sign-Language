import json
from pathlib import Path

# ===== 修改为你的路径 =====
DATA_ROOT = Path(r"E:\CE-CSL\CE-CSL")
MANIFEST_PATH = DATA_ROOT / "manifests" / "train.jsonl"
OUT_PATH = DATA_ROOT / "manifests" / "gloss_vocab.json"
# ==========================


def main():
    gloss_set = set()

    # 1️⃣ 读取 train.jsonl
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            gloss_tokens = record["gloss"]
            for token in gloss_tokens:
                gloss_set.add(token)

    # 2️⃣ 排序（保证稳定）
    gloss_list = sorted(list(gloss_set))

    # 3️⃣ 构建 vocab（blank=0）
    vocab = {"<blank>": 0}
    for i, token in enumerate(gloss_list, start=1):
        vocab[token] = i

    # 4️⃣ 保存
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("Vocab size (including blank):", len(vocab))
    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
