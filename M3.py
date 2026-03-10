
# ============================================================
# M2: RGB + Skeleton + Cross‑Modal Perception Alignment (Full)
# CE‑CSL Continuous Sign Language Recognition
# ============================================================

import os
import json
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CONFIG = {
    "base_dir": r"E:\CE-CSL\CE-CSL",

    "train_manifest": r"E:\CE-CSL\CE-CSL\manifests\train.jsonl",
    "dev_manifest": r"E:\CE-CSL\CE-CSL\manifests\dev.jsonl",
    "test_manifest": r"E:\CE-CSL\CE-CSL\manifests\test.jsonl",

    "skeleton_root": r"E:\CE-CSL\CE-CSL\skeleton_tasks75",

    "vocab_path": r"E:\CE-CSL\CE-CSL\vocab_m0_ctc.json",

    "experiment_name": "m2_align",
    "output_dir": r"E:\CE-CSL\CE-CSL\experiments\m2_align",

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "epochs": 55,
    "batch_size": 2,
    "num_workers": 0,

    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    "image_size": 224,
    "sample_stride": 2,
    "max_frames": 96,

    "frame_feature_dim": 576,
    "fusion_dim": 512,

    "skeleton_num_joints": 75,
    "skeleton_channels": 3,

    "tcn_hidden_dim": 512,
    "tcn_layers": 4,

    "log_interval": 20,
}


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i

    for j in range(m+1):
        dp[0][j] = j

    for i in range(1,n+1):
        for j in range(1,m+1):

            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )

    return dp[n][m]


def compute_wer(refs, hyps):

    total_words = 0
    total_err = 0

    for r,h in zip(refs,hyps):

        total_words += len(r)

        total_err += edit_distance(r,h)

    return total_err / total_words if total_words>0 else 0


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class CECSLDataset(Dataset):

    def __init__(self, manifest, vocab, cfg, split):

        self.items = read_jsonl(manifest)
        self.vocab = vocab
        self.cfg = cfg
        self.split = split

    def __len__(self):
        return len(self.items)

    def load_video(self, path):

        cap = cv2.VideoCapture(path)

        frames = []

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.cfg["image_size"], self.cfg["image_size"]))

            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2,0,1))

            frames.append(frame)

        cap.release()

        frames = frames[::self.cfg["sample_stride"]]

        if len(frames) > self.cfg["max_frames"]:

            idx = np.linspace(0,len(frames)-1,self.cfg["max_frames"]).astype(int)

            frames = [frames[i] for i in idx]

        frames = np.stack(frames)

        return torch.tensor(frames)

    def load_skeleton(self, sid, T):

        path = os.path.join(self.cfg["skeleton_root"], self.split, sid + ".npy")

        skel = np.load(path)

        idx = np.linspace(0,skel.shape[0]-1,T).astype(int)

        skel = skel[idx]

        return torch.tensor(skel).float()

    def __getitem__(self, idx):

        item = self.items[idx]

        frames = self.load_video(item["video"])

        T = frames.shape[0]

        skeleton = self.load_skeleton(item["id"], T)

        gloss = item["gloss"].split()

        target = [self.vocab[g] for g in gloss if g in self.vocab]

        return {
            "frames": frames,
            "skeleton": skeleton,
            "target": torch.tensor(target),
            "gloss": gloss,
        }


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class RGBEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        net = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.features = net.features
        self.pool = net.avgpool

    def forward(self,x):

        B,T,C,H,W = x.shape

        x = x.view(B*T,C,H,W)

        f = self.features(x)

        f = self.pool(f).flatten(1)

        f = f.view(B,T,-1)

        return f


class SkeletonEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(75*3,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )

    def forward(self,x):

        B,T,J,C = x.shape

        x = x.view(B,T,J*C)

        return self.net(x)


class CrossModalAttention(nn.Module):

    def __init__(self,dim):
        super().__init__()

        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)

        self.scale = dim ** -0.5

    def forward(self,rgb,skel):

        Q = self.q(rgb)
        K = self.k(skel)
        V = self.v(skel)

        attn = torch.matmul(Q,K.transpose(-2,-1))*self.scale
        attn = torch.softmax(attn,dim=-1)

        out = torch.matmul(attn,V)

        return out


class TCN(nn.Module):

    def __init__(self,dim,layers):
        super().__init__()

        blocks=[]

        for _ in range(layers):
            blocks.append(nn.Conv1d(dim,dim,3,padding=1))

        self.net = nn.Sequential(*blocks)

    def forward(self,x):

        x = x.transpose(1,2)

        x = self.net(x)

        x = x.transpose(1,2)

        return x


class M2Model(nn.Module):

    def __init__(self,vocab_size,cfg):

        super().__init__()

        self.rgb = RGBEncoder()

        self.skel = SkeletonEncoder()

        self.rgb_proj = nn.Linear(cfg["frame_feature_dim"],cfg["fusion_dim"])
        self.skel_proj = nn.Linear(256,cfg["fusion_dim"])

        self.align = CrossModalAttention(cfg["fusion_dim"])

        self.fusion = nn.Linear(cfg["fusion_dim"]*2,cfg["fusion_dim"])

        self.tcn = TCN(cfg["fusion_dim"],cfg["tcn_layers"])

        self.cls = nn.Linear(cfg["fusion_dim"],vocab_size)

    def forward(self,frames,skeleton):

        rgb = self.rgb(frames)
        rgb = self.rgb_proj(rgb)

        skel = self.skel(skeleton)
        skel = self.skel_proj(skel)

        aligned = self.align(rgb,skel)

        x = torch.cat([rgb,aligned],dim=-1)

        x = self.fusion(x)

        x = self.tcn(x)

        logits = self.cls(x)

        return F.log_softmax(logits,dim=-1)


# ------------------------------------------------------------
# Decode
# ------------------------------------------------------------

def greedy_decode(log_probs):

    pred = log_probs.argmax(-1)

    hyps=[]

    for seq in pred:

        last=None
        out=[]

        for p in seq:

            p=p.item()

            if p!=0 and p!=last:
                out.append(p)

            last=p

        hyps.append(out)

    return hyps


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

def train():

    cfg=CONFIG

    set_seed(cfg["seed"])

    device=cfg["device"]

    os.makedirs(cfg["output_dir"],exist_ok=True)

    with open(cfg["vocab_path"]) as f:
        vocab=json.load(f)

    inv_vocab={v:k for k,v in vocab.items()}

    train_ds=CECSLDataset(cfg["train_manifest"],vocab,cfg,"train")
    dev_ds=CECSLDataset(cfg["dev_manifest"],vocab,cfg,"dev")

    train_loader=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True)
    dev_loader=DataLoader(dev_ds,batch_size=cfg["batch_size"])

    model=M2Model(len(vocab),cfg).to(device)

    criterion=nn.CTCLoss(blank=0)

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    best_wer=1.0

    for epoch in range(cfg["epochs"]):

        model.train()

        for i,batch in enumerate(train_loader):

            frames=batch["frames"].to(device)
            skeleton=batch["skeleton"].to(device)

            targets=batch["target"]

            optimizer.zero_grad()

            log_probs=model(frames,skeleton)

            T=log_probs.size(1)

            log_probs=log_probs.permute(1,0,2)

            input_lengths=torch.full((frames.size(0),),T).to(device)

            target_lengths=torch.tensor([len(t) for t in targets]).to(device)

            loss=criterion(
                log_probs,
                torch.cat(targets).to(device),
                input_lengths,
                target_lengths
            )

            loss.backward()

            optimizer.step()

            if i%cfg["log_interval"]==0:
                print(f"Epoch {epoch} Batch {i} Loss {loss.item():.4f}")


        # -------- DEV EVAL --------

        model.eval()

        refs=[]
        hyps=[]

        with torch.no_grad():

            for batch in dev_loader:

                frames=batch["frames"].to(device)
                skeleton=batch["skeleton"].to(device)

                log_probs=model(frames,skeleton)

                pred=greedy_decode(log_probs.cpu())

                hyps+=pred

                for g in batch["gloss"]:

                    refs.append([vocab[x] for x in g if x in vocab])

        wer=compute_wer(refs,hyps)

        print("DEV WER:",wer)

        if wer<best_wer:

            best_wer=wer

            torch.save(
                model.state_dict(),
                os.path.join(cfg["output_dir"],"best_model.pt")
            )

            print("Saved best model")


if __name__=="__main__":
    train()
