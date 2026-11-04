"""
Helper bersama untuk evaluasi cross-dataset LID-DS (Step 4-6).

Menyediakan: definisi model (identik dengan robust_encoder.py), pemuatan
data LID-DS, remap token native->ADFA (Mode B), split attack recording-level
(anti-leakage), subsample benign, dan komputasi metrik.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

PREPARED_LIDDS = Path("prepared_data_lidds")
PREPARED_ADFA = Path("prepared_data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Partition id (sesuai lidds_preparation.py)
P_TRAIN, P_VAL, P_TEST_NORMAL, P_TEST_ATTACK = 0, 1, 2, 3

MAX_LEN = 130  # konsisten dengan ADFA (model pakai AdaptiveMaxPool, fleksibel)

# Arsitektur ADFA (untuk zero-shot/fine-tune). vocab_size dibaca dari data yang
# disiapkan (kosakata dibangun dari trace latih saja), bukan di-hard-code.
ADFA_VOCAB_SIZE = json.load(open(PREPARED_ADFA / "metadata.json"))["vocab_size"]
ADFA_CFG = dict(vocab_size=ADFA_VOCAB_SIZE, embed_dim=64, conv_channels=64,
                kernels=[3, 5, 7], hidden_dim=128, num_classes=2, dropout=0.3)


# ===========================================================================
# Model (identik dengan robust_encoder.py / baselines_ablation.py)
# ===========================================================================
class MultiScaleEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, conv_channels=64,
                 kernels=(3, 5, 7), hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, conv_channels, k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            for k in kernels
        ])
        rep_dim = conv_channels * len(kernels)
        self.fc_rep = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

    def encode(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        conv_outs = [block(e).squeeze(-1) for block in self.conv_blocks]
        h = torch.cat(conv_outs, dim=1)
        return self.fc_rep(h)

    def forward(self, x):
        h = self.encode(x)
        return self.classifier(h), h


def load_adfa_model(tag="A4_full", seed=42, device="cpu"):
    model = MultiScaleEncoder(**{k: (tuple(v) if k == "kernels" else v)
                                 for k, v in ADFA_CFG.items()}).to(device)
    ckpt = torch.load(MODEL_DIR / f"{tag}_seed{seed}.pt",
                      weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


# ===========================================================================
# Data loading
# ===========================================================================
def load_scenario(scenario):
    d = np.load(PREPARED_LIDDS / f"{scenario}.npz")
    return {
        "windows": d["windows"].astype(np.int64),
        "labels": d["labels"].astype(np.int64),
        "partition": d["partition"].astype(np.int64),
        "rec_ids": d["rec_ids"].astype(np.int64),
    }


def build_adfa_remap():
    """Array remap: native_token_id -> ADFA model input token.
    PAD(0)->0, UNK(1)->1, syscall name -> model_input_token (aligned)."""
    native = json.load(open(PREPARED_LIDDS / "vocabulary_native.json"))
    aligned = json.load(open(PREPARED_LIDDS / "vocabulary_aligned.json"))
    id2name = {v: k for k, v in native.items()}
    size = len(native)
    remap = np.ones(size, dtype=np.int64)  # default ke <UNK>=1
    remap[0] = 0  # PAD
    remap[1] = 1  # UNK
    for tid in range(2, size):
        name = id2name[tid]
        if name in aligned:
            remap[tid] = aligned[name]["model_input_token"]
        else:
            remap[tid] = 1
    return remap


def remap_windows(windows, remap):
    """Vectorized remap native->ADFA token space."""
    return remap[windows]


def token_coverage(windows, partition_mask=None):
    """Coverage berbobot-frekuensi: distribusi kind atas token aktual.
    Return dict kind->count (mengabaikan PAD=0)."""
    aligned = json.load(open(PREPARED_LIDDS / "vocabulary_aligned.json"))
    native = json.load(open(PREPARED_LIDDS / "vocabulary_native.json"))
    id2name = {v: k for k, v in native.items()}
    w = windows if partition_mask is None else windows[partition_mask]
    flat = w.reshape(-1)
    flat = flat[flat != 0]  # buang PAD
    counts = {"direct": 0, "socket_collapsed": 0, "ipc_collapsed": 0,
              "abi_unmapped": 0, "unk_native": 0}
    # hitung via bincount lalu map id->kind
    binc = np.bincount(flat, minlength=len(native))
    for tid in range(1, len(native)):
        c = int(binc[tid])
        if c == 0:
            continue
        if tid == 1:
            counts["unk_native"] += c
            continue
        name = id2name[tid]
        kind = aligned.get(name, {}).get("kind", "abi_unmapped")
        counts[kind] += c
    return counts


# ===========================================================================
# Recording-level split (anti-leakage) untuk fine-tune & from-scratch
# ===========================================================================
def split_attack_recordings(data, attack_frac_train=0.5, seed=42):
    """Bagi recording di partisi TEST_ATTACK ke train vs eval pada level
    recording (bukan window) agar tidak ada kebocoran.
    Return (train_mask, eval_mask) untuk window-window di partisi attack."""
    part = data["partition"]
    rec = data["rec_ids"]
    attack_mask = part == P_TEST_ATTACK
    attack_recs = np.unique(rec[attack_mask])
    rng = np.random.RandomState(seed)
    shuffled = attack_recs.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * attack_frac_train)
    train_recs = set(shuffled[:n_train].tolist())

    train_mask = np.zeros(len(part), dtype=bool)
    eval_mask = np.zeros(len(part), dtype=bool)
    for i in np.where(attack_mask)[0]:
        if rec[i] in train_recs:
            train_mask[i] = True
        else:
            eval_mask[i] = True
    return train_mask, eval_mask


def subsample_benign(windows, labels, max_benign=200_000, seed=42, rec_ids=None):
    """Subsample window benign (label 0), pertahankan semua malicious.
    Jika rec_ids diberikan, rec_ids ikut disubsample dan dikembalikan."""
    rng = np.random.RandomState(seed)
    benign_idx = np.where(labels == 0)[0]
    mal_idx = np.where(labels == 1)[0]
    if len(benign_idx) > max_benign:
        benign_idx = rng.choice(benign_idx, size=max_benign, replace=False)
    keep = np.concatenate([benign_idx, mal_idx])
    rng.shuffle(keep)
    if rec_ids is not None:
        return windows[keep], labels[keep], rec_ids[keep]
    return windows[keep], labels[keep]


def recording_val_split(rec_ids, labels, val_frac=0.15, seed=42):
    """Split validasi tingkat-RECORDING (window yang overlap dari satu recording
    tidak pernah terbagi ke train dan val). Stratifikasi: recording yang memuat
    window malicious vs tidak. Return (train_idx, val_idx)."""
    recs = np.unique(rec_ids)
    has_mal = {r: False for r in recs}
    for r in np.unique(rec_ids[labels == 1]):
        has_mal[r] = True
    rng = np.random.RandomState(seed)
    val_recs = []
    for flag in (False, True):
        grp = np.array([r for r in recs if has_mal[r] == flag])
        rng.shuffle(grp)
        val_recs.extend(grp[:max(1, int(round(len(grp) * val_frac)))] if len(grp) else [])
    vmask = np.isin(rec_ids, val_recs)
    return np.where(~vmask)[0], np.where(vmask)[0]


# ===========================================================================
# Metrics
# ===========================================================================
@torch.no_grad()
def infer_probs(model, windows, device, batch_size=512):
    model.eval()
    X = torch.tensor(windows, dtype=torch.long)
    out = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size].to(device)
        logits, _ = model(batch)
        out.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    try:
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    try:
        auprc = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    except ValueError:
        auprc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1,
            "fpr": fpr, "auc": auc, "auprc": auprc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pad_to(windows, length=MAX_LEN):
    if windows.shape[1] >= length:
        return windows[:, :length]
    pad = np.zeros((windows.shape[0], length - windows.shape[1]), dtype=windows.dtype)
    return np.concatenate([windows, pad], axis=1)
