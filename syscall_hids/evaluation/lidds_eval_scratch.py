"""
LID-DS Step 6: From-scratch Training & Evaluation

Melatih arsitektur Multi-Scale CNN dari nol pada LID-DS menggunakan
vocabulary native (69 token). Ini adalah baseline cross-dataset paling fair
(representasi konsisten dalam-dataset) dan menjadi upper bound performa LID-DS.

Split anti-leakage (recording-level):
  - Test set : test/normal (benign) + 50% attack recording (held out)
  - Train pool: training+validation (benign) + 50% attack recording
  - Benign di-subsample ke max 200K (keputusan: hemat compute untuk SQL),
    malicious dipertahankan penuh; seed dicatat untuk reproduktibilitas.
  - Train pool dibagi 85/15 tingkat-recording untuk pemilihan checkpoint val-F1.

Output: results/lidds_scratch_report.json, models/lidds_scratch_<scenario>.pt
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import syscall_hids.data.lidds_common as C

SCENARIOS = ["CVE-2014-0160", "Bruteforce_CWE-307", "CWE-89-SQL-injection"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MAX_BENIGN = 200_000
EPOCHS = 20
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
VAL_FRAC = 0.15
MAX_CLASS_WEIGHT = 20.0  # cap bobot malicious (hindari over-prediksi)


def build_pools(data):
    """Return (train_win, train_lbl, test_win, test_lbl) dengan split
    recording-level pada partisi attack + benign dari train/val/test_normal."""
    part = data["partition"]; win = data["windows"]; lbl = data["labels"]
    tr_attack_mask, ev_attack_mask = C.split_attack_recordings(data, 0.5, SEED)

    # Train pool: benign (train+val partition) + attack-train
    benign_train_mask = (part == C.P_TRAIN) | (part == C.P_VAL)
    train_idx = np.where(benign_train_mask | tr_attack_mask)[0]
    # Test pool: benign (test_normal) + attack-eval
    test_idx = np.where((part == C.P_TEST_NORMAL) | ev_attack_mask)[0]

    return (win[train_idx], lbl[train_idx], win[test_idx], lbl[test_idx],
            data["rec_ids"][train_idx])


def train_one(scn, vocab_size):
    data = C.load_scenario(scn)
    tr_w, tr_l, te_w, te_l, tr_r = build_pools(data)

    # Subsample benign di train
    tr_w, tr_l, tr_r = C.subsample_benign(tr_w, tr_l, MAX_BENIGN, SEED, rec_ids=tr_r)

    # Pad
    tr_w = C.pad_to(tr_w, C.MAX_LEN); te_w = C.pad_to(te_w, C.MAX_LEN)

    # Split train/val tingkat-recording (tanpa kebocoran window overlap)
    tr_idx, val_idx = C.recording_val_split(tr_r, tr_l, VAL_FRAC, SEED)
    Xtr, ytr = tr_w[tr_idx], tr_l[tr_idx]
    Xva, yva = tr_w[val_idx], tr_l[val_idx]

    # Class weights (inverse freq, di-cap agar tidak degenerate pada
    # skenario dengan malicious sangat langka spt CVE-2014-0160/Heartbleed)
    n0 = int((ytr == 0).sum()); n1 = int((ytr == 1).sum())
    raw_w = n0 / max(n1, 1)
    mal_w = float(min(max(raw_w, 1.0), MAX_CLASS_WEIGHT))
    w = torch.tensor([1.0, mal_w], dtype=torch.float32, device=DEVICE)

    model = C.MultiScaleEncoder(vocab_size=vocab_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.long),
                      torch.tensor(ytr, dtype=torch.long)),
        batch_size=BATCH, shuffle=True)

    best_f1, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
        # Val
        vp = C.infer_probs(model, Xva, DEVICE)
        vm = C.compute_metrics(vp, yva)
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"    epoch {ep:2d}: val_F1={vm['f1']:.4f} val_AUC={vm['auc']:.4f}")

    model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "scenario": scn,
                "vocab_size": vocab_size, "best_val_f1": best_f1},
               C.MODEL_DIR / f"lidds_scratch_{scn}.pt")

    # Cari threshold optimal-F1 pada val (base-rate antar-skenario sangat
    # berbeda, threshold tetap 0,5 tidak adil). AUC/AUPRC tetap utama.
    vp = C.infer_probs(model, Xva, DEVICE)
    best_thr, best_vf1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        f1 = C.compute_metrics(vp, yva, thr)["f1"]
        if f1 > best_vf1:
            best_vf1, best_thr = f1, float(thr)

    # Test pada threshold optimal-val
    tp = C.infer_probs(model, te_w, DEVICE)
    tm = C.compute_metrics(tp, te_l, best_thr)
    tm_05 = C.compute_metrics(tp, te_l, 0.5)
    return tm, {"train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
                "test_windows": int(len(te_l)), "test_malicious": int(te_l.sum()),
                "best_val_f1": best_f1, "tuned_threshold": best_thr,
                "metrics_at_0.5": tm_05}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    vocab = json.load(open(C.PREPARED_LIDDS / "vocabulary_native.json"))
    vocab_size = len(vocab)

    print("=" * 64)
    print("LID-DS Step 6: From-scratch Training (Mode A, native vocab)")
    print(f"Device: {DEVICE}  | vocab_size={vocab_size}")
    print("=" * 64)

    report = {"mode": "from_scratch", "vocab_size": vocab_size,
              "config": {"epochs": EPOCHS, "lr": LR, "max_benign": MAX_BENIGN,
                         "seed": SEED}, "scenarios": {}}

    for scn in SCENARIOS:
        print(f"\n[{scn}] training...")
        tm, info = train_one(scn, vocab_size)
        report["scenarios"][scn] = {"metrics": tm, "info": info}
        print(f"  TEST: Recall={tm['recall']:.4f} FPR={tm['fpr']:.4f} "
              f"F1={tm['f1']:.4f} AUC={tm['auc']:.4f} AUPRC={tm['auprc']:.4f}")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "lidds_scratch_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 64)
    print("Ringkasan From-scratch:")
    print(f"  {'Skenario':<24}{'Recall':>8}{'FPR':>8}{'F1':>8}{'AUC':>8}{'AUPRC':>8}")
    print("  " + "-" * 64)
    for scn, r in report["scenarios"].items():
        m = r["metrics"]
        print(f"  {scn:<24}{m['recall']:>8.3f}{m['fpr']:>8.3f}"
              f"{m['f1']:>8.3f}{m['auc']:>8.3f}{m['auprc']:>8.3f}")
    print(f"\n  Saved: results/lidds_scratch_report.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
