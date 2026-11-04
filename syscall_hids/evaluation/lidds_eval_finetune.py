"""
LID-DS Step 5: Fine-tune Transfer Evaluation

Model A4 ADFA dijadikan titik awal: blok konvolusi (ekstraktor pola lokal
yang dipelajari dari ADFA) DIBEKUKAN, sementara embedding + representation
layer + classifier di-fine-tune pada LID-DS (vocab aligned Mode B).

Menguji: seberapa banyak filter konvolusi ADFA dapat ditransfer ke LID-DS
dengan adaptasi minimal. Berada di antara zero-shot (gagal) dan from-scratch.

Split anti-leakage identik dengan from-scratch (recording-level).
Metrik dilaporkan pada threshold optimal-val; AUC/AUPRC threshold-independent.

Output: results/lidds_finetune_report.json, models/lidds_finetune_<scenario>.pt
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import syscall_hids.data.lidds_common as C
from syscall_hids.evaluation.lidds_eval_scratch import build_pools

SCENARIOS = ["CVE-2014-0160", "Bruteforce_CWE-307", "CWE-89-SQL-injection"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MAX_BENIGN = 200_000
EPOCHS = 15
BATCH = 256
LR = 1e-4               # lebih kecil dari from-scratch (fine-tuning)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
VAL_FRAC = 0.15
MAX_CLASS_WEIGHT = 20.0


def finetune_one(scn, remap):
    data = C.load_scenario(scn)
    tr_w, tr_l, te_w, te_l, tr_r = build_pools(data)
    tr_w, tr_l, tr_r = C.subsample_benign(tr_w, tr_l, MAX_BENIGN, SEED, rec_ids=tr_r)

    # Remap native -> ADFA token space, lalu pad
    tr_w = C.pad_to(C.remap_windows(tr_w, remap), C.MAX_LEN)
    te_w = C.pad_to(C.remap_windows(te_w, remap), C.MAX_LEN)

    tr_idx, val_idx = C.recording_val_split(tr_r, tr_l, VAL_FRAC, SEED)  # recording-level
    Xtr, ytr = tr_w[tr_idx], tr_l[tr_idx]
    Xva, yva = tr_w[val_idx], tr_l[val_idx]

    n0 = int((ytr == 0).sum()); n1 = int((ytr == 1).sum())
    mal_w = float(min(max(n0 / max(n1, 1), 1.0), MAX_CLASS_WEIGHT))
    w = torch.tensor([1.0, mal_w], dtype=torch.float32, device=DEVICE)

    # Mulai dari model ADFA terlatih, bekukan blok konvolusi
    model = C.load_adfa_model("A4_full", SEED, DEVICE)
    for p in model.conv_blocks.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
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
        vp = C.infer_probs(model, Xva, DEVICE)
        vm = C.compute_metrics(vp, yva)
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"    epoch {ep:2d}: val_F1={vm['f1']:.4f} val_AUC={vm['auc']:.4f}")

    model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "scenario": scn,
                "best_val_f1": best_f1}, C.MODEL_DIR / f"lidds_finetune_{scn}.pt")

    # Threshold optimal-val
    vp = C.infer_probs(model, Xva, DEVICE)
    best_thr, best_vf1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        f1 = C.compute_metrics(vp, yva, thr)["f1"]
        if f1 > best_vf1:
            best_vf1, best_thr = f1, float(thr)

    tp = C.infer_probs(model, te_w, DEVICE)
    tm = C.compute_metrics(tp, te_l, best_thr)
    return tm, {"test_windows": int(len(te_l)), "test_malicious": int(te_l.sum()),
                "best_val_f1": best_f1, "tuned_threshold": best_thr,
                "frozen": "conv_blocks"}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    remap = C.build_adfa_remap()

    print("=" * 64)
    print("LID-DS Step 5: Fine-tune Transfer (ADFA -> LID-DS, conv frozen)")
    print(f"Device: {DEVICE}")
    print("=" * 64)

    report = {"mode": "fine_tune", "init": "A4_full_seed42",
              "frozen": "conv_blocks",
              "config": {"epochs": EPOCHS, "lr": LR, "max_benign": MAX_BENIGN,
                         "seed": SEED}, "scenarios": {}}

    for scn in SCENARIOS:
        print(f"\n[{scn}] fine-tuning...")
        tm, info = finetune_one(scn, remap)
        report["scenarios"][scn] = {"metrics": tm, "info": info}
        print(f"  TEST: Recall={tm['recall']:.4f} FPR={tm['fpr']:.4f} "
              f"F1={tm['f1']:.4f} AUC={tm['auc']:.4f} AUPRC={tm['auprc']:.4f}")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "lidds_finetune_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 64)
    print("Ringkasan Fine-tune:")
    print(f"  {'Skenario':<24}{'Recall':>8}{'FPR':>8}{'F1':>8}{'AUC':>8}{'AUPRC':>8}")
    print("  " + "-" * 64)
    for scn, r in report["scenarios"].items():
        m = r["metrics"]
        print(f"  {scn:<24}{m['recall']:>8.3f}{m['fpr']:>8.3f}"
              f"{m['f1']:>8.3f}{m['auc']:>8.3f}{m['auprc']:>8.3f}")
    print(f"\n  Saved: results/lidds_finetune_report.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
