"""
Stage 4: Detection-to-Prevention Framework

Takes the trained robust encoder and converts its detection scores into
actionable prevention decisions.  Three metrics structure the analysis:

  ABR  — Attack Blocking Rate:  TP / (TP + FN)
          "Of all attacks attempted, what fraction do we stop?"

  FPR  — False Prevention Rate: FP / (FP + TN)
          "Of all benign processes, what fraction do we mistakenly kill?"

  U    — Prevention Utility:     ABR - C_fp * FPR
          "Is prevention worth it, given the cost of mistakes?"

The framework sweeps the decision threshold from 0.01 to 0.99, producing
cost-benefit curves that show the full operating range of the system under
both clean and adversarial conditions.

Outputs:
  results/prevention_report.json   — full threshold sweep + optimal points
  results/prevention_curves.json   — data for plotting
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = np.arange(0.01, 1.00, 0.01).tolist()
COST_RATIOS = [1, 2, 5, 7, 10, 20]

SEED = 42

ATTACK_CATEGORIES = [
    "benign", "Adduser", "Hydra_FTP", "Hydra_SSH",
    "Java_Meterpreter", "Meterpreter", "Web_Shell",
]


# ===========================================================================
# Model (must match robust_encoder.py)
# ===========================================================================
import torch.nn as nn


class MultiScaleEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, conv_channels, kernels,
                 hidden_dim, num_classes, dropout):
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


# ===========================================================================
# Inference helpers
# ===========================================================================
@torch.no_grad()
def get_scores(model, loader, device):
    """Return probabilities and metadata for all samples."""
    model.eval()
    all_probs, all_labels, all_cats = [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, c = [t.to(device) for t in batch]
        else:
            x, y = [t.to(device) for t in batch]
            c = torch.zeros(x.size(0), dtype=torch.long)
        logits, _ = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(malicious)
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())
        all_cats.append(c.cpu())
    return (torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_cats).numpy())


def compute_metrics_at_threshold(probs, labels, threshold):
    """Compute TP, FP, FN, TN at a given threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn


def prevention_metrics(tp, fp, fn, tn):
    """Compute ABR, FPR, Precision, Recall, F1 from confusion matrix."""
    abr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = abr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0
    return {"abr": abr, "fpr": fpr, "precision": prec, "recall": rec,
            "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ===========================================================================
# Main
# ===========================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Stage 4: Detection-to-Prevention Framework")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model and data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading model and data...")

    with open(PREPARED_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(RESULTS_DIR / "evaluation_report.json") as f:
        eval_config = json.load(f)["config"]

    model = MultiScaleEncoder(
        vocab_size=meta["vocab_size"],
        embed_dim=eval_config["embed_dim"],
        conv_channels=eval_config["conv_channels"],
        kernels=eval_config["conv_kernels"],
        hidden_dim=eval_config["hidden_dim"],
        num_classes=2,
        dropout=eval_config["dropout"],
    ).to(DEVICE)

    ckpt = torch.load(MODEL_DIR / "encoder_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded model from epoch {ckpt['epoch']} (F1={ckpt['val_f1']:.4f})")

    # Load test data
    test = np.load(PREPARED_DIR / "test.npz")
    X_test = torch.tensor(test["windows"], dtype=torch.long)
    y_test = torch.tensor(test["labels"], dtype=torch.long)
    c_test = torch.tensor(test["cat_ids"], dtype=torch.long)

    # Pad to 130
    max_len = 130
    if X_test.size(1) < max_len:
        pad = torch.zeros(X_test.size(0), max_len - X_test.size(1), dtype=torch.long)
        X_test = torch.cat([X_test, pad], dim=1)

    test_loader = DataLoader(TensorDataset(X_test, y_test, c_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # Load adversarial test sets
    adv_strategies = ["distribution_aware", "random", "position_aware"]
    adv_loaders = {}
    for strat in adv_strategies:
        fpath = ADVERSARIAL_DIR / f"{strat}_test.npz"
        if fpath.exists():
            d = np.load(fpath)
            x = torch.tensor(d["windows"], dtype=torch.long)
            y = torch.ones(x.size(0), dtype=torch.long)  # all malicious
            c = torch.tensor(d["cat_ids"], dtype=torch.long)
            adv_loaders[strat] = DataLoader(
                TensorDataset(x, y, c), batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 2. Get scores
    # ------------------------------------------------------------------
    print("\n[2/5] Running inference...")

    clean_probs, clean_labels, clean_cats = get_scores(model, test_loader, DEVICE)
    print(f"  Clean test: {len(clean_probs)} samples")

    adv_scores = {}
    for strat, loader in adv_loaders.items():
        p, l, c = get_scores(model, loader, DEVICE)
        adv_scores[strat] = {"probs": p, "labels": l, "cats": c}
        print(f"  Adversarial ({strat}): {len(p)} samples")

    # ------------------------------------------------------------------
    # 3. Threshold sweep — clean test
    # ------------------------------------------------------------------
    print("\n[3/5] Threshold sweep (clean test)...")

    sweep_clean = []
    for t in THRESHOLDS:
        tp, fp, fn, tn = compute_metrics_at_threshold(clean_probs, clean_labels, t)
        m = prevention_metrics(tp, fp, fn, tn)
        m["threshold"] = t
        m["utility"] = {}
        for c_fp in COST_RATIOS:
            m["utility"][str(c_fp)] = m["abr"] - c_fp * m["fpr"]
        sweep_clean.append(m)

    # Operating points are CHOSEN on the trace-level validation split (seed 42,
    # the same split used for checkpoint selection) and only REPORTED on the test
    # set. The test sweep above is kept as a descriptive curve, not for selection.
    split = json.load(open(RESULTS_DIR / "splits" / "val_traces_seed42.json"))
    trn = np.load(PREPARED_DIR / "train.npz")
    vmask = np.isin(trn["trace_ids"], split["val_trace_ids"])
    Xv = torch.tensor(trn["windows"][vmask], dtype=torch.long)
    Xv = torch.cat([Xv, torch.zeros(Xv.size(0), max_len - Xv.size(1),
                                    dtype=torch.long)], dim=1)
    yv = torch.tensor(trn["labels"][vmask], dtype=torch.long)
    val_probs, val_labels, _ = get_scores(
        model, DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE), DEVICE)
    print(f"  Validation (threshold selection): {len(val_probs)} windows, "
          f"{split['n_val_traces']} traces")

    optimal = {}
    for c_fp in COST_RATIOS:
        best_u, best_t = -999, 0.5
        for t in THRESHOLDS:
            m = prevention_metrics(*compute_metrics_at_threshold(val_probs, val_labels, t))
            u = m["abr"] - c_fp * m["fpr"]
            if u > best_u:
                best_u, best_t = u, t
        m = prevention_metrics(*compute_metrics_at_threshold(clean_probs, clean_labels, best_t))
        optimal[str(c_fp)] = {"threshold": best_t, "val_utility": best_u,
                              "utility": m["abr"] - c_fp * m["fpr"], **m}

    print("\n  Optimal thresholds by cost of false prevention:")
    print(f"  {'C_fp':>5}  {'Threshold':>9}  {'ABR':>6}  {'FPR':>6}  {'F1':>6}  {'U':>7}")
    print("  " + "-" * 50)
    for c_fp in COST_RATIOS:
        o = optimal[str(c_fp)]
        print(f"  {c_fp:>5}  {o['threshold']:>9.2f}  {o['abr']:>6.4f}  "
              f"{o['fpr']:>6.4f}  {o['f1']:>6.4f}  {o['utility']:>7.4f}")

    # ------------------------------------------------------------------
    # 4. Threshold sweep — adversarial test (attack-only ABR)
    # ------------------------------------------------------------------
    print("\n[4/5] Adversarial ABR at key thresholds...")

    key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    adv_sweep = {}
    for strat in adv_strategies:
        if strat not in adv_scores:
            continue
        p = adv_scores[strat]["probs"]
        l = adv_scores[strat]["labels"]
        c = adv_scores[strat]["cats"]
        strat_results = []
        for t in key_thresholds:
            tp, fp, fn, tn = compute_metrics_at_threshold(p, l, t)
            m = prevention_metrics(tp, fp, fn, tn)
            m["threshold"] = t
            strat_results.append(m)
        adv_sweep[strat] = strat_results

    # Print adversarial ABR
    print(f"\n  {'Threshold':>9}  " + "  ".join(f"{s[:12]:>12}" for s in adv_strategies))
    print("  " + "-" * (9 + 14 * len(adv_strategies)))
    for t in key_thresholds:
        vals = []
        for strat in adv_strategies:
            if strat in adv_sweep:
                entry = next(e for e in adv_sweep[strat] if abs(e["threshold"] - t) < 0.001)
                vals.append(f"{entry['abr']:>12.4f}")
            else:
                vals.append(f"{'N/A':>12}")
        print(f"  {t:>9.2f}  " + "  ".join(vals))

    # ------------------------------------------------------------------
    # 5. Per-category breakdown
    # ------------------------------------------------------------------
    print("\n[5/5] Per-category prevention metrics (clean test, threshold=0.50)...")

    cat_map = {v: k for k, v in meta["category_map"].items()}
    cat_results = {}
    for cat_id in sorted(set(clean_cats)):
        mask = clean_cats == cat_id
        cat_probs = clean_probs[mask]
        cat_labels = clean_labels[mask]
        cat_name = cat_map.get(int(cat_id), f"cat_{cat_id}")

        # At default threshold 0.5
        tp, fp, fn, tn = compute_metrics_at_threshold(cat_probs, cat_labels, 0.5)
        m = prevention_metrics(tp, fp, fn, tn)
        cat_results[cat_name] = {
            "count": int(mask.sum()),
            "threshold_0.5": m,
        }

        # At optimal threshold for C_fp=5
        opt_t = optimal["5"]["threshold"]
        tp2, fp2, fn2, tn2 = compute_metrics_at_threshold(cat_probs, cat_labels, opt_t)
        m2 = prevention_metrics(tp2, fp2, fn2, tn2)
        cat_results[cat_name][f"threshold_{opt_t:.2f}"] = m2

    print(f"\n  {'Category':25s}  {'Count':>5}  --- threshold=0.50 ---  --- threshold={opt_t:.2f} ---")
    print(f"  {'':25s}  {'':>5}  {'ABR':>6}  {'FPR':>6}  {'F1':>6}  {'ABR':>6}  {'FPR':>6}  {'F1':>6}")
    print("  " + "-" * 80)
    for cat_name in ATTACK_CATEGORIES:
        if cat_name not in cat_results:
            continue
        cr = cat_results[cat_name]
        m1 = cr["threshold_0.5"]
        key2 = [k for k in cr if k.startswith("threshold_") and k != "threshold_0.5"]
        m2 = cr[key2[0]] if key2 else m1
        print(f"  {cat_name:25s}  {cr['count']:>5}  {m1['abr']:>6.4f}  {m1['fpr']:>6.4f}  "
              f"{m1['f1']:>6.4f}  {m2['abr']:>6.4f}  {m2['fpr']:>6.4f}  {m2['f1']:>6.4f}")

    # Per-category adversarial ABR
    print(f"\n  Per-category ABR under Distribution-Aware mimicry (threshold=0.50):")
    for cat_id in sorted(set(clean_cats)):
        cat_name = cat_map.get(int(cat_id), f"cat_{cat_id}")
        if cat_name == "benign":
            continue
        # From adversarial data
        if "distribution_aware" in adv_scores:
            adv_c = adv_scores["distribution_aware"]["cats"]
            adv_p = adv_scores["distribution_aware"]["probs"]
            adv_l = adv_scores["distribution_aware"]["labels"]
            cat_mask = adv_c == cat_id
            if cat_mask.any():
                tp, fp, fn, tn = compute_metrics_at_threshold(
                    adv_p[cat_mask], adv_l[cat_mask], 0.5)
                abr = tp / (tp + fn) if (tp + fn) else 0
                print(f"    {cat_name:25s}  ABR={abr:.4f}  ({tp}/{tp+fn} blocked)")

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    report = {
        "optimal_thresholds": optimal,
        "clean_threshold_sweep": sweep_clean,
        "adversarial_sweep": {k: v for k, v in adv_sweep.items()},
        "per_category": cat_results,
        "cost_ratios_tested": COST_RATIOS,
        "key_thresholds": key_thresholds,
        "model_info": {
            "best_epoch": ckpt["epoch"],
            "best_f1": ckpt["val_f1"],
            "best_auc": ckpt["val_auc"],
        },
    }

    # Curves data (lighter, for plotting)
    curves = {
        "thresholds": THRESHOLDS,
        "clean_abr_curve": [e["abr"] for e in sweep_clean],
        "clean_fpr_curve": [e["fpr"] for e in sweep_clean],
        "clean_f1_curve": [e["f1"] for e in sweep_clean],
        "clean_utility_curves": {
            str(c): [e["utility"][str(c)] for e in sweep_clean]
            for c in COST_RATIOS
        },
        "adversarial_abr_curves": {
            strat: [e["abr"] for e in entries]
            for strat, entries in adv_sweep.items()
        },
        "optimal_points": optimal,
    }

    with open(RESULTS_DIR / "prevention_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(RESULTS_DIR / "prevention_curves.json", "w") as f:
        json.dump(curves, f, indent=2, default=float)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stage 4 complete.")
    print("=" * 60)
    print(f"\n  Saved: {RESULTS_DIR}/prevention_report.json")
    print(f"  Saved: {RESULTS_DIR}/prevention_curves.json")

    # Print final summary
    print("\n  === PREVENTION SUMMARY ===")
    print(f"\n  At default threshold (0.50):")
    tp, fp, fn, tn = compute_metrics_at_threshold(clean_probs, clean_labels, 0.5)
    m = prevention_metrics(tp, fp, fn, tn)
    print(f"    ABR  = {m['abr']:.4f} ({m['abr']*100:.1f}% attacks blocked)")
    print(f"    FPR  = {m['fpr']:.4f} ({m['fpr']*100:.1f}% benign disrupted)")
    print(f"    F1   = {m['f1']:.4f}")

    # Best operating point for moderate cost
    o5 = optimal["5"]
    print(f"\n  At optimal threshold for C_fp=5 (threshold={o5['threshold']:.2f}):")
    print(f"    ABR  = {o5['abr']:.4f} ({o5['abr']*100:.1f}% attacks blocked)")
    print(f"    FPR  = {o5['fpr']:.4f} ({o5['fpr']*100:.1f}% benign disrupted)")
    print(f"    F1   = {o5['f1']:.4f}")
    print(f"    U(5) = {o5['utility']:.4f}")

    # Adversarial ABR at 0.5
    if "distribution_aware" in adv_scores:
        tp, fp, fn, tn = compute_metrics_at_threshold(
            adv_scores["distribution_aware"]["probs"],
            adv_scores["distribution_aware"]["labels"], 0.5)
        abr_adv = tp / (tp + fn) if (tp + fn) else 0
        print(f"\n  Adversarial ABR (Distribution-Aware, threshold=0.50):")
        print(f"    ABR  = {abr_adv:.4f} ({abr_adv*100:.1f}% mimicry attacks blocked)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
