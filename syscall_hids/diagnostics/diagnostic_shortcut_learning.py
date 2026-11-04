"""
Diagnostic A1: Shortcut Learning Test

Membedakan apakah recall adversarial yang tinggi pada Full Model (A4)
berasal dari:
  (a) genuine adversarial robustness via L_robust + L_contrastive, ATAU
  (b) shortcut learning oleh arsitektur CNN multi-skala terhadap n-gram
      artifacts dari injeksi no-op.

Logika: jika Vanilla CNN (B3, alpha=0, beta=0) sudah mencapai AdvRec
yang dekat dengan Full Model, maka sebagian besar "robustness" berasal
dari arsitektur, bukan dari adversarial training.

Eksperimen:
- Load 6 checkpoint: B3 × {42,123,456} dan A4 × {42,123,456}
- Untuk setiap model+seed, hitung AdvRec pada 3 strategi (DA, Random, PA)
  dan recall/F1/FPR pada clean test
- Paired t-test + Wilcoxon + Cohen's d untuk gap A4-B3 per strategi
- Decision rule:
    gap >= 0.05 dan p < 0.05  -> GENUINE_ROBUSTNESS
    gap < 0.02                 -> SHORTCUT_DOMINANT
    selainnya                  -> MIXED

Bonus: A2 (cls+robust) dan A3 (cls+contrastive) juga dievaluasi
untuk memetakan kontribusi tiap komponen.

Output:
  results/diagnostic_a1_shortcut.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

VOCAB_SIZE = json.load(open(PREPARED_DIR / "metadata.json"))["vocab_size"]
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2
DROPOUT = 0.3
CONV_KERNELS = [3, 5, 7]
CONV_CHANNELS = 64
MAX_LEN = 130
BATCH_SIZE = 512
THRESHOLD = 0.5

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
MODELS = ["B3_vanilla_cnn", "A2_cls_robust", "A3_cls_contrastive", "A4_full"]
STRATEGIES = ["distribution_aware", "random", "position_aware"]

# Decision-rule thresholds
GENUINE_GAP_THRESH = 0.05
SHORTCUT_GAP_THRESH = 0.02
ALPHA_SIG = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model
# ===========================================================================
class MultiScaleEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 conv_channels=CONV_CHANNELS, kernels=CONV_KERNELS,
                 hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT):
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
        # Projector ada di state_dict tapi tidak dipakai untuk inference
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

    def forward(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        conv_outs = [block(e).squeeze(-1) for block in self.conv_blocks]
        h = torch.cat(conv_outs, dim=1)
        h = self.fc_rep(h)
        return self.classifier(h)


# ===========================================================================
# Helpers
# ===========================================================================
def pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.size(1) >= length:
        return x[:, :length]
    pad = torch.zeros(x.size(0), length - x.size(1), dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def load_model(model_name: str, seed: int) -> MultiScaleEncoder:
    path = MODEL_DIR / f"{model_name}_seed{seed}.pt"
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = MultiScaleEncoder().to(DEVICE)
    # Beberapa baseline (B3) tidak pernah memakai projector saat training,
    # tetapi state_dict-nya tetap menyertakan layer (initialised). Load
    # dengan strict=False agar aman terhadap variasi minor.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def get_probs(model, X: torch.Tensor) -> np.ndarray:
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    out = []
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def confusion_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = THRESHOLD):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return {
        "recall": recall, "fpr": fpr, "precision": prec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Effect size Cohen's d untuk paired sample (mean / std of differences)."""
    if diffs.std(ddof=1) == 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def verdict_from(gap: float, p_value: float) -> str:
    if gap >= GENUINE_GAP_THRESH and p_value < ALPHA_SIG:
        return "GENUINE_ROBUSTNESS"
    if gap < SHORTCUT_GAP_THRESH:
        return "SHORTCUT_DOMINANT"
    return "MIXED"


# ===========================================================================
# Main
# ===========================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic A1: Shortcut Learning Test")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading test data...")
    test = np.load(PREPARED_DIR / "test.npz")
    X_clean = pad_to(torch.tensor(test["windows"], dtype=torch.long), MAX_LEN)
    y_clean = test["labels"].astype(int)
    print(f"  Clean test: {len(y_clean)} samples "
          f"({int((y_clean == 0).sum())} benign + {int((y_clean == 1).sum())} malicious)")

    adv_data = {}
    for strat in STRATEGIES:
        d = np.load(ADVERSARIAL_DIR / f"{strat}_test.npz")
        X_adv = torch.tensor(d["windows"], dtype=torch.long)
        # Jika ukuran < MAX_LEN, pad. Sebenarnya sudah 130 by design.
        X_adv = pad_to(X_adv, MAX_LEN)
        y_adv = np.ones(X_adv.size(0), dtype=int)  # semua malicious
        adv_data[strat] = (X_adv, y_adv)
        print(f"  Adversarial {strat}: {len(y_adv)} samples (all malicious)")

    # ------------------------------------------------------------------
    # 2. Inference per (model, seed)
    # ------------------------------------------------------------------
    print("\n[2/4] Running inference for all model+seed combinations...")
    per_run = {}  # (model, seed) -> dict
    for model_name in MODELS:
        for seed in SEEDS:
            key = f"{model_name}_seed{seed}"
            print(f"  Evaluating {key}...", end=" ", flush=True)
            try:
                model = load_model(model_name, seed)
            except FileNotFoundError as e:
                print(f"SKIP (checkpoint missing: {e})")
                continue

            # Clean test
            clean_probs = get_probs(model, X_clean)
            clean = confusion_metrics(clean_probs, y_clean)

            # Adversarial sets
            adv_metrics = {}
            for strat, (X_adv, y_adv) in adv_data.items():
                p = get_probs(model, X_adv)
                m = confusion_metrics(p, y_adv)
                adv_metrics[strat] = m

            per_run[key] = {
                "model": model_name,
                "seed": seed,
                "clean": clean,
                "adversarial": adv_metrics,
            }
            print(f"clean F1={clean['f1']:.4f}  "
                  f"AdvRec(DA)={adv_metrics['distribution_aware']['recall']:.4f}")

            # Free GPU memory
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Paired analysis: A4 - B3 per strategy
    # ------------------------------------------------------------------
    print("\n[3/4] Paired statistical tests (A4 vs B3, 10 seeds)...")
    paired_diffs = {}
    for strat in STRATEGIES:
        b3_vals = np.array([
            per_run[f"B3_vanilla_cnn_seed{s}"]["adversarial"][strat]["recall"]
            for s in SEEDS if f"B3_vanilla_cnn_seed{s}" in per_run
        ])
        a4_vals = np.array([
            per_run[f"A4_full_seed{s}"]["adversarial"][strat]["recall"]
            for s in SEEDS if f"A4_full_seed{s}" in per_run
        ])

        if len(b3_vals) != len(a4_vals) or len(b3_vals) < 2:
            print(f"  {strat}: SKIP (insufficient seeds)")
            continue

        diffs = a4_vals - b3_vals
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1))

        # Paired t-test (3 datapoints — small but valid)
        try:
            t_stat, t_p = stats.ttest_rel(a4_vals, b3_vals)
            t_p = float(t_p)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")

        # Wilcoxon — needs >= some samples. With n=3, may warn, gunakan try.
        try:
            w_stat, w_p = stats.wilcoxon(a4_vals, b3_vals,
                                         zero_method="wilcox", alternative="two-sided")
            w_p = float(w_p)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        d = cohens_d_paired(diffs)
        v = verdict_from(mean_diff, t_p) if not np.isnan(t_p) else "INCONCLUSIVE"

        paired_diffs[strat] = {
            "b3_values": b3_vals.tolist(),
            "a4_values": a4_vals.tolist(),
            "diffs": diffs.tolist(),
            "mean": mean_diff,
            "std": std_diff,
            "ttest_stat": float(t_stat),
            "ttest_p": t_p,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p": w_p,
            "cohens_d": d,
            "verdict": v,
        }

    # Print summary table
    print()
    print(f"  {'Strategy':<22}{'Mean Diff':>11}{'Std Diff':>11}"
          f"{'t-test p':>11}{'Wilcoxon p':>13}{'Cohen d':>11}  Verdict")
    print("  " + "-" * 99)
    for strat, r in paired_diffs.items():
        wp = f"{r['wilcoxon_p']:.4f}" if r['wilcoxon_p'] is not None and not np.isnan(r['wilcoxon_p']) else "n/a"
        print(f"  {strat:<22}{r['mean']:>+11.4f}{r['std']:>11.4f}"
              f"{r['ttest_p']:>11.4f}{wp:>13}{r['cohens_d']:>+11.4f}  {r['verdict']}")

    # ------------------------------------------------------------------
    # 4. Save and summarize
    # ------------------------------------------------------------------
    print("\n[4/4] Saving report...")

    # Overall verdict: gunakan verdict pada Distribution-Aware sebagai primary
    primary = paired_diffs.get("distribution_aware", {})
    overall_verdict = primary.get("verdict", "INCONCLUSIVE")

    discussion_points = []
    if overall_verdict == "SHORTCUT_DOMINANT":
        discussion_points.append(
            "Gap A4 - B3 sangat kecil (< 2pp). Sebagian besar adversarial "
            "robustness berasal dari arsitektur CNN multi-skala, bukan dari "
            "L_robust + L_contrastive pada evaluasi ini."
        )
    elif overall_verdict == "GENUINE_ROBUSTNESS":
        discussion_points.append(
            "Gap A4 - B3 substantif (>= 5pp) dengan signifikansi statistik. "
            "Adversarial training memberikan kontribusi nyata di atas baseline arsitektur."
        )
    elif overall_verdict == "MIXED":
        discussion_points.append(
            "Gap A4 - B3 berada di zona menengah (2-5pp). Adversarial training "
            "memberikan peningkatan terukur tetapi marginal. Sebagian besar robustness "
            "sudah terlihat pada baseline CNN."
        )

    # Cek apakah ada strategi dengan verdict berbeda
    verdicts_set = {r["verdict"] for r in paired_diffs.values()}
    if len(verdicts_set) > 1:
        discussion_points.append(
            f"Verdict berbeda antar strategi: {verdicts_set}."
        )

    report = {
        "config": {
            "models": MODELS,
            "seeds": SEEDS,
            "strategies": STRATEGIES,
            "threshold": THRESHOLD,
            "max_len": MAX_LEN,
            "decision_rule": {
                "genuine_gap_threshold": GENUINE_GAP_THRESH,
                "shortcut_gap_threshold": SHORTCUT_GAP_THRESH,
                "alpha_significance": ALPHA_SIG,
            },
        },
        "per_run": per_run,
        "paired_diffs_a4_minus_b3": paired_diffs,
        "summary": {
            "overall_verdict": overall_verdict,
            "primary_strategy": "distribution_aware",
            "discussion_points": discussion_points,
        },
    }

    out_path = RESULTS_DIR / "diagnostic_a1_shortcut.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"  Saved: {out_path}")

    print("\n" + "=" * 70)
    print(f"OVERALL VERDICT (Distribution-Aware): {overall_verdict}")
    print("=" * 70)
    for i, point in enumerate(discussion_points, 1):
        print(f"\n[{i}] {point}")
    print()


if __name__ == "__main__":
    main()
