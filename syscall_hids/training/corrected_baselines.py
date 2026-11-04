"""Corrected B1 (n-gram+RF) and B2 (LSTM) for Tabel 4.7 — on the regenerated
adversarial data, and (B2) with validation-split checkpoint selection.
Output: results/corrected_baselines.json"""
import json
from pathlib import Path
import numpy as np

from syscall_hids.training.corrected_pipeline import train_corrected, load_all, STRATEGIES
from syscall_hids.training.baselines_ablation import LSTMEncoder, train_ngram_baseline, load_data

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
RES = Path("results")


def agg(vals):
    v = np.array(vals, dtype=float)
    return [float(v.mean()), float(v.std())]


def main():
    out = {"seeds": SEEDS, "B1_ngram": {}, "B2_lstm": {}}

    # B1 — n-gram + RF (classical; uses regenerated adv_test)
    data_ba = load_data()
    b1 = [train_ngram_baseline(data_ba, seed=s) for s in SEEDS]
    out["B1_ngram"] = {
        **{k: agg([r["clean"][k] for r in b1])
           for k in ["auc", "f1", "recall", "fpr", "precision"]},
        **{f"advrec_{s}": agg([r["adversarial"][s]["recall"] for r in b1])
           for s in STRATEGIES}}

    # B2 — LSTM with corrected val-split selection (alpha=beta=0, no adv aug)
    data = load_all()
    b2 = [train_corrected(data, 0.0, 0.0, False, "none", s, "C_B2_lstm",
                          encoder=LSTMEncoder) for s in SEEDS]
    out["B2_lstm"] = {
        **{k: agg([r["clean"][k] for r in b2])
           for k in ["auc", "f1", "recall", "fpr", "precision"]},
        **{f"advrec_{s}": agg([r["adv"][s]["recall"] for r in b2])
           for s in STRATEGIES},
        "params": int(sum(p.numel() for p in LSTMEncoder().parameters()))}

    with open(RES / "corrected_baselines.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    for name in ["B1_ngram", "B2_lstm"]:
        d = out[name]
        print(f"{name}: AUC={d['auc'][0]:.4f} F1={d['f1'][0]:.4f} "
              f"recall={d['recall'][0]:.4f} FPR={d['fpr'][0]:.4f} "
              f"prec={d['precision'][0]:.4f} DA={d['advrec_distribution_aware'][0]:.4f}")
    print("Saved: results/corrected_baselines.json")


if __name__ == "__main__":
    main()
