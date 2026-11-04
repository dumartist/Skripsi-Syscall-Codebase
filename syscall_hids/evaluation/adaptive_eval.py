"""Adaptive white-box evasion across seeds on a COMMON sample set.

For each seed, every pipeline is attacked on exactly the same windows: malicious
test windows detected by ALL pipelines on clean input, one window per trace
(independent traces), sampled with a seed-fixed RNG (corrected_pipeline.
common_attack_set). Evasion is therefore conditional on one shared population;
the full-test-set post-attack recall is reported separately as a labelled
estimate (clean recall x (1 - evasion)), never mixed with the conditional rate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from syscall_hids.training.corrected_pipeline import run_adaptive, CanonModel, common_attack_set
from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model

RESULTS_DIR = Path("results")
PIPELINES = [("C_B0_vanilla", "none"), ("C_B0_sym", "none"), ("C_AT_augment", "none"),
             ("C_A2_pairedKL", "none"), ("C_A3_contrastive", "none"),
             ("C_A4_full", "none"), ("C_CANON_dedup", "dedup"),
             ("C_CANON_strip", "strip")]
KEYS = ["recall_clean_full_test", "evasion_le_10pct", "evasion_le_20pct",
        "evasion_le_30pct", "evasion_final", "post_attack_recall_est_le_10pct",
        "post_attack_recall_est_le_30pct"]


def agg(runs, k):
    v = np.array([r[k] for r in runs], dtype=float)
    return [float(v.mean()), float(v.std())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n", type=int, default=200, help="independent traces per seed")
    ap.add_argument("--out", default="adaptive_full.json")
    args = ap.parse_args()

    out = {"seeds": args.seeds, "n_traces": args.n,
           "population": "malicious test windows detected by ALL pipelines on clean "
                         "input; one window per trace; identical across pipelines",
           "sample_info": {}, "runs": {t: [] for t, _ in PIPELINES}, "aggregate": {}}
    for seed in args.seeds:
        models = {}
        for tag, mode in PIPELINES:
            m = load_model(tag, seed)
            models[tag] = CanonModel(m, mode) if mode != "none" else m
        idx, clean, info = common_attack_set(models, seed, args.n)
        out["sample_info"][str(seed)] = info
        print(f"seed {seed}: {info}")
        for tag, _ in PIPELINES:
            r = run_adaptive(models[tag], idx, clean[tag])
            out["runs"][tag].append(r)
            print(f"  {tag:18s} clean={clean[tag]:.3f} evasion<=10/20/30 = "
                  f"{r['evasion_le_10pct']:.2f}/{r['evasion_le_20pct']:.2f}/"
                  f"{r['evasion_le_30pct']:.2f}")
    for tag, _ in PIPELINES:
        out["aggregate"][tag] = {k: agg(out["runs"][tag], k) for k in KEYS}

    with open(RESULTS_DIR / args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n%-18s %7s  %-34s %s" % ("pipeline", "clean", "evasion<=10/20/30% (mean+-std)",
                                     "post-attack recall est. <=10%"))
    for tag, _ in PIPELINES:
        a = out["aggregate"][tag]
        print("%-18s %6.3f   %.2f+-%.2f / %.2f+-%.2f / %.2f+-%.2f   %.3f" % (
            tag, a["recall_clean_full_test"][0],
            *a["evasion_le_10pct"], *a["evasion_le_20pct"], *a["evasion_le_30pct"],
            a["post_attack_recall_est_le_10pct"][0]))
    print(f"\nSaved: results/{args.out}")


if __name__ == "__main__":
    main()
