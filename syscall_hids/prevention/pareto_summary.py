"""Multi-objective comparison for the model-choice discussion.
Collects, per model, the objectives already measured elsewhere (no new experiment)
and marks which models are Pareto-optimal (not dominated on every objective).
Output: results/examiner/pareto.json
"""
import json
from pathlib import Path
import numpy as np

R, E = Path("results"), Path("results/examiner")
full = json.load(open(R / "corrected_full.json"))
adf = json.load(open(R / "adaptive_full.json"))["aggregate"]
inj = json.load(open(R / "corrected_analysis.json"))["injected_fpr_4_11"]
canon = json.load(open(E / "canon_injected_fpr.json"))
pauc = json.load(open(E / "pauc.json"))
b1 = json.load(open(E / "b1_rf_svm.json"))

def mean(tag, sec, k):
    return float(np.mean([r[sec][k] for r in full["runs"][tag]]))

def da(tag):
    return float(np.mean([r["adv"]["distribution_aware"]["recall"] for r in full["runs"][tag]]))

rows = {}
for tag, name in [("C_B0_vanilla", "B0"), ("C_B0_sym", "B0-sym"), ("C_AT_augment", "AT"),
                  ("C_A2_pairedKL", "A2"), ("C_A3_contrastive", "A3"), ("C_A4_full", "A4"),
                  ("C_CANON_dedup", "Kanon. dedup"), ("C_CANON_strip", "Kanon. strip")]:
    dfpr = (inj[tag]["increase_noop_30"]["diff"] if tag in inj
            else canon[tag]["increase_noop_30"]["diff"])
    rows[name] = {"f1": mean(tag, "clean", "f1"), "fpr": mean(tag, "clean", "fpr"),
                  "pauc05": pauc[tag]["pauc05"][0], "adv_da": da(tag),
                  "evasion_le10": adf[tag]["evasion_le_10pct"][0], "dfpr_noop30": dfpr,
                  "n_seeds_adaptive": 10}
for name in ("RF", "SVM"):
    rs = b1[name]
    ad = [r["adaptive"] for r in rs if "adaptive" in r]
    rows[f"B1-{name}"] = {"f1": float(np.mean([r["f1"] for r in rs])),
                          "fpr": float(np.mean([r["fpr"] for r in rs])),
                          "pauc05": float(np.mean([r["pauc05"] for r in rs])),
                          "adv_da": float(np.mean([r["advrec_distribution_aware"] for r in rs])),
                          "evasion_le10": float(np.mean([a["le_10"] for a in ad])),
                          "dfpr_noop30": float(np.mean([r["inj_fpr"]["noop_30"] - r["inj_fpr"]["clean"] for r in rs])),
                          "n_seeds_adaptive": len(ad)}

# higher is better for f1, pauc05, adv_da; lower is better for the rest
SIGN = {"f1": 1, "pauc05": 1, "adv_da": 1, "fpr": -1, "evasion_le10": -1, "dfpr_noop30": -1}
def dominates(a, b):
    ge = all(SIGN[k] * a[k] >= SIGN[k] * b[k] for k in SIGN)
    gt = any(SIGN[k] * a[k] > SIGN[k] * b[k] for k in SIGN)
    return ge and gt
for n, r in rows.items():
    r["dominated_by"] = [m for m, o in rows.items() if m != n and dominates(o, r)]
    r["pareto_optimal"] = not r["dominated_by"]
    r["best_on"] = [k for k in SIGN if all(SIGN[k] * r[k] >= SIGN[k] * o[k] for o in rows.values())]
json.dump(rows, open(E / "pareto.json", "w"), indent=2)
for n, r in rows.items():
    print(f"{n:13s} F1={r['f1']:.4f} FPR={r['fpr']:.4f} pAUC={r['pauc05']:.4f} DA={r['adv_da']:.4f} "
          f"ev10={r['evasion_le10']:.3f} dFPR={r['dfpr_noop30']:+.3f} pareto={r['pareto_optimal']} "
          f"dominated_by={r['dominated_by']} best_on={r['best_on']}")
