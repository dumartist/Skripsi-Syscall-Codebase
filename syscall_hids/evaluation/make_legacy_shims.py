"""Summarize baseline and ablation metrics across training seeds."""
import json
import numpy as np
from pathlib import Path

R = Path("results")
full = json.load(open(R / "corrected_full.json"))
bl = json.load(open(R / "corrected_baselines.json"))
SEEDS = full["config"]["seeds"]
STRATS = ["distribution_aware", "random", "position_aware"]
PROV = {"source": "corrected_pipeline.py 10-seed run (AUTHORITATIVE)",
        "exported_from": ["results/corrected_full.json", "results/corrected_baselines.json"],
        "note": "B3_vanilla_cnn corresponds to the clean-only B0 baseline.",
        "seeds": SEEDS}

CLEAN_KEYS = ["auc", "f1", "recall", "fpr", "precision", "accuracy"]
import syscall_hids.data.lidds_common as _C
from syscall_hids.training.baselines_ablation import LSTMEncoder as _LSTM
_CNN = sum(p.numel() for p in _C.MultiScaleEncoder(**{k: (tuple(v) if k == "kernels" else v)
           for k, v in _C.ADFA_CFG.items()}).parameters())
PARAMS = {"B3_vanilla_cnn": _CNN, "A1_cls_only": _CNN, "A2_cls_robust": _CNN,
          "A3_cls_contrastive": _CNN, "A4_full": _CNN,
          "B2_lstm": sum(p.numel() for p in _LSTM().parameters())}


def ms(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "values": [float(v) for v in a]}


def node_from_full(tag):
    runs = full["runs"][tag]
    d = {"clean_agg": {k: ms([r["clean"][k] for r in runs]) for k in CLEAN_KEYS}}
    for s in STRATS:
        d[f"adv_{s}_agg"] = {"recall": ms([r["adv"][s]["recall"] for r in runs])}
    d["runs"] = [{"seed": sd, "clean": r["clean"]} for sd, r in zip(SEEDS, runs)]
    return d


def node_from_baselines(key):
    d = {"clean_agg": {k: {"mean": bl[key][k][0], "std": bl[key][k][1]}
                       for k in CLEAN_KEYS if k in bl[key]}}
    for s in STRATS:
        d[f"adv_{s}_agg"] = {"recall": {"mean": bl[key][f"advrec_{s}"][0],
                                        "std": bl[key][f"advrec_{s}"][1]}}
    return d


# map corrected tags -> legacy names
C2L = {"C_B0_vanilla": "B3_vanilla_cnn", "C_A2_pairedKL": "A2_cls_robust",
       "C_A3_contrastive": "A3_cls_contrastive", "C_A4_full": "A4_full"}

baselines = {"_provenance": PROV,
             "B1_ngram": {**node_from_baselines("B1_ngram"), "params": None},
             "B2_lstm": {**node_from_baselines("B2_lstm"), "params": PARAMS["B2_lstm"]},
             "B3_vanilla_cnn": {**node_from_full("C_B0_vanilla"), "params": _CNN}}

ablations = {"_provenance": PROV}
for ctag, ltag in [("C_B0_vanilla", "A1_cls_only"), ("C_A2_pairedKL", "A2_cls_robust"),
                   ("C_A3_contrastive", "A3_cls_contrastive"), ("C_A4_full", "A4_full")]:
    ablations[ltag] = {**node_from_full(ctag), "params": _CNN}

full_cmp = {"_provenance": PROV, "seeds": SEEDS, "baselines": baselines, "ablations": ablations}

for name, obj in [("baselines_report.json", baselines),
                  ("ablation_report.json", ablations),
                  ("full_comparison_report.json", full_cmp)]:
    json.dump(obj, open(R / name, "w"), indent=2, default=float)

# No adaptive_attack.json shim: it mixed full-test clean recall with conditional
# evasion (different populations). See adaptive_full.json for the common-population evaluation.
stale = R / "adaptive_attack.json"
if stale.exists():
    stale.rename(R / "adaptive_attack.legacy_mixed_population.json")

print("wrote authoritative legacy exports: baselines_report.json, ablation_report.json, "
      "full_comparison_report.json")
print(f"  B3/B0 clean recall={baselines['B3_vanilla_cnn']['clean_agg']['recall']['mean']:.4f} "
      f"DA={baselines['B3_vanilla_cnn']['adv_distribution_aware_agg']['recall']['mean']:.4f}")
print(f"  A4   clean recall={ablations['A4_full']['clean_agg']['recall']['mean']:.4f} "
      f"DA={ablations['A4_full']['adv_distribution_aware_agg']['recall']['mean']:.4f}")
