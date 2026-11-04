"""Expose the corrected_pipeline checkpoints under the legacy tag names that the
downstream scripts load (diagnostics, prevention, ONNX export, LID-DS), and
write models/encoder_best.pt + results/evaluation_report.json from the SAME
run (C_A4_full, seed 42). No retraining: this is a copy, so every downstream
number comes from the authoritative corrected run.
Run after: corrected_pipeline.py --out corrected_full.json
"""
import json
import shutil
from pathlib import Path

import torch

M, R = Path("models"), Path("results")
C2L = {"C_B0_vanilla": "B3_vanilla_cnn", "C_A2_pairedKL": "A2_cls_robust",
       "C_A3_contrastive": "A3_cls_contrastive", "C_A4_full": "A4_full"}

full = json.load(open(R / "corrected_full.json"))
seeds = full["config"]["seeds"]
for ctag, ltag in C2L.items():
    for s in seeds:
        shutil.copyfile(M / f"{ctag}_seed{s}.pt", M / f"{ltag}_seed{s}.pt")

i42 = seeds.index(42)
run = full["runs"]["C_A4_full"][i42]
ck = torch.load(M / "C_A4_full_seed42.pt", weights_only=False, map_location="cpu")
torch.save({"epoch": ck.get("epoch"), "model_state": ck["model_state"],
            "val_f1": float(ck.get("val_f1", 0.0)), "val_auc": None,
            "source": "C_A4_full_seed42.pt (corrected_pipeline)"}, M / "encoder_best.pt")
meta = json.load(open("prepared_data/metadata.json"))
# AUPRC is not part of corrected_full.json; compute it from the same checkpoint so
# every consumer (Tabel 4.3, Tabel G.1, Gambar G.3) reads one value.
import numpy as np
from sklearn.metrics import average_precision_score
from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model, get_probs, pad_to, MAX_LEN
te = np.load("prepared_data/test.npz")
p = get_probs(load_model("C_A4_full", 42), pad_to(torch.tensor(te["windows"], dtype=torch.long), MAX_LEN))
run["clean"]["auprc"] = float(average_precision_score(te["labels"], p))
json.dump({"clean": run["clean"], "adversarial": run["adv"],
           "config": {"vocab_size": meta["vocab_size"], "source": "corrected_full.json C_A4_full seed 42",
                      "embed_dim": 64, "conv_channels": 64, "conv_kernels": [3, 5, 7],
                      "hidden_dim": 128, "dropout": 0.3, "alpha": 1.0, "beta": 0.1,
                      **{k: v for k, v in full["config"].items() if k != "seeds"}}},
          open(R / "evaluation_report.json", "w"), indent=2, default=float)
print(f"staged {len(C2L)} tags x {len(seeds)} seeds; encoder_best.pt <- C_A4_full_seed42")
