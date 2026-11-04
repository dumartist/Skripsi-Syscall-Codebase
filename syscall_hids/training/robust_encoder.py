"""
Stage 3: Robust Representation Encoder (CORRECTED)

Trains the multi-scale 1D-CNN "Full Model" (A4) on ADFA-LD syscall windows with
the combined objective:

    L_total = L_cls  +  alpha * L_robust  +  beta * L_contrastive

  L_cls          — Cross-entropy on benign vs malicious windows.
  L_robust       — TRADES-style KL divergence between a clean-malicious window and
                    ITS OWN adversarial (mimicry) version — a PAIRED loss. (The
                    original implementation here compared two unrelated slices of a
                    shuffled batch; that defect is fixed.)
  L_contrastive  — SimCLR-style NT-Xent on two no-op-augmented views of each window.

Corrections vs. the original:
  * L_robust is paired (matched clean/adversarial windows), not batch-arbitrary.
  * Best checkpoint is selected on a stratified VALIDATION split carved from the
    training set, never on the test set.
  * The no-op token set is the corrected i386 vocabulary (getuid32/geteuid32).
  * Adversarial windows are used only as paired targets for L_robust (not added as
    extra label-malicious classification samples).

The actual training is delegated to corrected_pipeline.train_corrected so that this
script and the ablation runner share one audited implementation.

Outputs:
  models/encoder_best.pt         — best checkpoint (by validation F1)
  results/evaluation_report.json — final test + adversarial evaluation
"""

import json
from pathlib import Path

import torch

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Architecture / training hyperparameters (full model = A4)
EMBED_DIM = 64
HIDDEN_DIM = 128
CONV_KERNELS = [3, 5, 7]
CONV_CHANNELS = 64
DROPOUT = 0.3
BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.5
ALPHA = 1.0   # L_robust weight
BETA = 0.1    # L_contrastive weight
SEED = 42


def train():
    from syscall_hids.training.corrected_pipeline import train_corrected, load_all
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Stage 3: Robust Representation Encoder (corrected, full model A4)")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)

    data = load_all()
    r = train_corrected(data, alpha=ALPHA, beta=BETA, augment_adv=False,
                        canon="none", seed=SEED, tag="encoder_full",
                        epochs=NUM_EPOCHS)

    # train_corrected saved models/encoder_full_seed{SEED}.pt; expose it as the
    # canonical encoder_best.pt (schema expected by downstream scripts).
    ck = torch.load(MODEL_DIR / f"encoder_full_seed{SEED}.pt",
                    weights_only=False, map_location="cpu")
    torch.save({"epoch": NUM_EPOCHS, "model_state": ck["model_state"],
                "val_f1": float(ck.get("val_f1", 0.0)),
                "val_auc": float(r["clean"]["auc"])},
               MODEL_DIR / "encoder_best.pt")

    report = {
        "clean": r["clean"],
        "adversarial": r["adv"],
        "config": {
            "vocab_size": data["vocab"], "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM, "conv_kernels": CONV_KERNELS,
            "conv_channels": CONV_CHANNELS, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "alpha": ALPHA, "beta": BETA, "temperature": TEMPERATURE,
        },
    }
    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n  val_f1={ck.get('val_f1', 0.0):.4f}  "
          f"test F1={r['clean']['f1']:.4f}  recall={r['clean']['recall']:.4f}  "
          f"FPR={r['clean']['fpr']:.4f}  DA_recall={r['adv']['distribution_aware']['recall']:.4f}")
    print(f"  Saved: {MODEL_DIR}/encoder_best.pt")
    print(f"  Saved: {RESULTS_DIR}/evaluation_report.json")
    print("Stage 3 complete.")


if __name__ == "__main__":
    train()
