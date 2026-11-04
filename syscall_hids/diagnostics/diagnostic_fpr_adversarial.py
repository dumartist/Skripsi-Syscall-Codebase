"""
Diagnostic A3: FPR pada Kondisi Adversarial (Section 4.3.11)

Menutup lubang fundamental: FPR tidak terdefinisi pada himpunan adversarial
yang 100% malicious (Tabel 4.3). Untuk mengukur FPR adversarial, perturbasi
no-op diterapkan pada sampel BENIGN.

Justifikasi semantik: penyisipan syscall no-op TIDAK boleh mengubah kelas
trace benign. Jika model membalik benign -> malicious karena injeksi, itu
mengindikasikan model overspecific terhadap artefak n-gram injeksi (shortcut),
bukan ketahanan. Ini menutup loop diagnostic A1/A2: jika model mendeteksi
artefak injeksi (bukan maliciousness), maka benign+no-op akan memicu FPR.

Strategi: random no-op injection pada rasio 5%, 10%, 30% (no-op level lemah,
tanpa optimasi distribusi — karena perturbasi pada benign tidak punya niat
adversarial; ini murni uji ketahanan model terhadap injeksi netral).

No-op: getpid, getuid32, geteuid32, getppid, time, gettimeofday (i386); token
id diturunkan dari vocabulary.json (getuid/geteuid 16-bit 24/49 tidak muncul di
ADFA-LD, varian 32-bit 199/201 yang dipakai).

Output: results/diagnostic_a3_fpr_adversarial.json
"""

import json
import numpy as np
import torch

import syscall_hids.data.lidds_common as C

PREPARED = C.PREPARED_ADFA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
THRESHOLD = 0.5
RATIOS = [0.05, 0.10, 0.30]
# Token ids of the corrected i386 no-ops (getpid, getuid32, geteuid32, getppid,
# time, gettimeofday), derived from the vocabulary rather than hard-coded.
_VOCAB = json.load(open(PREPARED / "vocabulary.json"))
NOOP_TOKENS = [_VOCAB[str(s)] for s in (20, 199, 201, 64, 13, 78)]
MAX_LEN = 130
MODELS = ["B3_vanilla_cnn", "A4_full"]


def inject_noops(window, ratio, rng):
    """Sisipkan no-op token acak ke posisi acak dalam region non-pad.
    Mengembalikan window baru (list token, dipotong/dipad ke MAX_LEN)."""
    seq = [t for t in window.tolist() if t != 0]  # buang PAD
    L = len(seq)
    if L == 0:
        return window[:MAX_LEN]
    k = int(np.floor(ratio * L))
    for _ in range(k):
        pos = rng.randint(0, len(seq) + 1)
        tok = NOOP_TOKENS[rng.randint(0, len(NOOP_TOKENS))]
        seq.insert(pos, tok)
    seq = seq[:MAX_LEN]
    out = np.zeros(MAX_LEN, dtype=np.int64)
    out[:len(seq)] = seq
    return out


def perturb_benign(X_benign, ratio, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros((len(X_benign), MAX_LEN), dtype=np.int64)
    for i in range(len(X_benign)):
        out[i] = inject_noops(X_benign[i], ratio, rng)
    return out


def fpr_of(model, X):
    probs = C.infer_probs(model, X, DEVICE)
    preds = (probs >= THRESHOLD).astype(int)
    # X seluruhnya benign -> setiap prediksi malicious = false prevention
    return float(preds.mean()), int(preds.sum())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("Diagnostic A3: FPR pada Kondisi Adversarial (benign + no-op)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    test = np.load(PREPARED / "test.npz")
    benign_mask = test["labels"] == 0
    X_benign = test["windows"][benign_mask].astype(np.int64)
    n_benign = len(X_benign)
    print(f"\nBenign test: {n_benign} window")

    # Pra-bangkitkan perturbasi (sekali, dipakai semua model agar adil)
    X_clean = C.pad_to(X_benign.copy(), MAX_LEN)
    perturbed = {r: perturb_benign(X_benign, r, SEED) for r in RATIOS}
    print(f"Perturbasi dibangkitkan untuk rasio: {RATIOS}")

    report = {"config": {"seed": SEED, "threshold": THRESHOLD, "ratios": RATIOS,
                         "noop_tokens": NOOP_TOKENS, "n_benign": n_benign},
              "models": {}}

    for tag in MODELS:
        model = C.load_adfa_model(tag, SEED, DEVICE)
        fpr_clean, n_clean = fpr_of(model, X_clean)
        entry = {"fpr_clean": fpr_clean, "n_flipped_clean": n_clean,
                 "fpr_perturbed": {}, "delta_fpr": {}, "n_flipped": {}}
        for r in RATIOS:
            fpr_p, n_p = fpr_of(model, perturbed[r])
            entry["fpr_perturbed"][f"{r}"] = fpr_p
            entry["delta_fpr"][f"{r}"] = fpr_p - fpr_clean
            entry["n_flipped"][f"{r}"] = n_p
        # Verdict
        max_delta = max(entry["delta_fpr"].values())
        entry["verdict"] = "OVERSPECIFIC" if max_delta > 0.02 else "ROBUST"
        report["models"][tag] = entry
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Model':<20}{'FPR-clean':>10}{'FPR@5%':>9}{'FPR@10%':>9}"
          f"{'FPR@30%':>9}{'max-d':>9}  Verdict")
    print("  " + "-" * 78)
    for tag in MODELS:
        e = report["models"][tag]
        md = max(e["delta_fpr"].values())
        print(f"  {tag:<20}{e['fpr_clean']:>10.4f}"
              f"{e['fpr_perturbed']['0.05']:>9.4f}"
              f"{e['fpr_perturbed']['0.1']:>9.4f}"
              f"{e['fpr_perturbed']['0.3']:>9.4f}{md:>+9.4f}  {e['verdict']}")  # noqa

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "diagnostic_a3_fpr_adversarial.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: results/diagnostic_a3_fpr_adversarial.json")

    print("\n" + "=" * 70)
    print("INTERPRETASI:")
    for tag in MODELS:
        e = report["models"][tag]
        print(f"  {tag}: FPR {e['fpr_clean']:.4f} (clean) -> "
              f"{e['fpr_perturbed']['0.3']:.4f} (30% no-op), "
              f"delta={e['delta_fpr']['0.3']:+.4f} [{e['verdict']}]")
    print("\n  Jika FPR naik dengan injeksi no-op pada benign, model terbukti")
    print("  overspecific terhadap artefak injeksi — konsisten dengan A1/A2:")
    print("  model mendeteksi 'sekuens termodifikasi', bukan 'maliciousness'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
