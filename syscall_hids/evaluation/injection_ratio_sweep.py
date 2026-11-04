"""
D1: Injection Ratio Sweep (Section 4.2.4 / 4.3.x)

Menutup kritik "kenapa hanya uji budget injeksi 30%?". Tanpa latih ulang —
membangkitkan adversarial test set Distribution-Aware pada beberapa budget
{0, 5, 10, 20, 30}% lalu inference pada checkpoint yang sudah ada (B3, A4).

Untuk tiap rasio dilaporkan:
  - JSD (Distribution-Aware) malicious vs distribusi benign
  - Recall B3 (Vanilla CNN) dan A4 (Full Model)
  - Paired recovery rate & flip rate (relatif ke clean, rasio 0%)
  - FPR benign + injeksi no-op (random, token-space) pada rasio yang sama

Output: results/injection_ratio_sweep.json
"""

import json
import pickle
import numpy as np
import torch
from collections import Counter
from scipy.spatial.distance import jensenshannon

import syscall_hids.data.lidds_common as C
from syscall_hids.attacks.adversarial_mimicry import (
    DistributionAwareMimicryAttack, build_bigram_ref, TRIGRAM_N, NOOP_SYSCALLS,
)
from syscall_hids.data.adfa_ld_preparation import load_traces, DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
THRESHOLD = 0.5
RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30]
MAX_LEN = C.MAX_LEN
# Token ids of the SAME six corrected no-ops the generator uses (raw NOOP_SYSCALLS
# from adversarial_mimicry), derived from the vocabulary rather than hard-coded.
_VOCAB = json.load(open(C.PREPARED_ADFA / "vocabulary.json"))
NOOP_TOKENS = [_VOCAB[str(s)] for s in NOOP_SYSCALLS]
N_JSD_SAMPLE = 500               # sampel window untuk estimasi JSD
N_BENIGN = 5000                  # subsample benign untuk FPR


def encode_pad(windows_raw, vocab):
    out = []
    for w in windows_raw:
        enc = [vocab.get(str(s), 1) for s in w]
        enc = enc[:MAX_LEN] + [0] * max(0, MAX_LEN - len(enc))
        out.append(enc)
    return np.array(out, dtype=np.int64)


def mean_jsd(windows_raw, ref_keys, ref_vec, n_sample):
    idx = np.linspace(0, len(windows_raw) - 1, min(n_sample, len(windows_raw))).astype(int)
    vals = []
    for i in idx:
        w = windows_raw[i]
        counts = Counter()
        for j in range(len(w) - TRIGRAM_N + 1):
            counts[tuple(w[j:j + TRIGRAM_N])] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        p = np.array([counts.get(k, 0) / total for k in ref_keys])
        vals.append(float(jensenshannon(p, ref_vec) ** 2))
    return float(np.mean(vals)) if vals else 0.0


def inject_benign_token(window, ratio, rng):
    seq = [t for t in window.tolist() if t != 0]
    L = len(seq)
    k = int(np.floor(ratio * L))
    for _ in range(k):
        seq.insert(rng.randint(0, len(seq) + 1), NOOP_TOKENS[rng.randint(0, len(NOOP_TOKENS))])
    seq = seq[:MAX_LEN]
    out = np.zeros(MAX_LEN, dtype=np.int64)
    out[:len(seq)] = seq
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 74)
    print("D1: Injection Ratio Sweep (Distribution-Aware)")
    print(f"Device: {DEVICE} | ratios: {[int(r*100) for r in RATIOS]}%")
    print("=" * 74)

    # --- Data & referensi ---
    vocab = json.load(open(C.PREPARED_ADFA / "vocabulary.json"))
    benign_tri = pickle.load(open(C.PREPARED_ADFA / "benign_trigrams.pkl", "rb"))
    benign_bi = build_bigram_ref(load_traces(DATA_ROOT / "Training_Data_Master"))
    ref_keys = sorted(benign_tri.keys())
    ref_vec = np.array([benign_tri[k] for k in ref_keys])

    test = np.load(C.PREPARED_ADFA / "test.npz", allow_pickle=True)
    test_mal_raw = test["raw_windows"][test["labels"] == 1]
    benign_tok = test["windows"][test["labels"] == 0].astype(np.int64)
    rng_b = np.random.RandomState(SEED)
    if len(benign_tok) > N_BENIGN:
        sel = rng_b.choice(len(benign_tok), N_BENIGN, replace=False)
        benign_tok = benign_tok[sel]
    print(f"\nTest malicious: {len(test_mal_raw)} | Benign sample: {len(benign_tok)}")

    # --- Model ---
    b3 = C.load_adfa_model("B3_vanilla_cnn", SEED, DEVICE)
    a4 = C.load_adfa_model("A4_full", SEED, DEVICE)

    # Baseline clean malicious predictions (rasio 0) untuk paired
    clean_enc = encode_pad([w.tolist() for w in test_mal_raw], vocab)
    b3_clean_pred = (C.infer_probs(b3, clean_enc, DEVICE) >= THRESHOLD).astype(int)
    a4_clean_pred = (C.infer_probs(a4, clean_enc, DEVICE) >= THRESHOLD).astype(int)

    rows = []
    for r in RATIOS:
        # Generate DA adversarial malicious pada rasio r
        atk = DistributionAwareMimicryAttack(benign_tri, benign_bi, NOOP_SYSCALLS, max_ratio=r)
        adv_raw = [atk.generate(w.tolist()) for w in test_mal_raw]
        jsd = mean_jsd(adv_raw, ref_keys, ref_vec, N_JSD_SAMPLE)

        adv_enc = encode_pad(adv_raw, vocab)
        b3_pred = (C.infer_probs(b3, adv_enc, DEVICE) >= THRESHOLD).astype(int)
        a4_pred = (C.infer_probs(a4, adv_enc, DEVICE) >= THRESHOLD).astype(int)
        b3_recall = float(b3_pred.mean())
        a4_recall = float(a4_pred.mean())

        # Paired (relatif clean)
        def paired(clean_pred, adv_pred):
            det = clean_pred == 1; miss = clean_pred == 0
            recovery = float((miss & (adv_pred == 1)).sum() / miss.sum()) if miss.sum() else 0.0
            flip = float((det & (adv_pred == 0)).sum() / det.sum()) if det.sum() else 0.0
            return recovery, flip
        b3_rec, b3_flip = paired(b3_clean_pred, b3_pred)
        a4_rec, a4_flip = paired(a4_clean_pred, a4_pred)

        # FPR benign + injeksi no-op (token-space, random)
        rng = np.random.RandomState(SEED)
        ben_inj = np.array([inject_benign_token(w, r, rng) for w in benign_tok])
        b3_fpr = float((C.infer_probs(b3, ben_inj, DEVICE) >= THRESHOLD).mean())
        a4_fpr = float((C.infer_probs(a4, ben_inj, DEVICE) >= THRESHOLD).mean())

        rows.append({
            "ratio": r, "jsd": jsd,
            "recall_b3": b3_recall, "recall_a4": a4_recall,
            "margin_a4_minus_b3": a4_recall - b3_recall,
            "recovery_b3": b3_rec, "flip_b3": b3_flip,
            "recovery_a4": a4_rec, "flip_a4": a4_flip,
            "fpr_benign_b3": b3_fpr, "fpr_benign_a4": a4_fpr,
        })
        print(f"  ratio {int(r*100):>2}% : JSD={jsd:.4f} | "
              f"B3rec={b3_recall:.4f} A4rec={a4_recall:.4f} "
              f"(margin={a4_recall-b3_recall:+.4f}) | "
              f"FPRben B3={b3_fpr:.4f} A4={a4_fpr:.4f}")

    # --- Tabel ringkas (format review user) ---
    print("\n" + "=" * 74)
    print("TABEL SWEEP (Distribution-Aware):")
    hdr = "  {:<26}" + "".join(f"{int(r*100):>9}%" for r in RATIOS)
    print(hdr.format("Rasio Injeksi"))
    print("  " + "-" * 70)
    def line(label, key):
        s = f"  {label:<26}"
        for row in rows:
            s += f"{row[key]:>10.4f}"
        return s
    print(line("JSD (DA)", "jsd"))
    print(line("Recall B3 Vanilla CNN", "recall_b3"))
    print(line("Recall A4 Full Model", "recall_a4"))
    print(line("Margin A4 - B3", "margin_a4_minus_b3"))
    print(line("Recovery rate A4", "recovery_a4"))
    print(line("Flip rate A4", "flip_a4"))
    print(line("FPR benign+noop B3", "fpr_benign_b3"))
    print(line("FPR benign+noop A4", "fpr_benign_a4"))

    report = {"config": {"seed": SEED, "ratios": RATIOS, "threshold": THRESHOLD,
                         "strategy": "distribution_aware",
                         "n_test_malicious": int(len(test_mal_raw)),
                         "n_benign_sample": int(len(benign_tok))},
              "rows": rows}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "injection_ratio_sweep.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: results/injection_ratio_sweep.json")
    print("=" * 74)


if __name__ == "__main__":
    main()
