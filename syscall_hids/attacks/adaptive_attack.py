"""
Adaptive white-box attack on syscall sequences.

Mimikri statis (adversarial_mimicry.py) menyuntik no-op SEKALI lalu defender
dilatih sesudahnya — itu data augmentation berlabel musuh, bukan robustness dalam
definisi Carlini/Tramèr. Di sini attacker bersifat ADAPTIF & white-box: ia tahu
model target (A4/B3) dan secara iteratif memilih penyisipan no-op yang paling
menurunkan probabilitas kelas malicious, sampai trace malicious dikira benign
(evasi) atau budget habis.

Greedy white-box evasion (aksi diskret -> bukan PGD kontinu):
  - Mulai dari trace malicious yang TERDETEKSI model (true positive).
  - Tiap langkah: bangun kandidat (posisi sisip × token no-op), evaluasi SEMUA
    via satu forward batched, pilih yang prob-malicious-nya paling rendah.
  - Komit penyisipan itu. Ulang sampai prob < THRESHOLD (evasi) atau budget ≤30%.
  - Hormati ΦF (urutan syscall asli dipertahankan, hanya INSERT), ΦO (hanya token
    no-op valid), ΦM (≤30% panjang asli).

Metrik: evasion success rate = fraksi malicious-terdeteksi yang berhasil dibuat
lolos. Recall-under-adaptive = (1 − evasion). Bandingkan A4 vs B3: jika A4 benar
"lebih robust", evasi terhadap A4 harus LEBIH SULIT (success lebih rendah / butuh
budget lebih besar).

Output: results/adaptive_attack.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from syscall_hids.diagnostics.diagnostic_shortcut_learning import (
    load_model, get_probs, pad_to, MAX_LEN, THRESHOLD, DEVICE,
)
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import load_noop_tokens, NOOP_SYSCALLS_RAW

PREPARED_DIR = Path("prepared_data")
RESULTS_DIR = Path("results")

MODELS = ["B3_vanilla_cnn", "A4_full"]
SEED = 42
MAX_RATIO = 0.30          # ΦM
N_POS_CANDIDATES = 24     # posisi sisip yang dievaluasi per langkah
MAX_SAMPLES = 300         # malicious yang diserang (subset untuk kecepatan)


@torch.no_grad()
def prob_malicious_batch(model, seqs):
    """seqs: list[list[int]] -> np array prob malicious (encode+pad ke MAX_LEN)."""
    arr = np.zeros((len(seqs), MAX_LEN), dtype=np.int64)
    for i, s in enumerate(seqs):
        t = s[:MAX_LEN]
        arr[i, :len(t)] = t
    x = torch.tensor(arr, dtype=torch.long, device=DEVICE)
    logits = model(x)
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def attack_one(model, base_seq, noop_tokens, max_ratio=MAX_RATIO):
    """Greedy white-box evasion pada satu trace malicious (list token, tanpa PAD).
    Return dict: evaded, n_inserted, ratio_used, prob_before, prob_after."""
    seq = list(base_seq)
    real_len = len(seq)
    budget = int(np.floor(max_ratio * real_len))
    prob_before = float(prob_malicious_batch(model, [seq])[0])
    cur_prob = prob_before

    rng = np.random.default_rng(SEED)
    inserted = 0
    for _ in range(budget):
        L = len(seq)
        # Kandidat posisi sisip (sampel merata + ujung), token no-op semua
        if L + 1 > N_POS_CANDIDATES:
            positions = sorted(set(np.linspace(0, L, N_POS_CANDIDATES).astype(int).tolist()))
        else:
            positions = list(range(L + 1))

        candidates, meta = [], []
        for pos in positions:
            for noop in noop_tokens:
                c = seq[:pos] + [int(noop)] + seq[pos:]
                candidates.append(c)
                meta.append((pos, int(noop)))

        probs = prob_malicious_batch(model, candidates)
        j = int(np.argmin(probs))
        if probs[j] >= cur_prob:   # tak ada perbaikan -> berhenti (local optimum)
            break
        pos, noop = meta[j]
        seq = seq[:pos] + [noop] + seq[pos:]
        cur_prob = float(probs[j])
        inserted += 1
        if cur_prob < THRESHOLD:   # evasi berhasil
            break

    return {
        "evaded": cur_prob < THRESHOLD,
        "n_inserted": inserted,
        "ratio_used": inserted / real_len if real_len else 0.0,
        "prob_before": prob_before,
        "prob_after": cur_prob,
    }


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 72)
    print("#3 Adaptive White-box Attacker (greedy evasion) — uji 'robust'")
    print(f"Device: {DEVICE} | budget<=30% | n_pos={N_POS_CANDIDATES} | max_samples={MAX_SAMPLES}")
    print("=" * 72)

    noop_tokens, _ = load_noop_tokens()
    noop_unique = sorted(set(int(t) for t in noop_tokens))
    print(f"no-op tokens: {noop_unique} (raw {NOOP_SYSCALLS_RAW})")

    test = np.load(PREPARED_DIR / "test.npz")
    mal_enc = test["windows"][test["labels"] == 1]   # (N,100) terenkode
    # buang PAD jadi list token nyata
    mal_seqs_all = [w[w != 0].tolist() for w in mal_enc]

    report = {"config": {"max_ratio": MAX_RATIO, "n_pos": N_POS_CANDIDATES,
                         "max_samples": MAX_SAMPLES, "noop_tokens": noop_unique,
                         "threshold": THRESHOLD}, "per_model": {}}

    for model_name in MODELS:
        model = load_model(model_name, SEED)

        # Hanya serang malicious yang TERDETEKSI (true positive) pada clean
        Xmc = pad_to(torch.tensor(mal_enc, dtype=torch.long), MAX_LEN)
        p_clean = get_probs(model, Xmc)
        detected_idx = np.where(p_clean >= THRESHOLD)[0]
        if len(detected_idx) > MAX_SAMPLES:
            detected_idx = np.random.RandomState(SEED).choice(
                detected_idx, MAX_SAMPLES, replace=False)
        print(f"\n[{model_name}] detected malicious (clean): "
              f"{int((p_clean>=THRESHOLD).sum())}/{len(p_clean)}; "
              f"menyerang {len(detected_idx)} sampel...")

        results = [attack_one(model, mal_seqs_all[i], noop_unique) for i in detected_idx]

        evaded = np.array([r["evaded"] for r in results])
        ratios = np.array([r["ratio_used"] for r in results])
        evasion_rate = float(evaded.mean())
        recall_clean = float((p_clean >= THRESHOLD).mean())

        agg = {
            "n_attacked": int(len(results)),
            "evasion_success_rate": evasion_rate,
            "recall_under_adaptive": float(1.0 - evasion_rate),
            "recall_clean": recall_clean,
            "mean_ratio_used_overall": float(ratios.mean()),
            "mean_ratio_used_evaded": float(ratios[evaded].mean()) if evaded.any() else 0.0,
            "mean_prob_drop": float(np.mean([r["prob_before"] - r["prob_after"] for r in results])),
        }
        report["per_model"][model_name] = agg
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  evasion success      = {evasion_rate:.1%}")
        print(f"  recall under adaptive = {agg['recall_under_adaptive']:.1%} "
              f"(vs clean {recall_clean:.1%})")
        print(f"  mean budget (evaded)  = {agg['mean_ratio_used_evaded']:.1%}")

    # Diskusi A4 vs B3
    disc = []
    b3 = report["per_model"].get("B3_vanilla_cnn", {})
    a4 = report["per_model"].get("A4_full", {})
    if b3 and a4:
        disc.append(
            f"Evasion success: B3={b3['evasion_success_rate']:.1%}, "
            f"A4={a4['evasion_success_rate']:.1%}. "
            + ("A4 LEBIH SULIT dievasi -> adversarial training memberi robustness "
               "nyata terhadap serangan adaptif (bukan cuma artefak FPR)."
               if a4['evasion_success_rate'] < b3['evasion_success_rate'] - 0.02 else
               "A4 TIDAK lebih sulit dievasi dari B3 -> di bawah attacker adaptif, "
               "keunggulan A4 (yang terlihat pada mimikri statis) HILANG. Ini batas "
               "robustness yang harus diakui jujur: 'tahan' hanya valid utk serangan statis.")
        )
        disc.append(
            f"Recall A4 turun dari {a4['recall_clean']:.1%} (clean) menjadi "
            f"{a4['recall_under_adaptive']:.1%} di bawah serangan adaptif white-box "
            f"≤30% — inilah angka robustness yang sebenarnya (definisi Carlini/Tramèr), "
            "menggantikan AdvRec mimikri-statis yang optimistis."
        )
    report["summary"] = {"discussion_points": disc}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "adaptive_attack.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    for i, d in enumerate(disc, 1):
        print(f"[{i}] {d}\n")


if __name__ == "__main__":
    main()
