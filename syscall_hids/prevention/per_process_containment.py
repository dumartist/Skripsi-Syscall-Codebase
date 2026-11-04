"""
Section 4.4.3 + 4.4.6: Per-Process Risk Aggregation (B5) &
                        Containment Effectiveness Simulation (B3)

Keduanya beroperasi pada level trace (satu file ADFA = satu proses),
sehingga digabung dalam satu script.

B5 (Per-Process Aggregation): keputusan per-window independen rentan
terhadap false-positive terisolasi. Agregasi per-PID (max / mean atas
window dalam satu proses) menghasilkan keputusan yang lebih reliable.
ADFA tidak punya PID eksplisit -> file = surrogate PID.

B3 (Containment): tiap trace serangan dibagi N segmen sekuensial;
Containment Score = 1 - (segmen_pertama_terblokir / N). Mengukur seberapa
awal serangan dihentikan (relevan untuk APT kill-chain).

Output: results/per_process_report.json, results/containment_report.json
Thesis : Tabel 4.18 (per-process), 4.21 (containment).
"""

import json
import numpy as np
import torch

import syscall_hids.data.lidds_common as C
from syscall_hids.data.adfa_ld_preparation import (
    read_trace, sliding_windows, encode, load_traces,
    ATTACK_CATEGORIES, CAT_TO_ID, DATA_ROOT, TRAIN_ATTACK_MAX_FOLD,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
THRESHOLD = 0.5
N_SEG = 10  # jumlah segmen untuk containment


def load_test_traces_by_process(vocab):
    """Return list of (category, encoded_trace) untuk himpunan uji:
    benign (validation) + attack fold 6-10."""
    procs = []
    # Benign: validation partition
    for t in load_traces(DATA_ROOT / "Validation_Data_Master"):
        procs.append(("benign", encode(t, vocab)))
    # Attack: folds > TRAIN_ATTACK_MAX_FOLD
    attack_dir = DATA_ROOT / "Attack_Data_Master"
    for cat in ATTACK_CATEGORIES:
        for sub in sorted(attack_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith(cat + "_"):
                fold = int(sub.name.rsplit("_", 1)[-1])
                if fold <= TRAIN_ATTACK_MAX_FOLD:
                    continue
                for t in load_traces(sub):
                    procs.append((cat, encode(t, vocab)))
    return procs


@torch.no_grad()
def risk_for_windows(model, windows):
    """windows: list of token-lists (panjang 100). Return array risk."""
    if not windows:
        return np.array([])
    X = C.pad_to(np.array(windows, dtype=np.int64), C.MAX_LEN)
    return C.infer_probs(model, X, DEVICE)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("Section 4.4.3 + 4.4.6: Per-Process Aggregation (B5) & Containment (B3)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    vocab = json.load(open(C.PREPARED_ADFA / "vocabulary.json"))
    model = C.load_adfa_model("A4_full", SEED, DEVICE)
    procs = load_test_traces_by_process(vocab)
    n_benign = sum(1 for c, _ in procs if c == "benign")
    n_attack = len(procs) - n_benign
    print(f"\nProses uji: {len(procs)} ({n_benign} benign, {n_attack} attack)")

    # ------------------------------------------------------------------
    # Hitung risk per-window per-proses (sekali)
    # ------------------------------------------------------------------
    proc_data = []  # (category, is_mal, risks[])
    all_window_risks = []  # untuk baseline per-window
    all_window_labels = []
    for cat, enc in procs:
        wins = sliding_windows(enc)
        risks = risk_for_windows(model, wins)
        is_mal = 0 if cat == "benign" else 1
        proc_data.append((cat, is_mal, risks))
        all_window_risks.extend(risks.tolist())
        all_window_labels.extend([is_mal] * len(risks))

    all_window_risks = np.array(all_window_risks)
    all_window_labels = np.array(all_window_labels)

    # ==================================================================
    # B5: Per-Process Aggregation
    # ==================================================================
    def proc_decision(agg_fn):
        flags, labels = [], []
        for cat, is_mal, risks in proc_data:
            if len(risks) == 0:
                continue
            agg = agg_fn(risks)
            flags.append(int(agg >= THRESHOLD))
            labels.append(is_mal)
        return np.array(flags), np.array(labels)

    def abr_fpr(flags, labels):
        mal = labels == 1; ben = labels == 0
        abr = float((flags[mal] == 1).mean()) if mal.sum() else 0.0
        fpr = float((flags[ben] == 1).mean()) if ben.sum() else 0.0
        return abr, fpr

    # Per-window baseline
    win_pred = (all_window_risks >= THRESHOLD).astype(int)
    pw_abr = float((win_pred[all_window_labels == 1] == 1).mean())
    pw_fpr = float((win_pred[all_window_labels == 0] == 1).mean())

    # Per-process max & mean
    flags_max, labels_p = proc_decision(np.max)
    flags_mean, _ = proc_decision(np.mean)
    pmax_abr, pmax_fpr = abr_fpr(flags_max, labels_p)
    pmean_abr, pmean_fpr = abr_fpr(flags_mean, labels_p)

    # Isolated FP: benign proses yang max>=0.5 tapi mean<0.5 (diselamatkan mean-agg)
    isolated_fp = 0
    for cat, is_mal, risks in proc_data:
        if is_mal == 0 and len(risks) and risks.max() >= THRESHOLD and risks.mean() < THRESHOLD:
            isolated_fp += 1

    per_process = {
        "threshold": THRESHOLD,
        "per_window": {"abr": pw_abr, "fpr": pw_fpr,
                       "n_windows": int(len(all_window_labels))},
        "per_process_max": {"abr": pmax_abr, "fpr": pmax_fpr},
        "per_process_mean": {"abr": pmean_abr, "fpr": pmean_fpr},
        "n_processes": int(len(labels_p)),
        "isolated_fp_benign_saved_by_mean": isolated_fp,
    }

    print("\n[B5] Per-Process Aggregation (threshold=0.50):")
    print(f"  {'Granularity':<22}{'ABR':>8}{'FPR':>8}")
    print("  " + "-" * 38)
    print(f"  {'Per-Window':<22}{pw_abr:>8.4f}{pw_fpr:>8.4f}")
    print(f"  {'Per-Process (max)':<22}{pmax_abr:>8.4f}{pmax_fpr:>8.4f}")
    print(f"  {'Per-Process (mean)':<22}{pmean_abr:>8.4f}{pmean_fpr:>8.4f}")
    print(f"  Isolated FP benign diselamatkan mean-agg: {isolated_fp}")

    # ==================================================================
    # B3: Containment Effectiveness
    # ==================================================================
    def containment_of(enc):
        """Bagi trace ke N_SEG segmen, cari segmen pertama yang ada window
        ber-risk>=threshold. Return containment score (1=blok segmen awal)."""
        L = len(enc)
        if L == 0:
            return None
        seg_len = max(1, L // N_SEG)
        for i in range(N_SEG):
            start = i * seg_len
            end = L if i == N_SEG - 1 else (i + 1) * seg_len
            seg = enc[start:end]
            wins = sliding_windows(seg) if len(seg) >= 1 else []
            risks = risk_for_windows(model, wins)
            if len(risks) and risks.max() >= THRESHOLD:
                return 1.0 - (i / N_SEG)
        return 0.0  # tak pernah terblokir

    cont_by_cat = {cat: [] for cat in ATTACK_CATEGORIES}
    for cat, enc in procs:
        if cat == "benign":
            continue
        c = containment_of(enc)
        if c is not None:
            cont_by_cat[cat].append(c)

    containment = {"n_segments": N_SEG, "threshold": THRESHOLD, "per_category": {}}
    print(f"\n[B3] Containment Score per Kategori (N={N_SEG} segmen, threshold=0.50):")
    print(f"  {'Kategori':<20}{'Mean':>8}{'Median':>8}{'C>=0.8':>8}{'C=0':>6}{'n':>5}")
    print("  " + "-" * 55)
    for cat in ATTACK_CATEGORIES:
        scores = np.array(cont_by_cat[cat])
        if len(scores) == 0:
            continue
        entry = {
            "mean": float(scores.mean()), "median": float(np.median(scores)),
            "n_high_containment": int((scores >= 0.8).sum()),
            "n_never_blocked": int((scores == 0.0).sum()),
            "n_traces": int(len(scores)),
        }
        containment["per_category"][cat] = entry
        print(f"  {cat:<20}{entry['mean']:>8.3f}{entry['median']:>8.3f}"
              f"{entry['n_high_containment']:>8}{entry['n_never_blocked']:>6}"
              f"{entry['n_traces']:>5}")

    # ------------------------------------------------------------------
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "per_process_report.json", "w") as f:
        json.dump(per_process, f, indent=2, default=float)
    with open(C.RESULTS_DIR / "containment_report.json", "w") as f:
        json.dump(containment, f, indent=2, default=float)
    print(f"\n  Saved: results/per_process_report.json, containment_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
