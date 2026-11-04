"""
Multi-Tier Action Mapping (B1)

Mengimplementasikan fungsi kebijakan berjenjang pi(r) untuk memetakan skor risiko ke tindakan:

    pi(r) = Normal     jika r < tau1
            Alert      jika tau1 <= r < tau2
            Quarantine jika tau2 <= r < tau3
            Terminate  jika r >= tau3

Setiap aksi memiliki biaya asimetris terhadap ground-truth (benign/malicious).
Threshold (tau1,tau2,tau3) dioptimasi via grid-search untuk meminimalkan
expected cost pada prior serangan eksplisit (prevalensi uji). Threshold dikalibrasi
pada himpunan VALIDASI tingkat-trace (bukan data yang dipakai melatih model) dan
dilaporkan pada himpunan TEST -> tidak ada kebocoran test; sensitivitas terhadap
prior dilaporkan terpisah.

Tiga profil biaya disimulasikan: Balanced, Low-Disruption (gangguan benign
mahal), Zero-Tolerance (serangan terlewat sangat mahal).

Output: results/multitier_policy_report.json
"""

import json
import itertools
import numpy as np
import torch

import syscall_hids.data.lidds_common as C

PREPARED = C.PREPARED_ADFA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MODEL_TAG = "A4_full"

ACTIONS = ["Normal", "Alert", "Quarantine", "Terminate"]
BLOCKING_ACTIONS = {2, 3}  # Quarantine, Terminate menghentikan proses

# Cost matrix per profil: cost[action_idx][truth]  (truth 0=benign,1=malicious)
COST_PROFILES = {
    "Balanced": {
        "Normal":     [0.0, 100.0],
        "Alert":      [0.1, 50.0],
        "Quarantine": [5.0, 5.0],
        "Terminate":  [20.0, 0.0],
    },
    "Low-Disruption": {   # gangguan proses benign lebih mahal
        "Normal":     [0.0, 100.0],
        "Alert":      [0.1, 50.0],
        "Quarantine": [10.0, 5.0],
        "Terminate":  [40.0, 0.0],
    },
    "Zero-Tolerance": {   # serangan terlewat sangat mahal
        "Normal":     [0.0, 200.0],
        "Alert":      [0.1, 100.0],
        "Quarantine": [5.0, 5.0],
        "Terminate":  [20.0, 0.0],
    },
}

TAU_GRID = np.round(np.arange(0.10, 0.90, 0.05), 2)


def cost_array(profile):
    """4x2 numpy array dari dict profil."""
    return np.array([COST_PROFILES[profile][a] for a in ACTIONS], dtype=float)


def assign_actions(probs, taus):
    """taus=(t1,t2,t3) -> action idx per sampel via digitize."""
    t1, t2, t3 = taus
    return np.digitize(probs, [t1, t2, t3])  # 0..3


def _prior_cost(per_sample, labels, prior):
    """Expected cost per decision at an explicit attack prior: class-conditional
    mean costs weighted by `prior`, so the result does not silently depend on the
    class mix of whichever split it is computed on."""
    return float(prior * per_sample[labels == 1].mean()
                 + (1.0 - prior) * per_sample[labels == 0].mean())


def expected_cost(probs, labels, taus, cmat, prior):
    return _prior_cost(cmat[assign_actions(probs, taus), labels], labels, prior)


def block_metrics(probs, labels, taus):
    """ABR/FPR dengan 'blocked' = Quarantine|Terminate."""
    actions = assign_actions(probs, taus)
    blocked = np.isin(actions, list(BLOCKING_ACTIONS))
    mal = labels == 1
    ben = labels == 0
    abr = float((blocked & mal).sum() / mal.sum()) if mal.sum() else 0.0
    fpr = float((blocked & ben).sum() / ben.sum()) if ben.sum() else 0.0
    return abr, fpr


def confusion_4x2(probs, labels, taus):
    actions = assign_actions(probs, taus)
    mat = np.zeros((4, 2), dtype=int)
    for a in range(4):
        for t in range(2):
            mat[a, t] = int(((actions == a) & (labels == t)).sum())
    return mat


def grid_search_multitier(probs, labels, cmat, prior):
    """Cari (t1,t2,t3) dengan t1<t2<t3 yang minimasi expected cost."""
    best_cost, best_taus = float("inf"), None
    for t1, t2, t3 in itertools.combinations(TAU_GRID, 3):
        c = expected_cost(probs, labels, (t1, t2, t3), cmat, prior)
        if c < best_cost:
            best_cost, best_taus = c, (float(t1), float(t2), float(t3))
    return best_taus, best_cost


def single_cost(probs, labels, tau, cmat, prior):
    actions = np.where(probs >= tau, 3, 0)  # Terminate / Normal
    return _prior_cost(cmat[actions, labels], labels, prior)


def grid_search_single(probs, labels, cmat, prior):
    """Baseline single-threshold (Terminate jika r>=tau, else Normal), dicari pada
    grid, data, dan prior yang SAMA dengan multi-tier agar perbandingan adil."""
    best_cost, best_tau = float("inf"), None
    for tau in TAU_GRID:
        c = single_cost(probs, labels, tau, cmat, prior)
        if c < best_cost:
            best_cost, best_tau = c, float(tau)
    return best_tau, best_cost


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("Multi-Tier Action Mapping (B1)")
    print(f"Device: {DEVICE} | Model: {MODEL_TAG}")
    print("=" * 70)

    # Thresholds tuned on the trace-level VALIDATION split (held out from model
    # fitting; same split as checkpoint selection), reported on TEST.
    model = C.load_adfa_model(MODEL_TAG, SEED, DEVICE)
    tr = np.load(PREPARED / "train.npz")
    te = np.load(PREPARED / "test.npz")
    split = json.load(open(C.RESULTS_DIR / "splits" / f"val_traces_seed{SEED}.json"))
    vmask = np.isin(tr["trace_ids"], split["val_trace_ids"])
    Xva = C.pad_to(tr["windows"][vmask].astype(np.int64), C.MAX_LEN)
    Xte = C.pad_to(te["windows"].astype(np.int64), C.MAX_LEN)
    ytr = tr["labels"][vmask].astype(int)          # validation labels
    yte = te["labels"].astype(int)
    ptr = C.infer_probs(model, Xva, DEVICE)        # validation probs (tuning)
    pte = C.infer_probs(model, Xte, DEVICE)
    # Expected cost is computed at an explicit attack prior: the test prevalence
    # (deployment-like). Sensitivity to other priors is reported below.
    PRIOR = float(yte.mean())
    print(f"\nValidation: {len(ytr)} ({int((ytr==1).sum())} mal, "
          f"{split['n_val_traces']} traces) | Test: {len(yte)} ({int((yte==1).sum())} mal)"
          f" | attack prior = {PRIOR:.4f}")

    report = {"model": f"{MODEL_TAG}_seed{SEED}", "tuning_split": "validation (trace-level)",
              "attack_prior": PRIOR, "tau_grid": TAU_GRID.tolist(),
              "cost_profiles": COST_PROFILES, "profiles": {}}

    print(f"\n{'Profile':<16}{'tau1':>6}{'tau2':>6}{'tau3':>6}"
          f"{'MT-cost':>9}{'ST-cost':>9}{'cost_red':>9}{'ABR':>7}{'FPR':>7}")
    print("-" * 76)

    for profile in COST_PROFILES:
        cmat = cost_array(profile)
        # Multi-tier: tune on validation, report on test
        taus, _ = grid_search_multitier(ptr, ytr, cmat, PRIOR)
        mt_cost_test = expected_cost(pte, yte, taus, cmat, PRIOR)
        abr, fpr = block_metrics(pte, yte, taus)
        cm = confusion_4x2(pte, yte, taus)

        # Single-threshold baseline (same cost matrix, grid, data, prior)
        st_tau, _ = grid_search_single(ptr, ytr, cmat, PRIOR)
        st_cost_test = single_cost(pte, yte, st_tau, cmat, PRIOR)

        cost_red = 100 * (st_cost_test - mt_cost_test) / st_cost_test if st_cost_test else 0.0

        report["profiles"][profile] = {
            "multitier_thresholds": {"tau1": taus[0], "tau2": taus[1], "tau3": taus[2]},
            "multitier_expected_cost_test": mt_cost_test,
            "single_threshold": st_tau,
            "single_expected_cost_test": st_cost_test,
            "cost_reduction_pct": cost_red,
            "abr": abr, "fpr": fpr,
            "confusion_4x2": {ACTIONS[a]: {"benign": int(cm[a, 0]),
                                           "malicious": int(cm[a, 1])}
                              for a in range(4)},
        }

        print(f"{profile:<16}{taus[0]:>6.2f}{taus[1]:>6.2f}{taus[2]:>6.2f}"
              f"{mt_cost_test:>9.3f}{st_cost_test:>9.3f}{cost_red:>8.1f}%"
              f"{abr:>7.3f}{fpr:>7.3f}")

    # Confusion matrix detail untuk Balanced
    print("\nConfusion Matrix 4x2 (Balanced, test set):")
    print(f"  {'Action':<12}{'Benign':>10}{'Malicious':>11}")
    cmb = report["profiles"]["Balanced"]["confusion_4x2"]
    for a in ACTIONS:
        print(f"  {a:<12}{cmb[a]['benign']:>10}{cmb[a]['malicious']:>11}")

    # Sensitivity of the cost reduction to the assumed attack prevalence.
    print("\nPrevalence sensitivity (cost reduction MT vs ST, tuned+reported at prior):")
    print(f"  {'prior':>7}" + "".join(f"{p:>16}" for p in COST_PROFILES))
    report["prevalence_sensitivity"] = {}
    for prior in [0.01, PRIOR, 0.20, 0.50]:
        row = {}
        for profile in COST_PROFILES:
            cmat = cost_array(profile)
            taus, _ = grid_search_multitier(ptr, ytr, cmat, prior)
            st_tau, _ = grid_search_single(ptr, ytr, cmat, prior)
            mt = expected_cost(pte, yte, taus, cmat, prior)
            st = single_cost(pte, yte, st_tau, cmat, prior)
            row[profile] = {"mt_cost": mt, "st_cost": st,
                            "cost_reduction_pct": 100 * (st - mt) / st if st else 0.0}
        report["prevalence_sensitivity"][f"{prior:.4f}"] = row
        print(f"  {prior:>7.4f}" + "".join(f"{row[p]['cost_reduction_pct']:>15.1f}%"
                                         for p in COST_PROFILES))

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "multitier_policy_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: results/multitier_policy_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
