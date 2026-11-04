"""
Diagnostic A2: Paired Robustness Evaluation (Section 4.3.10)

Mengganti single-population comparison (recall clean vs recall adv secara
terpisah) dengan paired evaluation per-sampel — standar adversarial ML modern.

Untuk sampel malicious x_i dengan x_i_adv = perturbasi adversarial-nya:
  Robustness Rate (RR) = |{f(x)=mal AND f(x_adv)=mal}| / |{f(x)=mal}|
  Flip Rate (FR)       = |{f(x)=mal AND f(x_adv)=benign}| / |{f(x)=mal}|
  (RR + FR = 1 untuk sampel yang clean-terdeteksi malicious)

Invarian alignment (terverifikasi): adversarial_data/<strat>_test.npz[i]
adalah perturbasi dari sampel malicious ke-i di prepared_data/test.npz
(urutan identik, cat_ids cocok).

Output: results/diagnostic_a2_paired.json
"""

import json
import numpy as np
import torch

import syscall_hids.data.lidds_common as C

PREPARED = C.PREPARED_ADFA
ADV = __import__("pathlib").Path("adversarial_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["B3_vanilla_cnn", "A2_cls_robust", "A3_cls_contrastive", "A4_full"]
STRATEGIES = ["distribution_aware", "random", "position_aware"]
SEED = 42
THRESHOLD = 0.5


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("Diagnostic A2: Paired Robustness Evaluation")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Clean malicious (paired reference)
    test = np.load(PREPARED / "test.npz")
    mal_mask = test["labels"] == 1
    X_clean_mal = C.pad_to(test["windows"][mal_mask].astype(np.int64), C.MAX_LEN)
    clean_cats = test["cat_ids"][mal_mask]
    n_mal = len(X_clean_mal)
    print(f"\nClean malicious: {n_mal} sampel")

    # Adversarial sets (paired by order)
    adv_sets = {}
    for strat in STRATEGIES:
        d = np.load(ADV / f"{strat}_test.npz")
        Xa = C.pad_to(d["windows"].astype(np.int64), C.MAX_LEN)
        assert len(Xa) == n_mal, f"{strat}: jumlah tidak cocok"
        assert np.array_equal(d["cat_ids"], clean_cats), f"{strat}: cat_ids tidak align"
        adv_sets[strat] = Xa
    print(f"Adversarial sets: {list(adv_sets.keys())} (semua aligned)")

    report = {"config": {"seed": SEED, "threshold": THRESHOLD, "n_malicious": n_mal},
              "models": {}, "comparison_table": []}

    for tag in MODELS:
        model = C.load_adfa_model(tag, SEED, DEVICE)
        clean_probs = C.infer_probs(model, X_clean_mal, DEVICE)
        clean_pred = (clean_probs >= THRESHOLD).astype(int)
        clean_detected = clean_pred == 1
        n_detected = int(clean_detected.sum())

        report["models"][tag] = {}
        for strat, Xa in adv_sets.items():
            adv_probs = C.infer_probs(model, Xa, DEVICE)
            adv_pred = (adv_probs >= THRESHOLD).astype(int)

            # Paired (hanya sampel yang clean-terdeteksi malicious)
            robust = int((clean_detected & (adv_pred == 1)).sum())
            flipped = int((clean_detected & (adv_pred == 0)).sum())
            rr = robust / n_detected if n_detected else 0.0
            fr = flipped / n_detected if n_detected else 0.0

            # Recovery: sampel yang LOLOS deteksi clean tapi terdeteksi
            # setelah injeksi no-op. Tingginya nilai ini mengindikasikan
            # model mendeteksi ARTEFAK injeksi, bukan maliciousness.
            clean_missed = ~clean_detected
            n_missed = int(clean_missed.sum())
            n_recovered = int((clean_missed & (adv_pred == 1)).sum())
            recovery_rate = n_recovered / n_missed if n_missed else 0.0

            sp_clean = float(clean_pred.mean())
            sp_adv = float(adv_pred.mean())

            entry = {
                "n_clean_detected": n_detected,
                "n_robust": robust, "n_flipped": flipped,
                "robustness_rate": rr, "flip_rate": fr,
                "n_clean_missed": n_missed,
                "n_recovered_by_injection": n_recovered,
                "recovery_rate": recovery_rate,
                "single_pop_recall_clean": sp_clean,
                "single_pop_recall_adv": sp_adv,
                "delta_pop_recall": sp_adv - sp_clean,
            }
            report["models"][tag][strat] = entry
            report["comparison_table"].append({
                "model": tag, "strategy": strat,
                "RR": rr, "FR": fr,
                "single_pop_clean": sp_clean, "single_pop_adv": sp_adv,
            })

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Print
    print(f"\n  {'Model':<20}{'Strategy':<20}{'RR':>8}{'FR':>8}"
          f"{'SP-clean':>10}{'SP-adv':>9}")
    print("  " + "-" * 75)
    for tag in MODELS:
        for strat in STRATEGIES:
            e = report["models"][tag][strat]
            warn = "  <-- RR<SP-adv" if e["robustness_rate"] < e["single_pop_recall_adv"] - 1e-9 else ""
            print(f"  {tag:<20}{strat:<20}{e['robustness_rate']:>8.4f}"
                  f"{e['flip_rate']:>8.4f}{e['single_pop_recall_clean']:>10.4f}"
                  f"{e['single_pop_recall_adv']:>9.4f}{warn}")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "diagnostic_a2_paired.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: results/diagnostic_a2_paired.json")

    # Ringkasan interpretatif (fokus DA, A4 vs B3)
    print("\n" + "=" * 70)
    print("INTERPRETASI (Distribution-Aware):")
    for tag in ["B3_vanilla_cnn", "A4_full"]:
        e = report["models"][tag]["distribution_aware"]
        print(f"  {tag}: RR={e['robustness_rate']:.4f} (FR={e['flip_rate']:.4f}) | "
              f"clean-missed={e['n_clean_missed']} -> recovered-by-injection="
              f"{e['n_recovered_by_injection']} (recovery={e['recovery_rate']:.1%})")
    print("\n  Kesimpulan: RR ~ 1.0 menunjukkan injeksi no-op TIDAK mengelabui")
    print("  model (robustness genuine pada sampel clean-terdeteksi). Kenaikan")
    print("  single-pop recall justru berasal dari RECOVERY sampel yang lolos")
    print("  deteksi clean — model mendeteksi artefak injeksi, bukan maliciousness.")
    print("  Ini mengkonfirmasi verdict shortcut-dominant A1 dari sudut paired.")
    print("=" * 70)


if __name__ == "__main__":
    main()
