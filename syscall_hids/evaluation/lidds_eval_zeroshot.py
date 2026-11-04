"""
LID-DS Step 4: Zero-shot Transfer Evaluation (Section 4.5.3)

Model A4 yang dilatih pada ADFA-LD diuji LANGSUNG pada LID-DS tanpa
pelatihan ulang, menggunakan vocabulary aligned (Mode B). Token LID-DS
dipetakan ke ruang token ADFA; socket->102, ipc->117, ABI_UNMAPPED-><UNK>.

Ekspektasi (jujur): kinerja turun drastis karena (a) domain shift workload,
(b) ketidakcocokan ABI i386<->x86_64 yang mengkolaps/menghilangkan token.
Ini adalah BUKTI domain shift untuk Section 4.5.6, bukan kegagalan metode.

Test set per skenario = partisi test/normal (benign) + test/normal_and_attack
(benign + malicious), dengan label window dari pelabelan timestamp-based.

Output: results/lidds_zeroshot_report.json
"""

import json
import numpy as np
import torch

import syscall_hids.data.lidds_common as C

SCENARIOS = ["CVE-2014-0160", "Bruteforce_CWE-307", "CWE-89-SQL-injection"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 64)
    print("LID-DS Step 4: Zero-shot Transfer (ADFA model -> LID-DS)")
    print(f"Device: {DEVICE}")
    print("=" * 64)

    model = C.load_adfa_model("A4_full", SEED, DEVICE)
    remap = C.build_adfa_remap()

    report = {"mode": "zero_shot", "model": "A4_full_seed42",
              "scenarios": {}}

    for scn in SCENARIOS:
        data = C.load_scenario(scn)
        part = data["partition"]
        # Test set: test_normal + test_attack
        test_mask = (part == C.P_TEST_NORMAL) | (part == C.P_TEST_ATTACK)
        win = data["windows"][test_mask]
        lbl = data["labels"][test_mask]

        # Coverage berbobot-frekuensi (token-level) pada test set
        cov = C.token_coverage(win)
        tot_tok = sum(cov.values())
        cov_pct = {k: round(100 * v / tot_tok, 2) for k, v in cov.items()} if tot_tok else cov

        # Remap native -> ADFA, lalu pad ke MAX_LEN
        win_adfa = C.remap_windows(win, remap)
        win_adfa = C.pad_to(win_adfa, C.MAX_LEN)

        probs = C.infer_probs(model, win_adfa, DEVICE)
        m = C.compute_metrics(probs, lbl)

        report["scenarios"][scn] = {
            "n_test": int(len(lbl)),
            "n_malicious": int(lbl.sum()),
            "n_benign": int((lbl == 0).sum()),
            "metrics": m,
            "token_coverage_pct": cov_pct,
        }

        print(f"\n[{scn}]")
        print(f"  Test: {len(lbl)} window ({int(lbl.sum())} mal / {int((lbl==0).sum())} benign)")
        print(f"  Token coverage (freq-weighted): "
              f"direct={cov_pct.get('direct',0):.1f}%  "
              f"socket={cov_pct.get('socket_collapsed',0):.1f}%  "
              f"ipc={cov_pct.get('ipc_collapsed',0):.1f}%  "
              f"unmapped={cov_pct.get('abi_unmapped',0):.1f}%  "
              f"unk={cov_pct.get('unk_native',0):.1f}%")
        print(f"  Recall={m['recall']:.4f}  FPR={m['fpr']:.4f}  "
              f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "lidds_zeroshot_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 64)
    print("Ringkasan Zero-shot (Section 4.5.3):")
    print(f"  {'Skenario':<24}{'Recall':>8}{'FPR':>8}{'F1':>8}{'AUC':>8}{'AUPRC':>8}")
    print("  " + "-" * 64)
    for scn, r in report["scenarios"].items():
        m = r["metrics"]
        print(f"  {scn:<24}{m['recall']:>8.3f}{m['fpr']:>8.3f}"
              f"{m['f1']:>8.3f}{m['auc']:>8.3f}{m['auprc']:>8.3f}")
    print(f"\n  Saved: results/lidds_zeroshot_report.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
