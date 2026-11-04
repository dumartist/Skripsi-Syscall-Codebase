"""Diagnostic quantities for the manuscript, recomputed on the corrected run.

  §4.9   static-mimicry comparison: paired (per-seed) differences vs B0 with 95% CI,
         Cohen's d and Holm-adjusted p. No pass/fail threshold — the old ">=5 pp =
         genuine" rule was unreachable near the recall ceiling and is removed.
  §4.10  paired robustness on Distribution-Aware mimicry (seed 42).
  §4.11  benign + injected-token FPR, with a LENGTH CONTROL: inserting the same
         number of benign-frequent non-no-op tokens. If FPR rises under no-op
         insertion but not under the control, sequence length/padding is not
         what drives it.
  §4.12  adaptive evasion: paired per-seed differences vs B0 (every pipeline is
         attacked on the same traces per seed — see adaptive_eval.py).

Inputs : results/corrected_full.json, results/adaptive_full.json, checkpoints.
Output : results/corrected_analysis.json
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import stats

from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model, get_probs, pad_to, THRESHOLD, MAX_LEN
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import inject_noop_tokens
from syscall_hids.training.corrected_pipeline import NOOP_TOKENS, STRATEGIES

PREP = Path("prepared_data"); ADV = Path("adversarial_data"); RES = Path("results")
FPR_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
BENIGN_N = 8000                      # seed-fixed subsample of benign test windows
RATIOS = (0.05, 0.10, 0.30)


def paired(a, b):
    """a - b per seed: mean, 95% CI (t), raw p (paired t-test), Cohen's d_z."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = float(stats.ttest_rel(a, b).pvalue) if se > 0 else 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "p": p, "cohens_dz": float(d.mean() / (d.std(ddof=1) + 1e-12)), "n_seeds": n}


def holm(results):
    """Holm-adjust the 'p' of a dict of results in place (family = the dict)."""
    keys = sorted(results, key=lambda k: results[k]["p"])
    m, running = len(keys), 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * results[k]["p"]))
        results[k]["p_holm"] = running
    return results


def static_comparisons(full):
    def rec(tag, s):
        return [r["adv"][s]["recall"] for r in full["runs"][tag]]
    fam = {}
    for tag in ["C_A2_pairedKL", "C_A4_full"]:
        for s in STRATEGIES:
            fam[f"{tag}_vs_B0__{s}"] = paired(rec(tag, s), rec("C_B0_vanilla", s))
    return holm(fam)


def adaptive_comparisons(adf):
    fam = {}
    b0 = adf["runs"]["C_B0_vanilla"]
    for tag in ["C_B0_sym", "C_AT_augment", "C_A2_pairedKL", "C_A3_contrastive",
                "C_A4_full", "C_CANON_dedup", "C_CANON_strip"]:
        for cap in ["evasion_le_10pct", "evasion_le_20pct", "evasion_le_30pct"]:
            fam[f"{tag}_vs_B0__{cap}"] = paired(
                [r[cap] for r in adf["runs"][tag]], [r[cap] for r in b0])
    return holm(fam)


def paired_robustness(tags, seed=42):
    test = np.load(PREP / "test.npz")
    m1 = test["labels"] == 1
    mal, tids = test["windows"][m1], test["trace_ids"][m1]
    Xc = pad_to(torch.tensor(mal, dtype=torch.long), MAX_LEN)
    Xa = pad_to(torch.tensor(np.load(ADV / "distribution_aware_test.npz")["windows"],
                             dtype=torch.long), MAX_LEN)
    out = {"n_windows": int(len(mal)), "n_traces": int(len(np.unique(tids)))}
    for tag in tags:
        m = load_model(tag, seed)
        pc = get_probs(m, Xc) >= THRESHOLD
        pa = get_probs(m, Xa) >= THRESHOLD
        missed = ~pc
        rr = float(pa[pc].mean()) if pc.any() else 0.0
        out[tag] = {"RR": rr, "FR": 1 - rr, "clean_detected": int(pc.sum()),
                    "clean_missed": int(missed.sum()), "recovered": int(pa[missed].sum()),
                    "recovery_rate": float(pa[missed].sum() / max(int(missed.sum()), 1))}
    return out


def benign_token_pool():
    """Benign TRAIN token frequencies excluding PAD, UNK and the no-op tokens."""
    tr = np.load(PREP / "train.npz")
    w = tr["windows"][tr["labels"] == 0].ravel()
    excl = {0, 1, *NOOP_TOKENS}
    vals, cnt = np.unique(w[~np.isin(w, list(excl))], return_counts=True)
    return vals, cnt / cnt.sum()


def inject_pool(window, pool, probs, ratio, rng):
    """Same procedure as inject_noop_tokens, but tokens drawn from `pool`."""
    real = int((window != 0).sum())
    seq = window[:real].tolist()
    for _ in range(int(np.floor(ratio * real))):
        seq.insert(int(rng.integers(0, len(seq) + 1)), int(rng.choice(pool, p=probs)))
    seq = seq[:MAX_LEN]
    return np.array(seq + [0] * (MAX_LEN - len(seq)), dtype=np.int64)


def injected_fpr(tags):
    test = np.load(PREP / "test.npz")
    ben = test["windows"][test["labels"] == 0]
    ben = ben[np.random.default_rng(0).choice(len(ben), min(BENIGN_N, len(ben)),
                                              replace=False)]
    sets = {"clean": pad_to(torch.tensor(ben, dtype=torch.long), MAX_LEN)}
    for r in RATIOS:
        rng = np.random.default_rng(42)
        sets[f"noop_{int(r*100)}"] = torch.tensor(np.stack(
            [inject_noop_tokens(w, NOOP_TOKENS, r, rng) for w in ben]), dtype=torch.long)
    pool, probs = benign_token_pool()
    rng = np.random.default_rng(43)
    sets["benigntok_30"] = torch.tensor(np.stack(
        [inject_pool(w, pool, probs, 0.30, rng) for w in ben]), dtype=torch.long)

    out = {"n_benign_windows": int(len(ben)), "seeds": FPR_SEEDS}
    for tag in tags:
        per = {k: [] for k in sets}
        for seed in FPR_SEEDS:
            m = load_model(tag, seed)
            for k, X in sets.items():
                per[k].append(float((get_probs(m, X) >= THRESHOLD).mean()))
        res = {k: [float(np.mean(v)), float(np.std(v))] for k, v in per.items()}
        res["increase_noop_30"] = paired(per["noop_30"], per["clean"])
        res["increase_benigntok_30"] = paired(per["benigntok_30"], per["clean"])
        out[tag] = res
    return out


def main():
    full = json.load(open(RES / "corrected_full.json"))
    adf = json.load(open(RES / "adaptive_full.json"))
    report = {
        "seeds": full["config"]["seeds"],
        "static_vs_B0_4_9": static_comparisons(full),
        "paired_robustness_4_10": paired_robustness(
            ["C_B0_vanilla", "C_A2_pairedKL", "C_A3_contrastive", "C_A4_full"]),
        "injected_fpr_4_11": injected_fpr(
            ["C_B0_vanilla", "C_B0_sym", "C_AT_augment", "C_A2_pairedKL",
             "C_A3_contrastive", "C_A4_full"]),
        "adaptive_vs_B0_4_12": adaptive_comparisons(adf),
    }
    with open(RES / "corrected_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n== §4.9 static recall vs B0 (paired, 10 seeds, Holm) ==")
    for k, d in report["static_vs_B0_4_9"].items():
        print(f"  {k:42s} diff={d['diff']*100:+.2f}pp CI=[{d['ci95'][0]*100:+.2f},"
              f"{d['ci95'][1]*100:+.2f}] dz={d['cohens_dz']:+.2f} p_holm={d['p_holm']:.3g}")
    print("\n== §4.10 paired robustness (DA, seed 42) ==")
    pr = report["paired_robustness_4_10"]
    print(f"  windows={pr['n_windows']} traces={pr['n_traces']}")
    for t in ["C_B0_vanilla", "C_A2_pairedKL", "C_A3_contrastive", "C_A4_full"]:
        d = pr[t]
        print(f"  {t:18s} RR={d['RR']:.4f} missed->recovered {d['clean_missed']}->"
              f"{d['recovered']} ({d['recovery_rate']:.1%})")
    print("\n== §4.11 benign FPR: clean / +no-op 5,10,30% / +benign-token 30% (control) ==")
    for t, d in report["injected_fpr_4_11"].items():
        if not isinstance(d, dict):
            continue
        print(f"  {t:18s} {d['clean'][0]:.3f} / {d['noop_5'][0]:.3f} {d['noop_10'][0]:.3f} "
              f"{d['noop_30'][0]:.3f} / ctrl {d['benigntok_30'][0]:.3f}  "
              f"inc_noop={d['increase_noop_30']['diff']*100:+.1f}pp "
              f"inc_ctrl={d['increase_benigntok_30']['diff']*100:+.1f}pp")
    print("\n== §4.12 adaptive evasion vs B0 (paired on common traces, Holm) ==")
    for k, d in report["adaptive_vs_B0_4_12"].items():
        if "le_10" in k:
            print(f"  {k:44s} diff={d['diff']*100:+.1f}pp CI=[{d['ci95'][0]*100:+.1f},"
                  f"{d['ci95'][1]*100:+.1f}] p_holm={d['p_holm']:.3g}")
    print("\nSaved: results/corrected_analysis.json")


if __name__ == "__main__":
    main()
