"""Evaluate adaptive evasion, false positives, and uncertainty across models.

Stages (run all, or pick with --stages):
  pertrace  Re-run the score-based adaptive attack on the SAME populations as
            adaptive_eval.py, keeping per-trace outcomes, and assert that the
            aggregates reproduce results/adaptive_full.json exactly.
  alphabeta alpha/beta sensitivity grid (5 seeds). Existing checkpoints are reused
            for (0,0)=B0, (1,0)=A2, (0,0.1)=A3, (1,0.1)=A4; other cells are trained
            with the same corrected trainer.
  b1        B1 n-gram baseline: Random Forest AND linear SVM reported separately
            (no test-set model selection), incl. static mimicry recall, injected-FPR
            and the score-based adaptive attack (5 seeds).
  pauc      Standardized partial AUC (FPR <= 5%, McClish) for every model x seed.
  boot      Trace-level bootstrap (test-sample uncertainty) and two-level
            seed x trace bootstrap for the key paired differences.
  prev      Expected precision and operational burden at 7% / 1% / 0.1% prevalence.

Outputs: results/examiner/*.json (one file per stage).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from syscall_hids.attacks.adaptive_attack import attack_one
from syscall_hids.diagnostics.analyze_corrected import inject_pool, benign_token_pool, paired, holm
from syscall_hids.training.corrected_pipeline import (CanonModel, common_attack_set, load_all, train_corrected,
                                NOOP_TOKENS, STRATEGIES)
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import inject_noop_tokens
from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model, get_probs, pad_to, THRESHOLD, MAX_LEN

PREP, ADV, RES = Path("prepared_data"), Path("adversarial_data"), Path("results")
OUT = RES / "examiner"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
SEEDS5 = SEEDS[:5]
PIPELINES = [("C_B0_vanilla", "none"), ("C_B0_sym", "none"), ("C_AT_augment", "none"),
             ("C_A2_pairedKL", "none"), ("C_A3_contrastive", "none"),
             ("C_A4_full", "none"), ("C_CANON_dedup", "dedup"),
             ("C_CANON_strip", "strip")]
CAPS = (0.10, 0.20, 0.30)
N_BOOT = 2000


def dump(name, obj):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  saved results/examiner/{name}")


def load_pipe(tag, mode, seed):
    m = load_model(tag, seed)
    return CanonModel(m, mode) if mode != "none" else m


def test_arrays():
    te = np.load(PREP / "test.npz")
    return te["windows"], te["labels"].astype(int), te["trace_ids"]


def mal_test():
    w, y, t = test_arrays()
    m = y == 1
    return w[m], t[m]


def attack_outcomes(model, sample_idx):
    mal_enc, _ = mal_test()
    seqs = [w[w != 0].tolist() for w in mal_enc[sample_idx]]
    res = [attack_one(model, s, NOOP_TOKENS) for s in seqs]
    ev = np.array([r["evaded"] for r in res], bool)
    ratio = np.array([r["ratio_used"] for r in res], float)
    return {f"le_{int(c*100)}": (ev & (ratio <= c + 1e-9)).astype(int).tolist() for c in CAPS}


# --------------------------------------------------------------------------- pertrace
def stage_pertrace():
    ref = json.load(open(RES / "adaptive_full.json"))
    out = {"seeds": SEEDS, "samples": {}, "outcomes": {t: {} for t, _ in PIPELINES}}
    for seed in SEEDS:
        models = {t: load_pipe(t, m, seed) for t, m in PIPELINES}
        idx, _, _ = common_attack_set(models, seed, 200)
        _, tids = mal_test()
        out["samples"][str(seed)] = {"window_idx": idx.tolist(), "trace_ids": tids[idx].tolist()}
        for i, (tag, _) in enumerate(PIPELINES):
            o = attack_outcomes(models[tag], idx)
            out["outcomes"][tag][str(seed)] = o
            for c in CAPS:
                k = f"le_{int(c*100)}"
                got = float(np.mean(o[k]))
                exp = ref["runs"][tag][SEEDS.index(seed)][f"evasion_{k}pct"]
                assert abs(got - exp) < 1e-9, (tag, seed, k, got, exp)
        print(f"  seed {seed}: per-trace outcomes reproduce adaptive_full.json")
    dump("adaptive_pertrace.json", out)


# --------------------------------------------------------------------------- injected FPR
_BEN = None


def benign_sets():
    global _BEN
    if _BEN is None:
        w, y, _ = test_arrays()
        ben = w[y == 0]
        ben = ben[np.random.default_rng(0).choice(len(ben), 8000, replace=False)]
        rng = np.random.default_rng(42)
        noop = np.stack([inject_noop_tokens(x, NOOP_TOKENS, 0.30, rng) for x in ben])
        pool, probs = benign_token_pool()
        rng = np.random.default_rng(43)
        ctrl = np.stack([inject_pool(x, pool, probs, 0.30, rng) for x in ben])
        _BEN = {"clean": pad_to(torch.tensor(ben, dtype=torch.long), MAX_LEN).numpy(),
                "noop_30": noop, "benigntok_30": ctrl,
                "_clean_unpadded": ben}          # B1 was trained/evaluated on 100-wide windows
    return _BEN


def fpr_sets(score_fn, unpadded_clean=False):
    s = benign_sets()
    sets = {"clean": s["_clean_unpadded"] if unpadded_clean else s["clean"],
            "noop_30": s["noop_30"], "benigntok_30": s["benigntok_30"]}
    return {k: float((score_fn(X) >= THRESHOLD).mean()) for k, X in sets.items()}


def torch_scorer(model):
    return lambda X: get_probs(model, torch.tensor(np.asarray(X), dtype=torch.long))


# --------------------------------------------------------------------------- alpha/beta
EXISTING = {(0.0, 0.0): "C_B0_vanilla", (1.0, 0.0): "C_A2_pairedKL",
            (0.0, 0.1): "C_A3_contrastive", (1.0, 0.1): "C_A4_full"}
ALPHAS, BETAS = [0.0, 0.5, 1.0, 2.0], [0.0, 0.1, 0.3]


def stage_alphabeta():
    data = load_all()
    full = json.load(open(RES / "corrected_full.json"))
    pt = json.load(open(OUT / "adaptive_pertrace.json"))
    mal_enc, _ = mal_test()
    out = {"seeds": SEEDS5, "alphas": ALPHAS, "betas": BETAS, "cells": {}}
    for a in ALPHAS:
        for b in BETAS:
            tag = EXISTING.get((a, b), f"C_AB_a{a}_b{b}")
            runs = []
            for seed in SEEDS5:
                if (a, b) in EXISTING:
                    r = full["runs"][tag][SEEDS.index(seed)]
                    clean, adv = r["clean"], r["adv"]
                else:
                    ck = Path("models") / f"{tag}_seed{seed}.pt"
                    if ck.exists():                      # resume: re-evaluate the checkpoint
                        m = load_model(tag, seed)
                        Xt = pad_to(torch.tensor(test_arrays()[0], dtype=torch.long), MAX_LEN)
                        p = get_probs(m, Xt); y = test_arrays()[1]
                        pr = p >= THRESHOLD
                        tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
                        clean = {"recall": tp / (y == 1).sum(), "fpr": fp / (y == 0).sum(),
                                 "precision": tp / max(tp + fp, 1), "auc": roc_auc_score(y, p)}
                        clean["f1"] = 2 * clean["precision"] * clean["recall"] / (clean["precision"] + clean["recall"])
                        adv = {s: {"recall": float((get_probs(m, data["adv_test"][s][0]) >= THRESHOLD).mean())}
                               for s in STRATEGIES}
                    else:
                        r = train_corrected(data, a, b, False, "none", seed, tag)
                        clean, adv = r["clean"], r["adv"]
                m = load_model(tag, seed)
                idx = np.array(pt["samples"][str(seed)]["window_idx"])
                o = attack_outcomes(m, idx)                # all 200 are detected by A1-A4 cells? check below
                det = get_probs(m, pad_to(torch.tensor(mal_enc[idx], dtype=torch.long), MAX_LEN)) >= THRESHOLD
                f = fpr_sets(torch_scorer(m))
                runs.append({"seed": seed, "clean": {k: float(clean[k]) for k in ("auc", "f1", "recall", "fpr")},
                             "adv_da_recall": float(adv["distribution_aware"]["recall"]),
                             "n_attacked": int(len(idx)), "n_detected_clean": int(det.sum()),
                             "evasion_le_10": float(np.mean(o["le_10"])),
                             "evasion_le_30": float(np.mean(o["le_30"])),
                             "inj_fpr_increase_30": f["noop_30"] - f["clean"]})
                print(f"  a={a} b={b} seed={seed}: F1={runs[-1]['clean']['f1']:.4f} "
                      f"DA={runs[-1]['adv_da_recall']:.4f} ev10={runs[-1]['evasion_le_10']:.3f} "
                      f"dFPR={runs[-1]['inj_fpr_increase_30']:+.3f}")
            agg = {}
            for k in ("f1", "recall", "fpr", "auc"):
                v = [r["clean"][k] for r in runs]; agg[k] = [float(np.mean(v)), float(np.std(v))]
            for k in ("adv_da_recall", "evasion_le_10", "evasion_le_30", "inj_fpr_increase_30"):
                v = [r[k] for r in runs]; agg[k] = [float(np.mean(v)), float(np.std(v))]
            out["cells"][f"a{a}_b{b}"] = {"alpha": a, "beta": b, "tag": tag,
                                           "reused_existing": (a, b) in EXISTING,
                                           "runs": runs, "agg": agg}
    dump("alphabeta_sensitivity.json", out)


# --------------------------------------------------------------------------- B1 (RF / SVM)
def stage_b1():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    data = load_all()
    pt = json.load(open(OUT / "adaptive_pertrace.json"))
    ref = json.load(open(RES / "corrected_baselines.json"))["B1_ngram"]
    mal_enc, _ = mal_test()
    to_str = lambda W: [" ".join(map(str, w)) for w in np.asarray(W)]
    # Same inputs as baselines_ablation.train_ngram_baseline: UNPADDED 100-wide
    # train/test windows, 130-wide adversarial windows.
    tr_npz, te_npz = np.load(PREP / "train.npz"), np.load(PREP / "test.npz")
    Xtr = to_str(tr_npz["windows"]); ytr = tr_npz["labels"].astype(int)
    Xte = to_str(te_npz["windows"]); yte = te_npz["labels"].astype(int)
    out = {"note": "B1 = TF-IDF 3-7-gram; RF and SVM reported separately; no test-set "
                   "model selection. RF reproduces corrected_baselines.json.",
           "RF": [], "SVM": []}
    for seed in SEEDS:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(3, 7), max_features=5000, sublinear_tf=True)
        A = vec.fit_transform(Xtr); T = vec.transform(Xte)
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1).fit(A, ytr)
        svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed), cv=3).fit(A, ytr)
        for name, clf in (("RF", rf), ("SVM", svm)):
            # RF/SVM use their own .predict for the clean label (as in the original
            # baseline); probabilities are used for AUC / pAUC / score-based attack.
            p = clf.predict_proba(T)[:, 1]; pr = clf.predict(T)
            tp = int(((pr == 1) & (yte == 1)).sum()); fp = int(((pr == 1) & (yte == 0)).sum())
            rec = tp / (yte == 1).sum(); fpr = fp / (yte == 0).sum(); prec = tp / max(tp + fp, 1)
            r = {"seed": seed, "auc": roc_auc_score(yte, p), "pauc05": roc_auc_score(yte, p, max_fpr=0.05),
                 "recall": rec, "fpr": fpr, "precision": prec, "f1": 2 * prec * rec / (prec + rec)}
            for s in STRATEGIES:
                r[f"advrec_{s}"] = float((clf.predict(vec.transform(to_str(data["adv_test"][s][0].numpy()))) == 1).mean())
            score = lambda X, clf=clf: clf.predict_proba(vec.transform(to_str(X)))[:, 1]
            r["inj_fpr"] = fpr_sets(score, unpadded_clean=True)
            if seed in SEEDS5:                                   # score-based attack (CPU heavy)
                idx = np.array(pt["samples"][str(seed)]["window_idx"])
                det = score(pad_to(torch.tensor(mal_enc[idx], dtype=torch.long), MAX_LEN).numpy()) >= THRESHOLD
                wrapper = SklearnScorer(score)
                o = attack_outcomes(wrapper, idx[det])
                r["adaptive"] = {"n_detected_of_200": int(det.sum()),
                                 **{k: float(np.mean(v)) for k, v in o.items()}}
            out[name].append(r)
            print(f"  B1-{name} seed {seed}: F1={r['f1']:.4f} rec={r['recall']:.4f} FPR={r['fpr']:.4f} "
                  f"DA={r['advrec_distribution_aware']:.4f} adaptive={r.get('adaptive')}")
        if seed == SEEDS[0]:
            assert abs(out["RF"][0]["f1"] - ref["f1"][0]) < 0.01, "RF does not reproduce corrected_baselines"
    dump("b1_rf_svm.json", out)


class SklearnScorer(torch.nn.Module):
    """Lets attack_one query a sklearn model: forward returns 2-class logits whose
    softmax equals the model's predict_proba (score access only, no gradients)."""
    def __init__(self, score):
        super().__init__(); self.score = score

    def forward(self, x):
        p = np.clip(self.score(x.cpu().numpy()), 1e-6, 1 - 1e-6)
        return torch.tensor(np.log(np.stack([1 - p, p], 1)), dtype=torch.float32)


# --------------------------------------------------------------------------- pAUC
def stage_pauc():
    w, y, _ = test_arrays()
    X = pad_to(torch.tensor(w, dtype=torch.long), MAX_LEN)
    out = {}
    for tag, mode in PIPELINES:
        v = [roc_auc_score(y, get_probs(load_pipe(tag, mode, s), X), max_fpr=0.05) for s in SEEDS]
        out[tag] = {"pauc05": [float(np.mean(v)), float(np.std(v))], "per_seed": v}
        print(f"  {tag:18s} pAUC(FPR<=5%) = {np.mean(v):.4f} +- {np.std(v):.4f}")
    dump("pauc.json", out)


# --------------------------------------------------------------------------- bootstrap
def boot_ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def stage_boot():
    rng = np.random.default_rng(2026)
    w, y, tids = test_arrays()
    X = pad_to(torch.tensor(w, dtype=torch.long), MAX_LEN)
    Xa = pad_to(torch.tensor(np.load(ADV / "distribution_aware_test.npz")["windows"], dtype=torch.long), MAX_LEN)
    mal_tids = tids[y == 1]
    ben_tr, ben_inv = np.unique(tids[y == 0], return_inverse=True)
    mal_tr, mal_inv = np.unique(mal_tids, return_inverse=True)
    nb, nm = len(ben_tr), len(mal_tr)
    # Stratified trace resampling = per-trace COUNT weights (multinomial draws).
    Wb = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)])
    Wm = np.stack([np.bincount(rng.integers(0, nm, nm), minlength=nm) for _ in range(N_BOOT)])
    win_b = np.bincount(ben_inv, minlength=nb); win_m = np.bincount(mal_inv, minlength=nm)

    out = {"n_boot": N_BOOT, "unit": "trace (stratified by class)",
           "n_benign_traces": int(nb), "n_malicious_traces": int(nm), "seed42": {}}
    probs, adv_hit = {}, {}
    for tag, mode in PIPELINES:
        m = load_pipe(tag, mode, 42)
        probs[tag] = get_probs(m, X); adv_hit[tag] = get_probs(m, Xa) >= THRESHOLD

    def per_trace(tag):
        p = probs[tag] >= THRESHOLD
        fp_t = np.bincount(ben_inv, weights=p[y == 0], minlength=nb)
        tp_t = np.bincount(mal_inv, weights=p[y == 1], minlength=nm)
        da_t = np.bincount(mal_inv, weights=adv_hit[tag], minlength=nm)
        return fp_t, tp_t, da_t

    for tag in probs:
        fp_t, tp_t, da_t = per_trace(tag)
        TP, FP = Wm @ tp_t, Wb @ fp_t
        rec, fpr = TP / (Wm @ win_m), FP / (Wb @ win_b)
        prec = TP / np.maximum(TP + FP, 1); f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        da = (Wm @ da_t) / (Wm @ win_m)
        p_rec, p_fpr = tp_t.sum() / win_m.sum(), fp_t.sum() / win_b.sum()
        p_prec = tp_t.sum() / max(tp_t.sum() + fp_t.sum(), 1)
        out["seed42"][tag] = {
            "recall": [float(p_rec), *boot_ci(rec)], "fpr": [float(p_fpr), *boot_ci(fpr)],
            "precision": [float(p_prec), *boot_ci(prec)],
            "f1": [float(2 * p_prec * p_rec / (p_prec + p_rec)), *boot_ci(f1)],
            "adv_da_recall": [float(da_t.sum() / win_m.sum()), *boot_ci(da)]}
        print(f"  {tag:18s} recall {p_rec:.4f} CI{np.round(boot_ci(rec),4)}  FPR {p_fpr:.4f} CI{np.round(boot_ci(fpr),4)}")

    # paired trace bootstrap, seed 42: static DA recall (A - B0), same resampled traces
    out["seed42_paired_static_da"] = {}
    _, _, da_b0 = per_trace("C_B0_vanilla")
    for tag in ("C_A2_pairedKL", "C_A4_full"):
        _, _, da_a = per_trace(tag)
        d = (Wm @ (da_a - da_b0)) / (Wm @ win_m)
        out["seed42_paired_static_da"][tag] = [float((da_a - da_b0).sum() / win_m.sum()), *boot_ci(d)]

    # adaptive: trace-only (per seed) and two-level seed x trace bootstrap of A - B0
    pt = json.load(open(OUT / "adaptive_pertrace.json"))
    out["adaptive"] = {}
    for tag in ("C_A2_pairedKL", "C_A4_full", "C_AT_augment", "C_B0_sym", "C_A3_contrastive"):
        res = {}
        for cap in ("le_10", "le_20", "le_30"):
            A = np.array([pt["outcomes"][tag][str(s)][cap] for s in SEEDS], float)        # (10, 200)
            B = np.array([pt["outcomes"]["C_B0_vanilla"][str(s)][cap] for s in SEEDS], float)
            D = A - B
            point = float(D.mean())
            tr_only = [D[0, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(N_BOOT)]
            two = []
            for _ in range(N_BOOT):
                ss = rng.integers(0, D.shape[0], D.shape[0])
                two.append(np.mean([D[s, rng.integers(0, D.shape[1], D.shape[1])].mean() for s in ss]))
            seed_only = paired(A.mean(1), B.mean(1))["ci95"]
            res[cap] = {"diff": point, "ci_seed_only_t": seed_only,
                        "ci_trace_only_seed42": [float(D[0].mean()), *boot_ci(tr_only)],
                        "ci_seed_and_trace": boot_ci(two)}
        out["adaptive"][tag] = res
        r = res["le_10"]
        print(f"  adaptive {tag:18s} <=10%: diff {r['diff']*100:+.1f}pp seed-only {np.round(np.array(r['ci_seed_only_t'])*100,1)} "
              f"seed+trace {np.round(np.array(r['ci_seed_and_trace'])*100,1)}")
    dump("bootstrap.json", out)


# --------------------------------------------------------------------------- prevalence
def stage_prev():
    boot = json.load(open(OUT / "bootstrap.json"))["seed42"]["C_A4_full"]
    prevent = json.load(open(RES / "prevention_report.json"))
    perproc = json.load(open(RES / "per_process_report.json"))
    op67 = prevent["optimal_thresholds"]["5"]
    points = {
        "window_tau0.50": (boot["recall"][0], boot["fpr"][0]),
        "window_tau0.67_Cfp5": (op67["abr"], op67["fpr"]),
        "trace_mean_tau0.50": (perproc["per_process_mean"]["abr"], perproc["per_process_mean"]["fpr"]),
    }
    WPS = 470.0   # windows/s observed by the deployed daemon
    out = {"formula": "PPV = pi*TPR / (pi*TPR + (1-pi)*FPR)", "windows_per_second_host": WPS, "rows": []}
    for name, (tpr, fpr) in points.items():
        for pi in (0.0695, 0.01, 0.001):
            ppv = pi * tpr / (pi * tpr + (1 - pi) * fpr)
            row = {"operating_point": name, "prevalence": pi, "tpr": tpr, "fpr": fpr, "ppv": ppv,
                   "false_alarms_per_1000_benign": 1000 * fpr,
                   "false_per_true_alarm": (1 - ppv) / ppv if ppv else float("inf")}
            if name.startswith("window"):
                row["false_alarms_per_hour_at_host_rate"] = WPS * 3600 * (1 - pi) * fpr
            out["rows"].append(row)
            print(f"  {name:22s} pi={pi:<7} TPR={tpr:.3f} FPR={fpr:.3f} PPV={ppv:.4f} "
                  f"false/true={row['false_per_true_alarm']:.1f}")
    dump("prevalence.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["pertrace", "pauc", "boot", "prev", "alphabeta", "b1"])
    for st in ap.parse_args().stages:
        print(f"\n=== stage {st} ===")
        globals()[f"stage_{st}"]()
