"""
Diagnostic E4: Artifact-Only Injection + Honest Eval-Set Composition

Memeriksa dua kelemahan evaluasi:

  #0  KOMPOSISI EVAL SET.
      Set evaluasi adversarial saat ini (results/diagnostic_a1_shortcut.json)
      HANYA berisi sampel malicious (tn=0, fp=0) -> precision adversarial = 1.0
      secara trivial. Di sini kita bangun set evaluasi CAMPURAN
      (benign clean + benign+artifact + malicious mimicry) dan melaporkan
      FPR & precision yang jujur.

  #1  E4 ARTIFACT-ONLY (LEVEL 6).
      Suntikkan no-op ke trace BENIGN (TANPA malware sama sekali) dengan rasio
      dan no-op set yang sama seperti generator mimikri. Jika model menandai
      trace benign ini sebagai *malicious*, maka "robustness" terhadap mimikri
      sebagian besar adalah sensitivitas terhadap artefak injeksi no-op
      (shortcut), bukan invariansi terhadap perilaku berbahaya.

Logika bukti shortcut (E4):
  - Ambil benign yang DIPREDIKSI benign oleh model (true negative pada clean).
  - Suntik no-op -> jika prediksi BERUBAH jadi malicious = "artifact flip".
  - artifact_flip_rate tinggi (terutama pada B3 vanilla) => SHORTCUT terkonfirmasi.

Catatan kesetaraan ruang token:
  Generator menyuntik no-op pada syscall MENTAH lalu encode via
  vocab.get(str(s), 1). Karena encoding adalah pemetaan per-elemen, menyuntik
  no-op mentah lalu encode == menyuntik TOKEN no-op langsung. Kita pakai jalur
  kedua (token) agar lepas dari ambiguitas syscall-mentah-0 vs PAD.

Output:
  results/diagnostic_e4_artifact_only.json
"""

import json
from pathlib import Path

import numpy as np
import torch

# Reuse infrastruktur diagnostic A1 (model, loader, metrik) — hindari duplikasi.
from syscall_hids.diagnostics.diagnostic_shortcut_learning import (
    MultiScaleEncoder,  # noqa: F401  (dipakai tak langsung via load_model)
    load_model,
    get_probs,
    pad_to,
    confusion_metrics,
    MAX_LEN,
    THRESHOLD,
    SEEDS,
    DEVICE,
)

PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
RESULTS_DIR = Path("results")

# Model yang diuji. B3 (vanilla, alpha=beta=0) adalah kunci: jika B3 saja sudah
# menandai artifact-benign sebagai malicious, shortcut berasal dari arsitektur.
MODELS = ["B3_vanilla_cnn", "A2_cls_robust", "A3_cls_contrastive", "A4_full"]

# No-op identik dengan generator (adversarial_mimicry.NOOP_SYSCALLS, raw syscall).
# getuid32=199/geteuid32=201 (i386) menggantikan 24/49 yang tak ada di vocab
# (dulu ter-encode jadi <UNK>). Lihat catatan di adversarial_mimicry.py.
NOOP_SYSCALLS_RAW = [20, 199, 201, 64, 13, 78]
INJECTION_RATIO = 0.30  # sama dengan max_injection_ratio generator
INJECTION_SEED = 42

# Ambang verdict shortcut artifact-only.
ARTIFACT_SHORTCUT_FLIP_THRESH = 0.30   # >30% benign flip => shortcut kuat
ARTIFACT_MINOR_FLIP_THRESH = 0.05      # <5% => model relatif imun artefak


def load_noop_tokens() -> list:
    with open(PREPARED_DIR / "vocabulary.json") as f:
        vocab = json.load(f)
    toks = [int(vocab.get(str(s), 1)) for s in NOOP_SYSCALLS_RAW]
    return toks, vocab


def inject_noop_tokens(window: np.ndarray, noop_tokens: list,
                       ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Suntik token no-op ke satu window terenkode (PAD=0 di ekor).

    Mengikuti perilaku generator: tentukan panjang nyata (token != PAD),
    sisipkan floor(ratio * panjang) no-op pada posisi acak dalam region nyata,
    lalu truncate/pad ke MAX_LEN.
    """
    real_len = int((window != 0).sum())
    if real_len == 0:
        return window[:MAX_LEN].copy()
    seq = window[:real_len].tolist()
    n_inject = int(np.floor(ratio * real_len))
    for _ in range(n_inject):
        pos = int(rng.integers(0, len(seq) + 1))
        noop = int(noop_tokens[rng.integers(0, len(noop_tokens))])
        seq.insert(pos, noop)
    seq = seq[:MAX_LEN]
    if len(seq) < MAX_LEN:
        seq = seq + [0] * (MAX_LEN - len(seq))
    return np.array(seq, dtype=np.int64)


def build_populations():
    """Bangun 4 populasi evaluasi (semua terenkode, panjang MAX_LEN)."""
    noop_tokens, _ = load_noop_tokens()
    rng = np.random.default_rng(INJECTION_SEED)

    test = np.load(PREPARED_DIR / "test.npz")
    enc = test["windows"]            # (N, 100) terenkode
    y = test["labels"].astype(int)

    benign_enc = enc[y == 0]
    malicious_enc = enc[y == 1]

    # P1: benign clean
    X_benign_clean = pad_to(torch.tensor(benign_enc, dtype=torch.long), MAX_LEN)

    # P2: benign + artifact (no-op injeksi, TANPA malware)  <-- E4
    artifact = np.stack([
        inject_noop_tokens(w, noop_tokens, INJECTION_RATIO, rng)
        for w in benign_enc
    ])
    X_benign_artifact = torch.tensor(artifact, dtype=torch.long)

    # P3: malicious clean
    X_mal_clean = pad_to(torch.tensor(malicious_enc, dtype=torch.long), MAX_LEN)

    # P4: malicious mimicry (distribution_aware, sudah terenkode oleh generator)
    d = np.load(ADVERSARIAL_DIR / "distribution_aware_test.npz")
    X_mal_mimicry = pad_to(torch.tensor(d["windows"], dtype=torch.long), MAX_LEN)

    return {
        "benign_clean": X_benign_clean,
        "benign_artifact": X_benign_artifact,
        "malicious_clean": X_mal_clean,
        "malicious_mimicry": X_mal_mimicry,
        "noop_tokens": noop_tokens,
        "n_benign": int(benign_enc.shape[0]),
        "n_malicious": int(malicious_enc.shape[0]),
    }


def verdict_from_flip(flip_rate: float) -> str:
    if flip_rate >= ARTIFACT_SHORTCUT_FLIP_THRESH:
        return "ARTIFACT_SHORTCUT_CONFIRMED"
    if flip_rate < ARTIFACT_MINOR_FLIP_THRESH:
        return "ARTIFACT_ROBUST"
    return "ARTIFACT_PARTIAL"


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 72)
    print("Diagnostic E4: Artifact-Only Injection + Honest Eval Composition")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    print("\n[1/3] Building evaluation populations...")
    pop = build_populations()
    print(f"  benign={pop['n_benign']}  malicious={pop['n_malicious']}")
    print(f"  no-op token ids (from vocab): {pop['noop_tokens']}  "
          f"(raw {NOOP_SYSCALLS_RAW})")
    print(f"  injection ratio: {INJECTION_RATIO:.0%}")

    Xbc = pop["benign_clean"]
    Xba = pop["benign_artifact"]
    Xmc = pop["malicious_clean"]
    Xmm = pop["malicious_mimicry"]

    print("\n[2/3] Per-model evaluation...")
    per_run = {}
    for model_name in MODELS:
        for seed in SEEDS:
            key = f"{model_name}_seed{seed}"
            try:
                model = load_model(model_name, seed)
            except FileNotFoundError:
                print(f"  {key}: SKIP (checkpoint missing)")
                continue

            p_bc = get_probs(model, Xbc)
            p_ba = get_probs(model, Xba)
            p_mc = get_probs(model, Xmc)
            p_mm = get_probs(model, Xmm)

            pred_bc = (p_bc >= THRESHOLD).astype(int)
            pred_ba = (p_ba >= THRESHOLD).astype(int)

            # FPR pada benign clean vs benign artifact
            fpr_clean = float(pred_bc.mean())
            fpr_artifact = float(pred_ba.mean())

            # E4 artifact flip: benign yang BENAR diprediksi benign pada clean,
            # lalu berubah jadi malicious setelah injeksi no-op.
            tn_mask = pred_bc == 0
            n_tn = int(tn_mask.sum())
            n_flip = int(((pred_ba == 1) & tn_mask).sum())
            flip_rate = (n_flip / n_tn) if n_tn else 0.0

            # #0 Honest mixed-set: benign_clean + benign_artifact (label 0)
            #     + malicious_mimicry (label 1).
            mixed_probs = np.concatenate([p_bc, p_ba, p_mm])
            mixed_labels = np.concatenate([
                np.zeros(len(p_bc), dtype=int),
                np.zeros(len(p_ba), dtype=int),
                np.ones(len(p_mm), dtype=int),
            ])
            honest = confusion_metrics(mixed_probs, mixed_labels)

            per_run[key] = {
                "model": model_name,
                "seed": seed,
                "fpr_benign_clean": fpr_clean,
                "fpr_benign_artifact": fpr_artifact,
                "fpr_delta_artifact": fpr_artifact - fpr_clean,
                "artifact_flip": {
                    "n_clean_true_negative": n_tn,
                    "n_flipped_to_malicious": n_flip,
                    "flip_rate": flip_rate,
                },
                "honest_mixed_set": honest,
                "recall_malicious_mimicry": float((p_mm >= THRESHOLD).mean()),
                "recall_malicious_clean": float((p_mc >= THRESHOLD).mean()),
            }
            print(f"  {key}: clean-FPR={fpr_clean:.3f}  "
                  f"artifact-FPR={fpr_artifact:.3f}  "
                  f"flip={flip_rate:.3f}  "
                  f"honest-precision={honest['precision']:.3f}")

            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print("\n[3/3] Aggregating across seeds + verdict...")
    agg = {}
    for model_name in MODELS:
        runs = [per_run[f"{model_name}_seed{s}"]
                for s in SEEDS if f"{model_name}_seed{s}" in per_run]
        if not runs:
            continue
        flip = np.array([r["artifact_flip"]["flip_rate"] for r in runs])
        fpr_d = np.array([r["fpr_delta_artifact"] for r in runs])
        hp = np.array([r["honest_mixed_set"]["precision"] for r in runs])
        hf1 = np.array([r["honest_mixed_set"]["f1"] for r in runs])
        agg[model_name] = {
            "flip_rate_mean": float(flip.mean()),
            "flip_rate_std": float(flip.std(ddof=1)) if len(flip) > 1 else 0.0,
            "fpr_delta_mean": float(fpr_d.mean()),
            "honest_precision_mean": float(hp.mean()),
            "honest_f1_mean": float(hf1.mean()),
            "verdict": verdict_from_flip(float(flip.mean())),
        }

    # Verdict global: fokus pada B3 (arsitektur murni) dan A4 (full model).
    b3 = agg.get("B3_vanilla_cnn", {})
    a4 = agg.get("A4_full", {})
    discussion = []
    if b3:
        discussion.append(
            f"B3 (vanilla CNN, tanpa adversarial training) menandai "
            f"{b3['flip_rate_mean']:.1%} benign-artifact sebagai malicious. "
            "Karena tidak ada malware pada trace ini, deteksi murni bereaksi "
            "terhadap artefak injeksi no-op."
        )
    if b3 and a4:
        ratio = (b3["flip_rate_mean"] / a4["flip_rate_mean"]
                 if a4["flip_rate_mean"] else float("inf"))
        discussion.append(
            f"NUANSA PENTING: flip-rate A4 (full) = {a4['flip_rate_mean']:.1%} "
            f"vs B3 (vanilla) = {b3['flip_rate_mean']:.1%} — A4 ~{ratio:.1f}x "
            "LEBIH rendah. Berbeda dari sumbu recall (diagnostic A1: A4 ~= B3, "
            "SHORTCUT_DOMINANT), pada sumbu FALSE-POSITIVE terhadap artefak no-op "
            "adversarial training (L_robust + L_contrastive) MEMBERI manfaat nyata: "
            "mengurangi kecenderungan menandai benign+no-op sebagai malicious. "
            "Diagnostic A1 hanya mengukur recall sehingga melewatkan kontribusi ini; "
            "E4 + komposisi eval jujur menyingkapnya. Inilah klaim kontribusi yang "
            "dapat dipertahankan: manfaat utama bukan AdvRec lebih tinggi, melainkan "
            "robustness FPR terhadap artefak injeksi."
        )
    discussion.append(
        "KOMPOSISI EVAL (#0): precision pada set campuran (benign clean + "
        "benign artifact + malicious mimicry) TIDAK lagi 1.0 trivial seperti "
        "pada diagnostic_a1 (yang tn=fp=0). Gunakan precision/F1 'honest' ini "
        "untuk semua klaim robustness."
    )

    overall = a4.get("verdict") or b3.get("verdict") or "INCONCLUSIVE"

    report = {
        "config": {
            "models": MODELS,
            "seeds": SEEDS,
            "noop_syscalls_raw": NOOP_SYSCALLS_RAW,
            "noop_tokens": pop["noop_tokens"],
            "injection_ratio": INJECTION_RATIO,
            "injection_seed": INJECTION_SEED,
            "threshold": THRESHOLD,
            "max_len": MAX_LEN,
            "flip_thresholds": {
                "shortcut_confirmed": ARTIFACT_SHORTCUT_FLIP_THRESH,
                "robust": ARTIFACT_MINOR_FLIP_THRESH,
            },
        },
        "populations": {
            "n_benign": pop["n_benign"],
            "n_malicious": pop["n_malicious"],
        },
        "per_run": per_run,
        "aggregate": agg,
        "summary": {
            "overall_verdict_a4": overall,
            "discussion_points": discussion,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diagnostic_e4_artifact_only.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  Saved: {out}")

    print("\n" + "=" * 72)
    print("AGGREGATE (mean across seeds)")
    print("=" * 72)
    print(f"  {'Model':<22}{'flip%':>9}{'dFPR':>9}{'honestP':>10}{'honestF1':>10}  Verdict")
    print("  " + "-" * 78)
    for m, a in agg.items():
        print(f"  {m:<22}{a['flip_rate_mean']*100:>8.1f}%"
              f"{a['fpr_delta_mean']:>+9.3f}{a['honest_precision_mean']:>10.3f}"
              f"{a['honest_f1_mean']:>10.3f}  {a['verdict']}")
    print()
    for i, d in enumerate(discussion, 1):
        print(f"[{i}] {d}\n")


if __name__ == "__main__":
    main()
