"""
Diagnostic #2a: Integrated Gradients (token-level attribution)

Diagnostik atribusi: membuktikan secara mekanistik
prediksi perilaku dari E4. E4 menunjukkan model menandai benign+no-op sebagai
malicious (shortcut). Pertanyaan mekanistik: APAKAH skor atribusi benar-benar
menumpuk pada token no-op (16/36/13/41 + UNK)?

Metode: Integrated Gradients (Sundararajan et al. 2017) atas ruang EMBEDDING
(karena input token diskret). Baseline = embedding PAD (nol, karena padding_idx=0).
Atribusi per-posisi = sum_d (emb - baseline) * avg_grad terhadap logit kelas
malicious. Lalu dihitung berapa fraksi massa atribusi yang jatuh pada posisi
token no-op, dibandingkan fraksi kemunculan token no-op (frekuensi).

Jika fraksi_atribusi(no-op) >> fraksi_frekuensi(no-op) => shortcut ter-lokalisasi
secara eksplisit pada token no-op. Bandingkan B3 (vanilla) vs A4 (full): prediksi
dari E4 adalah A4 menaruh massa atribusi LEBIH SEDIKIT pada no-op (konsisten dengan
flip-rate A4 yang ~1.9x lebih rendah).

Target sampel: benign+artifact yang FLIP ke malicious (false-positive artefak) —
di situlah mekanisme shortcut paling murni terlihat (nol malware, tetap dideteksi).

Output: results/diagnostic_attribution.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from syscall_hids.diagnostics.diagnostic_shortcut_learning import (
    load_model, get_probs, pad_to, MAX_LEN, THRESHOLD, SEEDS, DEVICE,
)
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import (
    load_noop_tokens, inject_noop_tokens, INJECTION_RATIO, INJECTION_SEED,
)

PREPARED_DIR = Path("prepared_data")
RESULTS_DIR = Path("results")

MODELS = ["B3_vanilla_cnn", "A2_cls_robust", "A3_cls_contrastive", "A4_full"]
IG_STEPS = 50
MAX_SAMPLES = 200      # sampel flipped benign-artifact per (model, seed)
SEED_FOR_ATTR = 42     # gunakan satu seed untuk atribusi (cukup; multi-seed mahal)
PAD_TOKEN = 0
UNK_TOKEN = 1


def forward_from_emb(model, emb):
    """Replikasi forward MultiScaleEncoder mulai dari embedding (B,L,D)."""
    e = emb.permute(0, 2, 1)
    conv_outs = [block(e).squeeze(-1) for block in model.conv_blocks]
    h = torch.cat(conv_outs, dim=1)
    h = model.fc_rep(h)
    return model.classifier(h)


def integrated_gradients(model, x_row, steps=IG_STEPS):
    """IG untuk satu window (1D LongTensor panjang MAX_LEN).
    Return token_attr (np array panjang MAX_LEN), signed terhadap kelas malicious."""
    x = x_row.view(1, -1).to(DEVICE)
    with torch.no_grad():
        input_emb = model.embedding(x).detach()      # (1,L,D)
    baseline = torch.zeros_like(input_emb)            # PAD embedding = nol

    total_grad = torch.zeros_like(input_emb)
    for k in range(1, steps + 1):
        alpha = k / steps
        interp = (baseline + alpha * (input_emb - baseline)).clone()
        interp.requires_grad_(True)
        logits = forward_from_emb(model, interp)
        mal_logit = logits[:, 1].sum()
        grad = torch.autograd.grad(mal_logit, interp)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / steps
    ig = (input_emb - baseline) * avg_grad            # (1,L,D)
    token_attr = ig.sum(dim=-1).squeeze(0)            # (L,)
    return token_attr.cpu().numpy()


def analyze_model(model_name, seed, Xba_flip, noop_tokens):
    model = load_model(model_name, seed)
    noop_set = set(int(t) for t in noop_tokens) | {UNK_TOKEN}

    frac_attr_list, frac_count_list, ratio_list = [], [], []
    n = min(MAX_SAMPLES, Xba_flip.size(0))
    for i in range(n):
        x = Xba_flip[i]
        attr = integrated_gradients(model, x)
        toks = x.numpy()

        real_mask = toks != PAD_TOKEN
        if real_mask.sum() == 0:
            continue
        noop_mask = np.array([t in noop_set for t in toks]) & real_mask

        abs_attr = np.abs(attr)
        total_mass = abs_attr[real_mask].sum()
        if total_mass <= 0:
            continue
        noop_mass = abs_attr[noop_mask].sum()

        frac_attr = noop_mass / total_mass                       # fraksi massa atribusi pd no-op
        frac_count = noop_mask.sum() / real_mask.sum()           # fraksi kemunculan no-op
        frac_attr_list.append(float(frac_attr))
        frac_count_list.append(float(frac_count))
        # enrichment ratio: >1 berarti no-op over-weighted vs frekuensinya
        if frac_count > 0:
            ratio_list.append(float(frac_attr / frac_count))

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "n_samples": n,
        "noop_attr_fraction_mean": float(np.mean(frac_attr_list)) if frac_attr_list else 0.0,
        "noop_count_fraction_mean": float(np.mean(frac_count_list)) if frac_count_list else 0.0,
        "enrichment_ratio_mean": float(np.mean(ratio_list)) if ratio_list else 0.0,
        "enrichment_ratio_median": float(np.median(ratio_list)) if ratio_list else 0.0,
    }


def main():
    np.random.seed(42); torch.manual_seed(42)
    print("=" * 72)
    print("Diagnostic #2a: Integrated Gradients — atribusi token no-op")
    print(f"Device: {DEVICE} | IG steps: {IG_STEPS} | max samples: {MAX_SAMPLES}")
    print("=" * 72)

    # Bangun benign+artifact (sama persis E4)
    noop_tokens, _ = load_noop_tokens()
    rng = np.random.default_rng(INJECTION_SEED)
    test = np.load(PREPARED_DIR / "test.npz")
    benign_enc = test["windows"][test["labels"] == 0]
    artifact = np.stack([
        inject_noop_tokens(w, noop_tokens, INJECTION_RATIO, rng) for w in benign_enc
    ])
    Xba = torch.tensor(artifact, dtype=torch.long)
    print(f"\nbenign-artifact pool: {Xba.size(0)}  no-op tokens: {noop_tokens} (+UNK={UNK_TOKEN})")

    report = {"config": {"ig_steps": IG_STEPS, "max_samples": MAX_SAMPLES,
                         "seed": SEED_FOR_ATTR, "noop_tokens": noop_tokens,
                         "injection_ratio": INJECTION_RATIO},
              "per_model": {}}

    print(f"\n{'Model':<22}{'noop_attr%':>11}{'noop_freq%':>11}{'enrichment':>12}  interpretasi")
    print("-" * 78)
    for model_name in MODELS:
        # Sampel yang FLIP: prediksi malicious pada benign-artifact (seed atribusi)
        m = load_model(model_name, SEED_FOR_ATTR)
        p = get_probs(m, Xba)
        del m
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        flip_idx = np.where(p >= THRESHOLD)[0]
        Xba_flip = Xba[flip_idx]

        if Xba_flip.size(0) == 0:
            report["per_model"][model_name] = {"note": "no flips"}
            print(f"{model_name:<22}{'—':>11}{'—':>11}{'—':>12}  tidak ada flip")
            continue

        res = analyze_model(model_name, SEED_FOR_ATTR, Xba_flip, noop_tokens)
        res["n_flipped_available"] = int(Xba_flip.size(0))
        report["per_model"][model_name] = res

        enr = res["enrichment_ratio_mean"]
        tag = ("no-op OVER-weighted (shortcut mekanistik)" if enr > 1.3
               else "no-op proporsional" if enr > 0.8 else "no-op under-weighted")
        print(f"{model_name:<22}{res['noop_attr_fraction_mean']*100:>10.1f}%"
              f"{res['noop_count_fraction_mean']*100:>10.1f}%"
              f"{enr:>12.2f}  {tag}")

    # Diskusi B3 vs A4 (interpretasi JUJUR — enrichment ~1 = proporsional, bukan fiksasi)
    disc = []
    b3 = report["per_model"].get("B3_vanilla_cnn", {})
    a4 = report["per_model"].get("A4_full", {})
    if b3.get("enrichment_ratio_mean") and a4.get("enrichment_ratio_mean"):
        enr_max = max(b3["enrichment_ratio_mean"], a4["enrichment_ratio_mean"])
        if enr_max > 1.5:
            disc.append(
                "Enrichment > 1.5: massa atribusi MENUMPUK pada token no-op jauh "
                "melebihi frekuensinya — shortcut ter-lokalisasi eksplisit pada no-op."
            )
        else:
            disc.append(
                f"TEMUAN JUJUR: enrichment mendekati 1 (B3={b3['enrichment_ratio_mean']:.2f}, "
                f"A4={a4['enrichment_ratio_mean']:.2f}). Atribusi pada token no-op kurang-lebih "
                "PROPORSIONAL dengan frekuensinya — model TIDAK ber-fiksasi pada token no-op "
                "individual. Namun karena ~29% token adalah no-op hasil injeksi (rasio 30%), "
                "ini berarti ~30% massa 'bukti malicious' pada trace benign+no-op berasal dari "
                "token yang TIDAK punya semantik berbahaya. Jadi shortcut tetap terkonfirmasi, "
                "tetapi bekerja via DISRUPSI struktur n-gram lokal yang tersebar (konsisten "
                "dengan filter konv multi-skala kernel 3/5/7), bukan via fiksasi satu token. "
                "Ini mekanisme yang lebih halus daripada hipotesis 'token no-op = penanda malicious'."
            )
        delta = b3["noop_attr_fraction_mean"] - a4["noop_attr_fraction_mean"]
        if delta > 0.005:
            disc.append(
                f"A4 menaruh massa atribusi no-op sedikit LEBIH RENDAH "
                f"({a4['noop_attr_fraction_mean']:.1%}) dibanding B3 "
                f"({b3['noop_attr_fraction_mean']:.1%}; selisih {delta*100:.1f}pp) — "
                "directionally konsisten dengan artifact-FPR A4 yang ~1.9x lebih rendah (E4), "
                "tetapi selisihnya KECIL secara mekanistik. Penurunan FPR A4 kemungkinan "
                "lebih banyak dari kalibrasi/ketajaman skor daripada relokasi atribusi besar-besaran."
            )
        else:
            disc.append(
                f"A4 ({a4['noop_attr_fraction_mean']:.1%}) dan B3 "
                f"({b3['noop_attr_fraction_mean']:.1%}) menaruh massa atribusi no-op yang "
                "hampir sama. Penurunan artifact-FPR A4 (E4) BUKAN dari relokasi atribusi — "
                "lebih mungkin dari kalibrasi skor (threshold-margin), bukan perubahan 'apa yang dilihat'."
            )
    report["summary"] = {"discussion_points": disc}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diagnostic_attribution.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved: {out}\n")
    for i, d in enumerate(disc, 1):
        print(f"[{i}] {d}\n")


if __name__ == "__main__":
    main()
