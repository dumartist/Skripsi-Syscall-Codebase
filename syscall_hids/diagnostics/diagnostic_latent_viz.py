"""
Diagnostic #2b: Latent-Space Visualization (t-SNE) + analisis kuantitatif

Diagnostik ruang laten: visualisasi representasi model.

Mengekstrak representasi laten 128-dim (output fc_rep) untuk empat populasi:
  benign_clean, benign_artifact (no-op, tanpa malware), malicious_clean,
  malicious_mimicry. Lalu:

  (1) Analisis KUANTITATIF di ruang laten 128-dim asli (lebih andal dari jarak
      t-SNE 2D): jarak centroid. Pertanyaan kunci shortcut:
      apakah benign+artifact BERGESER mendekati cluster malicious?
      Jika d(benign_artifact, malicious) < d(benign_clean, malicious),
      maka injeksi no-op menggeser benign menuju manifold malicious — bukti
      visual/geometris dari shortcut yang melengkapi E4 (perilaku) & IG (atribusi).

  (2) t-SNE 2D -> simpan koordinat+label ke JSON + render PNG (jika matplotlib ada).

Dijalankan untuk B3 (vanilla) dan A4 (full) agar bisa dibandingkan.

Output:
  results/diagnostic_latent_viz.json
  results/figures/latent_tsne_<model>.png  (jika matplotlib tersedia)
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model, pad_to, MAX_LEN, DEVICE
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import (
    load_noop_tokens, inject_noop_tokens, INJECTION_RATIO, INJECTION_SEED,
)

PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
RESULTS_DIR = Path("results")
FIG_DIR = Path("results/figures")

MODELS = ["B3_vanilla_cnn", "A4_full"]
SEED = 42
N_PER_POP = 600     # subsample per populasi untuk t-SNE
POP_COLORS = {
    "benign_clean": "tab:green",
    "benign_artifact": "tab:orange",
    "malicious_clean": "tab:red",
    "malicious_mimicry": "tab:purple",
}


@torch.no_grad()
def get_latent(model, X, batch=512):
    """Ekstrak laten 128-dim (output fc_rep)."""
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(DEVICE)
        e = model.embedding(xb).permute(0, 2, 1)
        conv_outs = [blk(e).squeeze(-1) for blk in model.conv_blocks]
        h = torch.cat(conv_outs, dim=1)
        h = model.fc_rep(h)
        out.append(h.cpu().numpy())
    return np.concatenate(out)


def build_populations(rng_seed=INJECTION_SEED):
    noop_tokens, _ = load_noop_tokens()
    rng = np.random.default_rng(rng_seed)
    test = np.load(PREPARED_DIR / "test.npz")
    enc, y = test["windows"], test["labels"].astype(int)
    benign_enc, mal_enc = enc[y == 0], enc[y == 1]

    sub = np.random.RandomState(SEED)
    b_idx = sub.choice(len(benign_enc), min(N_PER_POP, len(benign_enc)), replace=False)
    m_idx = sub.choice(len(mal_enc), min(N_PER_POP, len(mal_enc)), replace=False)

    benign_sample = benign_enc[b_idx]
    artifact = np.stack([
        inject_noop_tokens(w, noop_tokens, INJECTION_RATIO, rng) for w in benign_sample
    ])
    d = np.load(ADVERSARIAL_DIR / "distribution_aware_test.npz")
    mim = d["windows"]
    mim_idx = sub.choice(len(mim), min(N_PER_POP, len(mim)), replace=False)

    return {
        "benign_clean":      pad_to(torch.tensor(benign_sample, dtype=torch.long), MAX_LEN),
        "benign_artifact":   torch.tensor(artifact, dtype=torch.long),
        "malicious_clean":   pad_to(torch.tensor(mal_enc[m_idx], dtype=torch.long), MAX_LEN),
        "malicious_mimicry": pad_to(torch.tensor(mim[mim_idx], dtype=torch.long), MAX_LEN),
    }


def centroid(latents):
    return latents.mean(axis=0)


def analyze_model(model_name, pops):
    model = load_model(model_name, SEED)
    lat = {name: get_latent(model, X) for name, X in pops.items()}
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Jarak centroid di ruang 128-dim
    c = {k: centroid(v) for k, v in lat.items()}
    def dist(a, b): return float(np.linalg.norm(c[a] - c[b]))

    d_bc_mc = dist("benign_clean", "malicious_clean")
    d_ba_mc = dist("benign_artifact", "malicious_clean")
    d_bc_ba = dist("benign_clean", "benign_artifact")
    shift_ratio = d_ba_mc / d_bc_mc if d_bc_mc else float("inf")

    quant = {
        "dist_benignclean_to_malclean": d_bc_mc,
        "dist_benignartifact_to_malclean": d_ba_mc,
        "dist_benignclean_to_benignartifact": d_bc_ba,
        "shift_ratio_artifact_vs_clean": shift_ratio,
        "interpretation": (
            "benign+artifact LEBIH DEKAT ke malicious (shortcut geometris terkonfirmasi)"
            if shift_ratio < 0.9 else
            "benign+artifact TIDAK lebih dekat ke malicious"
            if shift_ratio > 1.1 else
            "pergeseran kecil/ambigu"
        ),
    }

    # t-SNE 2D gabungan
    names, mats = [], []
    for name, m in lat.items():
        names += [name] * len(m)
        mats.append(m)
    Z = np.concatenate(mats)
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=SEED, max_iter=1000)
    coords = tsne.fit_transform(Z)

    return quant, names, coords, lat


def render_png(model_name, names, coords):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = np.array(names)
    plt.figure(figsize=(8, 6))
    for pop, color in POP_COLORS.items():
        mask = names == pop
        plt.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                    label=pop, color=color)
    plt.legend(markerscale=2, fontsize=9)
    shown = {"B3_vanilla_cnn": "B0 Vanilla CNN", "A4_full": "A4 Full Model"}.get(model_name, model_name)
    plt.title(f"Latent t-SNE — {shown} (seed 42, ilustrasi kualitatif)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
    path = FIG_DIR / f"latent_tsne_{model_name}.png"
    plt.savefig(path, dpi=130); plt.close()
    return str(path)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 72)
    print("Diagnostic #2b: Latent t-SNE + analisis geometris shortcut")
    print(f"Device: {DEVICE} | {N_PER_POP}/populasi")
    print("=" * 72)

    pops = build_populations()
    report = {"config": {"models": MODELS, "seed": SEED, "n_per_pop": N_PER_POP},
              "per_model": {}}

    for model_name in MODELS:
        print(f"\n[{model_name}] extracting latents + t-SNE...")
        quant, names, coords, _ = analyze_model(model_name, pops)
        png = render_png(model_name, names, coords)
        report["per_model"][model_name] = {
            "quantitative": quant,
            "tsne_png": png,
            "tsne_coords": {
                "x": coords[:, 0].tolist(),
                "y": coords[:, 1].tolist(),
                "labels": names,
            },
        }
        print(f"  d(benign_clean -> mal)     = {quant['dist_benignclean_to_malclean']:.3f}")
        print(f"  d(benign_artifact -> mal)  = {quant['dist_benignartifact_to_malclean']:.3f}")
        print(f"  shift_ratio                = {quant['shift_ratio_artifact_vs_clean']:.3f}")
        print(f"  -> {quant['interpretation']}")
        if png:
            print(f"  PNG: {png}")

    # Diskusi
    disc = []
    for m in MODELS:
        q = report["per_model"][m]["quantitative"]
        disc.append(f"{m}: shift_ratio={q['shift_ratio_artifact_vs_clean']:.3f} "
                    f"({q['interpretation']}).")
    report["summary"] = {"discussion_points": disc}

    # Buang coords besar dari print, simpan semua ke JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diagnostic_latent_viz.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    for i, d in enumerate(disc, 1):
        print(f"[{i}] {d}")


if __name__ == "__main__":
    main()
