"""
Section 4.4.5: Analisis Latensi dan Feasibilitas Real-time (B2)

Memprofil latensi inference per komponen pipeline pada CPU dan GPU,
menghitung throughput (window/detik), dan membandingkannya dengan syscall
rate produksi Linux (10.000-100.000 syscall/detik) untuk menilai feasibilitas
deployment HIPS real-time.

Komponen: forward pass (embedding+conv+fc+classifier), risk computation
(softmax), policy decision (digitize). Tokenisasi diukur sebagai dict lookup.

Output: results/latency_report.json
Thesis : Tabel 4.20.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F

import syscall_hids.data.lidds_common as C

SEED = 42
N_ITER = 10_000          # iterasi pengukuran (window tunggal)
N_BATCH_ITER = 200       # iterasi batched
BATCH = 256
WARMUP = 50
MAX_LEN = C.MAX_LEN
TAUS = (0.30, 0.60, 0.85)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_forward(model, x, dev, n_iter, warmup=WARMUP):
    """Rata-rata latensi forward pass (detik) untuk tensor x."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        sync(dev)
        t1 = time.perf_counter()
    return (t1 - t0) / n_iter


def time_softmax(logits, n_iter):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        F.softmax(logits, dim=1)[:, 1]
    return (time.perf_counter() - t0) / n_iter


def time_policy(probs_np, n_iter):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        np.digitize(probs_np, TAUS)
    return (time.perf_counter() - t0) / n_iter


def time_tokenize(seq_list, vocab, n_iter):
    """Simulasi tokenisasi: dict lookup per token untuk satu window 100-token."""
    t0 = time.perf_counter()
    for _ in range(n_iter):
        [vocab.get(str(s), 1) for s in seq_list]
    return (time.perf_counter() - t0) / n_iter


def profile_device(model, dev, sample_window):
    """Return dict latensi (mikrodetik) untuk batch=1 dan batch=256."""
    model.to(dev)
    x1 = torch.tensor(sample_window[None, :], dtype=torch.long, device=dev)
    xB = x1.repeat(BATCH, 1)

    fwd1 = time_forward(model, x1, dev, N_ITER)
    fwdB = time_forward(model, xB, dev, N_BATCH_ITER)

    with torch.no_grad():
        logits1, _ = model(x1)
        logitsB, _ = model(xB)
    sm1 = time_softmax(logits1, N_ITER)
    smB = time_softmax(logitsB, N_BATCH_ITER)

    probs1 = np.array([0.5]); probsB = np.full(BATCH, 0.5)
    pol1 = time_policy(probs1, N_ITER)
    polB = time_policy(probsB, N_BATCH_ITER)

    us = 1e6
    return {
        "batch1": {
            "forward_us": fwd1 * us,
            "softmax_us": sm1 * us,
            "policy_us": pol1 * us,
            "total_us": (fwd1 + sm1 + pol1) * us,
            "throughput_wps": 1.0 / (fwd1 + sm1 + pol1),
        },
        "batch256_per_sample": {
            "forward_us": fwdB / BATCH * us,
            "softmax_us": smB / BATCH * us,
            "policy_us": polB / BATCH * us,
            "total_us": (fwdB + smB + polB) / BATCH * us,
            "throughput_wps": BATCH / (fwdB + smB + polB),
        },
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("Section 4.4.5: Latency & Real-time Feasibility (B2)")
    print("=" * 70)

    model = C.load_adfa_model("A4_full", SEED, "cpu")
    n_params = sum(p.numel() for p in model.parameters())

    # Sample window dari test
    te = np.load(C.PREPARED_ADFA / "test.npz")
    sample = C.pad_to(te["windows"][:1].astype(np.int64), MAX_LEN)[0]
    vocab = json.load(open(C.PREPARED_ADFA / "vocabulary.json"))
    raw_seq = te["windows"][0].tolist()

    report = {"model": "A4_full_seed42", "n_params": int(n_params),
              "config": {"n_iter": N_ITER, "batch": BATCH, "max_len": MAX_LEN},
              "devices": {}}

    # Tokenisasi (device-independent, CPU)
    tok_us = time_tokenize(raw_seq, vocab, N_ITER) * 1e6
    report["tokenization_us_per_window"] = tok_us

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        print(f"\n[{dev.type.upper()}] profiling...")
        prof = profile_device(model, dev, sample)
        report["devices"][dev.type] = prof
        b1 = prof["batch1"]; bB = prof["batch256_per_sample"]
        print(f"  batch=1   : forward={b1['forward_us']:.1f}us  "
              f"total={b1['total_us']:.1f}us  "
              f"throughput={b1['throughput_wps']:,.0f} win/s")
        print(f"  batch=256 : forward={bB['forward_us']:.2f}us/sample  "
              f"throughput={bB['throughput_wps']:,.0f} win/s")

    # Feasibility: konversi syscall rate -> window rate (stride=50 -> 1 window per 50 syscall)
    print(f"\n  Tokenisasi: {tok_us:.1f} us/window  | Params: {n_params:,}")
    print("\n  Feasibilitas (stride=50 -> 1 window / 50 syscall baru):")
    best_dev = "cuda" if "cuda" in report["devices"] else "cpu"
    tp_b1 = report["devices"][best_dev]["batch1"]["throughput_wps"]
    tp_bB = report["devices"][best_dev]["batch256_per_sample"]["throughput_wps"]
    for sysrate in [10_000, 100_000]:
        win_needed = sysrate / 50.0
        print(f"    {sysrate:>7,} syscall/s -> {win_needed:,.0f} window/s needed | "
              f"batch1 {'OK' if tp_b1>=win_needed else 'NO'} "
              f"({tp_b1:,.0f}) | batch256 {'OK' if tp_bB>=win_needed else 'NO'} "
              f"({tp_bB:,.0f})")
    report["feasibility"] = {
        "windows_per_syscall": 1.0 / 50,
        "best_device": best_dev,
        "throughput_batch1_wps": tp_b1,
        "throughput_batch256_wps": tp_bB,
    }

    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "latency_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: results/latency_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
