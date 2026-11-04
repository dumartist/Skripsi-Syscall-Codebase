#!/usr/bin/env python3
"""
#2b Fine-grained latency — breakdown per-komponen pipeline HIPS.

Mengukur latensi & throughput tiap tahap pipeline secara terisolasi:
  (1) tokenisasi (map syscall_nr -> token via vocab)
  (2) window build (sliding window per-PID, deque ops)
  (3) inferensi ONNX (forward)
  (4) policy decision (multi-tier digitize)

Tujuan: klaim host-level proof-of-concept yang defensible — menunjukkan tahap mana
yang dominan (prediksi: inferensi kecil ~ratusan µs; bottleneck nyata = overhead
per-event di daemon Python, lihat overhead_bench_v2.py).

Jalankan di VM (butuh onnxruntime + model ONNX + deploy vocab):
  python3 deploy/benchmarks/pipeline_latency_bench.py \
    --model models/lidds_Bruteforce_CWE-307.onnx \
    --vocab /tmp/lidds_vocab.json

Output: /tmp/pipeline_latency.json
"""
import argparse, json, time, collections
import numpy as np
import onnxruntime as ort

MAX_LEN = 130
WINDOW = 100
TAUS = (0.30, 0.60, 0.85)
N_ITER = 20000


def load_vocab(path):
    raw = json.load(open(path))
    nr2tok = {}
    for k, v in raw.items():
        try: nr2tok[int(k)] = int(v)
        except ValueError: pass
    return nr2tok, int(raw.get("<PAD>", 0)), int(raw.get("<UNK>", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--iters", type=int, default=N_ITER)
    ap.add_argument("--threads", type=int, default=1,
                    help="ORT intra/inter-op threads (1=single, 0=default multi)")
    args = ap.parse_args()

    nr2tok, PAD, UNK = load_vocab(args.vocab)
    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    print(f"[*] ORT intra_op_threads={args.threads or 'default'}")
    rng = np.random.default_rng(42)

    # Data sintetis: nomor syscall x86_64 acak dari domain vocab
    nrs = list(nr2tok.keys()) or [0, 1, 2, 3, 4, 5]
    raw_stream = rng.choice(nrs, size=args.iters + WINDOW).tolist()

    us = 1e6
    N = args.iters

    # (1) Tokenisasi: satu lookup per syscall
    t0 = time.perf_counter()
    for i in range(N):
        _ = nr2tok.get(raw_stream[i], UNK)
    t_tok = (time.perf_counter() - t0) / N * us

    # (2) Window build: deque maxlen + ekstraksi window
    buf = collections.deque(maxlen=WINDOW)
    for s in raw_stream[:WINDOW]:
        buf.append(nr2tok.get(s, UNK))
    t0 = time.perf_counter()
    for i in range(N):
        buf.append(nr2tok.get(raw_stream[i], UNK))
        _ = list(buf)
    t_win = (time.perf_counter() - t0) / N * us

    # (3) Inferensi ONNX (batch=1, di-pad ke MAX_LEN)
    win = list(buf) + [PAD] * (MAX_LEN - len(buf))
    x = np.array([win[:MAX_LEN]], dtype=np.int64)
    for _ in range(50):  # warmup
        sess.run(["risk"], {"tokens": x})
    t0 = time.perf_counter()
    for _ in range(N):
        out = sess.run(["risk"], {"tokens": x})
    t_inf = (time.perf_counter() - t0) / N * us
    risk = float(out[0][0])

    # (4) Policy decision (digitize multi-tier)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = int(np.digitize(risk, TAUS))
    t_pol = (time.perf_counter() - t0) / N * us

    total = t_tok + t_win + t_inf + t_pol
    result = {
        "iters": N,
        "per_step_us": {
            "tokenization": round(t_tok, 3),
            "window_build": round(t_win, 3),
            "onnx_inference": round(t_inf, 2),
            "policy_decision": round(t_pol, 3),
            "total_per_window": round(total, 2),
        },
        "throughput_windows_per_sec": round(1e6 / total, 0),
        "inference_share_pct": round(100 * t_inf / total, 1),
        "note": ("Latensi per-komponen TERISOLASI. Bandingkan total ini dengan "
                 "latensi end-to-end daemon (~3ms) di results/latency_report.json: "
                 "selisih = overhead per-event userspace Python (ring buffer poll, "
                 "callback, dispatch) yang menjadi bottleneck nyata."),
    }
    print(json.dumps(result, indent=2))
    with open("/tmp/pipeline_latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n[*] Saved /tmp/pipeline_latency.json")


if __name__ == "__main__":
    main()
