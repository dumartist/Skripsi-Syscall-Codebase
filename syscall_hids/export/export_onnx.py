"""
Deployment Fase 1: Export model A4 ke ONNX untuk inference real-time.

Membungkus MultiScaleEncoder agar mengeluarkan P(malicious) langsung
(softmax kelas-1), lalu mengekspor ke ONNX dengan batch dinamis.
Memverifikasi parity numerik antara PyTorch dan ONNX Runtime.

Output: models/a4_full.onnx  (dipakai oleh daemon HIPS di host Linux)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import syscall_hids.data.lidds_common as C

SEED = 42
MAX_LEN = C.MAX_LEN
OUT = C.MODEL_DIR / "a4_full.onnx"


class RiskScorer(nn.Module):
    """Wrapper: input token ids (B, L) -> P(malicious) (B,)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        logits, _ = self.encoder(x)
        return F.softmax(logits, dim=1)[:, 1]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 64)
    print("Deployment Fase 1: Export A4 -> ONNX")
    print("=" * 64)

    enc = C.load_adfa_model("A4_full", SEED, "cpu").eval()
    model = RiskScorer(enc).eval()

    # Dummy input: (batch=1, seq=130) int64
    dummy = torch.randint(0, C.ADFA_VOCAB_SIZE, (1, MAX_LEN), dtype=torch.long)

    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["tokens"], output_names=["risk"],
        dynamic_axes={"tokens": {0: "batch"}, "risk": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
        dynamo=False,  # legacy exporter mendukung AdaptiveMaxPool1d
    )
    print(f"\n  Exported: {OUT}")

    # --- Verifikasi parity PyTorch vs ONNX Runtime ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])

    # Uji pada beberapa window test nyata
    test = np.load(C.PREPARED_ADFA / "test.npz")
    X = C.pad_to(test["windows"][:256].astype(np.int64), MAX_LEN)

    with torch.no_grad():
        pt = model(torch.tensor(X)).numpy()
    onnx_out = sess.run(["risk"], {"tokens": X})[0]

    max_diff = float(np.abs(pt - onnx_out).max())
    print(f"\n  Parity check (256 window test):")
    print(f"    max |PyTorch - ONNX| = {max_diff:.2e}")
    print(f"    {'PASS' if max_diff < 1e-4 else 'FAIL'} (toleransi 1e-4)")

    # Latensi ONNX batch=1 (indikatif, di CPU Windows)
    import time
    x1 = X[:1]
    for _ in range(50):
        sess.run(["risk"], {"tokens": x1})
    t0 = time.perf_counter()
    for _ in range(2000):
        sess.run(["risk"], {"tokens": x1})
    us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"\n  ONNX latency batch=1 (CPU): {us:.1f} us/window")
    print("=" * 64)


if __name__ == "__main__":
    main()
