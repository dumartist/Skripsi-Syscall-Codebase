"""
Export model LID-DS native (x86_64) ke ONNX untuk deployment.
Memperbaiki domain shift: model ini dilatih pada syscall x86_64 native,
cocok dengan host VM (berbeda dari A4 ADFA yang i386).

Default: Bruteforce_CWE-307 (AUC 0.893, terbaik) — cocok untuk demo
serangan brute-force nyata (Hydra) di Langkah B.

Output: models/lidds_<scenario>.onnx
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import syscall_hids.data.lidds_common as C

SEED = 42
MAX_LEN = C.MAX_LEN
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "Bruteforce_CWE-307"


class RiskScorer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, x):
        logits, _ = self.encoder(x)
        return F.softmax(logits, dim=1)[:, 1]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 64)
    print(f"Export LID-DS native model -> ONNX | scenario={SCENARIO}")
    print("=" * 64)

    ckpt = torch.load(C.MODEL_DIR / f"lidds_scratch_{SCENARIO}.pt",
                      weights_only=False, map_location="cpu")
    vocab_size = ckpt["vocab_size"]
    print(f"  vocab_size={vocab_size}  best_val_f1={ckpt['best_val_f1']:.4f}")

    enc = C.MultiScaleEncoder(vocab_size=vocab_size).eval()
    enc.load_state_dict(ckpt["model_state"], strict=False)
    model = RiskScorer(enc).eval()

    out = C.MODEL_DIR / f"lidds_{SCENARIO}.onnx"
    dummy = torch.randint(0, vocab_size, (1, MAX_LEN), dtype=torch.long)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["tokens"], output_names=["risk"],
        dynamic_axes={"tokens": {0: "batch"}, "risk": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  Exported: {out}")

    # Parity check pada data LID-DS skenario tsb
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    data = C.load_scenario(SCENARIO)
    X = C.pad_to(data["windows"][:256].astype(np.int64), MAX_LEN)
    with torch.no_grad():
        pt = model(torch.tensor(X)).numpy()
    ox = sess.run(["risk"], {"tokens": X})[0]
    diff = float(np.abs(pt - ox).max())
    print(f"  Parity max|PyTorch-ONNX| = {diff:.2e}  {'PASS' if diff<1e-4 else 'FAIL'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
