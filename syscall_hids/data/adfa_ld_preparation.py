"""
Stage 1: ADFA-LD Data Preparation

Parses ADFA-LD partitions (Training/Validation/Attack), builds syscall
vocabulary with <PAD>=0 and <UNK>=1, applies sliding-window segmentation
(WINDOW_SIZE=100, STRIDE=50), and enforces a TESSERACT-conformant
time-aware train/test split.

Temporal split rationale:
  - Training_Data_Master was collected first (benign baseline).
  - Attack subfolders are numbered _1.._10 in collection order.
  - We use folds 1-5 for training, folds 6-10 for testing.
  - Validation_Data_Master (benign, collected later) goes to test only.
  This ensures the model never sees future data during training.
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("archive-kaggle/ADFA-LD")
OUTPUT_DIR = Path("prepared_data")
WINDOW_SIZE = 100
STRIDE = 50
TRAIN_ATTACK_MAX_FOLD = 5  # folds 1..5 → train, 6..10 → test
SEED = 42
TRIGRAM_N = 3

# i386 no-ops that actually occur in ADFA-LD (getuid32/geteuid32, not 16-bit 24/49)
NOOP_SYSCALLS = [20, 199, 201, 64, 13, 78]  # getpid, getuid32, geteuid32, getppid, time, gettimeofday

ATTACK_CATEGORIES = [
    "Adduser", "Hydra_FTP", "Hydra_SSH",
    "Java_Meterpreter", "Meterpreter", "Web_Shell",
]

CAT_TO_ID = {"benign": 0}
for _i, _cat in enumerate(ATTACK_CATEGORIES, 1):
    CAT_TO_ID[_cat] = _i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_trace(filepath):
    with open(filepath, "r") as f:
        content = f.read().strip()
    return [int(x) for x in content.split()] if content else []


def load_traces(directory):
    traces = []
    for fp in sorted(directory.glob("*.txt")):
        t = read_trace(fp)
        if t:
            traces.append(t)
    return traces


def sliding_windows(trace, w=WINDOW_SIZE, s=STRIDE):
    if len(trace) < w:
        return [trace + [0] * (w - len(trace))]
    return [trace[i : i + w] for i in range(0, len(trace) - w + 1, s)]


def build_vocab(all_ids):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sid in sorted(all_ids):
        vocab[str(sid)] = len(vocab)
    return vocab


def encode(trace, vocab):
    return [vocab.get(str(s), 1) for s in trace]


def trigram_distribution(traces, n=TRIGRAM_N):
    counts = Counter()
    for t in traces:
        for i in range(len(t) - n + 1):
            counts[tuple(t[i : i + n])] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Stage 1: ADFA-LD Data Preparation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load traces
    # ------------------------------------------------------------------
    print("\n[1/5] Loading traces...")

    train_traces = load_traces(DATA_ROOT / "Training_Data_Master")
    val_traces = load_traces(DATA_ROOT / "Validation_Data_Master")

    # Attack traces keyed by category → fold → list of traces
    attack_data = {}
    for cat in ATTACK_CATEGORIES:
        attack_data[cat] = {}
        attack_dir = DATA_ROOT / "Attack_Data_Master"
        for sub in sorted(attack_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith(cat + "_"):
                fold = int(sub.name.rsplit("_", 1)[-1])
                traces = load_traces(sub)
                if traces:
                    attack_data[cat][fold] = traces

    n_attack = sum(
        sum(len(v) for v in folds.values()) for folds in attack_data.values()
    )
    print(f"  Benign training:   {len(train_traces):>5} traces")
    print(f"  Benign validation: {len(val_traces):>5} traces")
    print(f"  Attack:            {n_attack:>5} traces  ({len(ATTACK_CATEGORIES)} categories)")

    # ------------------------------------------------------------------
    # 2. Build vocabulary from all observed syscall IDs
    # ------------------------------------------------------------------
    print("\n[2/5] Building vocabulary...")

    # Vocabulary from TRAINING traces only (benign Training_Data_Master + attack
    # folds 1..TRAIN_ATTACK_MAX_FOLD). Syscalls first seen in test traces map to
    # <UNK>, so no test information enters the token set.
    train_ids, test_ids = set(), set()
    for t in train_traces:
        train_ids.update(t)
    for t in val_traces:
        test_ids.update(t)
    for folds in attack_data.values():
        for fold, traces in folds.items():
            for t in traces:
                (train_ids if fold <= TRAIN_ATTACK_MAX_FOLD else test_ids).update(t)
    test_only = sorted(test_ids - train_ids)

    vocab = build_vocab(train_ids)
    print(f"  Vocabulary size: {len(vocab)}  (PAD=0, UNK=1, +{len(vocab)-2} syscalls; "
          f"built from training traces only)")
    print(f"  Syscalls seen only in test (-> <UNK>): {len(test_only)}")

    # ------------------------------------------------------------------
    # 3. Segment into windows with TESSERACT time-aware split
    # ------------------------------------------------------------------
    print("\n[3/5] Segmenting windows (TESSERACT time-aware split)...")

    # Each window records the id of the trace it came from (unique across the
    # whole dataset), so later splits can keep overlapping windows of one trace
    # on the same side.
    next_tid = [0]

    def add_trace(t, label, cat, enc_l, raw_l, lab_l, cat_l, tid_l):
        enc = encode(t, vocab)
        for w, rw in zip(sliding_windows(enc), sliding_windows(t)):
            enc_l.append(w)
            raw_l.append(rw)
            lab_l.append(label)
            cat_l.append(CAT_TO_ID[cat])
            tid_l.append(next_tid[0])
        next_tid[0] += 1

    # -- Training set: benign + attack folds 1..TRAIN_ATTACK_MAX_FOLD --
    train_enc, train_raw, train_labels, train_cats, train_tids = [], [], [], [], []
    tr_lists = (train_enc, train_raw, train_labels, train_cats, train_tids)
    for t in train_traces:
        add_trace(t, 0, "benign", *tr_lists)
    for cat in ATTACK_CATEGORIES:
        for fold in sorted(attack_data[cat]):
            if fold > TRAIN_ATTACK_MAX_FOLD:
                continue
            for t in attack_data[cat][fold]:
                add_trace(t, 1, cat, *tr_lists)

    # -- Test set: benign validation + attack folds (TRAIN_ATTACK_MAX_FOLD+1)..10 --
    test_enc, test_raw, test_labels, test_cats, test_tids = [], [], [], [], []
    te_lists = (test_enc, test_raw, test_labels, test_cats, test_tids)
    for t in val_traces:
        add_trace(t, 0, "benign", *te_lists)
    for cat in ATTACK_CATEGORIES:
        for fold in sorted(attack_data[cat]):
            if fold <= TRAIN_ATTACK_MAX_FOLD:
                continue
            for t in attack_data[cat][fold]:
                add_trace(t, 1, cat, *te_lists)

    train_enc = np.array(train_enc, dtype=np.int32)
    train_raw = np.array(train_raw, dtype=np.int32)
    train_labels = np.array(train_labels, dtype=np.int32)
    train_cats = np.array(train_cats, dtype=np.int32)

    test_enc = np.array(test_enc, dtype=np.int32)
    test_raw = np.array(test_raw, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)
    test_cats = np.array(test_cats, dtype=np.int32)

    tr_ben = int((train_labels == 0).sum())
    tr_att = int((train_labels == 1).sum())
    te_ben = int((test_labels == 0).sum())
    te_att = int((test_labels == 1).sum())

    print(f"  Train: {len(train_enc):>6} windows  ({tr_ben} benign, {tr_att} attack)")
    print(f"  Test:  {len(test_enc):>6} windows  ({te_ben} benign, {te_att} attack)")
    print("  Per-category breakdown (test):")
    for cat in ATTACK_CATEGORIES:
        cid = CAT_TO_ID[cat]
        print(f"    {cat:25s} {(test_cats == cid).sum():>5} windows")

    # ------------------------------------------------------------------
    # 4. Compute benign trigram reference distribution
    # ------------------------------------------------------------------
    print("\n[4/5] Computing benign trigram reference...")
    benign_tri = trigram_distribution(train_traces)
    print(f"  {len(benign_tri)} unique trigrams")

    # ------------------------------------------------------------------
    # 5. Save everything
    # ------------------------------------------------------------------
    print("\n[5/5] Saving to prepared_data/...")

    with open(OUTPUT_DIR / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "train.npz",
        windows=train_enc,
        raw_windows=train_raw,
        labels=train_labels,
        cat_ids=train_cats,
        trace_ids=np.array(train_tids, dtype=np.int32),
    )

    np.savez_compressed(
        OUTPUT_DIR / "test.npz",
        windows=test_enc,
        raw_windows=test_raw,
        labels=test_labels,
        cat_ids=test_cats,
        trace_ids=np.array(test_tids, dtype=np.int32),
    )

    with open(OUTPUT_DIR / "benign_trigrams.pkl", "wb") as f:
        pickle.dump(benign_tri, f)

    meta = {
        "vocab_size": len(vocab),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "trigram_n": TRIGRAM_N,
        "train_size": int(len(train_enc)),
        "test_size": int(len(test_enc)),
        "train_benign": tr_ben,
        "train_attack": tr_att,
        "test_benign": te_ben,
        "test_attack": te_att,
        "attack_folds_train": list(range(1, TRAIN_ATTACK_MAX_FOLD + 1)),
        "attack_folds_test": list(range(TRAIN_ATTACK_MAX_FOLD + 1, 11)),
        "category_map": CAT_TO_ID,
        "noop_syscalls": NOOP_SYSCALLS,
        "seed": SEED,
        "vocab_source": "training traces only (benign Training_Data_Master + attack folds 1-5)",
        "test_only_syscalls_mapped_to_unk": test_only,
        "train_traces": int(len(set(train_tids))),
        "test_traces": int(len(set(test_tids))),
        "trigram_reference_source": "benign Training_Data_Master traces only",
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  vocabulary.json           {len(vocab)} entries")
    print(f"  train.npz                 windows {train_enc.shape}, raw {train_raw.shape}")
    print(f"  test.npz                  windows {test_enc.shape}, raw {test_raw.shape}")
    print(f"  benign_trigrams.pkl       {len(benign_tri)} trigrams")
    print(f"  metadata.json             split config + stats")

    print("\n" + "=" * 60)
    print("Stage 1 complete. Next: python -m syscall_hids.attacks.adversarial_mimicry")
    print("=" * 60)


if __name__ == "__main__":
    main()
