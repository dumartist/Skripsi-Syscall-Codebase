"""
LID-DS Step 1+2: Parsing, Labeling, Windowing, Native Vocabulary

Membaca recording hasil ekstraksi (LID-DS_extracted/) dan menghasilkan
jendela token siap-model, dengan pelabelan timestamp-based intra-trace
sesuai metadata .json.

Pelabelan berdasarkan metadata recording:
  - exploit=false  -> semua token benign
  - exploit=true   -> token dengan ts >= exploit_absolute -> malicious
                      token sebelum itu -> benign

Parsing .sc:
  - Hanya event ENTER (arah '>') diambil -> satu token per invokasi syscall
  - Token = nama syscall (string), dipetakan ke ID via native vocab

Windowing: WINDOW_SIZE=100, STRIDE=50 (konsisten dengan ADFA-LD).
Label window: malicious jika mengandung >=1 token malicious (konservatif).

Output:
  prepared_data_lidds/vocabulary_native.json
  prepared_data_lidds/<scenario>.npz
      windows     (N, 100) int32
      labels      (N,)     int8   (0=benign, 1=malicious)
      partition   (N,)     int8   (0=train,1=val,2=test_normal,3=test_attack)
      rec_ids     (N,)     int32  (indeks recording, untuk split recording-level)
  prepared_data_lidds/metadata.json

Penggunaan:
  python -m syscall_hids.data.lidds_preparation
  python -m syscall_hids.data.lidds_preparation --scenarios CVE-2014-0160
"""

import argparse
import json
from pathlib import Path

import numpy as np

EXTRACTED_ROOT = Path("LID-DS_extracted")
OUTPUT_DIR = Path("prepared_data_lidds")

WINDOW_SIZE = 100
STRIDE = 50

PARTITION_MAP = {
    "training": 0,
    "validation": 1,
    "test/normal": 2,
    "test/normal_and_attack": 3,
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEFAULT_SCENARIOS = [
    "CVE-2014-0160", "Bruteforce_CWE-307", "CWE-89-SQL-injection",
]


# ===========================================================================
# Parsing
# ===========================================================================
def parse_sc(sc_path: Path):
    """Parse file .sc -> list (timestamp_sec: float, syscall_name: str)
    untuk event enter ('>') saja."""
    events = []
    with open(sc_path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            # Cari marker arah ('>' atau '<'); syscall = token sebelum marker.
            # Robust terhadap procname yang (jarang) mengandung spasi.
            dir_idx = None
            for i in range(5, min(len(parts), 9)):
                if parts[i] in (">", "<"):
                    dir_idx = i
                    break
            if dir_idx is None or dir_idx == 0:
                continue
            if parts[dir_idx] != ">":   # ambil enter saja
                continue
            syscall = parts[dir_idx - 1]
            try:
                ts_sec = int(parts[0]) / 1e9
            except ValueError:
                continue
            events.append((ts_sec, syscall))
    return events


def read_exploit_time(json_path: Path):
    """Return (is_exploit: bool, exploit_abs: float|None)."""
    try:
        with open(json_path, "r", errors="ignore") as f:
            meta = json.load(f)
    except Exception:
        return False, None
    is_exploit = bool(meta.get("exploit", False))
    exploit_abs = None
    t = meta.get("time", {})
    expl = t.get("exploit")
    if isinstance(expl, list) and expl:
        exploit_abs = expl[0].get("absolute")
    elif isinstance(expl, dict):
        exploit_abs = expl.get("absolute")
    return is_exploit, exploit_abs


def label_tokens(events, is_exploit, exploit_abs):
    """Return (names: list[str], labels: list[int])."""
    names = [e[1] for e in events]
    if not is_exploit or exploit_abs is None:
        labels = [0] * len(events)
    else:
        labels = [1 if ts >= exploit_abs else 0 for ts, _ in events]
    return names, labels


# ===========================================================================
# Recording iteration
# ===========================================================================
def iter_recordings(scenario: str):
    """Yield (partition_name, rec_dir) untuk semua recording skenario."""
    scn_root = EXTRACTED_ROOT / scenario
    for part_name in PARTITION_MAP:
        part_dir = scn_root / part_name
        if not part_dir.exists():
            continue
        for rec_dir in sorted(part_dir.iterdir()):
            if rec_dir.is_dir():
                yield part_name, rec_dir


def load_recording(rec_dir: Path):
    """Return (names, labels) atau (None, None) jika file tidak lengkap."""
    sc_files = list(rec_dir.glob("*.sc"))
    json_files = list(rec_dir.glob("*.json"))
    if not sc_files:
        return None, None
    events = parse_sc(sc_files[0])
    if not events:
        return None, None
    is_exploit, exploit_abs = (False, None)
    if json_files:
        is_exploit, exploit_abs = read_exploit_time(json_files[0])
    return label_tokens(events, is_exploit, exploit_abs)


# ===========================================================================
# Windowing
# ===========================================================================
def make_windows(token_ids, labels, win=WINDOW_SIZE, stride=STRIDE):
    """Sliding window. Return list (window_ids, window_label).
    Window label = malicious jika ada >=1 token malicious."""
    out = []
    n = len(token_ids)
    if n == 0:
        return out
    # Jika lebih pendek dari window: satu window dengan padding (PAD=0)
    if n < win:
        padded = token_ids + [0] * (win - n)
        wlabel = 1 if any(labels) else 0
        out.append((padded, wlabel))
        return out
    for start in range(0, n - win + 1, stride):
        chunk_ids = token_ids[start:start + win]
        chunk_lab = labels[start:start + win]
        wlabel = 1 if any(chunk_lab) else 0
        out.append((chunk_ids, wlabel))
    # Tangani sisa di ujung (tail) bila tidak terbagi rata
    last_start = ((n - win) // stride) * stride
    if last_start + win < n:
        chunk_ids = token_ids[n - win:n]
        chunk_lab = labels[n - win:n]
        wlabel = 1 if any(chunk_lab) else 0
        out.append((chunk_ids, wlabel))
    return out


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("LID-DS Step 1+2: Parse / Label / Window / Vocab")
    print(f"Skenario: {args.scenarios}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Pass 1: parse semua recording, kumpulkan nama untuk vocab
    # ------------------------------------------------------------------
    print("\n[Pass 1] Parsing recording & membangun vocabulary native...")
    parsed = {}  # scenario -> list of (partition, rec_idx, names, labels)
    vocab_counter = {}
    for scn in args.scenarios:
        recs = []
        rec_idx = 0
        n_skip = 0
        for part_name, rec_dir in iter_recordings(scn):
            names, labels = load_recording(rec_dir)
            if names is None:
                n_skip += 1
                continue
            for nm in names:
                vocab_counter[nm] = vocab_counter.get(nm, 0) + 1
            recs.append((part_name, rec_idx, names, labels))
            rec_idx += 1
        parsed[scn] = recs
        print(f"  {scn:24s}: {len(recs)} recording diparse ({n_skip} dilewati)")

    # Build vocab: PAD=0, UNK=1, lalu syscall urut frekuensi desc
    sorted_names = sorted(vocab_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for name, _cnt in sorted_names:
        vocab[name] = len(vocab)
    print(f"\n  Vocabulary native: {len(vocab)} token "
          f"({len(vocab) - 2} syscall unik + PAD + UNK)")

    with open(OUTPUT_DIR / "vocabulary_native.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # ------------------------------------------------------------------
    # Pass 2: tokenisasi + windowing per skenario
    # ------------------------------------------------------------------
    print("\n[Pass 2] Tokenisasi & windowing...")
    meta = {
        "window_size": WINDOW_SIZE, "stride": STRIDE,
        "vocab_size": len(vocab),
        "partition_map": PARTITION_MAP,
        "scenarios": {},
    }

    for scn in args.scenarios:
        all_windows, all_labels, all_part, all_rec = [], [], [], []
        for part_name, rec_idx, names, labels in parsed[scn]:
            token_ids = [vocab.get(nm, 1) for nm in names]  # UNK=1
            wins = make_windows(token_ids, labels)
            pid = PARTITION_MAP[part_name]
            for w_ids, w_lab in wins:
                all_windows.append(w_ids)
                all_labels.append(w_lab)
                all_part.append(pid)
                all_rec.append(rec_idx)

        windows = np.array(all_windows, dtype=np.int32)
        lbls = np.array(all_labels, dtype=np.int8)
        parts = np.array(all_part, dtype=np.int8)
        recs = np.array(all_rec, dtype=np.int32)

        np.savez_compressed(
            OUTPUT_DIR / f"{scn}.npz",
            windows=windows, labels=lbls, partition=parts, rec_ids=recs,
        )

        # Statistik
        def part_stat(pid):
            m = parts == pid
            return int(m.sum()), int(lbls[m].sum())

        stat = {p: {"windows": part_stat(i)[0], "malicious": part_stat(i)[1]}
                for p, i in PARTITION_MAP.items()}
        meta["scenarios"][scn] = {
            "total_windows": int(len(windows)),
            "total_malicious": int(lbls.sum()),
            "by_partition": stat,
        }
        print(f"\n  [{scn}] total {len(windows)} window, "
              f"{int(lbls.sum())} malicious")
        for p, i in PARTITION_MAP.items():
            w, mal = part_stat(i)
            print(f"    {p:28s}: {w:6d} window, {mal:5d} malicious")

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 64)
    print(f"Selesai. Output di {OUTPUT_DIR}/")
    print(f"  vocabulary_native.json ({len(vocab)} token)")
    print(f"  {len(args.scenarios)} file .npz + metadata.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
