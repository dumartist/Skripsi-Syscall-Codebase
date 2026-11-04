#!/usr/bin/env python3
"""
Bangun deploy vocab: syscall_nr (x86_64) -> token native LID-DS.

Host eBPF menangkap NOMOR syscall x86_64. Model LID-DS native di-key
oleh NAMA syscall. Script ini menjembatani: parse tabel syscall x86_64
authoritative dari kernel header, lalu komposisikan dengan vocab native.

Jalankan DI VM (butuh linux-headers terinstal):
  python3 deploy/runtime/build_deploy_vocab.py \
    --native /home/arthur/vocabulary_native.json \
    --out    /home/arthur/lidds_deploy_vocab.json

Output JSON kompatibel dengan hips_daemon_v2.py --vocab:
  {"<PAD>":0, "<UNK>":1, "<nr>": <native_token>, ...}
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Lokasi header syscall x86_64 (urutan prioritas)
HEADER_CANDIDATES = [
    "/usr/include/x86_64-linux-gnu/asm/unistd_64.h",
    "/usr/include/asm/unistd_64.h",
    "/usr/include/asm-generic/unistd.h",
]


def parse_syscall_table():
    """Parse #define __NR_<name> <nr> -> {name: nr}."""
    for path in HEADER_CANDIDATES:
        p = Path(path)
        if not p.exists():
            continue
        name2nr = {}
        pat = re.compile(r"#define\s+__NR_(\w+)\s+(\d+)")
        for line in p.read_text().splitlines():
            m = pat.search(line)
            if m:
                name2nr[m.group(1)] = int(m.group(2))
        if name2nr:
            print(f"[*] Parsed {len(name2nr)} syscalls from {path}", file=sys.stderr)
            return name2nr
    print("[!] No syscall header found. Install linux-headers.", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="vocabulary_native.json")
    ap.add_argument("--out",    required=True, help="output deploy vocab JSON")
    args = ap.parse_args()

    name2nr = parse_syscall_table()

    # Alias: nama sysdig (LID-DS) -> nama kernel x86_64.
    # sysdig menormalkan beberapa nama; kernel pakai suffix 64.
    ALIASES = {
        "pread":  "pread64",
        "pwrite": "pwrite64",
        # 'procexit' = pseudo-event sysdig, sengaja tidak dipetakan
    }
    for sysdig_name, kernel_name in ALIASES.items():
        if kernel_name in name2nr and sysdig_name not in name2nr:
            name2nr[sysdig_name] = name2nr[kernel_name]

    native = json.load(open(args.native))  # {name: token}

    PAD = native.get("<PAD>", 0)
    UNK = native.get("<UNK>", 1)

    deploy = {"<PAD>": PAD, "<UNK>": UNK}
    mapped, unmapped = [], []
    for name, token in native.items():
        if name in ("<PAD>", "<UNK>"):
            continue
        if name in name2nr:
            deploy[str(name2nr[name])] = token
            mapped.append(name)
        else:
            unmapped.append(name)

    json.dump(deploy, open(args.out, "w"), indent=2)

    print(f"[*] Deploy vocab written: {args.out}")
    print(f"[*] Mapped:   {len(mapped)}/{len(native)-2} syscall names")
    print(f"[*] Unmapped: {unmapped}")
    print(f"    (unmapped -> <UNK> di host; 'procexit' wajar tak ada — "
          f"itu pseudo-event sysdig, bukan syscall nyata)")


if __name__ == "__main__":
    main()
