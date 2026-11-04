"""
LID-DS Step 2b: Vocabulary Alignment (Mode B — ADFA-compatible lossy)

Memetakan nama syscall LID-DS (x86_64) ke ruang token ADFA-LD (nomor
syscall i386), untuk memungkinkan skenario zero-shot (model ADFA membaca
LID-DS) dan fine-tune.

Strategi alignment vocabulary:
  - Pemetaan name -> nomor i386, MEMILIH varian yang benar-benar ada di
    vocab ADFA (mis. stat64=195 bukan stat=106, _newselect=142, dll).
  - Socket-family (socket/connect/accept/sendto/recvfrom/getpeername/
    getsockname/getsockopt/setsockopt/shutdown) -> 102 (socketcall).
    CATATAN: granularitas hilang — semua socket op kolaps ke satu token.
  - semop -> 117 (ipc, juga multiplexed). CATATAN: granularitas hilang.
  - Syscall tanpa padanan i386 di vocab ADFA, pseudo-event (procexit),
    dan syscall modern (epoll_create1, io_submit, io_getevents) ->
    <ABI_UNMAPPED>.
  - Saat membangun tensor untuk model ADFA (zero-shot), <ABI_UNMAPPED>
    dikolaps ke <UNK>=1 karena embedding ADFA tidak punya baris untuknya;
    namun dilacak terpisah untuk analisis (Tabel 4.22).

Output:
  prepared_data_lidds/vocabulary_aligned.json
      lidds_name -> {adfa_token, adfa_number, kind, model_input_token}
  prepared_data_lidds/alignment_stats.json   (untuk Tabel 4.22 thesis)
"""

import json
from pathlib import Path

ADFA_VOCAB = Path("prepared_data/vocabulary.json")
LIDDS_VOCAB = Path("prepared_data_lidds/vocabulary_native.json")
OUTPUT_DIR = Path("prepared_data_lidds")

UNK_TOKEN_IDX = 1  # <UNK> di vocab ADFA

# Socket-family LID-DS yang di-i386 di-multiplex lewat socketcall(102)
SOCKET_FAMILY = {
    "socket", "connect", "accept", "accept4", "bind", "listen",
    "sendto", "recvfrom", "sendmsg", "recvmsg", "getpeername",
    "getsockname", "getsockopt", "setsockopt", "shutdown",
}
SOCKETCALL_I386 = 102

# IPC-family (semop/semget/msgsnd/...) di-multiplex lewat ipc(117) di i386
IPC_FAMILY = {"semop", "semget", "semctl", "msgsnd", "msgrcv", "msgget",
              "shmat", "shmdt", "shmget", "shmctl"}
IPC_I386 = 117

# Tabel name -> nomor i386, MEMILIH varian yang dipakai glibc 32-bit modern
# (yaitu varian yang muncul di vocab ADFA). Hanya mencakup 67 syscall LID-DS
# + beberapa cadangan. Sumber: arch/x86/entry/syscalls/syscall_32.tbl.
I386_TABLE = {
    "read": 3, "write": 4, "open": 5, "close": 6,
    "unlink": 10, "chdir": 12, "chmod": 15, "lseek": 19,
    "getpid": 20, "utime": 30, "access": 33, "rename": 38,
    "times": 43, "brk": 45, "getppid": 64,
    "munmap": 91, "ftruncate": 93, "setitimer": 104,
    "wait4": 114, "fsync": 118, "clone": 120, "mprotect": 125,
    "getdents": 141, "writev": 146, "sched_yield": 158,
    "nanosleep": 162, "poll": 168, "rt_sigaction": 174,
    "rt_sigprocmask": 175, "pread": 180, "pwrite": 181,
    "select": 142,  # _newselect (varian 32-bit di vocab ADFA)
    "getcwd": 183, "mmap": 192, "ftruncate64": 194,
    # varian 32-bit yang ada di vocab ADFA:
    "stat": 195, "lstat": 196, "fstat": 197,
    "getuid": 199, "geteuid": 201, "setuid": 213, "setgid": 214,
    "setgroups": 206, "madvise": 219, "getdents64": 220,
    "fcntl": 221, "gettid": 224, "futex": 240,
    "umask": 60, "flock": 143, "exit_group": 252,
    "epoll_ctl": 255, "epoll_wait": 256, "openat": 295,
    "set_robust_list": 311,
    # Modern / tanpa padanan stabil di vocab ADFA (dipetakan, dicek nanti):
    "epoll_create1": 329, "io_submit": 248, "io_getevents": 247,
    # pseudo-event LID-DS (bukan syscall): tangani sebagai unmapped
    # "procexit" sengaja tidak di tabel -> ABI_UNMAPPED
}


def main():
    adfa = json.load(open(ADFA_VOCAB))
    lidds = json.load(open(LIDDS_VOCAB))
    adfa_numbers = {k for k in adfa if k not in ("<PAD>", "<UNK>")}

    aligned = {}
    stats = {"direct": [], "socket_collapsed": [], "ipc_collapsed": [],
             "abi_unmapped": []}

    for name in lidds:
        if name in ("<PAD>", "<UNK>"):
            continue

        if name in SOCKET_FAMILY:
            num = SOCKETCALL_I386
            kind = "socket_collapsed"
        elif name in IPC_FAMILY:
            num = IPC_I386
            kind = "ipc_collapsed"
        elif name in I386_TABLE:
            num = I386_TABLE[name]
            kind = "candidate"
        else:
            num = None
            kind = "abi_unmapped"

        # Verifikasi nomor benar-benar ada di vocab ADFA
        if num is not None and str(num) in adfa_numbers:
            adfa_token = adfa[str(num)]
            if kind == "candidate":
                kind = "direct"
            model_input_token = adfa_token
        else:
            # Tidak ada di vocab ADFA -> ABI_UNMAPPED
            kind = "abi_unmapped"
            num = None
            adfa_token = None
            model_input_token = UNK_TOKEN_IDX  # kolaps ke <UNK> untuk model

        aligned[name] = {
            "adfa_number": num,
            "adfa_token": adfa_token,
            "kind": kind,
            "model_input_token": model_input_token,
        }
        stats[kind].append(name)

    # Statistik untuk Tabel 4.22
    n_total = len(aligned)
    summary = {
        "lidds_syscalls_total": n_total,
        "adfa_vocab_size": len(adfa),
        "counts": {k: len(v) for k, v in stats.items()},
        "pct": {k: round(100 * len(v) / n_total, 1) for k, v in stats.items()},
        "lists": stats,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(aligned, open(OUTPUT_DIR / "vocabulary_aligned.json", "w"), indent=2)
    json.dump(summary, open(OUTPUT_DIR / "alignment_stats.json", "w"), indent=2)

    # ------------------------------------------------------------------
    # Print laporan
    # ------------------------------------------------------------------
    print("=" * 64)
    print("LID-DS Step 2b: Vocabulary Alignment (Mode B)")
    print("=" * 64)
    print(f"\nTotal syscall LID-DS       : {n_total}")
    print(f"ADFA vocab size            : {len(adfa)}")
    print(f"\n{'Kategori':<22}{'Jumlah':>8}{'Persen':>9}")
    print("-" * 40)
    label = {
        "direct": "Direct (1:1)",
        "socket_collapsed": "Socket -> 102",
        "ipc_collapsed": "IPC -> 117",
        "abi_unmapped": "ABI_UNMAPPED",
    }
    for k in ["direct", "socket_collapsed", "ipc_collapsed", "abi_unmapped"]:
        print(f"{label[k]:<22}{len(stats[k]):>8}{summary['pct'][k]:>8.1f}%")

    print(f"\nSocket-collapsed ({len(stats['socket_collapsed'])}): "
          f"{', '.join(sorted(stats['socket_collapsed']))}")
    print(f"IPC-collapsed ({len(stats['ipc_collapsed'])}): "
          f"{', '.join(sorted(stats['ipc_collapsed']))}")
    print(f"ABI_UNMAPPED ({len(stats['abi_unmapped'])}): "
          f"{', '.join(sorted(stats['abi_unmapped']))}")

    mapped = len(stats["direct"]) + len(stats["socket_collapsed"]) + len(stats["ipc_collapsed"])
    print(f"\nCoverage: {mapped}/{n_total} ter-map ke ruang token ADFA "
          f"({100*mapped/n_total:.1f}%), {len(stats['abi_unmapped'])} unmapped")
    print("=" * 64)
    print(f"\nSaved: {OUTPUT_DIR}/vocabulary_aligned.json")
    print(f"Saved: {OUTPUT_DIR}/alignment_stats.json  (-> Tabel 4.22)")


if __name__ == "__main__":
    main()
