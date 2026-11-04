#!/usr/bin/env python3
"""
Fase 3: Attack Simulator
Mensimulasikan proses berbahaya dengan pola syscall yang mencurigakan.
Tujuan: memicu TERMINATE dari HIPS daemon.

Strategi: kombinasi syscall yang tidak lazim untuk proses normal:
- fork() berulang (clone)
- execve mencoba binary
- openat ke file sensitif (/etc/passwd, /etc/shadow)
- mmap besar
- socket + connect berulang

Jalankan TERPISAH dari daemon (terminal/tmux lain):
  python3 deploy/validation/attack_simulator.py
"""

import os
import sys
import time
import ctypes
import ctypes.util
import socket
import struct

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Ubah nama proses di level kernel (comm field yang dibaca eBPF)
# PR_SET_NAME = 15
def set_proc_name(name: str):
    libc.prctl(15, name.encode()[:15], 0, 0, 0)

def log(msg):
    print(f"[ATTACK-SIM] {msg}", flush=True)

def spam_syscalls(duration=30):
    """
    Spam syscall mencurigakan selama `duration` detik.
    Pattern ini meniru trace Meterpreter/reverse shell.
    """
    # Set nama proses di kernel → eBPF akan baca "evil_proc" bukan "python3"
    set_proc_name("evil_proc")
    log(f"Starting attack simulation (PID={os.getpid()}) for {duration}s")
    log("Process name set to 'evil_proc' via prctl")
    log("Pattern: openat(/etc/passwd) + mmap + socket + getuid spam")

    t_end = time.time() + duration
    iteration = 0

    while time.time() < t_end:
        iteration += 1

        # 1. Buka file sensitif berulang (seperti privilege check)
        try:
            with open("/etc/passwd", "rb") as f:
                _ = f.read(16)
        except Exception:
            pass

        # 2. Coba buka /etc/shadow (akan ditolak, tapi syscall tetap tercatat)
        try:
            with open("/etc/shadow", "rb") as f:
                _ = f.read(16)
        except PermissionError:
            pass  # expected

        # 3. Socket ke localhost berulang (network recon pattern)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.01)
            s.connect(("127.0.0.1", 9999))  # port tidak ada → refused
            s.close()
        except Exception:
            pass

        # 4. mmap + munmap berulang (memory allocation pattern malware)
        try:
            mm = libc.mmap(0, 4096, 3, 34, -1, 0)  # PROT_READ|WRITE, MAP_ANON|PRIVATE
            if mm != -1:
                libc.munmap(mm, 4096)
        except Exception:
            pass

        # 5. getuid/getgid spam (privilege enumeration)
        for _ in range(10):
            os.getuid()
            os.getgid()
            os.getpid()

        # 6. fork + immediate exit (fork bomb lite — buat traffic PID baru)
        if iteration % 20 == 0:
            pid = os.fork()
            if pid == 0:
                # child: lakukan beberapa syscall lalu exit
                for _ in range(50):
                    os.getuid()
                    try:
                        open("/etc/passwd").close()
                    except Exception:
                        pass
                sys.exit(0)
            else:
                os.waitpid(pid, 0)

        if iteration % 10 == 0:
            elapsed = time.time() - (t_end - duration)
            log(f"  iteration={iteration} elapsed={elapsed:.1f}s PID={os.getpid()}")

        time.sleep(0.05)  # 20 iter/detik

    log(f"Simulation done. Total iterations: {iteration}")
    log("If HIPS is running with --enforce, this process should have been killed.")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    spam_syscalls(duration)
