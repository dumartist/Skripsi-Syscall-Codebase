#!/usr/bin/env python3
"""
Overhead benchmark untuk prototipe HIPS.
Mengukur dampak HIPS daemon terhadap host: CPU% dan RAM.

Mengukur 3 kondisi (jalankan terpisah):
  1. baseline  — host idle, daemon TIDAK jalan
  2. monitored — daemon jalan tanpa beban kerja
  3. loaded    — daemon jalan + beban kerja sintetis

Usage:
  python3 deploy/benchmarks/overhead_bench.py --label baseline  --duration 30
  python3 deploy/benchmarks/overhead_bench.py --label monitored --duration 30
  python3 deploy/benchmarks/overhead_bench.py --label loaded    --duration 30

Mengukur CPU host-wide via /proc/stat dan RAM via /proc/meminfo
(tanpa dependency psutil agar portable).
"""

import argparse
import time
import json
import os


def read_cpu():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return idle, total


def read_mem_used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.strip().split()[0])  # kB
    total = info["MemTotal"]
    avail = info.get("MemAvailable", info["MemFree"])
    used_mb = (total - avail) / 1024.0
    return used_mb


def find_daemon_cpu():
    """Cari proses hips_daemon dan jumlahkan %CPU (dari /proc/<pid>/stat)."""
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline") as f:
                cmd = f.read()
            if "hips_daemon" in cmd:
                pids.append(int(d))
        except (FileNotFoundError, PermissionError):
            pass
    return pids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True,
                   choices=["baseline", "monitored", "loaded"])
    p.add_argument("--duration", type=int, default=30)
    args = p.parse_args()

    print(f"[BENCH] label={args.label} duration={args.duration}s")
    daemon_pids = find_daemon_cpu()
    print(f"[BENCH] daemon PIDs detected: {daemon_pids or 'none'}")

    idle0, total0 = read_cpu()
    mem_samples = [read_mem_used_mb()]
    t_end = time.time() + args.duration

    while time.time() < t_end:
        time.sleep(2)
        mem_samples.append(read_mem_used_mb())

    idle1, total1 = read_cpu()

    cpu_busy_pct = 100.0 * (1 - (idle1 - idle0) / (total1 - total0))
    mem_mean = sum(mem_samples) / len(mem_samples)
    mem_max = max(mem_samples)

    result = {
        "label": args.label,
        "duration_s": args.duration,
        "cpu_busy_pct": round(cpu_busy_pct, 2),
        "mem_used_mb_mean": round(mem_mean, 1),
        "mem_used_mb_max": round(mem_max, 1),
        "daemon_running": bool(daemon_pids),
    }

    print("\n[RESULT]")
    print(json.dumps(result, indent=2))

    # Append ke file hasil
    outpath = "/tmp/overhead_results.jsonl"
    with open(outpath, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\n[BENCH] appended to {outpath}")
    print("[BENCH] Jalankan ketiga label, lalu: cat /tmp/overhead_results.jsonl")


if __name__ == "__main__":
    main()
