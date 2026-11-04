#!/usr/bin/env python3
"""
#2a Fine-grained overhead — ISOLASI proses daemon (bukan host-wide).

overhead_bench.py (v1) mengukur CPU host-wide via /proc/stat, sehingga overhead
daemon HIPS tercampur dengan beban host lain. v2 mengukur resource MILIK daemon
saja: CPU% (utime+stime delta dari /proc/<pid>/stat) dan RSS (VmRSS dari
/proc/<pid>/status), dijumlahkan atas semua proses daemon (BCC kerap multi-thread/
multi-proc). Membayar permintaan: "pisahkan overhead HIPS sendiri dari host-wide".

Jalankan saat daemon HIPS berjalan (di tmux):
  python3 deploy/benchmarks/overhead_bench_v2.py --duration 30
Opsional bandingkan dengan beban kerja aktif di terminal lain.

Output: /tmp/overhead_v2.json
"""
import argparse, json, time, os

CLK = os.sysconf("SC_CLK_TCK")  # jiffies/detik
PAGE = os.sysconf("SC_PAGE_SIZE")
NCPU = os.cpu_count()


def find_daemon_pids():
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline") as f:
                if "hips_daemon" in f.read():
                    pids.append(int(d))
        except (FileNotFoundError, PermissionError):
            pass
    return pids


def proc_cpu_jiffies(pid):
    """utime+stime (jiffies). Parsing robust: ambil field SETELAH ')' penutup comm
    (comm bisa mengandung spasi/kurung). utime=field 14, stime=15 -> indeks 11,12
    pada bagian setelah ')'."""
    try:
        raw = open(f"/proc/{pid}/stat").read()
        after = raw[raw.rfind(")") + 1:].split()
        return int(after[11]) + int(after[12])
    except Exception:
        return 0


def proc_rss_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    pids = find_daemon_pids()
    if not pids:
        print("[!] Daemon hips_daemon tidak ditemukan. Start daemon dulu di tmux.")
        return
    print(f"[*] daemon PIDs: {pids} | CLK={CLK} NCPU={NCPU} | window={args.duration}s")

    j0 = {p: proc_cpu_jiffies(p) for p in pids}
    t0 = time.time()
    rss_samples = []
    while time.time() - t0 < args.duration:
        time.sleep(2)
        rss_samples.append(sum(proc_rss_kb(p) for p in pids))
    elapsed = time.time() - t0
    j1 = {p: proc_cpu_jiffies(p) for p in pids}

    total_jiffies = sum(j1[p] - j0.get(p, 0) for p in pids)
    cpu_seconds = total_jiffies / CLK
    # %CPU relatif 1 core, dan relatif total host
    cpu_pct_one_core = 100.0 * cpu_seconds / elapsed
    cpu_pct_host = cpu_pct_one_core / NCPU
    rss_mb = (max(rss_samples) / 1024.0) if rss_samples else 0.0

    result = {
        "daemon_pids": pids,
        "duration_s": round(elapsed, 1),
        "daemon_cpu_seconds": round(cpu_seconds, 2),
        "daemon_cpu_pct_of_one_core": round(cpu_pct_one_core, 1),
        "daemon_cpu_pct_of_host": round(cpu_pct_host, 1),
        "daemon_rss_mb": round(rss_mb, 1),
        "n_cpu": NCPU,
    }
    print("\n[RESULT — overhead daemon TERISOLASI]")
    print(json.dumps(result, indent=2))
    with open("/tmp/overhead_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n[*] Saved /tmp/overhead_v2.json")
    print(f"  -> Daemon memakai {result['daemon_cpu_pct_of_one_core']}% dari 1 core "
          f"({result['daemon_cpu_pct_of_host']}% dari {NCPU}-core host), "
          f"RSS {result['daemon_rss_mb']} MB.")


if __name__ == "__main__":
    main()
