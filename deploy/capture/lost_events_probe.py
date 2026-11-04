#!/usr/bin/env python3
"""
#2c Lost events — drop rate perf-buffer eBPF di tahap CAPTURE di bawah beban.

Tahap eBPF->userspace dapat MENJATUHKAN event jika userspace tak mengejar laju
syscall (perf buffer penuh). Probe ini mengukur: total event diterima vs event
HILANG (lost) selama N detik. Jalankan SAMBIL membebani host untuk melihat drop
di kondisi stres — mengkuantifikasi batas tahap capture (klaim host-level jujur).

Jalankan (root, sistem BCC):
  sudo "$(command -v python3)" deploy/capture/lost_events_probe.py --duration 30
  # di terminal lain, beri beban: ( for i in $(seq 5); do find / -type f >/dev/null 2>&1; done )

Output: /tmp/lost_events.json
"""
import argparse, ctypes, json, time
from bcc import BPF

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
struct evt_t { u32 pid; u64 nr; };
BPF_PERF_OUTPUT(events);
RAW_TRACEPOINT_PROBE(sys_enter) {
    struct evt_t e = {};
    e.pid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    e.nr  = (u64)ctx->args[1];
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}
"""

class Evt(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint32), ("nr", ctypes.c_uint64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--page_cnt", type=int, default=64,
                    help="ukuran perf buffer (page); kecil = lebih mudah drop")
    args = ap.parse_args()

    b = BPF(text=BPF_PROGRAM)
    received = [0]
    lost = [0]

    def on_event(cpu, data, size):
        received[0] += 1

    def on_lost(cpu, count):
        lost[0] += count

    b["events"].open_perf_buffer(on_event, lost_cb=on_lost, page_cnt=args.page_cnt)
    print(f"[*] Probing lost events {args.duration}s (page_cnt={args.page_cnt})...")
    t0 = time.time()
    while time.time() - t0 < args.duration:
        b.perf_buffer_poll(timeout=100)
    elapsed = time.time() - t0

    total = received[0] + lost[0]
    drop_rate = (lost[0] / total) if total else 0.0
    result = {
        "duration_s": round(elapsed, 1),
        "events_received": received[0],
        "events_lost": lost[0],
        "total_events": total,
        "drop_rate_pct": round(100 * drop_rate, 3),
        "received_per_sec": round(received[0] / elapsed, 0),
        "page_cnt": args.page_cnt,
    }
    print("\n[RESULT — lost events tahap capture]")
    print(json.dumps(result, indent=2))
    with open("/tmp/lost_events.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n[*] Saved /tmp/lost_events.json")
    if drop_rate > 0:
        print(f"  -> {result['drop_rate_pct']}% event JATUH di bawah beban — "
              "batas tahap capture; mitigasi: perbesar buffer / filter di sisi eBPF / hot-path C.")
    else:
        print("  -> 0 drop pada kondisi ini; ulangi dengan beban lebih berat / page_cnt lebih kecil.")


if __name__ == "__main__":
    main()
