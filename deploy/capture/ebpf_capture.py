#!/usr/bin/env python3
"""
Fase 1: eBPF Syscall Capturer
Capture syscall events per-PID secara real-time menggunakan BCC.
Output: stream (pid, comm, syscall_nr, timestamp_ns) ke stdout/ring buffer.

Usage:
  sudo python3 deploy/capture/ebpf_capture.py              # print semua syscall
  sudo python3 deploy/capture/ebpf_capture.py --pid 1234   # filter PID tertentu
  sudo python3 deploy/capture/ebpf_capture.py --comm nginx # filter nama proses
"""

import argparse
import ctypes
import sys
import signal
from bcc import BPF

# eBPF program (C)
BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct event_t {
    u32 pid;
    u32 tgid;
    u64 timestamp_ns;
    u64 syscall_nr;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    struct event_t evt = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    evt.pid    = pid_tgid & 0xFFFFFFFF;
    evt.tgid   = pid_tgid >> 32;
    evt.timestamp_ns = bpf_ktime_get_ns();
    evt.syscall_nr   = args->id;
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));

    FILTER_PID
    FILTER_COMM

    events.perf_submit(args, &evt, sizeof(evt));
    return 0;
}
"""

# Struct Python yang cocok dengan C struct
class Event(ctypes.Structure):
    _fields_ = [
        ("pid",          ctypes.c_uint32),
        ("tgid",         ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
        ("syscall_nr",   ctypes.c_uint64),
        ("comm",         ctypes.c_char * 16),
    ]


def build_bpf(pid_filter=None, comm_filter=None):
    prog = BPF_PROGRAM

    if pid_filter:
        prog = prog.replace("FILTER_PID",
            f"if (evt.pid != {pid_filter}) return 0;")
    else:
        prog = prog.replace("FILTER_PID", "")

    if comm_filter:
        # BPF tak bisa strcmp langsung — filter di userspace lebih aman
        prog = prog.replace("FILTER_COMM", "")
    else:
        prog = prog.replace("FILTER_COMM", "")

    return BPF(text=prog)


def main():
    parser = argparse.ArgumentParser(description="eBPF Syscall Capturer")
    parser.add_argument("--pid",  type=int,  default=None, help="Filter by PID")
    parser.add_argument("--comm", type=str,  default=None, help="Filter by process name")
    parser.add_argument("--out",  type=str,  default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    print(f"[*] Loading eBPF program...", file=sys.stderr)
    b = build_bpf(pid_filter=args.pid, comm_filter=args.comm)
    print(f"[*] Capturing syscalls. Ctrl+C to stop.", file=sys.stderr)
    if args.pid:
        print(f"[*] Filter: PID={args.pid}", file=sys.stderr)
    if args.comm:
        print(f"[*] Filter: comm={args.comm}", file=sys.stderr)

    outfile = open(args.out, "w") if args.out else sys.stdout

    # Header CSV
    print("timestamp_ns,pid,tgid,syscall_nr,comm", file=outfile)

    event_count = [0]

    def handle_event(cpu, data, size):
        evt = ctypes.cast(data, ctypes.POINTER(Event)).contents
        comm = evt.comm.decode("utf-8", errors="replace").strip("\x00")

        # Filter comm di userspace
        if args.comm and comm != args.comm:
            return

        print(f"{evt.timestamp_ns},{evt.pid},{evt.tgid},{evt.syscall_nr},{comm}",
              file=outfile)
        event_count[0] += 1
        if event_count[0] % 1000 == 0:
            print(f"[*] {event_count[0]} events captured", file=sys.stderr)

    b["events"].open_perf_buffer(handle_event, page_cnt=256)

    def on_exit(sig, frame):
        print(f"\n[*] Total events: {event_count[0]}", file=sys.stderr)
        if args.out:
            outfile.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    while True:
        b.perf_buffer_poll(timeout=100)


if __name__ == "__main__":
    main()
