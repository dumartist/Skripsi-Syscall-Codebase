#!/usr/bin/env python3
"""
Fase 2 + 3: HIPS Daemon — Online Inference + Enforcement
=========================================================
Pipeline:
  eBPF (sys_enter) → per-PID sliding window → ONNX inference
  → multi-tier policy π(r) → enforce (log/SIGSTOP/SIGKILL)

Usage (root required for eBPF + enforcement):
  sudo "$(command -v python3)" deploy/runtime/hips_daemon.py --model models/lidds_Bruteforce_CWE-307.onnx \
       --vocab /tmp/lidds_vocab.json

Config flags:
  --window   W (default 100) — window size
  --stride   S (default 50)  — stride
  --tau1     float (default 0.30) — Normal/Alert boundary
  --tau2     float (default 0.60) — Alert/Quarantine boundary
  --tau3     float (default 0.85) — Quarantine/Terminate boundary
  --enforce  flag — aktifkan enforcement nyata (default: dry-run log saja)
  --pid      int  — hanya monitor PID tertentu (optional)
  --comm     str  — hanya monitor nama proses tertentu (optional)
"""

import argparse
import collections
import ctypes
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from bcc import BPF

# ---------------------------------------------------------------------------
_handlers = [logging.StreamHandler(sys.stdout)]
# File log is best-effort: a per-user default avoids the fs.protected_regular
# block on other-owned files in /tmp, and any open error degrades to stdout-only
# rather than crashing a security daemon.
_logfile = os.environ.get("HIPS_LOG", f"/tmp/hips_daemon_{os.getuid()}.log")
try:
    _handlers.append(logging.FileHandler(_logfile))
except OSError as _e:
    print(f"[warn] file log disabled ({_e}); stdout only", file=sys.stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_handlers,
)
log = logging.getLogger("hips")

# ---------------------------------------------------------------------------
# eBPF program
# ---------------------------------------------------------------------------
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
    u64 id = bpf_get_current_pid_tgid();
    evt.pid  = id & 0xFFFFFFFF;
    evt.tgid = id >> 32;
    evt.timestamp_ns = bpf_ktime_get_ns();
    evt.syscall_nr   = args->id;
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    FILTER_PLACEHOLDER
    events.perf_submit(args, &evt, sizeof(evt));
    return 0;
}
"""

class Event(ctypes.Structure):
    _fields_ = [
        ("pid",          ctypes.c_uint32),
        ("tgid",         ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
        ("syscall_nr",   ctypes.c_uint64),
        ("comm",         ctypes.c_char * 16),
    ]

# ---------------------------------------------------------------------------
# Vocabulary: syscall_nr (x86_64) → token index
# vocabulary.json keys = syscall ID string, values = token index
# ---------------------------------------------------------------------------
def load_vocab(path):
    with open(path) as f:
        raw = json.load(f)
    # raw: {"<PAD>":0, "<UNK>":1, "1":2, "3":3, ...}
    # Kita butuh map: int(syscall_nr) → token_index
    nr2tok = {}
    for k, v in raw.items():
        try:
            nr2tok[int(k)] = int(v)
        except ValueError:
            pass  # skip "<PAD>", "<UNK>"
    PAD = raw.get("<PAD>", 0)
    UNK = raw.get("<UNK>", 1)
    return nr2tok, int(PAD), int(UNK)

# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------
def load_model(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    log.info(f"Model loaded: {path}")
    return sess

def infer(sess, window_tokens, pad_token, window_size=100):
    """window_tokens: list[int] (panjang <= window_size) → risk score float"""
    arr = list(window_tokens)[-window_size:]
    arr = arr + [pad_token] * max(0, window_size - len(arr))
    # Model expect 130 tokens (max_len)
    arr = arr + [pad_token] * max(0, 130 - len(arr))
    arr = arr[:130]
    x = np.array([arr], dtype=np.int64)
    risk = sess.run(["risk"], {"tokens": x})[0][0]
    return float(risk)

# ---------------------------------------------------------------------------
# Multi-tier policy π(r)
# ---------------------------------------------------------------------------
ACTIONS = {0: "NORMAL", 1: "ALERT", 2: "QUARANTINE", 3: "TERMINATE"}

def policy(risk, tau1=0.30, tau2=0.60, tau3=0.85):
    if risk < tau1:   return 0  # NORMAL
    elif risk < tau2: return 1  # ALERT
    elif risk < tau3: return 2  # QUARANTINE
    else:             return 3  # TERMINATE

def enforce(action, pid, comm, risk, dry_run=True):
    action_name = ACTIONS[action]
    msg = f"PID={pid} comm={comm} risk={risk:.4f} action={action_name}"

    if action == 0:
        return  # NORMAL — tidak log

    if action == 1:  # ALERT
        log.warning(f"[ALERT]      {msg}")

    elif action == 2:  # QUARANTINE
        log.warning(f"[QUARANTINE] {msg}")
        if not dry_run:
            try:
                os.kill(pid, signal.SIGSTOP)
                log.warning(f"  → SIGSTOP sent to PID {pid}")
            except ProcessLookupError:
                log.info(f"  → PID {pid} already gone")
            except PermissionError:
                log.error(f"  → Permission denied for PID {pid}")

    elif action == 3:  # TERMINATE
        log.error(f"[TERMINATE]  {msg}")
        if not dry_run:
            try:
                os.kill(pid, signal.SIGKILL)
                log.error(f"  → SIGKILL sent to PID {pid}")
            except ProcessLookupError:
                log.info(f"  → PID {pid} already gone")
            except PermissionError:
                log.error(f"  → Permission denied for PID {pid}")

# ---------------------------------------------------------------------------
# Per-PID state
# ---------------------------------------------------------------------------
class PIDState:
    def __init__(self, comm, window_size=100, stride=50):
        self.comm = comm
        self.buffer = collections.deque()  # semua token sejak mulai
        self.window_size = window_size
        self.stride = stride
        self.n_syscalls = 0
        self.last_risk = 0.0
        self.windows_processed = 0

    def add(self, token):
        self.buffer.append(token)
        self.n_syscalls += 1

    def should_infer(self):
        """Trigger inference setiap stride syscall setelah window pertama."""
        if len(self.buffer) < self.window_size:
            return False
        return (self.n_syscalls - self.window_size) % self.stride == 0

    def get_window(self):
        return list(self.buffer)[-self.window_size:]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=str(Path(__file__).resolve().parents[2] / "models/a4_full.onnx"))
    parser.add_argument("--vocab",   default=str(Path(__file__).resolve().parents[2] / "prepared_data/vocabulary.json"))
    parser.add_argument("--window",  type=int,   default=100)
    parser.add_argument("--stride",  type=int,   default=50)
    parser.add_argument("--tau1",    type=float, default=0.30)
    parser.add_argument("--tau2",    type=float, default=0.60)
    parser.add_argument("--tau3",    type=float, default=0.85)
    parser.add_argument("--enforce", action="store_true",
                        help="Enforcement nyata (SIGSTOP/SIGKILL). Default: dry-run.")
    parser.add_argument("--pid",     type=int,   default=None)
    parser.add_argument("--comm",    type=str,   default=None)
    args = parser.parse_args()

    dry_run = not args.enforce
    mode = "DRY-RUN (log only)" if dry_run else "ENFORCEMENT ACTIVE"
    log.info(f"HIPS Daemon starting | mode={mode}")
    log.info(f"Policy: tau1={args.tau1} tau2={args.tau2} tau3={args.tau3}")
    log.info(f"Window: W={args.window} S={args.stride}")

    # Load vocab & model
    nr2tok, PAD, UNK = load_vocab(args.vocab)
    sess = load_model(args.model)

    # Per-PID state
    pid_states = {}

    # Statistik
    stats = {
        "total_syscalls": 0,
        "total_inferences": 0,
        "alerts": 0, "quarantines": 0, "terminates": 0,
        "start_time": time.time(),
    }

    # eBPF setup
    bpf_prog = BPF_PROGRAM
    if args.pid:
        bpf_prog = bpf_prog.replace("FILTER_PLACEHOLDER",
            f"if (evt.pid != {args.pid}) return 0;")
    else:
        bpf_prog = bpf_prog.replace("FILTER_PLACEHOLDER", "")

    log.info("Loading eBPF program...")
    b = BPF(text=bpf_prog)
    log.info("eBPF loaded. Monitoring syscalls...")

    # Latency tracking untuk pengukuran end-to-end.
    latencies_us = []
    comm_scores = {}   # comm -> [n_windows, max_risk]; printed at exit (calibration)
    MY_PID = os.getpid()

    def handle_event(cpu, data, size):
        t_receive = time.perf_counter()
        evt = ctypes.cast(data, ctypes.POINTER(Event)).contents
        pid  = evt.pid
        comm = evt.comm.decode("utf-8", errors="replace").strip("\x00")
        nr   = int(evt.syscall_nr)

        # Filter comm userspace; never score the daemon's own syscalls
        if pid == MY_PID or (args.comm and comm != args.comm):
            return

        stats["total_syscalls"] += 1

        # Token mapping
        token = nr2tok.get(nr, UNK)

        # Per-PID state
        if pid not in pid_states:
            pid_states[pid] = PIDState(comm, args.window, args.stride)
        state = pid_states[pid]
        state.add(token)

        if not state.should_infer():
            return

        # Inference
        window = state.get_window()
        risk = infer(sess, window, PAD, args.window)
        state.last_risk = risk
        cs = comm_scores.setdefault(comm, [0, 0.0])   # [n_windows, max_risk]
        cs[0] += 1; cs[1] = max(cs[1], risk)
        state.windows_processed += 1
        stats["total_inferences"] += 1

        # Latency end-to-end (capture → decision)
        t_decision = time.perf_counter()
        lat_us = (t_decision - t_receive) * 1e6
        latencies_us.append(lat_us)

        # Policy
        action = policy(risk, args.tau1, args.tau2, args.tau3)

        # Stats
        if action == 1: stats["alerts"] += 1
        elif action == 2: stats["quarantines"] += 1
        elif action == 3: stats["terminates"] += 1

        # Enforce
        enforce(action, pid, comm, risk, dry_run=dry_run)

        # Periodic summary (setiap 100 inference)
        if stats["total_inferences"] % 100 == 0:
            elapsed = time.time() - stats["start_time"]
            lat_mean = np.mean(latencies_us[-100:]) if latencies_us else 0
            log.info(
                f"[STATS] t={elapsed:.0f}s | "
                f"syscalls={stats['total_syscalls']} | "
                f"inferences={stats['total_inferences']} | "
                f"alerts={stats['alerts']} quarantines={stats['quarantines']} "
                f"terminates={stats['terminates']} | "
                f"latency_e2e={lat_mean:.1f}us"
            )

    b["events"].open_perf_buffer(handle_event, page_cnt=512)

    # The signal handler only sets a flag. Calling sys.exit() inside it raised
    # SystemExit within the BCC ctypes callback, where Python swallows it, so
    # Ctrl+C never stopped the loop.
    stop = {"flag": False}

    def request_stop(sig, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT,  request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while not stop["flag"]:
        b.perf_buffer_poll(timeout=100)

    def summary():
        elapsed = time.time() - stats["start_time"]
        log.info("=" * 60)
        log.info("HIPS Daemon stopping.")
        log.info(f"  Runtime:      {elapsed:.1f}s")
        log.info(f"  Syscalls:     {stats['total_syscalls']}")
        log.info(f"  Inferences:   {stats['total_inferences']}")
        log.info(f"  Alerts:       {stats['alerts']}")
        log.info(f"  Quarantines:  {stats['quarantines']}")
        log.info(f"  Terminates:   {stats['terminates']}")
        if latencies_us:
            log.info(f"  Latency e2e:  mean={np.mean(latencies_us):.1f}us "
                     f"p50={np.percentile(latencies_us,50):.1f}us "
                     f"p99={np.percentile(latencies_us,99):.1f}us")
        if comm_scores:
            log.info("  Max risk per process name (windows, max):")
            for c, (n, mx) in sorted(comm_scores.items(), key=lambda kv: -kv[1][1])[:25]:
                log.info(f"    {c:<16} n={n:<6} max={mx:.4f}")
        log.info("=" * 60)

    summary()


if __name__ == "__main__":
    main()
