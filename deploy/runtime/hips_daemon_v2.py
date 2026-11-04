#!/usr/bin/env python3
"""
Fase 3: HIPS Daemon v2 — Enforcement nyata + Whitelist
=======================================================
Perubahan dari v1:
- Whitelist proses normal (tidak akan di-flag)
- Flag --enforce mengaktifkan SIGSTOP/SIGKILL nyata
- Exclusion diri sendiri (daemon tidak monitor dirinya)
- Log lebih bersih

Usage:
  # Dry-run (aman, log saja)
  sudo PYTHONPATH=/usr/lib/python3/dist-packages:/home/arthur/venv_ubuntu/lib/python3.12/site-packages \
    python3 deploy/runtime/hips_daemon_v2.py \
    --model /home/arthur/a4_full.onnx \
    --vocab /home/arthur/vocabulary.json

  # Enforcement nyata (blokir proses)
  sudo PYTHONPATH=... python3 deploy/runtime/hips_daemon_v2.py \
    --model /home/arthur/a4_full.onnx \
    --vocab /home/arthur/vocabulary.json \
    --enforce
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
# Whitelist — proses normal yang TIDAK dimonitor
# Tambah nama proses di sini jika ada false positive
# ---------------------------------------------------------------------------
WHITELIST_COMM = {
    # System daemons
    "systemd", "systemd-journal", "systemd-udevd", "systemd-resolve",
    "systemd-network", "systemd-logind", "systemd-timesyn",
    # Login & shell
    "bash", "sh", "login", "sshd", "su",
    # Kernel threads (comm biasanya dalam [...])
    "kthreadd", "kworker", "ksoftirqd", "migration", "rcu_sched",
    "cpuhp", "idle_inject", "irq",
    # System services
    "multipathd", "dbus-daemon", "cron", "atd", "rsyslogd",
    "networkd", "resolved", "accounts-daemon",
    # VMware tools
    "vmtoolsd", "vmware-vmblock",
    # Monitoring tools
    "ps", "top", "htop", "awk", "grep", "head", "tail",
    # Python & pip (daemon itu sendiri)
    "python3", "pip", "pip3",
    # SSH
    "sshd", "ssh",
    # Package manager
    "apt", "apt-get", "dpkg",
    # sudo (proses admin)
    "sudo",
}

# Whitelist PID — tambah PID proses penting saat runtime
WHITELIST_PID = {
    1,   # systemd
    2,   # kthreadd
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/hips_v2.log"),
    ],
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

// RAW_TRACEPOINT_PROBE: baca syscall id dari ctx->args[1], TIDAK bergantung
// parsing format tracepoint (kebal error 'incomplete struct' di sebagian VM).
RAW_TRACEPOINT_PROBE(sys_enter) {
    struct event_t evt = {};
    u64 id = bpf_get_current_pid_tgid();
    evt.pid  = id & 0xFFFFFFFF;
    evt.tgid = id >> 32;
    evt.timestamp_ns = bpf_ktime_get_ns();
    evt.syscall_nr   = (u64)ctx->args[1];   // args[0]=pt_regs, args[1]=syscall nr
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    events.perf_submit(ctx, &evt, sizeof(evt));
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
# Vocab & Model
# ---------------------------------------------------------------------------
def load_vocab(path):
    with open(path) as f:
        raw = json.load(f)
    nr2tok = {}
    for k, v in raw.items():
        try:
            nr2tok[int(k)] = int(v)
        except ValueError:
            pass
    PAD = int(raw.get("<PAD>", 0))
    UNK = int(raw.get("<UNK>", 1))
    return nr2tok, PAD, UNK

def load_model(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    log.info(f"Model loaded: {path}")
    return sess

def infer(sess, tokens, pad, maxlen=130):
    arr = list(tokens)[-100:]
    arr = arr + [pad] * max(0, 100 - len(arr))
    arr = arr + [pad] * max(0, maxlen - len(arr))
    arr = arr[:maxlen]
    x = np.array([arr], dtype=np.int64)
    return float(sess.run(["risk"], {"tokens": x})[0][0])

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
ACTIONS = {0: "NORMAL", 1: "ALERT", 2: "QUARANTINE", 3: "TERMINATE"}

def policy(risk, tau1, tau2, tau3):
    if risk < tau1:   return 0
    elif risk < tau2: return 1
    elif risk < tau3: return 2
    else:             return 3

# Proses yang sudah pernah di-terminate (jangan spam)
terminated_pids = set()
quarantined_pids = set()

def enforce(action, pid, comm, risk, dry_run):
    if action == 0:
        return

    name = ACTIONS[action]
    msg = f"PID={pid:<6} comm={comm:<15} risk={risk:.4f} -> {name}"

    if action == 1:
        log.warning(f"[ALERT]      {msg}")

    elif action == 2:
        if pid in quarantined_pids:
            return  # sudah pernah, skip spam
        log.warning(f"[QUARANTINE] {msg}")
        quarantined_pids.add(pid)
        if not dry_run:
            try:
                os.kill(pid, signal.SIGSTOP)
                log.warning(f"  >>> SIGSTOP -> PID {pid} ({comm})")
            except (ProcessLookupError, PermissionError) as e:
                log.debug(f"  SIGSTOP failed: {e}")

    elif action == 3:
        if pid in terminated_pids:
            return
        log.error(f"[TERMINATE]  {msg}")
        terminated_pids.add(pid)
        if not dry_run:
            try:
                os.kill(pid, signal.SIGKILL)
                log.error(f"  >>> SIGKILL -> PID {pid} ({comm})")
            except (ProcessLookupError, PermissionError) as e:
                log.debug(f"  SIGKILL failed: {e}")

# ---------------------------------------------------------------------------
# Per-PID state
# ---------------------------------------------------------------------------
class PIDState:
    def __init__(self, comm, W=100, S=50):
        self.comm = comm
        self.buf  = collections.deque(maxlen=W * 4)
        self.W, self.S = W, S
        self.n = 0
        self.last_risk = 0.0

    def add(self, tok):
        self.buf.append(tok)
        self.n += 1

    def should_infer(self):
        return self.n >= self.W and (self.n - self.W) % self.S == 0

    def window(self):
        return list(self.buf)[-self.W:]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="/home/arthur/a4_full.onnx")
    parser.add_argument("--vocab",   default="/home/arthur/vocabulary.json")
    parser.add_argument("--tau1",    type=float, default=0.30)
    parser.add_argument("--tau2",    type=float, default=0.60)
    parser.add_argument("--tau3",    type=float, default=0.85)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    dry_run = not args.enforce
    my_pid  = os.getpid()
    WHITELIST_PID.add(my_pid)

    log.info(f"HIPS Daemon v2 | mode={'ENFORCE' if args.enforce else 'DRY-RUN'}")
    log.info(f"Policy tau: {args.tau1}/{args.tau2}/{args.tau3}")
    log.info(f"My PID: {my_pid} (excluded)")
    log.info(f"Whitelist comms: {len(WHITELIST_COMM)} entries")

    nr2tok, PAD, UNK = load_vocab(args.vocab)
    sess = load_model(args.model)

    pid_states = {}
    stats = {"syscalls": 0, "inferences": 0,
             "alerts": 0, "quarantines": 0, "terminates": 0,
             "skipped_whitelist": 0}
    latencies = []
    t_start = time.time()

    b = BPF(text=BPF_PROGRAM)
    log.info("eBPF loaded. Monitoring...")

    def handle_event(cpu, data, size):
        t0 = time.perf_counter()
        evt  = ctypes.cast(data, ctypes.POINTER(Event)).contents
        pid  = evt.pid
        comm = evt.comm.decode("utf-8", errors="replace").strip("\x00")
        nr   = int(evt.syscall_nr)

        # Whitelist check
        if pid in WHITELIST_PID:
            return
        comm_base = comm.split("/")[-1][:15]
        if any(comm_base.startswith(w) for w in WHITELIST_COMM):
            stats["skipped_whitelist"] += 1
            return

        stats["syscalls"] += 1
        token = nr2tok.get(nr, UNK)

        if pid not in pid_states:
            pid_states[pid] = PIDState(comm)
        state = pid_states[pid]
        state.add(token)

        if not state.should_infer():
            return

        risk = infer(sess, state.window(), PAD)
        state.last_risk = risk
        stats["inferences"] += 1

        lat = (time.perf_counter() - t0) * 1e6
        latencies.append(lat)

        action = policy(risk, args.tau1, args.tau2, args.tau3)
        if action == 1: stats["alerts"] += 1
        elif action == 2: stats["quarantines"] += 1
        elif action == 3: stats["terminates"] += 1

        enforce(action, pid, comm, risk, dry_run)

        # Progress setiap 50 inference
        if stats["inferences"] % 50 == 0:
            t = time.time() - t_start
            lat_m = np.mean(latencies[-50:]) if latencies else 0
            log.info(
                f"[STATS] t={t:.0f}s syscalls={stats['syscalls']} "
                f"infer={stats['inferences']} "
                f"A={stats['alerts']} Q={stats['quarantines']} "
                f"T={stats['terminates']} "
                f"lat={lat_m:.0f}us skip={stats['skipped_whitelist']}"
            )

    b["events"].open_perf_buffer(handle_event, page_cnt=512)

    def on_exit(sig, frame):
        t = time.time() - t_start
        log.info("=" * 60)
        log.info(f"Stopped. Runtime={t:.1f}s")
        log.info(f"  Syscalls monitored : {stats['syscalls']}")
        log.info(f"  Skipped (whitelist): {stats['skipped_whitelist']}")
        log.info(f"  Inferences         : {stats['inferences']}")
        log.info(f"  Alerts             : {stats['alerts']}")
        log.info(f"  Quarantines        : {stats['quarantines']}")
        log.info(f"  Terminates         : {stats['terminates']}")
        if latencies:
            log.info(f"  Latency e2e mean   : {np.mean(latencies):.1f} us")
            log.info(f"  Latency e2e p50    : {np.percentile(latencies,50):.1f} us")
            log.info(f"  Latency e2e p99    : {np.percentile(latencies,99):.1f} us")
        log.info("=" * 60)
        sys.exit(0)

    signal.signal(signal.SIGINT,  on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    while True:
        b.perf_buffer_poll(timeout=100)

if __name__ == "__main__":
    main()
