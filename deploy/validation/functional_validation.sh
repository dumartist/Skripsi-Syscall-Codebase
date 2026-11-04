#!/usr/bin/env bash
# Check whether the workload still succeeds with syscall injection enabled.
#
# Menjalankan serangan NYATA (Hydra SSH brute-force, kelas = ADFA Hydra_SSH)
# dua kali: (A) baseline, (B) dengan shim no-op injection. Membuktikan:
#   1. Serangan TETAP berhasil dengan no-op disuntik  -> ΦF (functional preservation)
#   2. Trace shimmed berisi no-op nyata + lebih banyak syscall -> ΦO + overhead (ΦM cost)
# via strace -c (histogram syscall) untuk kedua run.
#
# Aman & terisolasi: target = user 'victim' lokal yang kita buat sendiri,
# wordlist kecil berisi password asli. Tidak ada korban eksternal.
#
# Jalankan di VM:  sudo bash deploy/validation/functional_validation.sh
set -u

SHIM=/tmp/noop_shim.so
SRC="$(dirname "$0")/noop_shim.c"
WL=/tmp/fv_wordlist.txt
BASE=/tmp/fv_strace_base.txt
SHIMMED=/tmp/fv_strace_shim.txt
RESULT=/tmp/functional_validation_result.json
VICTIM=victim
REAL_PASS=hunter2
NOOP_PER_CALL="${NOOP_PER_CALL:-5}"

echo "=== [0/6] Dependencies ==="
apt-get install -y hydra strace gcc >/dev/null 2>&1 || {
    echo "WARN: apt install gagal; pastikan hydra/strace/gcc ada."; }

echo "=== [1/6] Compile shim ==="
gcc -shared -fPIC -o "$SHIM" "$SRC" -ldl && echo "  built $SHIM" || { echo "FAIL build"; exit 1; }

echo "=== [2/6] Setup target user '$VICTIM' (terisolasi) ==="
id "$VICTIM" >/dev/null 2>&1 || useradd -m "$VICTIM"
echo "$VICTIM:$REAL_PASS" | chpasswd
printf 'admin\npassword\n123456\n%s\nletmein\nroot\n' "$REAL_PASS" > "$WL"
echo "  wordlist: $(tr '\n' ' ' < "$WL")"

run_hydra() {  # $1=outfile  $2=preload(opsional)
    local out="$1" preload="$2"
    if [ -n "$preload" ]; then
        strace -f -c -o "$out" -E LD_PRELOAD="$SHIM" \
            hydra -l "$VICTIM" -P "$WL" -t 1 -W 1 ssh://127.0.0.1 2>&1
    else
        strace -f -c -o "$out" \
            hydra -l "$VICTIM" -P "$WL" -t 1 -W 1 ssh://127.0.0.1 2>&1
    fi
}

echo "=== [3/6] Baseline (tanpa shim) ==="
BASE_OUT="$(run_hydra "$BASE" "")"
echo "$BASE_OUT" | grep -i "password:" && BASE_OK=1 || BASE_OK=0

echo "=== [4/6] Shimmed (no-op injection, NOOP_PER_CALL=$NOOP_PER_CALL) ==="
SHIM_OUT="$(NOOP_PER_CALL="$NOOP_PER_CALL" run_hydra "$SHIMMED" "$SHIM")"
echo "$SHIM_OUT" | grep -i "password:" && SHIM_OK=1 || SHIM_OK=0

echo "=== [5/6] Parse strace histogram ==="
# Ambil count syscall tertentu dari ringkasan strace -c (kolom: %time secs usecs calls errors syscall)
getcount() { awk -v s="$2" '$NF==s {print $4}' "$1" | tail -1; }
gettotal() { awk '/total/{print $4}' "$1" | tail -1; }

BASE_TOTAL=$(gettotal "$BASE");  SHIM_TOTAL=$(gettotal "$SHIMMED")
for sc in getpid getuid gettimeofday; do
    eval "BASE_$sc=$(getcount "$BASE" "$sc" 2>/dev/null || echo 0)"
    eval "SHIM_$sc=$(getcount "$SHIMMED" "$sc" 2>/dev/null || echo 0)"
done
BASE_getpid=${BASE_getpid:-0}; SHIM_getpid=${SHIM_getpid:-0}
BASE_getuid=${BASE_getuid:-0}; SHIM_getuid=${SHIM_getuid:-0}
BASE_gettimeofday=${BASE_gettimeofday:-0}; SHIM_gettimeofday=${SHIM_gettimeofday:-0}
BASE_TOTAL=${BASE_TOTAL:-0}; SHIM_TOTAL=${SHIM_TOTAL:-0}

echo "=== [6/6] Verdict + tulis JSON ==="
PHI_F=$([ "$BASE_OK" = 1 ] && [ "$SHIM_OK" = 1 ] && echo true || echo false)

cat > "$RESULT" <<JSON
{
  "attack": "hydra_ssh_bruteforce",
  "adfa_class": "Hydra_SSH",
  "noop_per_call": $NOOP_PER_CALL,
  "baseline": {
    "password_found": $([ "$BASE_OK" = 1 ] && echo true || echo false),
    "total_syscalls": ${BASE_TOTAL:-0},
    "getpid": ${BASE_getpid:-0}, "getuid": ${BASE_getuid:-0}, "gettimeofday": ${BASE_gettimeofday:-0}
  },
  "shimmed": {
    "password_found": $([ "$SHIM_OK" = 1 ] && echo true || echo false),
    "total_syscalls": ${SHIM_TOTAL:-0},
    "getpid": ${SHIM_getpid:-0}, "getuid": ${SHIM_getuid:-0}, "gettimeofday": ${SHIM_gettimeofday:-0}
  },
  "phi_F_functional_preserved": $PHI_F
}
JSON

echo
echo "------------------------------------------------------------"
echo "HASIL VALIDASI FUNGSIONAL (ΦF)"
echo "  Baseline  password ditemukan : $([ "$BASE_OK" = 1 ] && echo YA || echo TIDAK)  (total syscalls=$BASE_TOTAL)"
echo "  Shimmed   password ditemukan : $([ "$SHIM_OK" = 1 ] && echo YA || echo TIDAK)  (total syscalls=$SHIM_TOTAL)"
echo "  no-op getpid       base=$BASE_getpid  shim=$SHIM_getpid"
echo "  no-op getuid       base=$BASE_getuid  shim=$SHIM_getuid"
echo "  no-op gettimeofday base=$BASE_gettimeofday  shim=$SHIM_gettimeofday"
echo "  ΦF (fungsi terjaga di bawah injeksi no-op): $PHI_F"
echo "  JSON: $RESULT"
echo "------------------------------------------------------------"

echo "=== cleanup user '$VICTIM' ==="
userdel -r "$VICTIM" 2>/dev/null && echo "  victim dihapus"
