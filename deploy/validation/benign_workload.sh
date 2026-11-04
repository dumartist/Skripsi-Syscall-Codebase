#!/usr/bin/env bash
# Benign workload for HIPS threshold calibration.
# Simulates ordinary interactive/admin/dev use of the VM so the daemon can record
# the risk scores of NORMAL processes. Run as a normal user (no sudo). Everything
# it writes stays in a private temp dir that is removed at the end. A local HTTP
# server on 127.0.0.1 is started and stopped by the script itself.
#
# Usage: bash deploy/validation/benign_workload.sh [seconds]   (default 120)
set -u
DUR=${1:-120}
WD=$(mktemp -d /tmp/benign_wl.XXXXXX)
PORT=$((20000 + RANDOM % 20000))
cleanup() { kill "${HTTP_PID:-0}" 2>/dev/null; rm -rf "$WD"; }
trap cleanup EXIT INT TERM
cd "$WD" || exit 1

# a small local web server = benign network/server activity
python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
HTTP_PID=$!
sleep 1

end=$((SECONDS + DUR)); round=0
while [ $SECONDS -lt $end ]; do
  round=$((round + 1))
  # --- shell / file browsing
  ls -la /etc >/dev/null; ls -R /usr/share/doc 2>/dev/null | head -200 >/dev/null
  cat /etc/os-release /etc/hostname >/dev/null
  grep -ri "ssh" /etc/ssh/ssh_config 2>/dev/null >/dev/null
  find /usr/share/doc -maxdepth 2 -name "*.gz" 2>/dev/null | head -50 >/dev/null
  # --- system status
  ps aux >/dev/null; df -h >/dev/null; free -m >/dev/null; uptime >/dev/null
  ip addr >/dev/null 2>&1; who >/dev/null
  dpkg -l 2>/dev/null | head -100 >/dev/null; apt-cache policy bash >/dev/null 2>&1
  # --- text processing / editing
  cp /etc/services services.txt
  sort -u services.txt | uniq -c | sort -rn | head -20 > top.txt
  sed -i 's/tcp/TCP/g' services.txt; wc -l services.txt >/dev/null
  awk '{print $1}' services.txt | sort | uniq | head -50 >/dev/null
  # --- archives and checksums
  tar czf docs_$round.tgz -C /usr/share/doc bash coreutils 2>/dev/null
  gzip -t docs_$round.tgz 2>/dev/null; sha256sum docs_$round.tgz >/dev/null
  rm -f docs_$round.tgz
  # --- a bit of Python "development"
  python3 - <<'PY' >/dev/null
import json, os, statistics, hashlib
d = {f"k{i}": i * i for i in range(2000)}
s = json.dumps(d); json.loads(s)
statistics.mean(d.values()); hashlib.sha1(s.encode()).hexdigest()
os.listdir("/etc")
PY
  # --- local web requests
  curl -s "http://127.0.0.1:$PORT/" >/dev/null 2>&1 || python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/').read()" 2>/dev/null
  # --- version control, if git is installed
  if command -v git >/dev/null; then
    [ -d repo ] || { git init -q repo; git -C repo config user.email t@t; git -C repo config user.name t; }
    date > repo/notes.txt; git -C repo add -A; git -C repo commit -qm "round $round" >/dev/null 2>&1
    git -C repo log --oneline | head -5 >/dev/null
  fi
  echo "[benign] round $round done ($((end - SECONDS))s left)"
  sleep 0.5
done
echo "[benign] finished: $round rounds in ${DUR}s"
