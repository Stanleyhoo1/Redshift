#!/usr/bin/env bash
# Stop Redshift: abort the running mission through the backend, stop the backend / Vite, and reap any
# simulator or autonomy process left behind. All patterns are anchored so unrelated shells are never hit.
#
#   ./stop_all.sh [--port 8000] [--quiet] [--leftovers-only]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000 QUIET=0 LEFTOVERS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift ;;
    --quiet) QUIET=1 ;;
    --leftovers-only) LEFTOVERS=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
say() { (( QUIET )) || printf '\033[1;35m[redshift]\033[0m %s\n' "$*"; }

SIM_PATTERNS=(
  "^ros2 launch exoswarm_sim"
  "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch exoswarm_sim"
  "^gz sim"
  "^/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge"
  "^/usr/bin/python3 $ROOT/ros2_ws/install/"
)
APP_PATTERNS=(
  "^python3 -m uvicorn backend_bridge.app:app"
  "^python3 -u $ROOT/web_ui/dev/fake_backend.py"
  "^node $ROOT/web_ui/node_modules/.bin/vite"
)

alive() { for p in "$@"; do pgrep -f "$p" >/dev/null && return 0; done; return 1; }

if (( ! LEFTOVERS )); then
  if command -v curl >/dev/null && curl -s -m 2 -o /dev/null -X POST "http://127.0.0.1:$PORT/api/mission/abort"; then
    say "mission aborted via backend"
  fi
  for p in "${APP_PATTERNS[@]}"; do pkill -INT -f "$p" 2>/dev/null; done
fi

for p in "${SIM_PATTERNS[@]:0:2}"; do pkill -INT -f "$p" 2>/dev/null; done
for _ in $(seq 1 30); do alive "${SIM_PATTERNS[@]}" || break; sleep 0.25; done
for p in "${SIM_PATTERNS[@]}"; do pkill -KILL -f "$p" 2>/dev/null; done
if (( ! LEFTOVERS )); then
  for _ in $(seq 1 20); do alive "${APP_PATTERNS[@]}" || break; sleep 0.25; done
  for p in "${APP_PATTERNS[@]}"; do pkill -KILL -f "$p" 2>/dev/null; done
fi
if alive "${SIM_PATTERNS[@]}"; then say "warning: some simulator processes survived"; exit 1; fi
say "all Redshift processes stopped"
