#!/usr/bin/env bash
# Redshift: one-command bring-up of Mission Control (FastAPI + rclpy backend serving the built web UI).
# Missions themselves (Gazebo or mock world + autonomy) are launched from the UI's DEPLOY FLEET button.
#
#   ./start_all.sh                 build what is stale, serve http://localhost:8000
#   ./start_all.sh --dev           also run the Vite dev server (hot reload) on :5173
#   ./start_all.sh --rebuild       force colcon + web builds
#   ./start_all.sh --ui-demo       no ROS needed: serve the UI against the scripted fake backend
#   ./start_all.sh --port 8080 --host 0.0.0.0
#
# Ctrl+C stops the backend, which aborts any running mission and reaps every simulator process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000 HOST=127.0.0.1 DEV=0 REBUILD=0 UI_DEMO=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) DEV=1 ;;
    --rebuild) REBUILD=1 ;;
    --ui-demo) UI_DEMO=1 ;;
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1;35m[redshift]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[redshift] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- Node (nvm first: WSL may expose Windows npm)
pick_node() {
  local nvm_dir="${NVM_DIR:-$HOME/.nvm}" best=""
  if [[ -d "$nvm_dir/versions/node" ]]; then
    best="$(ls -d "$nvm_dir"/versions/node/v* 2>/dev/null | sort -V | tail -1)"
  fi
  if [[ -n "$best" && -x "$best/bin/node" ]]; then
    export PATH="$best/bin:$PATH"
  fi
  command -v node >/dev/null || fail "Node.js not found (install nvm + Node ≥ 20)"
  case "$(command -v node)" in /mnt/*) fail "only a Windows node.exe is on PATH; install Node inside WSL (nvm)";; esac
  local major; major="$(node -p 'process.versions.node.split(".")[0]')"
  (( major >= 20 )) || fail "Node $(node -v) is too old; need ≥ 20"
}

newer_than() {  # newer_than <stamp-file> <paths...>: any source file newer than the stamp (or no stamp)?
  local stamp="$1"; shift
  [[ -e "$stamp" ]] || return 0
  [[ -n "$(find "$@" -type f -newer "$stamp" -not -path '*/__pycache__/*' -not -path '*/node_modules/*' \
             -not -name '*.pyc' 2>/dev/null | head -1)" ]]
}

build_web() {
  pick_node
  cd "$ROOT/web_ui"
  if [[ ! -d node_modules || package-lock.json -nt node_modules/.package-lock.json ]]; then
    say "installing web dependencies (npm ci)"
    npm ci --no-audit --no-fund --loglevel=error
  fi
  if (( REBUILD )) || newer_than dist/index.html src index.html package.json vite.config.ts tsconfig.json public; then
    say "building Mission Control UI"
    npm run build --silent >/dev/null
  else
    say "web UI up to date"
  fi
  cd "$ROOT"
}

check_port() {
  if command -v ss >/dev/null && ss -ltn "sport = :$1" | grep -q LISTEN; then
    fail "port $1 is already in use ($(ss -ltnp "sport = :$1" 2>/dev/null | grep -o 'users:.*' | head -1)); try ./stop_all.sh or --port"
  fi
}

PIDS=()
cleanup() {
  local why="$1" p alive
  set +e                                         # every step below must run, whatever an earlier one returned
  trap '' INT TERM                               # a second Ctrl+C must not cut the teardown short
  trap - EXIT
  say "shutting down"
  # On Ctrl+C the terminal already delivered SIGINT to the backend; a second one would make uvicorn
  # force-exit before its shutdown hook has aborted the mission.
  if [[ "$why" != INT ]]; then
    for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -INT "$p" 2>/dev/null; done
  fi
  (( DEV )) && pkill -TERM -f "^node $ROOT/web_ui/node_modules/.bin/vite" 2>/dev/null
  for _ in $(seq 1 60); do                       # up to 15 s: the backend aborts the mission on shutdown
    alive=0
    for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && alive=1; done
    (( alive )) || break
    sleep 0.25
  done
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null; done
  "$ROOT/stop_all.sh" --quiet --leftovers-only || true
  return 0
}
set_traps() { trap 'cleanup INT' INT; trap 'cleanup TERM' TERM; trap 'cleanup EXIT' EXIT; }

start_vite() {
  if (( DEV )); then
    check_port 5173
    say "Vite dev server → http://localhost:5173 (proxying /api and /ws to :$PORT)"
    (cd "$ROOT/web_ui" && REDSHIFT_BACKEND="127.0.0.1:$PORT" exec npm run dev --silent -- --host "$HOST") &
    PIDS+=($!)
  fi
}

check_port "$PORT"
build_web

# ---------------------------------------------------------------- UI-only demo (no ROS / Gazebo)
if (( UI_DEMO )); then
  python3 -c 'import fastapi, uvicorn, numpy, cv2' 2>/dev/null || fail "UI demo needs python3 fastapi uvicorn numpy opencv"
  set_traps
  start_vite
  say "UI demo (scripted fake backend, no robots) → http://localhost:$PORT"
  python3 -u "$ROOT/web_ui/dev/fake_backend.py" --host "$HOST" --port "$PORT" &
  PIDS+=($!)
  wait "${PIDS[-1]}"
  exit 0
fi

# ---------------------------------------------------------------- ROS 2 workspace
[[ -f /opt/ros/jazzy/setup.bash ]] || fail "ROS 2 Jazzy not found at /opt/ros/jazzy (or run ./start_all.sh --ui-demo)"
set +u; source /opt/ros/jazzy/setup.bash; set -u
python3 - <<'PY' || fail "missing Python packages (see README → Requirements)"
import importlib, sys
missing = [m for m in ('fastapi', 'uvicorn', 'pydantic', 'yaml', 'numpy', 'cv2', 'scipy') if importlib.util.find_spec(m) is None]
if missing:
    print('missing:', ', '.join(missing), file=sys.stderr); sys.exit(1)
PY
command -v gz >/dev/null || say "warning: 'gz' (Gazebo Harmonic) not on PATH — only the mock world backend will be offered"

STAMP="$ROOT/ros2_ws/install/.redshift_build_stamp"
if (( REBUILD )) || [[ ! -f "$ROOT/ros2_ws/install/setup.bash" ]] || newer_than "$STAMP" "$ROOT/ros2_ws/src"; then
  say "building ROS 2 workspace (colcon, symlink install)"
  (cd "$ROOT/ros2_ws" && colcon build --symlink-install --event-handlers console_cohesion- 2>&1 | tail -3)
  touch "$STAMP"
else
  say "ROS 2 workspace up to date"
fi
set +u; source "$ROOT/ros2_ws/install/setup.bash"; set -u

# keep this demo's simulator and DDS traffic on this machine
export GZ_IP="${GZ_IP:-127.0.0.1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export PYTHONUNBUFFERED=1

set_traps
"$ROOT/stop_all.sh" --quiet --leftovers-only || true      # stale simulator processes from a crashed run
start_vite
say "Mission Control → http://localhost:$PORT   (Ctrl+C to stop)"
cd "$ROOT"
# exec-less child with SIGINT restored: background shells start jobs with SIGINT ignored
python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
  python3 -m uvicorn backend_bridge.app:app --host "$HOST" --port "$PORT" --log-level warning &
PIDS+=($!)
wait "${PIDS[-1]}"
