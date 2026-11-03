#!/usr/bin/env bash
# Does the duck get up after a shove? In Pollen's full simulator, one trial per speed, each from a
# freshly started simulator: shove the trunk sideways (m/s added to its velocity, - = the other
# way), then check it is really upright again (gravity z < -0.95, trunk above 0.10 m, held 2 s).
#
#   bash tools/push/push_trials.sh                 # 10 trials, ±1.0 … ±3.0 m/s
#   SPEEDS="2.0 2.0 2.0" bash tools/push/push_trials.sh
#
# The shove comes from tools/push/sitecustomize.py, loaded into the simulator's Python through
# PYTHONPATH; it changes nothing else. Expects ~/microduck and ~/microduck_rl like full_sim_run.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MICRODUCK="${MICRODUCK:-$HOME/microduck}"
export DUCK_SIM_RL="${DUCK_SIM_RL:-$HOME/microduck_rl}"
export DUCK_SIM_STATE="${DUCK_SIM_STATE:-$HOME/.cache/duck-sim}"
export DUCK_SIM_VIEWER="${DUCK_SIM_VIEWER:-0}"
SPEEDS="${SPEEDS:-1.0 1.5 2.0 2.5 3.0 -1.0 -1.5 -2.0 -2.5 -3.0}"
SIM="$MICRODUCK/scripts/duck-sim"
PUSH="$(mktemp -u)"
trap '"$SIM" down >/dev/null 2>&1 || true; rm -f "$PUSH"' EXIT
ups=0; n=0
for sp in $SPEEDS; do
    "$SIM" down >/dev/null 2>&1 || true
    rm -f "$PUSH"
    PYTHONPATH="$HERE" DUCK_PUSH_FILE="$PUSH" "$SIM" >/dev/null 2>&1
    sleep 8
    line="$(cd /tmp && "$DUCK_SIM_RL/.venv/bin/python" -I "$HERE/watch_fall.py" "$DUCK_SIM_STATE/duck-a.sock" "$PUSH" 20 "$sp" | tail -1)"
    printf 'push %5s m/s: %s\n' "$sp" "$line"
    n=$((n + 1)); case "$line" in *"up_after=NO"*) ;; *) ups=$((ups + 1)) ;; esac
done
echo "got up: $ups of $n"
