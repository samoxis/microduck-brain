#!/usr/bin/env bash
# The hand gesture ([gestures]) in Pollen's full simulator: a 9 x 9 cm "palm" (a mocap body the
# depth sensor sees, tools/scenes/make_hand_apartment.py) is held 10 cm in front of the sensor by
# tools/push/sitecustomize.py, on a schedule, while the brain runs:
#     at ~25 s: 1.5 s  -> a walking duck should stop      (paused)
#     at ~50 s: 1.5 s  -> a stopped duck should walk on   (forward)
#     at ~70 s: 6.0 s  -> too long: not a gesture, nothing toggles
# RUNS runs, each from a fresh simulator; prints what happened per run (tools/gesture_report.py).
# Side effect: writes its scene XML into $DUCK_SIM_RL (default ~/microduck_rl), next to Pollen's scenes.
#
#   RUNS=3 bash tools/gesture_trials.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RL="${DUCK_SIM_RL:-$HOME/microduck_rl}"
RUNS="${RUNS:-3}"
"$RL/.venv/bin/python" -I "$HERE/scenes/make_hand_apartment.py" "$RL/src/mjlab_microduck/robot/microduck/scene_apartment_hand.xml"
CFG="$(mktemp --suffix=.toml)"
HAND="$(mktemp -u)"
trap 'rm -f "$CFG" "$HAND"' EXIT
python3 - "$HERE/../brain.toml" "$CFG" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s, n = re.subn(r"(\[gestures\][^\[]*?)enabled = false", r"\1enabled = true", s, count=1, flags=re.S)
assert n == 1
open(sys.argv[2], "w").write(s)
PY
for i in $(seq 1 "$RUNS"); do
    rm -f "$HAND"
    before="$(ls -d "$HERE"/../runs/*-gesture 2>/dev/null | wc -l)"
    (   # the schedule, from the moment THIS run's brain first walks
        for _ in $(seq 480); do
            if [ "$(ls -d "$HERE"/../runs/*-gesture 2>/dev/null | wc -l)" -gt "$before" ] && \
               grep -q "state: forward" "$(ls -td "$HERE"/../runs/*-gesture/ | head -1)brain.log" 2>/dev/null; then break; fi
            sleep 0.5
        done
        sleep 23; echo "0.10 1.5" > "$HAND"
        sleep 25; echo "0.10 1.5" > "$HAND"
        sleep 20; echo "0.10 6.0" > "$HAND"
    ) &
    sched=$!
    out="$(DUCK_HAND_FILE="$HAND" DUCK_SIM_SCENE=apartment_hand MEASURE=1 RUN_LABEL=gesture BRAIN_CONFIG="$CFG" \
           SECONDS_TO_RUN="${SECONDS_TO_RUN:-90}" bash "$HERE/full_sim_run.sh" 2>&1)"
    wait "$sched" || true
    dir="$(printf '%s\n' "$out" | sed -n 's/^== done: //p' | tail -1)"
    printf 'run %s: ' "$i"
    (cd "$HERE/.." && uv run --quiet python tools/gesture_report.py "$dir")
done
