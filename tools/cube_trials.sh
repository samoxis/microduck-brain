#!/usr/bin/env bash
# The cube-after-turn test in Pollen's full simulator: a wall right ahead of the duck's start and a
# 7 cm cube on each side on the path it takes after turning in place, closer than the ~45 cm under
# which a level head can't see it (finding 3). RUNS runs with each config, each from a fresh
# simulator; prints the toy contacts and low objects seen per run.
#
#   OFF=level.toml ON=pitched.toml RUNS=5 bash tools/cube_trials.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RL="${DUCK_SIM_RL:-$HOME/microduck_rl}"
RUNS="${RUNS:-5}"
SECS="${SECONDS_TO_RUN:-40}"
cp "$HERE/scenes/cube_after_turn.xml" "$RL/src/mjlab_microduck/robot/microduck/scene_cube_after_turn.xml"
for label in off on; do
    cfg="${OFF}"; [ "$label" = on ] && cfg="${ON}"
    for i in $(seq 1 "$RUNS"); do
        out="$(DUCK_SIM_SCENE=cube_after_turn MEASURE=1 RUN_LABEL="cube-$label" BRAIN_CONFIG="$cfg" \
               SECONDS_TO_RUN="$SECS" bash "$HERE/full_sim_run.sh" 2>&1)"
        dir="$(printf '%s\n' "$out" | sed -n 's/^== done: //p' | tail -1)"
        toys="$(awk '$2 == "toy"' "$dir/contacts.txt" | wc -l)"
        low="$(python3 -c "import json;print(json.load(open('$dir/summary.json'))['replayed']['low_objects_seen'])")"
        printf '%s run %s: toy contacts %s, low objects seen %s  (%s)\n' "$label" "$i" "$toys" "$low" "$dir"
    done
done
