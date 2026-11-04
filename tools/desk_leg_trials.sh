#!/usr/bin/env bash
# The desk-leg wedge in Pollen's full simulator: the apartment moved so the duck starts 6 s (true pose) before
# the spot where it once ground against a desk leg until it fell (tools/scenes/make_shifted_apartment.py).
# RUNS runs with each config, each from a fresh simulator; prints furniture contacts, falls and
# the stuck / wedge stops per run.
#
#   OFF=a.toml ON=b.toml RUNS=5 bash tools/desk_leg_trials.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RL="${DUCK_SIM_RL:-$HOME/microduck_rl}"
RUNS="${RUNS:-5}"
SECS="${SECONDS_TO_RUN:-45}"
"$RL/.venv/bin/python" -I "$HERE/scenes/make_shifted_apartment.py" ${DESK_START:-3.06 0.48 31} \
    "$RL/src/mjlab_microduck/robot/microduck/scene_apartment_desk_leg.xml"
for i in $(seq 1 "$RUNS"); do
    for label in off on; do
        cfg="${OFF}"; [ "$label" = on ] && cfg="${ON}"
        out="$(DUCK_SIM_SCENE=apartment_desk_leg MEASURE=1 RUN_LABEL="desk-$label" BRAIN_CONFIG="$cfg" \
               SECONDS_TO_RUN="$SECS" bash "$HERE/full_sim_run.sh" 2>&1)"
        dir="$(printf '%s\n' "$out" | sed -n 's/^== done: //p' | tail -1)"
        furn="$(awk '$2 == "furniture"' "$dir/contacts.txt" | wc -l)"
        (cd "$HERE/.." && uv run --quiet python - "$dir" "$label" "$i" "$furn" "$cfg") <<'PY'
import sys
from microduck_brain.brain import Brain
from microduck_brain.config import load
from microduck_brain.record import read
d, label, i, furn, cfg = sys.argv[1:]
b, falls, was = Brain(load(cfg)), 0, False
for t in read(f"{d}/run.jsonl.gz"):
    b.tick(t.inputs)
    now = "fallen" in t.inputs.safe_reason
    falls += now and not was
    was = now
s = b.stats
print(f"{label} run {i}: furniture contacts {furn}, falls {falls}, stuck {s.stuck}, wedged {s.wedged}  ({d})")
PY
    done
done
