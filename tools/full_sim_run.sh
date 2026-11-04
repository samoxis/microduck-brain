#!/usr/bin/env bash
# One full-simulator run of microduck-brain, recorded, replayed and plotted (the README's Results).
#
# What it does:
#   1. starts Pollen's simulator headless in the apartment scene (scripts/duck-sim), unless one is
#      already running — the real robotd with the official walking policies, not the kinematic body;
#   2. checks the duck is standing and the world runs at real time (below 1.0x the policies can't
#      balance, and the run would be measuring the laptop, not the brain);
#   3. runs the brain for SECONDS with --record;
#   4. replays the recording (must match exactly), draws the path, and writes a summary.
#
# Environment: Linux or WSL (bash). Expects Pollen's repos at ~/microduck and ~/microduck_rl, with
# the simulator already working there once (policies downloaded, venv synced).
#
#   bash tools/full_sim_run.sh            # 300 s
#   SECONDS_TO_RUN=600 bash tools/full_sim_run.sh
#   BRAIN_CONFIG=/path/to/other.toml RUN_LABEL=on bash tools/full_sim_run.sh   # another config, labelled dir
#   MEASURE=1 bash tools/full_sim_run.sh     # also log contacts and the true trunk pose (a new simulator only)
#
# Output: runs/<timestamp>[-<label>]/ with run.jsonl.gz, run.svg, summary.json, brain.log, sim.txt, and
# brain.toml (the config the run used; the replay and the plot use it too)
set -euo pipefail

MICRODUCK="${MICRODUCK:-$HOME/microduck}"
export DUCK_SIM_RL="${DUCK_SIM_RL:-$HOME/microduck_rl}"
export DUCK_SIM_STATE="${DUCK_SIM_STATE:-$HOME/.cache/duck-sim}"
export DUCK_SIM_SCENE="${DUCK_SIM_SCENE:-apartment}"
export DUCK_SIM_VIEWER="${DUCK_SIM_VIEWER:-0}"
RUN_SECONDS="${SECONDS_TO_RUN:-300}"

BRAIN="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${BRAIN_CONFIG:-$BRAIN/brain.toml}"
OUT="$BRAIN/runs/$(date +%Y%m%d-%H%M%S)${RUN_LABEL:+-$RUN_LABEL}"
SIM="$MICRODUCK/scripts/duck-sim"
mkdir -p "$OUT"
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG" >&2; exit 1; }
cp "$CONFIG" "$OUT/brain.toml"
if [ "${MEASURE:-0}" = 1 ]; then
    # tools/push/sitecustomize.py, loaded into the simulator's Python: contacts with anything but the
    # floor, and where the trunk really is. Only takes effect if this script starts the simulator.
    export PYTHONPATH="$BRAIN/tools/push${PYTHONPATH:+:$PYTHONPATH}"
    export DUCK_CONTACT_FILE="$OUT/contacts.txt" DUCK_TRUTH_FILE="$OUT/truth.txt"
    : > "$OUT/contacts.txt"; : > "$OUT/truth.txt"
fi

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$SIM" ] || die "no $SIM — set MICRODUCK to your microduck checkout"
[ -d "$DUCK_SIM_RL" ] || die "no $DUCK_SIM_RL — set DUCK_SIM_RL to your microduck_rl checkout"
command -v uv >/dev/null || die "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"

STARTED_HERE=0
if [ -S "$DUCK_SIM_STATE/duck-a.sock" ]; then
    say "a simulator is already running — using it (scene as it was started)"
else
    say "starting the simulator: scene=$DUCK_SIM_SCENE, headless (first build can take a few minutes)"
    "$SIM" 2>&1 | tee "$OUT/sim-start.txt" | tail -n 5
    STARTED_HERE=1
fi
stop_sim() {
    if [ "$STARTED_HERE" = 1 ]; then
        say "stopping the simulator this script started"
        "$SIM" down >/dev/null 2>&1 || true
    fi
}
trap stop_sim EXIT

say "waiting for the duck to stand"
standing=0
for _ in $(seq 1 30); do
    "$SIM" status > "$OUT/status.txt" 2>&1 || true
    # match the success line exactly: the failure message also contains the word "standing"
    if grep -q ": standing (" "$OUT/status.txt"; then
        standing=1
        break
    fi
    sleep 2
done
[ "$standing" = 1 ] || die "the duck did not stand within 60 s — see $OUT/status.txt and '$SIM log'"

"$SIM" realtime > "$OUT/sim.txt" 2>&1 || true
cat "$OUT/sim.txt"
if ! grep -qE 'running at (1\.[0-9]+|0\.9[5-9])x' "$OUT/sim.txt"; then
    echo "WARNING: the simulator is below real time; the result will say more about this machine than the brain." | tee -a "$OUT/sim.txt"
fi

say "running the brain for $RUN_SECONDS s (Ctrl+C stops it cleanly)"
cd "$BRAIN"
# no pipe here: Ctrl+C must stop only the brain (it sends robot.stop), not the rest of this script
uv run microduck-brain -c "$OUT/brain.toml" --duration "$RUN_SECONDS" --record "$OUT/run.jsonl.gz" \
    > "$OUT/brain-summary.json" 2> "$OUT/brain.log" || true
[ -s "$OUT/run.jsonl.gz" ] || die "the brain wrote no recording — see $OUT/brain.log"

say "replaying the recording (must match exactly) and drawing the path"
uv run microduck-brain-replay -c "$OUT/brain.toml" "$OUT/run.jsonl.gz" > "$OUT/summary.json"
uv run python tools/explore_report.py -c "$OUT/brain.toml" "$OUT/run.jsonl.gz" > "$OUT/explore_report.json"
if [ -s "$OUT/truth.txt" ]; then
    uv run python tools/odom_truth.py "$OUT/run.jsonl.gz" "$OUT/truth.txt" --json "$OUT/odom_truth.json" > /dev/null || true
fi
uv run microduck-brain-plot -c "$OUT/brain.toml" "$OUT/run.jsonl.gz" -o "$OUT/run.svg" \
    --title "Full simulator, official walking policies, apartment ($RUN_SECONDS s)" >/dev/null

python3 - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
r, c = s["replayed"], s["coverage"]
print(f"""
  replay matches recording : {s['commands_differing_from_recording'] == 0}
  walked                   : {c['distance_m']} m, area covered {c['area_covered_m2']} m², box {c['bounding_box_m']} m
  avoidances               : {r['avoidances']} ({r['avoidances_with_stop']} with a stop), big turns {r['big_turns']}
  low objects / drops      : {r['low_objects_seen']} / {r['drops_seen']}
  falls or unhealthy holds : {r['holds_fall_or_unhealthy']}""")
PY

say "done: $OUT"
echo "Send back run.jsonl.gz (and look at run.svg). Falls, toys pushed, or bumps you saw on screen: note the time."
