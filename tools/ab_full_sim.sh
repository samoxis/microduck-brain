#!/usr/bin/env bash
# A/B in the full simulator: RUNS runs with exploration OFF and RUNS with it ON, on the same commit,
# alternating (off,on / on,off / ...) so a slow drift of the machine doesn't favour one side, then
# mean, spread and range per metric (tools/ab_summary.py).
#
#   bash tools/ab_full_sim.sh                       # 3 + 3 runs of 600 s (~70 min)
#   RUNS=5 SECONDS_TO_RUN=300 bash tools/ab_full_sim.sh
#
# Each run starts its own simulator (full_sim_run.sh stops it at the end), so every run starts from
# the same pose; that's why a simulator already running is refused. The configs are brain.toml with
# [explore] enabled = false / true, nothing else changed. Output: runs/<timestamp>-off|on/ and
# runs/ab-<timestamp>.json.
set -euo pipefail

BRAIN="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS:-3}"
export SECONDS_TO_RUN="${SECONDS_TO_RUN:-600}"
STATE="${DUCK_SIM_STATE:-$HOME/.cache/duck-sim}"
[ -S "$STATE/duck-a.sock" ] && { echo "ERROR: a simulator is already running; stop it first (each run must start fresh)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 - "$BRAIN/brain.toml" "$TMP" <<'PY'
import sys, pathlib
src, out = pathlib.Path(sys.argv[1]).read_text().splitlines(keepends=True), pathlib.Path(sys.argv[2])
for label, value in (("off", "false"), ("on", "true")):
    section, done, lines = None, False, []
    for line in src:
        s = line.strip()
        if s.startswith("["):
            section = s
        if section == "[explore]" and s.split("=")[0].strip() == "enabled":
            line, done = f"enabled = {value}\n", True
        lines.append(line)
    if not done:
        lines.append(f"\n[explore]\nenabled = {value}\n")
    (out / f"{label}.toml").write_text("".join(lines))
PY

cd "$BRAIN"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
DIRS=()
for i in $(seq 1 "$RUNS"); do
    if [ $((i % 2)) = 1 ]; then order="off on"; else order="on off"; fi
    for label in $order; do
        printf '\n######## run %s/%s: explore %s (commit %s)\n' "$i" "$RUNS" "$label" "$COMMIT"
        BRAIN_CONFIG="$TMP/$label.toml" RUN_LABEL="$label" bash tools/full_sim_run.sh
        DIRS+=("$(ls -td runs/*-"$label"/ | head -1)")
    done
done
OUTJSON="runs/ab-$(date +%Y%m%d-%H%M%S).json"
uv run python tools/ab_summary.py --commit "$COMMIT" --json "$OUTJSON" "${DIRS[@]}"
echo "saved: $OUTJSON"
