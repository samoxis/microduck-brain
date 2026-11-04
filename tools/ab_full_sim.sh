#!/usr/bin/env bash
# A/B in the full simulator: RUNS runs with exploration OFF and RUNS with it ON, on the same commit,
# alternating (off,on / on,off / ...) so a slow drift of the machine doesn't favour one side, then
# mean, spread and range per metric (tools/ab_summary.py).
#
#   bash tools/ab_full_sim.sh                       # 3 + 3 runs of 600 s (~70 min)
#   RUNS=5 SECONDS_TO_RUN=300 bash tools/ab_full_sim.sh
#   AB_KEY=head.scan RUNS=5 bash tools/ab_full_sim.sh   # A/B another boolean: [head] scan
#   AB_KEY=recover.enabled AB_SET="recover.enabled=true obstacle.stuck_seconds=4.0" bash tools/ab_full_sim.sh
#       # a profile: "off" is brain.toml as it is, "on" sets every section.key=value listed (AB_KEY: one of them, a boolean, labels the runs)
#
# Each run starts its own simulator (full_sim_run.sh stops it at the end), so every run starts from
# the same pose; that's why a simulator already running is refused. The configs are brain.toml with
# AB_KEY (default explore.enabled) = false / true, nothing else changed. Output:
# runs/<timestamp>-off|on/ and runs/ab-<timestamp>.json.
set -euo pipefail

BRAIN="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS:-3}"
AB_KEY="${AB_KEY:-explore.enabled}"
export SECONDS_TO_RUN="${SECONDS_TO_RUN:-600}"
STATE="${DUCK_SIM_STATE:-$HOME/.cache/duck-sim}"
[ -S "$STATE/duck-a.sock" ] && { echo "ERROR: a simulator is already running; stop it first (each run must start fresh)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 - "$BRAIN/brain.toml" "$TMP" "$AB_KEY" "${AB_SET:-}" <<'PY'
import sys, pathlib
src, out = pathlib.Path(sys.argv[1]).read_text().splitlines(keepends=True), pathlib.Path(sys.argv[2])
if sys.argv[4]:
    text = "".join(src)
    (out / "off.toml").write_text(text)
    for item in sys.argv[4].split():
        path, value = item.split("=", 1)
        sec, key = path.split(".")
        section, done, lines = None, False, []
        for line in text.splitlines(keepends=True):
            s = line.strip()
            if s.startswith("["):
                section = s
            if section == f"[{sec}]" and s.split("=")[0].strip() == key:
                line, done = f"{key} = {value}\n", True
            lines.append(line)
        if not done:
            raise SystemExit(f"no {key} in [{sec}] of brain.toml")
        text = "".join(lines)
    (out / "on.toml").write_text(text)
    raise SystemExit(0)
sec, key = sys.argv[3].split(".")
for label, value in (("off", "false"), ("on", "true")):
    section, done, lines = None, False, []
    for line in src:
        s = line.strip()
        if s.startswith("["):
            section = s
        if section == f"[{sec}]" and s.split("=")[0].strip() == key:
            line, done = f"{key} = {value}\n", True
        lines.append(line)
    if not done:
        raise SystemExit(f"no {key} in [{sec}] of brain.toml")
    (out / f"{label}.toml").write_text("".join(lines))
PY

cd "$BRAIN"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
DIRS=()
for i in $(seq 1 "$RUNS"); do
    if [ $((i % 2)) = 1 ]; then order="off on"; else order="on off"; fi
    for label in $order; do
        printf '\n######## run %s/%s: %s %s (commit %s)\n' "$i" "$RUNS" "$AB_KEY" "$label" "$COMMIT"
        BRAIN_CONFIG="$TMP/$label.toml" RUN_LABEL="$label" bash tools/full_sim_run.sh
        DIRS+=("$(ls -td runs/*-"$label"/ | head -1)")
    done
done
OUTJSON="runs/ab-$(date +%Y%m%d-%H%M%S).json"
uv run python tools/ab_summary.py --key "$AB_KEY" --commit "$COMMIT" --json "$OUTJSON" "${DIRS[@]}"
echo "saved: $OUTJSON"
