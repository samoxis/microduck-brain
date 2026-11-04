#!/usr/bin/env bash
# The first test of microduck-brain on a real Microduck, guided, for someone who has never used it.
#
# Run it ON THE DUCK'S OWN COMPUTER (where robotd runs), from the microduck-brain folder:
#
#   bash tools/first_test.sh
#
# It does, stopping at the first problem:
#   1. checks it is in the right place (robotd and the depth sensor are reachable) and installs uv
#      (the Python tool that runs the brain) if it is missing;
#   2. OBSERVE, 60 s, duck on a stand: the brain reads the sensors and decides, but sends the duck
#      NOTHING; then checks that recording (tools/first_robot_check.py). Any FAIL ends the test here;
#   3. only if everything passed, and only after you say yes: WALK, 30 s, on the floor in a clear
#      area. Ctrl+C stops the duck at once;
#   4. packs everything into one file to send back.
#
# Nothing is changed on the duck: no settings, no software. Everything it writes stays in
# first-test-<date>/ next to this folder's README.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1
OUT="first-test-$(date +%Y%m%d-%H%M%S)"
YES="${FIRST_TEST_YES:-0}"          # 1: answer every question with Enter / yes (for testing the script)
OBSERVE_S="${OBSERVE_SECONDS:-60}"
WALK_S="${WALK_SECONDS:-30}"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mSTOP\033[0m %s\n' "$*"; }
wait_enter() {
    [ "$YES" = 1 ] && return 0
    printf '\n  %s\n  Press Enter when ready (Ctrl+C to quit). ' "$1"
    read -r _
}
ask_yes() {
    [ "$YES" = 1 ] && return 0
    printf '\n  %s [type yes / anything else = no]: ' "$1"
    read -r answer
    [ "$answer" = "yes" ]
}
pack() {
    tar -czf "$OUT.tar.gz" "$OUT" 2>/dev/null && \
        printf '\n  Everything is in \033[1m%s/%s.tar.gz\033[0m - please send that file back.\n' "$HERE" "$OUT"
}

bold "microduck-brain - first test on a real duck"
echo "  This takes about 5 minutes. You need: the duck charged and switched on, a box or stand"
echo "  that holds it upright with its feet off the table, and a clear 2 x 2 m area of floor"
echo "  with no stairs, no table edges and nothing fragile."
mkdir -p "$OUT"

# --- 1. the right place --------------------------------------------------------------------
bold "Step 1 of 4: checking this computer"
[ -f pyproject.toml ] && [ -d src/microduck_brain ] || { bad "run this from the microduck-brain folder"; exit 1; }
ok "microduck-brain folder: $HERE"
if [ -n "${DUCK_SIM_STATE:-}" ] || [ -S "$HOME/.cache/duck-sim/duck-a.sock" ]; then
    ok "a simulator is running here: the test will use the simulated duck"
else
    [ -S /run/robotd.sock ] || { bad "robotd is not reachable (/run/robotd.sock): run this on the duck's own computer, with the duck switched on"; exit 1; }
    ok "robotd: /run/robotd.sock"
    [ -S /run/tofd/tof.sock ] || { bad "the depth sensor service is not reachable (/run/tofd/tof.sock)"; exit 1; }
    ok "depth sensor: /run/tofd/tof.sock"
fi
if ! command -v uv >/dev/null 2>&1; then
    [ -x "$HOME/.local/bin/uv" ] && export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "  uv (the tool that runs the brain) is not installed."
    ask_yes "Install it now from astral.sh?" || { bad "uv is needed: https://docs.astral.sh/uv/"; exit 1; }
    curl -LsSf https://astral.sh/uv/install.sh | sh || { bad "could not install uv"; exit 1; }
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version 2>/dev/null)"
uv sync --quiet >"$OUT/setup.log" 2>&1 || { bad "could not prepare Python (see $OUT/setup.log)"; pack; exit 1; }
ok "Python ready"

# --- 2. observe ------------------------------------------------------------------------------
bold "Step 2 of 4: watching, without moving (${OBSERVE_S} s)"
echo "  Put the duck on its stand, upright, feet in the air, the depth sensor (on its head)"
echo "  looking at the room. It will NOT move: nothing is sent to it."
wait_enter "Duck on the stand?"
echo "  Recording for ${OBSERVE_S} s... (wave a hand in front of its head now and then)"
uv run --quiet microduck-brain --observe --duration "$OBSERVE_S" --record "$OUT/observe.jsonl.gz" \
    >"$OUT/observe.log" 2>&1 || { bad "the brain stopped with an error (see $OUT/observe.log)"; pack; exit 1; }
ok "recorded"
uv run --quiet python tools/first_robot_check.py "$OUT/observe.jsonl.gz" --json "$OUT/check.json" | tee "$OUT/check.txt"
if grep -q '^FAIL' "$OUT/check.txt"; then
    bad "a check failed: the duck must NOT walk with the brain yet. Nothing else to do - thank you!"
    pack
    exit 1
fi
ok "all checks passed"

# --- 3. walk ---------------------------------------------------------------------------------
bold "Step 3 of 4: the first walk (${WALK_S} s)"
echo "  Now the brain WILL make the duck walk, on its own, for ${WALK_S} s, and stop by itself."
echo "  Put it on the floor in the middle of the clear area, standing. Stay next to it."
echo "  To stop it at any moment: press Ctrl+C here (or pick it up)."
if ! ask_yes "Ready for it to walk?"; then
    ok "skipped the walk - the observe test alone already helps a lot"
    pack
    exit 0
fi
for s in 3 2 1; do printf '  %s...\n' "$s"; [ "$YES" = 1 ] || sleep 1; done
uv run --quiet microduck-brain --duration "$WALK_S" --record "$OUT/walk.jsonl.gz" >"$OUT/walk.log" 2>&1
code=$?
[ $code -eq 0 ] && ok "walk finished, the duck was told to stop" || bad "the brain ended with code $code (see $OUT/walk.log)"
uv run --quiet microduck-brain-replay "$OUT/walk.jsonl.gz" >"$OUT/walk-replay.txt" 2>&1 || true
uv run --quiet microduck-brain-plot "$OUT/walk.jsonl.gz" -o "$OUT/walk.svg" >/dev/null 2>&1 || true

# --- 4. pack ---------------------------------------------------------------------------------
bold "Step 4 of 4: done"
if [ "$YES" != 1 ]; then
    printf '\n  In a few words, what did the duck do? (Enter to skip): '
    read -r note
    printf '%s\n' "$note" > "$OUT/notes.txt"
fi
pack
echo "  Thank you!"
