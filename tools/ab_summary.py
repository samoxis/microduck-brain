"""Mean, spread and range of full-simulator runs, grouped OFF / ON by a boolean config key (--key,
default explore.enabled).

Reads, from each run directory: summary.json (replay + coverage), explore_report.json (if present),
contacts.txt and odom_truth.json (MEASURE=1), and brain.toml (which says whether the key was on). A difference counts as a signal only when
the two groups' ranges don't overlap; with 3 runs a side that is all this says, and it says it
honestly.

    uv run python tools/ab_summary.py runs/2026...-off runs/2026...-on ... [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tomllib
from pathlib import Path

METRICS = [
    # (name, getter on (summary, report), higher is better?)
    ("area_m2", lambda s, r: s["coverage"]["area_covered_m2"], True),
    ("cells", lambda s, r: s["coverage"]["cells_visited"], True),
    ("distance_m", lambda s, r: s["coverage"]["distance_m"], True),
    ("bbox_m2", lambda s, r: s["coverage"]["bounding_box_m"][0] * s["coverage"]["bounding_box_m"][1], True),
    ("avoidances", lambda s, r: s["replayed"]["avoidances"], False),
    ("avoidances_with_stop", lambda s, r: s["replayed"]["avoidances_with_stop"], False),
    ("big_turns", lambda s, r: s["replayed"]["big_turns"], False),
    ("turn_s", lambda s, r: s["replayed"]["seconds_per_state"].get("turn", 0.0), False),
    ("falls_or_unhealthy", lambda s, r: s["replayed"]["holds_fall_or_unhealthy"], False),
    ("replay_differing", lambda s, r: s["commands_differing_from_recording"], False),
    ("avoids/min after turn in place", lambda s, r: (r or {}).get("after_every_turn_in_place", {}).get("avoids_per_min_after"), False),
    ("avoids/min elsewhere", lambda s, r: (r or {}).get("after_every_turn_in_place", {}).get("avoids_per_min_elsewhere"), False),
    ("contacts: furniture", lambda s, r: contacts(s["_dir"], "furniture"), False),
    ("contacts: toys", lambda s, r: contacts(s["_dir"], "toy"), False),
    ("contacts: stairs", lambda s, r: contacts(s["_dir"], "stair"), False),
    ("odometry error, final (m)", lambda s, r: s.get("_odom", {}).get("position_error_m", {}).get("final"), False),
    ("bends", lambda s, r: s["replayed"].get("exploration_bends"), None),
    ("bends_refused", lambda s, r: s["replayed"].get("exploration_bends_refused"), None),
    ("side_choices", lambda s, r: s["replayed"].get("exploration_side_choices"), None),
]


def contacts(d: Path, kind: str) -> int | None:
    """New contacts of a kind (furniture, toy, stair) logged by MEASURE=1 runs, while the brain ran."""
    p = d / "contacts.txt"
    if not p.exists():
        return None
    return sum(1 for line in p.read_text().splitlines() if line.split()[1:2] == [kind])


def load(d: Path, key: str = "explore.enabled") -> tuple[str, dict, dict | None]:
    cfg = tomllib.loads((d / "brain.toml").read_text()) if (d / "brain.toml").exists() else {}
    sec, k = key.split(".")
    label = "on" if cfg.get(sec, {}).get(k, False) else "off"
    s = json.loads((d / "summary.json").read_text())
    s["_dir"] = d
    ot = d / "odom_truth.json"
    if ot.exists():
        s["_odom"] = json.loads(ot.read_text())
    rp = d / "explore_report.json"
    r = json.loads(rp.read_text()) if rp.exists() and rp.stat().st_size else None
    return label, s, r


def describe(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 2),
        "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "values": [round(v, 2) for v in vals],
    }


def summarise(dirs: list[Path], key: str = "explore.enabled") -> dict:
    groups: dict[str, list[tuple[Path, dict, dict | None]]] = {"off": [], "on": []}
    for d in dirs:
        label, s, r = load(d, key)
        groups[label].append((d, s, r))
    out = {"key": key, "runs": {k: [str(d) for d, _, _ in v] for k, v in groups.items()}, "metrics": {}}
    for name, get, _better in METRICS:
        row = {}
        for label, runs in groups.items():
            vals = [v for _, s, r in runs if (v := get(s, r)) is not None]
            row[label] = describe([float(v) for v in vals])
        off, on = row["off"], row["on"]
        if off.get("n") and on.get("n"):
            row["diff_of_means"] = round(on["mean"] - off["mean"], 2)
            row["ranges_overlap"] = not (on["min"] > off["max"] or on["max"] < off["min"])
        out["metrics"][name] = row
    return out


def print_table(res: dict, commit: str) -> None:
    n_off, n_on = len(res["runs"]["off"]), len(res["runs"]["on"])
    print(f"\ncommit {commit or '?'} — {res.get('key', 'explore.enabled')} OFF: {n_off} runs, ON: {n_on} runs\n")
    print(f"{'metric':32s} {'OFF mean ± sd [min–max]':28s} {'ON mean ± sd [min–max]':28s} {'ON−OFF':>8s}  signal?")
    for name, row in res["metrics"].items():
        def cell(g):
            return f"{g['mean']:.2f} ± {g['sd']:.2f} [{g['min']:g}–{g['max']:g}]" if g.get("n") else "-"
        diff = row.get("diff_of_means")
        sig = "" if diff is None else ("noise (ranges overlap)" if row["ranges_overlap"] else "separated")
        print(f"{name:32s} {cell(row['off']):28s} {cell(row['on']):28s} {'' if diff is None else f'{diff:+.2f}':>8s}  {sig}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--commit", default="")
    ap.add_argument("--key", default="explore.enabled", help="the config switch the runs differ in")
    args = ap.parse_args()
    res = summarise(args.dirs, args.key)
    res["commit"] = args.commit
    print_table(res, args.commit)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
