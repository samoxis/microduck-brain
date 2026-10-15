"""Draw a recorded run as an SVG: the path (coloured by state), a rough wall map, and the low
objects and drops the brain remembered. Standard library only."""

from __future__ import annotations

from html import escape

from .config import Config
from .hazards import DROP, LOW
from .record import Tick, metrics, replay, wall_points

STATE_COLOURS = {
    "forward": "#2a6fdb",
    "brake": "#d1342f",
    "turn": "#e8890c",
    "bigturn": "#9b3fd1",
    "hold": "#8a8a8a",
}


def render_svg(ticks: list[Tick], cfg: Config, run_path: str, title: str = "", size: int = 820) -> str:
    rep = replay(run_path, cfg)
    walls = wall_points(ticks, cfg)
    path = [(t.inputs.pos, t.state) for t in ticks if t.inputs.pos is not None]
    low = sorted(rep.hazard_points.get(LOW, set()))
    drop = sorted(rep.hazard_points.get(DROP, set()))

    xs = [p[0] for p, _ in path] + [w[0] for w in walls] + [h[0] for h in low + drop]
    ys = [p[1] for p, _ in path] + [w[1] for w in walls] + [h[1] for h in low + drop]
    if not xs:
        raise ValueError("the recording has no odometry positions to draw")
    pad = 0.3
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    scale = (size - 40) / max(x1 - x0, y1 - y0, 0.5)
    top = 70

    def X(x: float) -> float:
        return round(20 + (x - x0) * scale, 1)

    def Y(y: float) -> float:          # world y up, SVG y down
        return round(top + (y1 - y) * scale, 1)

    height = int(top + (y1 - y0) * scale + 90)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{height}" '
        f'viewBox="0 0 {size} {height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{size}" height="{height}" fill="#ffffff"/>',
    ]
    m = metrics(ticks)
    st = rep.brain.stats
    out.append(f'<text x="20" y="28" font-size="18" font-weight="bold" fill="#1d1d1d">{escape(title or "microduck-brain run")}</text>')
    out.append(
        f'<text x="20" y="52" font-size="13" fill="#444">{m["duration_s"]} s · {m["distance_m"]} m walked · '
        f'{m["area_covered_m2"]} m² covered · {st.avoids} avoidances ({st.brakes} with a stop) · '
        f'{st.low_seen} low objects · {st.drops_seen} drops · {st.holds_unsafe} falls/unsafe holds</text>'
    )
    # 1 m scale bar
    out.append(f'<line x1="20" y1="{height - 50}" x2="{20 + scale:.1f}" y2="{height - 50}" stroke="#1d1d1d" stroke-width="2"/>')
    out.append(f'<text x="{24 + scale:.1f}" y="{height - 46}" font-size="11" fill="#1d1d1d">1 m</text>')

    for wx, wy in walls:
        out.append(f'<circle cx="{X(wx)}" cy="{Y(wy)}" r="1.4" fill="#b9b9b9"/>')
    # path, one segment per tick, coloured by the state the brain was in
    for (p, state), (q, _) in zip(path, path[1:]):
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) > 0.5:
            continue
        out.append(
            f'<line x1="{X(p[0])}" y1="{Y(p[1])}" x2="{X(q[0])}" y2="{Y(q[1])}" '
            f'stroke="{STATE_COLOURS.get(state, "#333")}" stroke-width="2.4" stroke-linecap="round"/>'
        )
    for hx, hy in drop:
        out.append(f'<rect x="{X(hx) - 3}" y="{Y(hy) - 3}" width="6" height="6" fill="#111" opacity="0.75"/>')
    for hx, hy in low:
        out.append(f'<circle cx="{X(hx)}" cy="{Y(hy)}" r="4" fill="none" stroke="#0a9b54" stroke-width="2"/>')
    if path:
        (sx, sy), _ = path[0]
        out.append(f'<circle cx="{X(sx)}" cy="{Y(sy)}" r="6" fill="#1d1d1d"/>')
        out.append(f'<text x="{X(sx) + 9}" y="{Y(sy) + 4}" font-size="12" fill="#1d1d1d">start</text>')

    legend = [(STATE_COLOURS[k], k) for k in ("forward", "turn", "brake", "bigturn", "hold")]
    lx = 20
    for colour, label in legend:
        out.append(f'<rect x="{lx}" y="{height - 33}" width="14" height="4" fill="{colour}"/>')
        out.append(f'<text x="{lx + 18}" y="{height - 27}" font-size="12" fill="#1d1d1d">{label}</text>')
        lx += 18 + 9 * len(label) + 16
    out.append(f'<circle cx="{lx + 4}" cy="{height - 31}" r="4" fill="none" stroke="#0a9b54" stroke-width="2"/>')
    out.append(f'<text x="{lx + 14}" y="{height - 27}" font-size="12" fill="#1d1d1d">low object</text>')
    lx += 100
    out.append(f'<rect x="{lx}" y="{height - 34}" width="6" height="6" fill="#111" opacity="0.75"/>')
    out.append(f'<text x="{lx + 12}" y="{height - 27}" font-size="12" fill="#1d1d1d">drop</text>')
    lx += 60
    out.append(f'<circle cx="{lx + 2}" cy="{height - 31}" r="2" fill="#b9b9b9"/>')
    out.append(f'<text x="{lx + 10}" y="{height - 27}" font-size="12" fill="#1d1d1d">wall returns</text>')
    out.append("</svg>")
    return "\n".join(out)
