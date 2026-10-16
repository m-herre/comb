"""Draw the Comb logo: a honeycomb of cells fed by table rows, and the "comb" wordmark in geometric strokes.

    python scripts/make_logo.py            # writes docs/assets/{logo,logo-dark,logo-mark}.svg

The mark is seven hexagonal cells. Their shade is the cell's row count (the model's state); the amber cell is the one
being edited, and one cell is an empty outline (rows forgotten). Rows of a table flow in from the left. The wordmark
is drawn as strokes (no font), so it looks the same everywhere.
"""
import math
from pathlib import Path

INK = {"light": "#1c2333", "dark": "#e8e6e1"}
AMBER = "#f2a516"
R, GAP = 21.0, 2.6  # hexagon circumradius and the gap between cells
W = math.sqrt(3) * (R + GAP / math.sqrt(3))  # centre spacing


def hexagon(cx, cy, r):
    pts = [(cx + r * math.cos(math.radians(60 * k - 90)), cy + r * math.sin(math.radians(60 * k - 90)))
           for k in range(6)]
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"


def comb_cells(cx, cy, ink):
    """Seven pointy-top cells: centre plus ring; shade = row count, amber = edited, outline = emptied."""
    ring = [(W, 0), (W / 2, 1.5 * (R + GAP / 2)), (-W / 2, 1.5 * (R + GAP / 2)), (-W, 0),
            (-W / 2, -1.5 * (R + GAP / 2)), (W / 2, -1.5 * (R + GAP / 2))]
    shades = [0.92, 0.55, 0.78, None, 0.35, 0.68]  # None: the emptied cell
    out = [f'<path d="{hexagon(cx, cy, R)}" fill="{AMBER}"/>',
           # the "+" on the edited cell
           f'<path d="M{cx - 7:.2f},{cy:.2f} H{cx + 7:.2f} M{cx:.2f},{cy - 7:.2f} V{cy + 7:.2f}" stroke="{INK["light"]}" '
           f'stroke-width="3.6" stroke-linecap="round"/>']
    for (dx, dy), shade in zip(ring, shades):
        if shade is None:
            out.append(f'<path d="{hexagon(cx + dx, cy + dy, R - 1.6)}" fill="none" stroke="{ink}" stroke-width="2.4" '
                       f'stroke-dasharray="4.2 3.4" stroke-linejoin="round" opacity="0.7"/>')
        else:
            out.append(f'<path d="{hexagon(cx + dx, cy + dy, R)}" fill="{ink}" opacity="{shade}"/>')
    return "\n  ".join(out)


def table_rows(x0, cy, ink):
    """Three table rows of small cells, converging on the comb."""
    out = []
    for i, dy in enumerate((-22, 0, 22)):
        for j in range(3):
            x = x0 + j * 11.5
            y = cy + dy * (1 - 0.18 * j) - 4
            op = 0.25 + 0.2 * j
            out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="8" height="8" rx="1.6" fill="{ink}" opacity="{op:.2f}"/>')
    return "\n  ".join(out)


def wordmark(x, base, ink, s=9.0):
    """'comb' in round-capped strokes: x-height 44, ascender 64."""
    r = 22 - s / 2  # bowl radius on the stroke centre line
    cy = base - 22
    c_x = x + 22
    a = math.radians(48)
    c = (f'M{c_x + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f} '
         f'A{r},{r} 0 1 0 {c_x + r * math.cos(a):.2f},{cy + r * math.sin(a):.2f}')
    o_x = x + 22 + 50
    o = f'M{o_x + r:.2f},{cy:.2f} A{r},{r} 0 1 0 {o_x - r:.2f},{cy:.2f} A{r},{r} 0 1 0 {o_x + r:.2f},{cy:.2f}'
    m_x = x + 50 + 50 + s / 2
    top, ar = base - 44 + s / 2, 13.0
    m = (f'M{m_x:.2f},{base - s / 2:.2f} V{top + ar:.2f} A{ar},{ar} 0 0 1 {m_x + 2 * ar:.2f},{top + ar:.2f} '
         f'V{base - s / 2:.2f} M{m_x + 2 * ar:.2f},{top + ar:.2f} A{ar},{ar} 0 0 1 {m_x + 4 * ar:.2f},{top + ar:.2f} '
         f'V{base - s / 2:.2f}')
    b_x = m_x + 4 * ar + 16
    b = (f'M{b_x:.2f},{base - 64 + s / 2:.2f} V{base - s / 2:.2f} '
         f'M{b_x:.2f},{cy:.2f} A{r},{r} 0 1 1 {b_x + 2 * r:.2f},{cy:.2f} A{r},{r} 0 1 1 {b_x:.2f},{cy:.2f}')
    width = b_x + 2 * r + s / 2 - x
    path = " ".join((c, o, m, b))
    return (f'<path d="{path}" fill="none" stroke="{ink}" stroke-width="{s}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'), width


def full_logo(theme):
    ink = INK[theme]
    cx, cy = 98, 60
    word, ww = wordmark(cx + W + R + 30, cy + 22, ink)
    width = cx + W + R + 30 + ww + 8
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} 120" width="{width:.0f}" height="120" '
            f'role="img" aria-label="Comb">\n  <title>Comb</title>\n  {table_rows(8, cy, ink)}\n  '
            f'{comb_cells(cx, cy, ink)}\n  {word}\n</svg>\n')


def mark(theme="light"):
    ink = INK[theme]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128" role="img" '
            f'aria-label="Comb">\n  <title>Comb</title>\n  {comb_cells(64, 64, ink)}\n</svg>\n')


def main():
    out = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logo.svg").write_text(full_logo("light"))
    (out / "logo-dark.svg").write_text(full_logo("dark"))
    (out / "logo-mark.svg").write_text(mark())
    print("wrote", ", ".join(str(p) for p in sorted(out.glob("*.svg"))))


if __name__ == "__main__":
    main()
