#!/usr/bin/env python3
"""Figures of the isometric texture coordinates (§3.12, §4.7).

  python make_isometric.py diagram
      writes diagram-isometric.svg: the forces on six rigid triangles of one chart mid-simulation (vertex to
      vertex springs, a repulsion impulse, velocity and spin).
  python make_isometric.py levels ATLAS_DIR ISOMETRIC_DIR OUT.png --levels 5 [--size 300]
      draws hierarchy levels saved by `meshlet-optimizer --save-levels` twice: with --no-isometric-uvs (the
      LSCM atlas; left, or the top row for several levels) and with the default isometric UVs (right, or the
      bottom row), and prints the checks of both for those levels: flipped triangles, overlapping pixels at
      1024^2, worst angle error, area ratio spread and the mean gap between corresponding corners.

Standard library only.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_diagrams import ACCENT, GOOD, MUTED, RED, TEXT, Svg  # noqa: E402
from plyfig import Canvas, read_ply, region_color, write_png  # noqa: E402

# ---------------------------------------------------------------- plane geometry


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, s):
    return (a[0] * s, a[1] * s)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def rot(a, t):
    c, s = math.cos(t), math.sin(t)
    return (c * a[0] - s * a[1], s * a[0] + c * a[1])


def tri_area(t):
    return 0.5 * cross2(sub(t[1], t[0]), sub(t[2], t[0]))


def tri_centroid(t):
    return ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0)


def flatten3(a, b, c):
    """The 3D triangle laid flat: a at the origin, b on +x, c above (counter-clockwise)."""
    d1 = [b[k] - a[k] for k in range(3)]
    d2 = [c[k] - a[k] for k in range(3)]
    l1 = math.sqrt(sum(v * v for v in d1))
    x = [v / l1 for v in d1]
    px = sum(d2[k] * x[k] for k in range(3))
    h = math.sqrt(max(0.0, sum(v * v for v in d2) - px * px))
    return [(0.0, 0.0), (l1, 0.0), (px, h)]


# ---------------------------------------------------------------- the diagram

FILL = "#6e7681"
SPRING = RED
PUSH = "#a371f7"
VELOCITY = ACCENT
TURN = GOOD


def arrow(svg, a, b, color, sw=1.6):
    svg.add('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="%.2f" stroke-linecap="round" '
            'marker-end="url(#arr-%s)"/>' % (a[0], a[1], b[0], b[1], color, sw, color[1:]))


class ClearSvg(Svg):
    """Svg with a transparent background (no background rectangle)."""

    def write(self, name):
        head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
                'font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif">' % (self.w, self.h, self.w, self.h))
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w", encoding="utf-8").write(
            head + "\n".join(self.items) + "</svg>\n")
        print("wrote", name)


class Frame:
    """Maps example coordinates (box) into an SVG rectangle, with a title above it and no panel box."""

    def __init__(self, svg, x, y, w, h, box, title):
        bx0, by0, bx1, by1 = box
        self.s = min(w / (bx1 - bx0), (h - 24) / (by1 - by0))
        self.ox = x + (w - self.s * (bx1 - bx0)) / 2 - self.s * bx0
        self.oy = y + 24 + (h - 24 - self.s * (by1 - by0)) / 2 - self.s * by0
        svg.text((x + 2, y + 13), title, 12.5, TEXT, weight="600")

    def p(self, q):
        return (self.ox + self.s * q[0], self.oy + self.s * q[1])


def clip_polygon(poly, tri):
    """poly clipped by the three half-planes of the counter-clockwise triangle tri (Sutherland-Hodgman)."""
    for e in range(3):
        a, b = tri[e], tri[(e + 1) % 3]
        edge = sub(b, a)
        side = lambda v: cross2(edge, sub(v, a))  # noqa: E731
        out = []
        for i in range(len(poly)):
            u, v = poly[i], poly[(i + 1) % len(poly)]
            su, sv = side(u), side(v)
            if su >= 0:
                out.append(u)
            if (su >= 0) != (sv >= 0):
                out.append(add(u, mul(sub(v, u), su / (su - sv))))
        poly = out
        if not poly:
            break
    return poly


def polygon_area(poly):
    return 0.5 * abs(sum(cross2(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))))


def polygon_centroid(poly):
    area, cx, cy = 0.0, 0.0, 0.0
    for i in range(len(poly)):
        u, v = poly[i], poly[(i + 1) % len(poly)]
        c = cross2(u, v)
        area += c
        cx += c * (u[0] + v[0])
        cy += c * (u[1] + v[1])
    return (cx / (3 * area), cy / (3 * area))


def separating_axis(p, q):
    """The unit axis along which triangles p and q overlap least, pointing from p to q, and that overlap."""
    best, axis = float("inf"), None
    for t in (p, q):
        for e in range(3):
            edge = sub(t[(e + 1) % 3], t[e])
            n = mul((-edge[1], edge[0]), 1.0 / math.hypot(*edge))
            pd = [n[0] * v[0] + n[1] * v[1] for v in p]
            qd = [n[0] * v[0] + n[1] * v[1] for v in q]
            if max(pd) - min(qd) < best:
                best, axis = max(pd) - min(qd), n
            if max(qd) - min(pd) < best:
                best, axis = max(qd) - min(pd), mul(n, -1.0)
    return axis, best


def diagram():
    """Six rigid isometric triangles of one chart around a shared vertex, mid-simulation (turned and moved, so the
    gaps show; one pair overlapping), with the forces on them. Transparent background."""
    outer = [(math.cos(math.radians(a)) * r, math.sin(math.radians(a)) * r)
             for a, r in ((0, 1.0), (52, 0.82), (118, 1.12), (183, 0.88), (236, 1.08), (305, 0.86))]
    base = [[(0.0, 0.0), outer[k], outer[(k + 1) % 6]] for k in range(6)]
    moves = [(6, (0.03, -0.02)), (-7, (0.0, 0.04)), (4, (-0.03, 0.02)),
             (-5, (-0.02, -0.03)), (8, (0.02, -0.04)), (-4, (0.04, 0.01))]
    scale = 0.9  # every triangle at the same scale: rigid bodies keep their size
    tris = []
    for t, (deg, shift) in zip(base, moves):
        c = tri_centroid(t)
        outward = mul(c, 0.1 / math.hypot(*c))  # apart from the shared vertex, so the gaps show
        tris.append([add(add(add(c, shift), outward), mul(rot(sub(q, c), math.radians(deg)), scale)) for q in t])
    # Triangle 3 slid toward triangle 2 until they overlap by about 12% of triangle 2's area.
    toward = sub(tri_centroid(tris[2]), tri_centroid(tris[3]))
    toward = mul(toward, 0.005 / math.hypot(*toward))
    overlap = []
    for _ in range(200):
        overlap = clip_polygon(list(tris[2]), tris[3])
        if len(overlap) >= 3 and polygon_area(overlap) > 0.12 * tri_area(tris[2]):
            break
        tris[3] = [add(q, toward) for q in tris[3]]

    svg = ClearSvg(1040, 360)
    svg.add('<defs>' + ''.join(
        '<marker id="arr-%s" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        '<path d="M0 0L10 5L0 10z" fill="%s"/></marker>' % (c[1:], c) for c in (SPRING, PUSH, VELOCITY, TURN)) + '</defs>')
    panel = Frame(svg, 10, 6, 640, 348, (-1.35, -1.3, 1.35, 1.3), "Forces on one chart's triangles")
    P = panel.p
    for t in tris:
        svg.poly([P(q) for q in t], FILL, "#0d1117", 1.0)
    if len(overlap) >= 3:
        svg.poly([P(q) for q in overlap], PUSH, PUSH, 1.0, extra='fill-opacity="0.55"')

    # Vertex <-> vertex springs: the copies of one atlas vertex pulled toward each other. Around the rim, the
    # corner pairs (corner 2 of triangle k, corner 1 of triangle k + 1); at the middle, the six copies of the
    # shared vertex, each pulled toward the others (drawn toward their mean).
    for k in range(6):
        a, b = tris[k][2], tris[(k + 1) % 6][1]
        if math.dist(P(a), P(b)) > 8:
            mid = mul(add(a, b), 0.5)
            arrow(svg, P(a), P(add(a, mul(sub(mid, a), 0.8))), SPRING, 1.4)
            arrow(svg, P(b), P(add(b, mul(sub(mid, b), 0.8))), SPRING, 1.4)
    centre = mul(tuple(map(sum, zip(*[t[0] for t in tris]))), 1.0 / 6)
    for t in tris:
        if math.dist(P(t[0]), P(centre)) > 8:
            arrow(svg, P(t[0]), P(add(t[0], mul(sub(centre, t[0]), 0.75))), SPRING, 1.4)

    # Repulsion: an impulse -j n on one triangle and +j n on the other, at the centroid of their overlap.
    if len(overlap) >= 3:
        n, _ = separating_axis(tris[2], tris[3])
        at = polygon_centroid(overlap)
        arrow(svg, P(at), P(add(at, mul(n, -0.3))), PUSH, 2.2)
        arrow(svg, P(at), P(add(at, mul(n, 0.3))), PUSH, 2.2)
        svg.dot(P(at), 3.0, PUSH)

    # Velocity of a body (its centroid), carried over from step to step.
    c = tri_centroid(tris[0])
    svg.dot(P(c), 2.6, VELOCITY)
    arrow(svg, P(c), P(add(c, (-0.24, -0.1))), VELOCITY, 2.0)

    # Spin about the centroid.
    c = P(tri_centroid(tris[4]))
    r = 20.0
    a0, a1 = math.radians(200), math.radians(330)
    p0 = (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0))
    p1 = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
    svg.add('<path d="M%.2f %.2f A%.1f %.1f 0 0 1 %.2f %.2f" fill="none" stroke="%s" stroke-width="1.8" '
            'marker-end="url(#arr-%s)"/>' % (p0[0], p0[1], r, r, p1[0], p1[1], TURN, TURN[1:]))
    svg.dot(c, 2.6, TURN)

    # Legend.
    x, y = 676, 70
    svg.text((x, y - 34), "Each triangle is a rigid body", 12.5, TEXT, weight="600")
    svg.text((x, y - 16), "(its angles and area stay its 3D ones, at the chart's scale)", 12, MUTED)
    rows = ((SPRING, "Vertex ↔ vertex spring", "corners that were one atlas vertex are pulled",
             "together (weak while overlaps are pushed apart)"),
            (PUSH, "Repulsion impulse", "an overlapping pair (shaded) is pushed apart,",
             "∓j n at the centroid of their overlap"),
            (VELOCITY, "Velocity", "forces and impulses change each body's",
             "velocity, which carries over between steps"),
            (TURN, "Spin", "torques turn each triangle about its centroid;",
             "its shape and size never change"))
    for k, (color, title, line1, line2) in enumerate(rows):
        yy = y + 26 + k * 62
        arrow(svg, (x, yy - 4), (x + 30, yy - 4), color, 2.0)
        svg.text((x + 40, yy), title, 12.5, TEXT, weight="600")
        svg.text((x + 40, yy + 17), line1, 12, MUTED)
        svg.text((x + 40, yy + 32), line2, 12, MUTED)
    svg.write("diagram-isometric.svg")


# ---------------------------------------------------------------- the levels


def corner_pairs(faces, regions):
    edges = {}
    for f, t in enumerate(faces):
        for c in range(3):
            edges.setdefault(tuple(sorted((t[c], t[(c + 1) % 3]))), []).append(f)
    pairs = []
    for (u, v), fs in edges.items():
        if len(fs) != 2 or regions[fs[0]] != regions[fs[1]]:
            continue
        a, b = fs
        for w in (u, v):
            pairs.append((a, list(faces[a]).index(w), b, list(faces[b]).index(w)))
    return pairs


def angles3(a, b, c):
    def ang(p, q, r):
        u = [q[k] - p[k] for k in range(3)]
        v = [r[k] - p[k] for k in range(3)]
        cx = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        return math.atan2(math.sqrt(sum(x * x for x in cx)), sum(u[k] * v[k] for k in range(3)))
    return (ang(a, b, c), ang(b, c, a), ang(c, a, b))


def checks(path, raster=1024):
    """Flips, overlapping pixels, angle error, area ratio spread and corner gaps of one saved level."""
    pos, faces, tex, regions = read_ply(path)
    regions = regions or [0] * len(faces)
    flips = 0
    worst_angle = 0.0
    ratio = []
    for f, t in enumerate(faces):
        uv = [(tex[f][2 * c], tex[f][2 * c + 1]) for c in range(3)]
        a2 = tri_area(uv)
        flips += a2 <= 0
        u3 = angles3(*(pos[i] for i in t))
        u2 = angles3(*((p[0], p[1], 0.0) for p in uv))
        worst_angle = max(worst_angle, max(abs(x - y) for x, y in zip(u2, u3)))
        x = flatten3(*(pos[i] for i in t))
        ratio.append(abs(a2) / tri_area(x) if tri_area(x) > 0 else 0.0)
    by_region = {}
    for f, r in enumerate(regions):
        by_region.setdefault(r, []).append(ratio[f])
    median = {r: sorted(v)[len(v) // 2] for r, v in by_region.items()}
    rel = [ratio[f] / median[regions[f]] for f in range(len(faces)) if median[regions[f]] > 0]
    pairs = corner_pairs(faces, regions)
    gap = sum(math.hypot(tex[fa][2 * ka] - tex[fb][2 * kb], tex[fa][2 * ka + 1] - tex[fb][2 * kb + 1])
              for fa, ka, fb, kb in pairs) / max(1, len(pairs))
    # pixels whose centre lies inside more than one triangle, at raster x raster
    cover = bytearray(raster * raster)
    overlap = 0
    for f in range(len(faces)):
        p = [(tex[f][2 * c] * raster, tex[f][2 * c + 1] * raster) for c in range(3)]
        a = tri_area(p)
        if a == 0:
            continue
        x0, x1 = max(0, int(min(q[0] for q in p))), min(raster - 1, int(max(q[0] for q in p)))
        y0, y1 = max(0, int(min(q[1] for q in p))), min(raster - 1, int(max(q[1] for q in p)))
        for y in range(y0, y1 + 1):
            py = y + 0.5
            for x in range(x0, x1 + 1):
                px = x + 0.5
                w0 = cross2(sub(p[1], (px, py)), sub(p[2], (px, py))) / (2 * a)
                w1 = cross2(sub(p[2], (px, py)), sub(p[0], (px, py))) / (2 * a)
                if w0 < 0 or w1 < 0 or 1 - w0 - w1 < 0:
                    continue
                i = y * raster + x
                if cover[i]:
                    overlap += 1
                cover[i] = min(255, cover[i] + 1)
    return {"triangles": len(faces), "charts": len(by_region), "flips": flips, "overlap": overlap,
            "angle": math.degrees(worst_angle), "ratio": (min(rel), max(rel)), "gap": gap * raster}


def render(path, size, supersample):
    _, faces, texcoords, regions = read_ply(path)
    s = size * supersample
    canvas = Canvas(s, s, background=(0.09, 0.105, 0.13))
    for k, uv in enumerate(texcoords):
        region = regions[k] if regions else 0
        pts = [(uv[2 * c] * s, (1.0 - uv[2 * c + 1]) * s, 0.0) for c in range(3)]
        canvas.triangle(pts[0], pts[1], pts[2], region_color(region, 0.45, 0.9), k, wire=0.55, wire_px=0.9)
    return canvas.downsample(supersample)


def levels(args):
    top, bottom = [], []
    for level in args.levels:
        for folder, row in ((args.atlas, top), (args.isometric, bottom)):
            row.append(render(os.path.join(folder, "level_%d.ply" % level), args.size, args.supersample))
    for level in args.levels:
        for name, folder in (("atlas", args.atlas), ("isometric", args.isometric)):
            c = checks(os.path.join(folder, "level_%d.ply" % level))
            print("level %d %-9s %6d triangles %4d charts  flips %d  overlapping pixels %d  worst angle error %.4f deg  "
                  "area ratio %.3f..%.3f  mean corner gap %.3f px"
                  % (level, name, c["triangles"], c["charts"], c["flips"], c["overlap"], c["angle"], c["ratio"][0],
                     c["ratio"][1], c["gap"]), flush=True)
    gap = 8
    if len(args.levels) == 1:
        # One level: the atlas and the isometric UVs side by side.
        write_png(args.out, [top[0], bottom[0]], gap=gap)
        return
    width = sum(c.w for c in top) + gap * (len(top) - 1)
    rows = Canvas(width, 2 * top[0].h + gap, background=(0.051, 0.067, 0.090))
    for row_index, row in enumerate((top, bottom)):
        x = 0
        for c in row:
            for y in range(c.h):
                dst = (row_index * (c.h + gap) + y) * rows.w + x
                rows.rgb[dst:dst + c.w] = c.rgb[y * c.w:(y + 1) * c.w]
            x += c.w + gap
    write_png(args.out, [rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = ap.add_subparsers(dest="command", required=True)
    commands.add_parser("diagram")
    lv = commands.add_parser("levels")
    lv.add_argument("atlas")
    lv.add_argument("isometric")
    lv.add_argument("out")
    lv.add_argument("--levels", type=int, nargs="+", required=True)
    lv.add_argument("--size", type=int, default=300)
    lv.add_argument("--supersample", type=int, default=2)
    args = ap.parse_args()
    if args.command == "diagram":
        diagram()
    else:
        levels(args)


if __name__ == "__main__":
    main()
