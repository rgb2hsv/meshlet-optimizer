#!/usr/bin/env python3
"""Renders hierarchy levels saved by `meshlet-optimizer --save-levels DIR` side by side, each face colored by
its region, region borders dark and triangle edges drawn faintly where triangles are large enough to see.

  python make_level_renders.py DIR OUT.png --levels 0 2 4 6 9 [--yaw 30 --pitch 20 --size 360]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plyfig import Canvas, read_ply, region_color, write_png  # noqa: E402


def rotation(yaw_deg, pitch_deg):
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))

    def rotate(p):
        x, y, z = p
        x, z = cy * x + sy * z, -sy * x + cy * z  # yaw about y
        y, z = cp * y - sp * z, sp * y + cp * z  # pitch about x
        return x, y, z

    return rotate


def render(path, size, supersample, rotate, frame, aspect=1.0):
    positions, faces, _, regions = read_ply(path)
    view = [rotate(p) for p in positions]
    (cx, cy), half = frame
    s = size * supersample
    sw = int(round(s * aspect))
    scale = 0.5 * s * 0.94 / half

    def screen(p):
        return (sw * 0.5 + (p[0] - cx) * scale, s * 0.5 - (p[1] - cy) * scale, -p[2])

    pts = [screen(p) for p in view]
    edge_px = 0.0
    for f in faces[:2000]:
        a, b = pts[f[0]], pts[f[1]]
        edge_px += math.hypot(a[0] - b[0], a[1] - b[1])
    edge_px /= max(1, min(len(faces), 2000) * supersample)
    wire = max(0.0, min(0.45, (edge_px - 3.0) / 8.0))  # faint edges only once triangles are big on screen
    light = (-0.35, 0.55, 0.76)
    canvas = Canvas(sw, s)
    for k, f in enumerate(faces):
        a, b, c = view[f[0]], view[f[1]], view[f[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        n = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        lambert = abs(nx * light[0] + ny * light[1] + nz * light[2]) / n
        shade = 0.38 + 0.62 * lambert
        r, g, bl = region_color(regions[k] if regions else 0)
        canvas.triangle(pts[f[0]], pts[f[1]], pts[f[2]], (r * shade, g * shade, bl * shade),
                        regions[k] if regions else 0, wire, 0.9 * supersample)
    canvas.outline_ids(width=supersample, darkness=0.82)
    return canvas.downsample(supersample), len(faces), (max(regions) + 1 if regions else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--levels", type=int, nargs="+", required=True)
    ap.add_argument("--yaw", type=float, default=30.0)
    ap.add_argument("--pitch", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--aspect", type=float, default=1.0, help="panel width / height")
    args = ap.parse_args()

    rotate = rotation(args.yaw, args.pitch)
    # One frame for every panel, from the finest level shown.
    positions, _, _, _ = read_ply(os.path.join(args.dir, "level_%d.ply" % args.levels[0]))
    view = [rotate(p) for p in positions]
    lo = [min(p[i] for p in view) for i in range(2)]
    hi = [max(p[i] for p in view) for i in range(2)]
    frame = (((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), max((hi[0] - lo[0]) / args.aspect, hi[1] - lo[1]) / 2)

    panels = []
    for level in args.levels:
        canvas, faces, regions = render(os.path.join(args.dir, "level_%d.ply" % level), args.size,
                                        args.supersample, rotate, frame, args.aspect)
        print("level %d: %d triangles, %d regions" % (level, faces, regions))
        panels.append(canvas)
    write_png(args.out, panels, gap=8)


if __name__ == "__main__":
    main()
