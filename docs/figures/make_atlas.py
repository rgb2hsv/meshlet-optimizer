#!/usr/bin/env python3
"""Draws the UV atlases of hierarchy levels saved by `meshlet-optimizer --save-levels DIR` side by side: every
triangle at its per-corner texture coordinates, colored by region, region outlines dark, on the unit square.

  python make_atlas.py DIR OUT.png --levels 0 3 6 9 [--size 300]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plyfig import Canvas, read_ply, region_color, write_png  # noqa: E402


def render(path, size, supersample):
    _, faces, texcoords, regions = read_ply(path)
    assert texcoords, path + " has no texture coordinates"
    s = size * supersample
    canvas = Canvas(s, s, background=(0.09, 0.105, 0.13))
    for k, uv in enumerate(texcoords):
        region = regions[k] if regions else 0
        pts = [(uv[2 * c] * s, (1.0 - uv[2 * c + 1]) * s, 0.0) for c in range(3)]
        canvas.triangle(pts[0], pts[1], pts[2], region_color(region, 0.45, 0.9), region)
    canvas.outline_ids(width=1, darkness=0.7)
    return canvas.downsample(supersample), len(faces), (max(regions) + 1 if regions else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--levels", type=int, nargs="+", required=True)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--supersample", type=int, default=2)
    args = ap.parse_args()
    panels = []
    for level in args.levels:
        canvas, faces, regions = render(os.path.join(args.dir, "level_%d.ply" % level), args.size, args.supersample)
        print("level %d: %d triangles, %d charts" % (level, faces, regions))
        panels.append(canvas)
    write_png(args.out, panels, gap=8)


if __name__ == "__main__":
    main()
