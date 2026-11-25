#!/usr/bin/env python3
"""Screenshots of meshlet-optimizer output in meshlet-viewer (its --screenshot option, no UI).

  python make_viewer_shots.py VIEWER metasphere.mlod cosplay.mlod

VIEWER is the meshlet-viewer executable (meshlets / meshlets.exe); the .mlod files are written by
`meshlet-optimizer assets/model/metasphere.ply` and `meshlet-optimizer assets/model/cosplay.ply`.
Writes viewer-*.png next to this script.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The figure from the front, aimed at the middle of its bounding box (the default target, the vertex centroid,
# sits at the chest and would crop the feet).
COSPLAY_VIEW = ("--yaw", 0, "--pitch", 5, "--target", 0, 0, 0, "--zoom", 1.1)


def shot(viewer, model, name, *args, width=900, height=900):
    out = os.path.join(HERE, name)
    cmd = [viewer, model, "--no-ui", "--width", str(width), "--height", str(height), *map(str, args),
           "--screenshot", out]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    viewer, metasphere, cosplay = sys.argv[1:]
    # Fixed DAG levels: each region is one meshlet (black outlines), colored by level.
    for level in (0, 2, 4, 6, 9):
        shot(viewer, metasphere, "viewer-metasphere-level%d.png" % level, "--lod-level", level,
             width=700, height=700)
    # Automatic LOD selection close up: levels 1-3 drawn side by side, meeting along shared borders.
    shot(viewer, metasphere, "viewer-metasphere-lod-cut.png", "--lod-error", 8, "--zoom", 1.7, "--pitch", 35)
    # The generated per-level atlas (no input UVs), shown as a UV grid.
    shot(viewer, metasphere, "viewer-metasphere-atlas-level2.png", "--lod-level", 2, "--uv-grid",
         "--uv-grid-cells", 48)
    # Cosplay at fixed DAG levels: meshlets in their level's color, outlined.
    for level in (1, 3, 5, 12):
        shot(viewer, cosplay, "viewer-cosplay-level%d.png" % level, "--lod-level", level, *COSPLAY_VIEW,
             width=700, height=1000)
    # Cosplay's own texture coordinates carried through the levels, shown as a UV grid.
    for level in (0, 4, 9):
        shot(viewer, cosplay, "viewer-cosplay-uv-level%d.png" % level, "--lod-level", level, "--uv-grid",
             "--uv-grid-cells", 32, *COSPLAY_VIEW, width=700, height=1000)


if __name__ == "__main__":
    main()
