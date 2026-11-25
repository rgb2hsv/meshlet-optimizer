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
    for level in (0, 3, 6):
        shot(viewer, metasphere, "viewer-metasphere-level%d.png" % level, "--lod-level", level)
    # Automatic LOD selection close up: levels 1-3 drawn side by side, meeting along shared borders.
    shot(viewer, metasphere, "viewer-metasphere-lod-cut.png", "--lod-error", 8, "--zoom", 1.7, "--pitch", 35)
    # The generated per-level atlas (no input UVs), shown as a UV grid.
    shot(viewer, metasphere, "viewer-metasphere-atlas-level2.png", "--lod-level", 2, "--uv-grid",
         "--uv-grid-cells", 48)
    # Cosplay's own texture coordinates carried through the levels, shown as a UV grid.
    for level in (0, 4, 9):
        shot(viewer, cosplay, "viewer-cosplay-uv-level%d.png" % level, "--lod-level", level, "--uv-grid",
             "--uv-grid-cells", 32, "--yaw", 0, "--pitch", 5, "--zoom", 0.9, width=700, height=1000)


if __name__ == "__main__":
    main()
