#!/usr/bin/env python3
"""Step diagrams of the method, computed on a small 2D example with the same algorithms the tool runs on
surfaces (simplified to the plane): restricted CVT charts, clique merging, remeshing one group, nested border
simplification, border relaxation, bin packing and the region DAG.

  python make_diagrams.py        (writes diagram-*.svg next to this script; standard library only)
"""

import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#151b23"
PANEL = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#9198a1"
FAINT = "#3d444d"
ACCENT = "#4493f8"
RED = "#ff7b72"
GOOD = "#3fb950"
W, H = 360.0, 240.0  # example domain


# ---------------------------------------------------------------- geometry


def circumcircle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return (0.0, 0.0, math.inf)
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2)


def delaunay(points):
    """Bowyer-Watson; returns triangles as index triples (counter-clockwise)."""
    pts = list(points)
    n = len(pts)
    big = 1e5
    pts += [(-big, -big), (big, -big), (0.0, big)]
    tris = [(n, n + 1, n + 2)]
    circ = {tris[0]: circumcircle(*(pts[i] for i in tris[0]))}
    for i in range(n):
        p = pts[i]
        bad = [t for t in tris if (p[0] - circ[t][0]) ** 2 + (p[1] - circ[t][1]) ** 2 < circ[t][2]]
        edges = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = tuple(sorted(e))
                edges[key] = None if key in edges else e
        bad_set = set(bad)
        tris = [t for t in tris if t not in bad_set]
        for t in bad:
            del circ[t]
        for e in edges.values():
            if e is None:
                continue
            t = (e[0], e[1], i)
            tris.append(t)
            circ[t] = circumcircle(*(pts[j] for j in t))
    out = []
    for t in tris:
        if max(t) >= n:
            continue
        a, b, c = (pts[j] for j in t)
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            t = (t[0], t[2], t[1])
        out.append(t)
    return out


def centroid(pts, tri):
    return tuple(sum(pts[i][k] for i in tri) / 3.0 for k in range(2))


def area(pts, tri):
    a, b, c = (pts[i] for i in tri)
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_in_tri(p, a, b, c):
    def s(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2, d3 = s(a, b, p), s(b, c, p), s(c, a, p)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def blue_noise(rng, count, radius, inside, existing=(), tries=40):
    """Dart throwing: up to count points, each at least radius from every other point and from existing."""
    out = []
    for _ in range(count * tries):
        if len(out) >= count:
            break
        p = inside(rng)
        if p is None:
            continue
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= radius * radius for q in list(existing) + out):
            out.append(p)
    return out


def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------- SVG


class Svg:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.items = []

    def add(self, s):
        self.items.append(s)

    def poly(self, pts, fill="none", stroke="none", sw=1.0, opacity=1.0, extra=""):
        d = " ".join("%.2f,%.2f" % p for p in pts)
        self.add('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
                 'opacity="%.3f" %s/>' % (d, fill, stroke, sw, opacity, extra))

    def line(self, a, b, stroke, sw=1.0, extra=""):
        self.add('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="%.2f" '
                 'stroke-linecap="round" %s/>' % (a[0], a[1], b[0], b[1], stroke, sw, extra))

    def path(self, pts, stroke, sw=1.0, extra=""):
        d = "M" + " L".join("%.2f %.2f" % p for p in pts)
        self.add('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round" '
                 'stroke-linecap="round" %s/>' % (d, stroke, sw, extra))

    def dot(self, p, r, fill, stroke="none", sw=1.0):
        self.add('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s" stroke="%s" stroke-width="%.2f"/>'
                 % (p[0], p[1], r, fill, stroke, sw))

    def text(self, p, s, size=12, fill=TEXT, anchor="start", weight="400"):
        self.add('<text x="%.1f" y="%.1f" font-size="%.1f" fill="%s" text-anchor="%s" font-weight="%s">%s</text>'
                 % (p[0], p[1], size, fill, anchor, weight, s))

    def rect(self, x, y, w, h, fill, stroke="none", sw=1.0, rx=0):
        self.add('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="%.1f" fill="%s" stroke="%s" '
                 'stroke-width="%.2f"/>' % (x, y, w, h, rx, fill, stroke, sw))

    def write(self, name):
        head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
                'font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif">'
                '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
                'orient="auto"><path d="M0 0L10 5L0 10z" fill="%s"/></marker></defs>'
                '<rect width="100%%" height="100%%" fill="%s"/>' % (self.w, self.h, self.w, self.h, MUTED, BG))
        open(os.path.join(HERE, name), "w", encoding="utf-8").write(head + "\n".join(self.items) + "</svg>\n")
        print("wrote", name)


def color(i, s=0.45, v=0.9):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.6180339887 + 0.08) % 1.0, s, v)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


class Panel:
    """A panel of a diagram: maps example coordinates (box) into an SVG rectangle."""

    def __init__(self, svg, x, y, w, h, box, title):
        self.svg = svg
        bx0, by0, bx1, by1 = box
        self.s = min(w / (bx1 - bx0), (h - 24) / (by1 - by0))
        self.ox = x + (w - self.s * (bx1 - bx0)) / 2 - self.s * bx0
        self.oy = y + 24 + (h - 24 - self.s * (by1 - by0)) / 2 - self.s * by0
        svg.rect(x, y + 20, w, h - 20, PANEL, FAINT, 1, 6)
        svg.text((x + 2, y + 13), title, 12.5, TEXT, weight="600")

    def p(self, q):
        return (self.ox + self.s * q[0], self.oy + self.s * q[1])


# ---------------------------------------------------------------- the example


def fine_mesh(rng):
    """Jittered grid plus boundary points, Delaunay triangulated: the example's 'input surface'."""
    pts = []
    step = 10.0
    nx, ny = int(W / step), int(H / step)
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = i * step, j * step
            if 0 < i < nx and 0 < j < ny:
                x += rng.uniform(-3.2, 3.2)
                y += rng.uniform(-3.2, 3.2)
            pts.append((x, y))
    return pts, delaunay(pts)


def label_faces(pts, tris, seeds):
    labels = []
    for t in tris:
        c = centroid(pts, t)
        labels.append(min(range(len(seeds)), key=lambda k: (c[0] - seeds[k][0]) ** 2 + (c[1] - seeds[k][1]) ** 2))
    return labels


def lloyd(pts, tris, seeds, iterations):
    for _ in range(iterations):
        labels = label_faces(pts, tris, seeds)
        acc = [[0.0, 0.0, 0.0] for _ in seeds]
        for t, l in zip(tris, labels):
            a = area(pts, t)
            c = centroid(pts, t)
            acc[l][0] += a * c[0]
            acc[l][1] += a * c[1]
            acc[l][2] += a
        seeds = [(x / m, y / m) if m > 0 else s for (x, y, m), s in zip(acc, seeds)]
    return seeds


def edge_faces(tris):
    ef = {}
    for f, t in enumerate(tris):
        for k in range(3):
            e = tuple(sorted((t[k], t[(k + 1) % 3])))
            ef.setdefault(e, []).append(f)
    return ef


def draw_mesh(panel, pts, tris, labels, colors, border_w=1.6, wire=True, border_color="#0d1117"):
    svg = panel.svg
    for t, l in zip(tris, labels):
        svg.poly([panel.p(pts[i]) for i in t], colors[l], colors[l] if not wire else "#0d1117",
                 0.35 if wire else 0.4, extra='stroke-opacity="%s"' % ("0.35" if wire else "1"))
    for e, fs in edge_faces(tris).items():
        if len(fs) == 2 and labels[fs[0]] != labels[fs[1]]:
            svg.line(panel.p(pts[e[0]]), panel.p(pts[e[1]]), border_color, border_w)


def adjacency(tris, labels, n):
    adj = [set() for _ in range(n)]
    for e, fs in edge_faces(tris).items():
        if len(fs) == 2 and labels[fs[0]] != labels[fs[1]]:
            a, b = labels[fs[0]], labels[fs[1]]
            adj[a].add(b)
            adj[b].add(a)
    return adj


def clique_groups(adj, weight, rng):
    """RegionHierarchy's grouping: BFS from a random node, largest clique (<= 4) of ungrouped nodes, lightest
    among equal sizes; leftover singles join the smallest neighbouring group."""
    n = len(adj)
    group = [-1] * n
    cliques = []
    order = list(range(n))
    rng.shuffle(order)
    queue = []
    for start in order:
        if group[start] >= 0:
            continue
        queue.append(start)
        while queue:
            v = queue.pop(0)
            if group[v] >= 0:
                continue
            free = [u for u in adj[v] if group[u] < 0]
            best = [v]
            for size in (4, 3, 2):
                cands = []
                import itertools
                for combo in itertools.combinations(free, size - 1):
                    if all(b in adj[a] for a, b in itertools.combinations(combo, 2)):
                        cands.append((sum(weight[u] for u in combo), [v] + list(combo)))
                if cands:
                    best = min(cands)[1]
                    break
            gid = len(cliques)
            cliques.append(best)
            for u in best:
                group[u] = gid
            for u in best:
                for w2 in adj[u]:
                    if group[w2] < 0:
                        queue.append(w2)
    absorbed = []
    sizes = [len(c) for c in cliques]
    for gid, c in enumerate(cliques):
        if len(c) == 1:
            v = c[0]
            nbr = [group[u] for u in adj[v] if sizes[group[u]] > 1]
            if nbr:
                g2 = min(nbr, key=lambda g: sizes[g])
                group[v] = g2
                absorbed.append(v)
    remap = {}
    for g in group:
        remap.setdefault(g, len(remap))
    return [remap[g] for g in group], cliques, absorbed


def boundary_edges(tris, faces):
    fs = set(faces)
    count = {}
    for f in faces:
        t = tris[f]
        for k in range(3):
            e = (t[k], t[(k + 1) % 3])
            key = tuple(sorted(e))
            count[key] = count.get(key, 0) + 1
    return [e for e, c in count.items() if c == 1]


def chain_loops(edges):
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen = set()
    loops = []
    for s in nbr:
        if s in seen:
            continue
        loop = [s]
        seen.add(s)
        prev, cur = None, s
        while True:
            nxt = [u for u in nbr[cur] if u != prev and u not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def vertex_labels(tris, labels, n_verts):
    around = [set() for _ in range(n_verts)]
    for t, l in zip(tris, labels):
        for v in t:
            around[v].add(l)
    return around


def junctions(pts, tris, labels):
    """Vertices where three or more regions meet, or where a border reaches the domain's edge."""
    around = vertex_labels(tris, labels, len(pts))
    on_edge = {v for e, fs in edge_faces(tris).items() if len(fs) == 1 for v in e}
    return [v for v in range(len(pts)) if len(around[v]) >= 3 or (len(around[v]) == 2 and v in on_edge)]


def straighten_seams(pts, tris, labels, weight=2.0):
    """SeamStraightening in the plane: each border chain between two regions is replaced by the cheapest
    edge path between its two junctions through the two regions, an edge costing its length stretched by its
    distance from the bisector of the regions' centroids; the faces between old and new path change sides."""
    import heapq
    labels = list(labels)
    ef = edge_faces(tris)
    mean_edge = sum(math.dist(pts[a], pts[b]) for a, b in ef) / len(ef)
    n = max(labels) + 1
    for a in range(n):
        for b in range(a + 1, n):
            chain = [e for e, fs in ef.items() if len(fs) == 2 and {labels[fs[0]], labels[fs[1]]} == {a, b}]
            if not chain:
                continue
            deg = {}
            for e in chain:
                for v in e:
                    deg[v] = deg.get(v, 0) + 1
            ends = [v for v, d in deg.items() if d == 1]
            if len(ends) != 2:
                continue  # several chains or a closed loop: keep
            cent = []
            for r in (a, b):
                fs = [f for f, l in enumerate(labels) if l == r]
                cent.append((sum(centroid(pts, tris[f])[0] for f in fs) / len(fs),
                             sum(centroid(pts, tris[f])[1] for f in fs) / len(fs)))
            mid = ((cent[0][0] + cent[1][0]) / 2, (cent[0][1] + cent[1][1]) / 2)
            nx, ny = cent[1][0] - cent[0][0], cent[1][1] - cent[0][1]
            nn = math.hypot(nx, ny) or 1.0
            nx, ny = nx / nn, ny / nn
            around = vertex_labels(tris, labels, len(pts))
            allowed = {v for v in range(len(pts)) if around[v] and around[v] <= {a, b}} | set(ends) | set(deg)
            nbr = {}
            for e, fs in ef.items():
                if all(labels[f] in (a, b) for f in fs) and e[0] in allowed and e[1] in allowed:
                    nbr.setdefault(e[0], []).append(e[1])
                    nbr.setdefault(e[1], []).append(e[0])
            src, dst = ends
            dist = {src: 0.0}
            prev = {}
            heap = [(0.0, src)]
            while heap:
                d, v = heapq.heappop(heap)
                if v == dst:
                    break
                if d > dist[v]:
                    continue
                for u in nbr.get(v, []):
                    m = ((pts[u][0] + pts[v][0]) / 2, (pts[u][1] + pts[v][1]) / 2)
                    off = abs((m[0] - mid[0]) * nx + (m[1] - mid[1]) * ny) / mean_edge
                    nd = d + math.dist(pts[u], pts[v]) * (1.0 + weight * off)
                    if nd < dist.get(u, math.inf):
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
            if dst not in prev:
                continue
            path = set()
            v = dst
            while v != src:
                path.add(tuple(sorted((v, prev[v]))))
                v = prev[v]
            # pieces of the two regions cut along the new path; each piece goes to the side of the centroid it is on
            faces = [f for f, l in enumerate(labels) if l in (a, b)]
            comp = {}
            for f in faces:
                if f in comp:
                    continue
                comp[f] = f
                stack = [f]
                while stack:
                    g = stack.pop()
                    t = tris[g]
                    for k in range(3):
                        e = tuple(sorted((t[k], t[(k + 1) % 3])))
                        if e in path:
                            continue
                        for h in ef[e]:
                            if h not in comp and labels[h] in (a, b):
                                comp[h] = f
                                stack.append(h)
            pieces = {}
            for f in faces:
                pieces.setdefault(comp[f], []).append(f)
            if len(pieces) != 2:
                continue
            new = list(labels)
            for fs in pieces.values():
                votes = sum(1 if labels[f] == a else -1 for f in fs)
                for f in fs:
                    new[f] = a if votes >= 0 else b
            if len({new[f] for f in faces}) == 2:
                labels = new
    return labels


def zipper(svg, panel, upper, lower, fill, sw=0.7):
    """Triangle strip between two x-sorted polylines."""
    i = j = 0
    while i < len(upper) - 1 or j < len(lower) - 1:
        if j == len(lower) - 1 or (i < len(upper) - 1 and upper[i + 1][0] <= lower[j + 1][0]):
            svg.poly([panel.p(upper[i]), panel.p(upper[i + 1]), panel.p(lower[j])], fill, "#0d1117", sw)
            i += 1
        else:
            svg.poly([panel.p(upper[i]), panel.p(lower[j + 1]), panel.p(lower[j])], fill, "#0d1117", sw)
            j += 1


# ---------------------------------------------------------------- diagrams


def diagram_cvt(pts, tris, seeds0, seeds1, labels1, straight):
    svg = Svg(1040, 262)
    box = (0, 0, W, H)
    n = len(seeds0)
    cols = [color(i) for i in range(n)]
    p = Panel(svg, 10, 6, 330, 250, box, "(a) Blue-noise seeds, nearest-seed labels")
    draw_mesh(p, pts, tris, label_faces(pts, tris, seeds0), cols)
    for s in seeds0:
        svg.dot(p.p(s), 3.2, "#ffffff", "#0d1117", 1.2)
    p = Panel(svg, 355, 6, 330, 250, box, "(b) Lloyd: seeds move to cell centroids")
    draw_mesh(p, pts, tris, labels1, cols)
    for a, b in zip(seeds0, seeds1):
        svg.line(p.p(a), p.p(b), "#ffffff", 1.0, 'marker-end="url(#arr)" opacity="0.7"')
    for s in seeds1:
        svg.dot(p.p(s), 3.2, "#ffffff", "#0d1117", 1.2)
    p = Panel(svg, 700, 6, 330, 250, box, "(c) Seams straightened between junctions")
    draw_mesh(p, pts, tris, straight, cols, border_w=2.2, wire=True)
    for v in junctions(pts, tris, straight):
        svg.dot(p.p(pts[v]), 2.6, "#ffffff", "#0d1117", 0.8)
    svg.write("diagram-cvt.svg")


def diagram_cliques(pts, tris, labels, seeds, group, cliques, absorbed):
    svg = Svg(1040, 262)
    box = (0, 0, W, H)
    n = len(seeds)
    cent = []
    for r in range(n):
        acc = [0.0, 0.0, 0.0]
        for t, l in zip(tris, labels):
            if l == r:
                a = area(pts, t)
                c = centroid(pts, t)
                acc[0] += a * c[0]
                acc[1] += a * c[1]
                acc[2] += a
        cent.append((acc[0] / acc[2], acc[1] / acc[2]))
    adj = adjacency(tris, labels, n)
    p = Panel(svg, 10, 6, 330, 250, box, "(a) Region adjacency graph")
    draw_mesh(p, pts, tris, labels, ["#2a313b"] * n, border_w=1.4, wire=False, border_color="#6e7681")
    for a in range(n):
        for b in adj[a]:
            if a < b:
                svg.line(p.p(cent[a]), p.p(cent[b]), MUTED, 1.2)
    for c in cent:
        svg.dot(p.p(c), 4, ACCENT, "#0d1117", 1)
    p = Panel(svg, 355, 6, 330, 250, box, "(b) Largest cliques (≤ 4), breadth first")
    gcols = [color(g * 3 + 1, 0.5, 0.85) for g in range(max(group) + 1)]
    draw_mesh(p, pts, tris, labels, ["#2a313b"] * n, border_w=1.4, wire=False, border_color="#6e7681")
    import itertools
    for c in cliques:
        if len(c) < 2:
            continue
        g = group[c[0]]
        for a, b in itertools.combinations(c, 2):
            svg.line(p.p(cent[a]), p.p(cent[b]), gcols[g], 3.2)
    for a in range(n):
        for b in adj[a]:
            if a < b and group[a] != group[b]:
                svg.line(p.p(cent[a]), p.p(cent[b]), FAINT, 1.0)
    for v in absorbed:
        svg.dot(p.p(cent[v]), 6.5, "none", RED, 1.8)
    for r, c in enumerate(cent):
        svg.dot(p.p(c), 4, gcols[group[r]], "#0d1117", 1)
    p = Panel(svg, 700, 6, 330, 250, box, "(c) Groups = regions of the next level")
    glabels = [group[l] for l in labels]
    draw_mesh(p, pts, tris, labels, [gcols[group[r]] for r in range(n)], border_w=0.9, wire=False,
              border_color="#0d1117")
    for e, fs in edge_faces(tris).items():
        if len(fs) == 2 and glabels[fs[0]] != glabels[fs[1]]:
            svg.line(p.p(pts[e[0]]), p.p(pts[e[1]]), "#0d1117", 3.0)
    svg.write("diagram-cliques.svg")


def diagram_remesh(pts, tris, glabels, gid, rng):
    faces = [f for f, g in enumerate(glabels) if g == gid]
    bedges = boundary_edges(tris, faces)
    bverts = sorted({v for e in bedges for v in e})
    verts = sorted({v for f in faces for v in tris[f]})
    interior = [v for v in verts if v not in set(bverts)]
    xs = [pts[v][0] for v in verts]
    ys = [pts[v][1] for v in verts]
    box = (min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4)
    region_tris = [tris[f] for f in faces]

    def inside(q):
        return any(point_in_tri(q, *(pts[i] for i in t)) for t in region_tris)

    def sample(r):
        for _ in range(100):
            q = (r.uniform(box[0], box[2]), r.uniform(box[1], box[3]))
            if inside(q):
                return q
        return None
    target = max(3, int(round(0.35 * len(interior))))
    bpts = [pts[v] for v in bverts]
    spacing = math.sqrt(sum(area(pts, t) for t in region_tris) / max(1, target)) * 0.75
    new = blue_noise(rng, target, spacing, sample, existing=bpts)
    before = list(new)
    # restricted 2D CVT: Lloyd with dense samples, boundary vertices as fixed generators
    samples = []
    step = 1.6
    y = box[1]
    while y < box[3]:
        x = box[0]
        while x < box[2]:
            q = (x + 0.5 * step, y + 0.5 * step)
            if inside(q):
                samples.append(q)
            x += step
        y += step
    gens = bpts + new
    for _ in range(10):
        acc = [[0.0, 0.0, 0] for _ in gens]
        for q in samples:
            k = min(range(len(gens)), key=lambda i: (q[0] - gens[i][0]) ** 2 + (q[1] - gens[i][1]) ** 2)
            acc[k][0] += q[0]
            acc[k][1] += q[1]
            acc[k][2] += 1
        for i in range(len(bpts), len(gens)):
            if acc[i][2]:
                c = (acc[i][0] / acc[i][2], acc[i][1] / acc[i][2])
                if inside(c):
                    gens[i] = c
    after = gens[len(bpts):]
    allp = bpts + after
    tri2 = [t for t in delaunay(allp) if inside(centroid(allp, t))]

    svg = Svg(1040, 280)
    titles = ["(a) Group: %d faces, %d interior vertices" % (len(faces), len(interior)),
              "(b) Border kept, %d blue-noise points" % len(before),
              "(c) Restricted 2D CVT (10 Lloyd steps)",
              "(d) Constrained Delaunay: %d triangles" % len(tri2)]
    col = color(gid * 3 + 1, 0.5, 0.85)
    for k in range(4):
        p = Panel(svg, 10 + k * 257, 6, 245, 268, box, titles[k])
        if k == 0:
            for t in region_tris:
                svg.poly([p.p(pts[i]) for i in t], col, "#0d1117", 0.6, extra='stroke-opacity="0.55"')
            for v in interior:
                svg.dot(p.p(pts[v]), 1.6, "#0d1117")
        elif k in (1, 2):
            for t in region_tris:
                svg.poly([p.p(pts[i]) for i in t], col, col, 0.4, opacity=0.35)
            if k == 2:
                for a, b in zip(before, after):
                    svg.line(p.p(a), p.p(b), "#ffffff", 0.9, 'opacity="0.6"')
            for q in (before if k == 1 else after):
                svg.dot(p.p(q), 3.0, "#ffffff", "#0d1117", 1)
        else:
            for t in tri2:
                svg.poly([p.p(allp[i]) for i in t], col, "#0d1117", 0.9)
        for e in bedges:
            svg.line(p.p(pts[e[0]]), p.p(pts[e[1]]), "#0d1117", 2.4)
        for v in bverts:
            svg.dot(p.p(pts[v]), 2.4, RED, "#0d1117", 0.6)
    svg.write("diagram-remesh.svg")


def diagram_borders(rng):
    """Nested border simplification on one border chain between an upper and a lower region."""
    xs = [0.0]
    while xs[-1] < 300:
        xs.append(xs[-1] + rng.uniform(9, 20))
    xs[-1] = 300.0
    fy = lambda x: 55 + 22 * math.sin(x / 40.0) + 7 * math.sin(x / 13.0)
    fine = [(x, fy(x)) for x in xs]
    keep = list(range(len(fine)))
    target = 48.0

    def mean_len():
        return sum(math.dist(fine[keep[i]], fine[keep[i + 1]]) for i in range(len(keep) - 1)) / (len(keep) - 1)
    # collapse the shortest edge, moving its non-junction end onto the other, until the target edge length
    while mean_len() < target and len(keep) > 2:
        i = min(range(len(keep) - 1), key=lambda i: math.dist(fine[keep[i]], fine[keep[i + 1]]))
        keep.pop(i + 1 if i + 1 < len(keep) - 1 else i)
    coarse = [fine[i] for i in keep]

    def snap(q):
        best = None
        for a, b in zip(coarse, coarse[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
            c = (a[0] + t * dx, a[1] + t * dy)
            if best is None or math.dist(q, c) < math.dist(q, best):
                best = c
        return best
    kept = set(keep)
    snapped = [q if i in kept else snap(q) for i, q in enumerate(fine)]

    svg = Svg(1040, 168)
    box = (-8, 18, 308, 100)
    p = Panel(svg, 10, 6, 330, 140, box, "(a) Level k: a border chain")
    svg.path([p.p(q) for q in fine], "#d2a8ff", 2.4)
    for q in fine:
        svg.dot(p.p(q), 3.0, "#d2a8ff", "#0d1117", 0.8)
    for q in (fine[0], fine[-1]):
        svg.dot(p.p(q), 5.5, GOOD, "#0d1117", 1)
    svg.text((175, 162), "green: junctions, never moved", 11, MUTED, "middle")

    p = Panel(svg, 355, 6, 330, 140, box, "(b) Level k+1: collapse, snap finer copies")
    svg.path([p.p(q) for q in fine], "#6e5a8a", 1.6)
    svg.path([p.p(q) for q in coarse], ACCENT, 2.6)
    for i, q in enumerate(fine):
        if i not in kept:
            svg.line(p.p(q), p.p(snapped[i]), RED, 1.2, 'marker-end="url(#arr)"')
            svg.dot(p.p(snapped[i]), 2.6, RED, "#0d1117", 0.6)
    for q in coarse:
        svg.dot(p.p(q), 3.4, ACCENT, "#0d1117", 0.8)
    svg.text((520, 162), "red: dropped vertices snap onto the coarse edge", 11, MUTED, "middle")

    p = Panel(svg, 700, 6, 330, 140, box, "(c) Fine above, coarse below: no gap")
    top = [(x, 24.0 + 3 * math.sin(x / 17.0)) for x in [i * 300 / 24 for i in range(25)]]
    bottom = [(x, 96.0) for x in [i * 300 / 5 for i in range(6)]]
    zipper(svg, p, top, snapped, "#c9b45a")
    zipper(svg, p, coarse, bottom, "#4fb7a5")
    svg.path([p.p(q) for q in coarse], "#0d1117", 2.2)
    svg.text((865, 162), "extra fine vertices lie on the coarse edges", 11, MUTED, "middle")
    svg.write("diagram-borders.svg")


def diagram_relax(rng):
    fy = lambda x: 52 + 18 * math.sin(x / 48.0)
    curve = [(x, fy(x)) for x in [i * 300 / 600 for i in range(601)]]
    arc = [0.0]
    for a, b in zip(curve, curve[1:]):
        arc.append(arc[-1] + math.dist(a, b))

    def at(s):
        for k in range(len(arc) - 1):
            if arc[k + 1] >= s:
                t = (s - arc[k]) / (arc[k + 1] - arc[k])
                return (curve[k][0] + t * (curve[k + 1][0] - curve[k][0]),
                        curve[k][1] + t * (curve[k + 1][1] - curve[k][1]))
        return curve[-1]
    n = 10
    gaps = [rng.uniform(0.3, 1.7) for _ in range(n - 1)]
    total = sum(gaps)
    s_before = [0.0]
    for g in gaps:
        s_before.append(s_before[-1] + arc[-1] * g / total)
    s_after = [arc[-1] * i / (n - 1) for i in range(n)]
    up = [(x, 10.0 + 3 * math.sin(x / 23.0)) for x in [i * 300 / 9 for i in range(10)]]
    dn = [(x, 98.0) for x in [i * 300 / 7 for i in range(8)]]
    svg = Svg(1040, 196)
    box = (-8, 4, 308, 102)
    for k, (title, ss) in enumerate((("(a) After the collapses: uneven border edges", s_before),
                                     ("(b) Relaxed: arclength per vertex, CVT energy over all levels", s_after))):
        p = Panel(svg, 10 + k * 520, 6, 505, 184, box, title)
        border = [at(s) for s in ss]
        zipper(svg, p, up, border, "#3b4a63", 0.9)
        zipper(svg, p, border, dn, "#33524a", 0.9)
        svg.path([p.p(q) for q in border], ACCENT, 2.6)
        if k == 1:
            for a, b in zip(s_before, s_after):
                if abs(a - b) > 3:
                    q0, q1 = at(a), at(b)
                    svg.line(p.p((q0[0], q0[1] - 7)), p.p((q1[0], q1[1] - 7)), RED, 1.3, 'marker-end="url(#arr)"')
        for i, q in enumerate(border):
            end = i in (0, n - 1)
            svg.dot(p.p(q), 5.5 if end else 3.6, GOOD if end else ACCENT, "#0d1117", 0.9)
    svg.write("diagram-relax.svg")


def min_area_rect_angle(hull):
    best = (math.inf, 0.0)
    for a, b in zip(hull, hull[1:] + hull[:1]):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(ang), math.sin(ang)
        us = [q[0] * c + q[1] * s for q in hull]
        vs = [-q[0] * s + q[1] * c for q in hull]
        ar = (max(us) - min(us)) * (max(vs) - min(vs))
        if ar < best[0]:
            best = (ar, ang)
    return best[1]


def maxrects(sizes, side):
    """MaxRects, best short side fit, 90-degree turns allowed; returns placements or None."""
    free = [(0.0, 0.0, side, side)]
    out = []
    for w, h in sizes:
        best = None
        for f in free:
            for turned, (rw, rh) in enumerate(((w, h), (h, w))):
                if rw <= f[2] + 1e-9 and rh <= f[3] + 1e-9:
                    key = (min(f[2] - rw, f[3] - rh), max(f[2] - rw, f[3] - rh))
                    if best is None or key < best[0]:
                        best = (key, (f[0], f[1], rw, rh), turned)
        if best is None:
            return None
        n = best[1]
        out.append((n, best[2]))
        new = []
        for f in free:
            if n[0] >= f[0] + f[2] or n[0] + n[2] <= f[0] or n[1] >= f[1] + f[3] or n[1] + n[3] <= f[1]:
                new.append(f)
                continue
            if n[0] > f[0]:
                new.append((f[0], f[1], n[0] - f[0], f[3]))
            if n[0] + n[2] < f[0] + f[2]:
                new.append((n[0] + n[2], f[1], f[0] + f[2] - n[0] - n[2], f[3]))
            if n[1] > f[1]:
                new.append((f[0], f[1], f[2], n[1] - f[1]))
            if n[1] + n[3] < f[1] + f[3]:
                new.append((f[0], n[1] + n[3], f[2], f[1] + f[3] - n[1] - n[3]))

        def contains(a, b):
            return b[0] >= a[0] - 1e-9 and b[1] >= a[1] - 1e-9 and b[0] + b[2] <= a[0] + a[2] + 1e-9 and \
                b[1] + b[3] <= a[1] + a[3] + 1e-9
        free = [f for i, f in enumerate(new)
                if not any(j != i and contains(g, f) and (not contains(f, g) or j < i) for j, g in enumerate(new))]
    return out


def diagram_binpack(pts, tris, glabels):
    ng = max(glabels) + 1
    charts = []
    for g in range(ng):
        faces = [f for f, l in enumerate(glabels) if l == g]
        loops = chain_loops(boundary_edges(tris, faces))
        loop = max(loops, key=len)
        poly = [pts[v] for v in loop]
        cx = sum(q[0] for q in poly) / len(poly)
        cy = sum(q[1] for q in poly) / len(poly)
        # a "flattened" chart: the region outline, given a random orientation as LSCM would
        ang = (g * 2.399) % (2 * math.pi)
        c, s = math.cos(ang), math.sin(ang)
        charts.append([((q[0] - cx) * c - (q[1] - cy) * s, (q[0] - cx) * s + (q[1] - cy) * c) for q in poly])
    svg = Svg(1040, 330)
    # (a) charts as flattened, side by side
    p = Panel(svg, 10, 6, 330, 318, (-10, -10, 330, 330), "(a) Flattened charts (LSCM, any orientation)")
    cols = [color(g * 3 + 1, 0.5, 0.85) for g in range(ng)]
    per_row = int(math.ceil(math.sqrt(ng)))
    cell = 320 / per_row
    for g, ch in enumerate(charts):
        ox = (g % per_row + 0.5) * cell
        oy = (g // per_row + 0.5) * cell
        k = 0.42 * cell / max(max(abs(q[0]), abs(q[1])) for q in ch)
        svg.poly([p.p((ox + k * q[0], oy + k * q[1])) for q in ch], cols[g], "#0d1117", 1.2)
    # (b) turned to their minimum-area rectangles
    turned = []
    for ch in charts:
        ang = min_area_rect_angle(convex_hull(ch))
        c, s = math.cos(ang), math.sin(ang)
        r = [(q[0] * c + q[1] * s, -q[0] * s + q[1] * c) for q in ch]
        lo = (min(q[0] for q in r), min(q[1] for q in r))
        turned.append([(q[0] - lo[0], q[1] - lo[1]) for q in r])
    p = Panel(svg, 355, 6, 330, 318, (-10, -10, 330, 330), "(b) Turned to their min-area rectangles")
    for g, ch in enumerate(turned):
        ox = (g % per_row + 0.5) * cell
        oy = (g // per_row + 0.5) * cell
        wch = max(q[0] for q in ch)
        hch = max(q[1] for q in ch)
        k = 0.42 * cell / max(wch, hch) * 2
        x0, y0 = ox - k * wch / 2, oy - k * hch / 2
        svg.rect(*p.p((x0, y0)), k * wch * p.s, k * hch * p.s, "none", MUTED, 1.0)
        svg.poly([p.p((x0 + k * q[0], y0 + k * q[1])) for q in ch], cols[g], "#0d1117", 1.2)
    # (c) packed with MaxRects into the smallest square, scaled to [0,1]^2
    pad = 0.02 * sum(max(max(q[0] for q in ch), max(q[1] for q in ch)) for ch in turned) / len(turned)
    sizes = [(max(q[0] for q in ch) + 2 * pad, max(q[1] for q in ch) + 2 * pad) for ch in turned]
    order = sorted(range(len(sizes)), key=lambda i: -max(sizes[i]))
    side = max(math.sqrt(sum(w * h for w, h in sizes)), max(max(s) for s in sizes))
    while True:
        placed = maxrects([sizes[i] for i in order], side)
        if placed:
            break
        side *= 1.02
    p = Panel(svg, 700, 6, 330, 318, (-0.03, -0.03, 1.03, 1.03), "(c) MaxRects into the unit square")
    svg.rect(*p.p((0, 0)), p.s, p.s, "none", ACCENT, 1.4)
    for (rect, t), i in zip(placed, order):
        ch = turned[i]
        hch = max(q[1] for q in ch)
        pts2 = []
        for q in ch:
            u, v = (hch - q[1], q[0]) if t else q
            pts2.append(((rect[0] + pad + u) / side, (rect[1] + pad + v) / side))
        svg.poly([p.p(q) for q in pts2], cols[i], "#0d1117", 1.2)
    svg.write("diagram-binpack.svg")


def diagram_dag():
    svg = Svg(1040, 320)
    levels = [16, 5, 2]
    ys = [262, 160, 58]
    xs = []
    for n, y in zip(levels, ys):
        span = 860.0
        xs.append([90 + span * (i + 0.5) / n for i in range(n)])
    parent0 = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    parent1 = [0, 0, 1, 1, 1]
    # cut: drawn = level-1 nodes 0, 1 and level-0 children of level-1 nodes 2..4 except group of node 3 -> level 1
    drawn = {(1, 0), (1, 1), (1, 3)} | {(0, i) for i, p in enumerate(parent0) if p in (2, 4)}
    cols = ["#e4d98c", "#7fd3a8", "#61b6e3"]
    for lvl in range(2):
        par = parent0 if lvl == 0 else parent1
        for i, pi in enumerate(par):
            svg.line((xs[lvl][i], ys[lvl] - 14), (xs[lvl + 1][pi], ys[lvl + 1] + 14), FAINT, 1.4)
    # groups: children of one parent, boxed
    for lvl, par in ((0, parent0), (1, parent1)):
        for g in sorted(set(par)):
            members = [i for i, p in enumerate(par) if p == g]
            x0 = xs[lvl][members[0]] - 24
            x1 = xs[lvl][members[-1]] + 24
            svg.rect(x0, ys[lvl] - 22, x1 - x0, 44, "none", MUTED, 1.0, 10)
    for lvl, n in enumerate(levels):
        for i in range(n):
            on = (lvl, i) in drawn
            svg.rect(xs[lvl][i] - 17, ys[lvl] - 14, 34, 28, cols[lvl], "#ffffff" if on else "#0d1117",
                     3.0 if on else 1.0, 5)
        svg.text((20, ys[lvl] + 4), "level %d" % lvl, 12.5, TEXT, weight="600")
    svg.text((955, ys[0] + 4), "meshlet = region", 11, MUTED)
    svg.text((955, ys[0] + 32 - 14), "", 11, MUTED)
    svg.text((520, 302), "boxes: groups (children of one region, swapped as a unit) · white outline: one crack-free cut "
             "selected by screen-space error", 11.5, MUTED, "middle")
    svg.write("diagram-dag.svg")


def main():
    rng = random.Random(7)
    pts, tris = fine_mesh(rng)
    n_seeds = 16
    seeds0 = blue_noise(rng, n_seeds, 46.0, lambda r: (r.uniform(0, W), r.uniform(0, H)))
    seeds1 = lloyd(pts, tris, seeds0, 12)
    labels1 = label_faces(pts, tris, seeds1)
    straight = straighten_seams(pts, tris, labels1)
    diagram_cvt(pts, tris, seeds0, seeds1, labels1, straight)
    labels1 = straight
    n = len(seeds1)
    adj = adjacency(tris, labels1, n)
    weight = [sum(1 for l in labels1 if l == r) for r in range(n)]
    group, cliques, absorbed = clique_groups(adj, weight, random.Random(3))
    diagram_cliques(pts, tris, labels1, seeds1, group, cliques, absorbed)
    glabels = [group[l] for l in labels1]
    sizes = {}
    for g in glabels:
        sizes[g] = sizes.get(g, 0) + 1
    # remesh the group nearest the middle
    cent = {}
    for f, g in enumerate(glabels):
        c = centroid(pts, tris[f])
        s = cent.setdefault(g, [0.0, 0.0, 0])
        s[0] += c[0]
        s[1] += c[1]
        s[2] += 1
    gid = min(cent, key=lambda g: (cent[g][0] / cent[g][2] - W / 2) ** 2 + (cent[g][1] / cent[g][2] - H / 2) ** 2)
    diagram_remesh(pts, tris, glabels, gid, random.Random(11))
    diagram_borders(random.Random(5))
    diagram_relax(random.Random(9))
    diagram_binpack(pts, tris, labels1)
    diagram_dag()


if __name__ == "__main__":
    main()
