"""Shared helpers of the figure scripts: binary PLY reading, a small z-buffered rasterizer and PNG output.

Standard library only (struct, zlib), so the figures rebuild without numpy or an image library.
"""

import colorsys
import math
import struct
import zlib

_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h", "ushort": "H",
    "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I", "float": "f", "float32": "f",
    "double": "d", "float64": "d",
}


def read_ply(path):
    """Binary little-endian PLY with vertex x/y/z and face vertex_indices; optional per-corner texcoord
    lists and an int region per face. Returns (positions, faces, texcoords or None, regions or None)."""
    data = open(path, "rb").read()
    end = data.index(b"end_header") + len(b"end_header")
    end = data.index(b"\n", end) + 1
    header = data[:end].decode("ascii").splitlines()
    assert "format binary_little_endian 1.0" in header, "binary little-endian PLY expected"
    elements = []
    for line in header:
        words = line.split()
        if words[:1] == ["element"]:
            elements.append((words[1], int(words[2]), []))
        elif words[:1] == ["property"]:
            if words[1] == "list":
                elements[-1][2].append((words[4], _TYPES[words[2]], _TYPES[words[3]]))
            else:
                elements[-1][2].append((words[2], None, _TYPES[words[1]]))
    pos = end
    out = {}
    for name, count, props in elements:
        if all(p[1] is None for p in props):
            fmt = "<" + "".join(p[2] for p in props)
            size = struct.calcsize(fmt)
            rows = list(struct.iter_unpack(fmt, data[pos:pos + size * count]))
            pos += size * count
            out[name] = {p[0]: [r[i] for r in rows] for i, p in enumerate(props)}
            continue
        columns = {p[0]: [] for p in props}
        unpack_from = struct.unpack_from
        for _ in range(count):
            for prop, count_type, item_type in props:
                if count_type is None:
                    value = unpack_from("<" + item_type, data, pos)[0]
                    pos += struct.calcsize(item_type)
                else:
                    n = unpack_from("<" + count_type, data, pos)[0]
                    pos += struct.calcsize(count_type)
                    value = unpack_from("<%d%s" % (n, item_type), data, pos)
                    pos += n * struct.calcsize(item_type)
                columns[prop].append(value)
        out[name] = columns
    v = out["vertex"]
    positions = list(zip(v["x"], v["y"], v["z"]))
    f = out.get("face", {})
    faces = f.get("vertex_indices") or f.get("vertex_index") or []
    return positions, faces, f.get("texcoord"), f.get("region")


def region_color(region, saturation=0.42, value=0.95):
    """A soft color per region id, hues spread by the golden ratio."""
    hue = (region * 0.6180339887) % 1.0
    lightness_jitter = 0.08 * (((region * 2654435761) >> 7) % 3 - 1)
    return colorsys.hsv_to_rgb(hue, saturation, min(1.0, value + lightness_jitter))


class Canvas:
    """RGB image with a depth buffer and a per-pixel id buffer (for border detection)."""

    def __init__(self, width, height, background=(0.051, 0.067, 0.090)):
        self.w, self.h = width, height
        self.bg = background
        self.rgb = [background] * (width * height)
        self.depth = [math.inf] * (width * height)
        self.ids = [-1] * (width * height)

    def triangle(self, p0, p1, p2, color, ident, wire=0.0, wire_px=1.0):
        """Fills screen-space triangle (x, y, depth) with color; with wire > 0 its edges are darkened by
        that much within wire_px pixels."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            return
        minx = max(0, int(math.floor(min(x0, x1, x2))))
        maxx = min(self.w - 1, int(math.ceil(max(x0, x1, x2))))
        miny = max(0, int(math.floor(min(y0, y1, y2))))
        maxy = min(self.h - 1, int(math.ceil(max(y0, y1, y2))))
        if minx > maxx or miny > maxy:
            return
        inv = 1.0 / area
        # pixel distance to each edge per unit of barycentric weight
        lens = (math.hypot(x2 - x1, y2 - y1), math.hypot(x0 - x2, y0 - y2), math.hypot(x1 - x0, y1 - y0))
        scale = [abs(area) / l if l > 0 else 1e9 for l in lens]
        depth, ids, rgb, w = self.depth, self.ids, self.rgb, self.w
        r, g, b = color
        for y in range(miny, maxy + 1):
            py = y + 0.5
            row = y * w
            for x in range(minx, maxx + 1):
                px = x + 0.5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv
                w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                i = row + x
                if z >= depth[i]:
                    continue
                depth[i] = z
                ids[i] = ident
                if wire > 0:
                    d = min(w0 * scale[0], w1 * scale[1], w2 * scale[2])
                    k = 1.0 - wire * max(0.0, 1.0 - d / wire_px)
                    rgb[i] = (r * k, g * k, b * k)
                else:
                    rgb[i] = color

    def outline_ids(self, width=1, darkness=0.85):
        """Darkens pixels whose id differs from a neighbour's within `width` pixels (region borders and
        silhouettes)."""
        w, h, ids = self.w, self.h, self.ids
        edge = [False] * (w * h)
        for y in range(h):
            for x in range(w):
                i = y * w + x
                a = ids[i]
                for dx, dy in ((1, 0), (0, 1)):
                    xx, yy = x + dx, y + dy
                    if xx < w and yy < h and ids[yy * w + xx] != a:
                        edge[i] = True
                        edge[yy * w + xx] = True
        if width > 1:
            grown = edge[:]
            for y in range(h):
                for x in range(w):
                    if edge[y * w + x]:
                        for dy in range(-width + 1, width):
                            for dx in range(-width + 1, width):
                                xx, yy = x + dx, y + dy
                                if 0 <= xx < w and 0 <= yy < h:
                                    grown[yy * w + xx] = True
            edge = grown
        k = 1.0 - darkness
        for i in range(w * h):
            if edge[i] and ids[i] >= 0:
                r, g, b = self.rgb[i]
                self.rgb[i] = (r * k, g * k, b * k)

    def downsample(self, factor):
        """Box-filtered copy, factor x smaller in each direction."""
        out = Canvas(self.w // factor, self.h // factor, self.bg)
        n = factor * factor
        for y in range(out.h):
            for x in range(out.w):
                r = g = b = 0.0
                for dy in range(factor):
                    row = (y * factor + dy) * self.w + x * factor
                    for dx in range(factor):
                        c = self.rgb[row + dx]
                        r += c[0]
                        g += c[1]
                        b += c[2]
                out.rgb[y * out.w + x] = (r / n, g / n, b / n)
        return out


def write_png(path, canvases, gap=0, background=(0.051, 0.067, 0.090)):
    """Writes canvases side by side as one 8-bit RGB PNG."""
    height = max(c.h for c in canvases)
    width = sum(c.w for c in canvases) + gap * (len(canvases) - 1)
    bg = bytes(int(round(255 * v)) for v in background)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for k, c in enumerate(canvases):
            if k:
                raw += bg * gap
            if y < c.h:
                for r, g, b in c.rgb[y * c.w:(y + 1) * c.w]:
                    raw += bytes((min(255, max(0, int(r * 255 + 0.5))), min(255, max(0, int(g * 255 + 0.5))),
                                  min(255, max(0, int(b * 255 + 0.5)))))
            else:
                raw += bg * c.w

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)
