#!/usr/bin/env python3
"""
Build a self-contained interactive HTML viewer from a .drawio wiring diagram
produced by wiring_to_drawio.py.

- The first page of the .drawio file is rendered to inline SVG (no external
  dependencies, works offline).
- Wire segments are grouped into circuits (nets): segments are joined where
  their end points meet, where an end point touches another wire (T junction),
  and across small gaps between same-colored wire ends (inline markers).
- Wire end points that terminate on a component are assigned to the nearest
  component callout, scoped by the dashed component boxes.
- Clicking a wire highlights its whole circuit; clicking a component label
  highlights every circuit connected to that component.

Usage:
    python build_viewer.py MT07-wiring.drawio -o MT07-wiring.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from dataclasses import dataclass, field

TOL = 3.0          # end point match tolerance (drawio px)
GAP_MAX = 40.0     # max gap bridged between same-colored wire ends
BOX_PAD = 16.0     # dashed box inflation when testing end points
NEAR_MAX = 160.0   # max distance from a loose end point to a component callout


# ---------------------------------------------------------------------------
# drawio loading
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    id: str
    raw: str
    style: dict
    attrs: dict
    value: str
    kind: str                      # "edge" | "vertex"
    pts: list = field(default_factory=list)
    geo: tuple = (0, 0, 0, 0)


def parse_style(s: str) -> dict:
    out = {}
    for kv in s.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
        elif kv:
            out[kv] = ""
    return out


def load_drawio(path: str, page: int = 0):
    root = ET.parse(path).getroot()
    diagram = root.findall("diagram")[page]
    model = diagram.find("mxGraphModel")
    if model is None:  # compressed diagram
        data = zlib.decompress(base64.b64decode(diagram.text), -15).decode("utf-8")
        model = ET.fromstring(urllib.parse.unquote(data))
    width = float(model.get("pageWidth", 1000))
    height = float(model.get("pageHeight", 1000))
    cells = []
    for el in model.find("root"):
        obj = el if el.tag == "mxCell" else el.find("mxCell")
        if obj is None or not obj.get("style"):
            continue
        attrs = dict(el.attrib)
        raw = obj.get("style")
        value = attrs.get("label", obj.get("value", "")) if el.tag != "mxCell" else obj.get("value", "")
        g = obj.find("mxGeometry")
        c = Cell(id=attrs.get("id") or obj.get("id"), raw=raw, style=parse_style(raw), attrs=attrs,
                 value=value, kind="edge" if obj.get("edge") else "vertex")
        if c.kind == "edge":
            src = tgt = None
            mids = []
            for p in g.iter("mxPoint"):
                xy = (float(p.get("x", 0)), float(p.get("y", 0)))
                if p.get("as") == "sourcePoint":
                    src = xy
                elif p.get("as") == "targetPoint":
                    tgt = xy
                else:
                    mids.append(xy)
            if src is None or tgt is None:
                continue
            c.pts = [src] + mids + [tgt]
        else:
            c.geo = tuple(float(g.get(k, 0)) for k in ("x", "y", "width", "height"))
        cells.append(c)
    return cells, width, height, diagram.get("name", "")


def decompress_stencil(s: str) -> str:
    return urllib.parse.unquote(zlib.decompress(base64.b64decode(s), -15).decode("ascii"))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)), t


def poly_dist(p, pts):
    """Return (distance, is_at_end) of point p to polyline pts."""
    best, at_end = 1e18, False
    n = len(pts) - 1
    for i in range(n):
        d, t = seg_dist(p, pts[i], pts[i + 1])
        if d < best:
            best = d
            at_end = (i == 0 and t < 1e-6) or (i == n - 1 and t > 1 - 1e-6)
    return best, at_end


def end_dir(pts, which):
    """Unit direction pointing outward from the given end (0 = start, 1 = end)."""
    if which == 0:
        a, b = pts[0], next((q for q in pts[1:] if q != pts[0]), pts[-1])
    else:
        a, b = pts[-1], next((q for q in reversed(pts[:-1]) if q != pts[-1]), pts[0])
    dx, dy = a[0] - b[0], a[1] - b[1]
    L = math.hypot(dx, dy) or 1
    return dx / L, dy / L


def rect_dist(p, r):
    x0, y0, x1, y1 = r
    dx = max(x0 - p[0], 0, p[0] - x1)
    dy = max(y0 - p[1], 0, p[1] - y1)
    return math.hypot(dx, dy)


def rect_gap(a, b):
    dx = max(a[0] - b[2], 0, b[0] - a[2])
    dy = max(a[1] - b[3], 0, b[1] - a[3])
    return math.hypot(dx, dy)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


class Grid:
    """Uniform grid of polyline segments for fast proximity queries."""

    def __init__(self, size=50.0):
        self.size = size
        self.cells = defaultdict(list)

    def add(self, key, a, b, pad=0.0):
        s = self.size
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        for gx in range(int((x0 - pad) // s), int((x1 + pad) // s) + 1):
            for gy in range(int((y0 - pad) // s), int((y1 + pad) // s) + 1):
                self.cells[(gx, gy)].append((key, a, b))

    def near(self, p):
        s = self.size
        gx, gy = int(p[0] // s), int(p[1] // s)
        seen = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for item in self.cells.get((gx + dx, gy + dy), ()):
                    if id(item) not in seen:
                        seen.add(id(item))
                        yield item


# ---------------------------------------------------------------------------
# Circuit analysis
# ---------------------------------------------------------------------------

def analyze(cells):
    # Wire segments: group base + stripe edges sharing the same geometry
    wire_cells = [c for c in cells if c.kind == "edge" and c.style.get("wire") == "1"]
    seg_index = {}
    segments = []  # {"pts", "cells", "color"}
    for c in wire_cells:
        key = tuple((round(x, 1), round(y, 1)) for x, y in c.pts)
        if key not in seg_index:
            seg_index[key] = len(segments)
            segments.append({"pts": c.pts, "cells": [], "color": c.attrs.get("wireColor", "?")})
        s = segments[seg_index[key]]
        s["cells"].append(c.id)
        if c.style.get("dashed") != "1" and c.attrs.get("wireColor"):
            s["color"] = c.attrs["wireColor"]

    n = len(segments)
    uf = UnionFind(n)
    grid = Grid()
    for i, s in enumerate(segments):
        for a, b in zip(s["pts"], s["pts"][1:]):
            grid.add(i, a, b, pad=TOL)

    # 1) end-to-end and T junctions
    connected_end = set()  # (segment, which_end)
    for i, s in enumerate(segments):
        for which, p in ((0, s["pts"][0]), (1, s["pts"][-1])):
            for j, a, b in grid.near(p):
                if j == i:
                    continue
                d, _ = seg_dist(p, a, b)
                if d <= TOL:
                    uf.union(i, j)
                    connected_end.add((i, which))
                    # mark the other side too when the touch is end-to-end
                    for w2, q in ((0, segments[j]["pts"][0]), (1, segments[j]["pts"][-1])):
                        if math.dist(p, q) <= TOL:
                            connected_end.add((j, w2))

    # 2) bridge small gaps between facing, same-colored loose ends (inline markers)
    loose = [(i, w) for i in range(n) for w in (0, 1) if (i, w) not in connected_end]
    used = set()
    for ai, (i, wi) in enumerate(loose):
        pi = segments[i]["pts"][0 if wi == 0 else -1]
        di = end_dir(segments[i]["pts"], wi)
        best = None
        for (j, wj) in loose[ai + 1:]:
            if j == i or segments[j]["color"] != segments[i]["color"]:
                continue
            pj = segments[j]["pts"][0 if wj == 0 else -1]
            gap = math.dist(pi, pj)
            if gap > GAP_MAX or gap < 1e-6:
                continue
            dj = end_dir(segments[j]["pts"], wj)
            ux, uy = (pj[0] - pi[0]) / gap, (pj[1] - pi[1]) / gap
            # both ends must point at each other along a straight line
            if di[0] * ux + di[1] * uy > 0.98 and -(dj[0] * ux + dj[1] * uy) > 0.98:
                if best is None or gap < best[0]:
                    best = (gap, j, wj)
        if best and (i, wi) not in used and (best[1], best[2]) not in used:
            uf.union(i, best[1])
            used.update({(i, wi), (best[1], best[2])})

    nets = {}
    seg_net = []
    for i in range(n):
        r = uf.find(i)
        nets.setdefault(r, len(nets))
        seg_net.append(nets[r])

    # every wire end: (segment, which_end, point, loose); loose ends touch no other wire
    ends = [(i, w, segments[i]["pts"][0 if w == 0 else -1],
             (i, w) not in connected_end and (i, w) not in used)
            for i in range(n) for w in (0, 1)]
    return segments, seg_net, len(nets), ends


def clip_outside(a, b, rects, shrink=2.0):
    """Return the parts of segment a-b that lie outside all rects (each shrunk a little)."""
    keep = [(0.0, 1.0)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    for r in rects:
        rx0, ry0, rx1, ry1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
        if rx0 >= rx1 or ry0 >= ry1:
            continue
        t0, t1 = 0.0, 1.0
        ok = True
        for p, q in ((-dx, a[0] - rx0), (dx, rx1 - a[0]), (-dy, a[1] - ry0), (dy, ry1 - a[1])):
            if p == 0:
                if q < 0:
                    ok = False
                    break
            else:
                t = q / p
                if p < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
        if not ok or t0 >= t1:
            continue
        nxt = []
        for s0, s1 in keep:
            if s1 <= t0 or s0 >= t1:
                nxt.append((s0, s1))
            else:
                if s0 < t0:
                    nxt.append((s0, t0))
                if t1 < s1:
                    nxt.append((t1, s1))
        keep = nxt
    pt = lambda t: (a[0] + dx * t, a[1] + dy * t)
    return [[pt(s0), pt(s1)] for s0, s1 in keep if s1 - s0 > 1e-6]


def stencil_outline(cell):
    """Polylines (absolute coordinates) approximating a stencil cell's path."""
    x0, y0 = cell.geo[0], cell.geo[1]
    root = ET.fromstring(decompress_stencil(cell.style["shape"][8:-1]))
    polys, cur, last = [], [], (0.0, 0.0)
    for el in root.iter():
        if el.tag == "move":
            if len(cur) > 1:
                polys.append(cur)
            last = (x0 + float(el.get("x")), y0 + float(el.get("y")))
            cur = [last]
        elif el.tag == "line":
            last = (x0 + float(el.get("x")), y0 + float(el.get("y")))
            cur.append(last)
        elif el.tag == "curve":
            c1 = (x0 + float(el.get("x1")), y0 + float(el.get("y1")))
            c2 = (x0 + float(el.get("x2")), y0 + float(el.get("y2")))
            e = (x0 + float(el.get("x3")), y0 + float(el.get("y3")))
            for k in range(1, 7):
                t = k / 6
                u = 1 - t
                cur.append((u ** 3 * last[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * e[0],
                            u ** 3 * last[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * e[1]))
            last = e
        elif el.tag == "close" and cur:
            cur.append(cur[0])
    if len(cur) > 1:
        polys.append(cur)
    return polys


class BoxCircuit:
    """Thin-line circuit inside one dashed component box.

    Lead lines are joined where they touch (and through small junction dots);
    component bodies (symbols such as bulbs, fuses or coils) are not crossed.
    Each callout is attached to the body nearest to it, so a pin belongs to the
    callouts whose bodies its lead lines reach.
    """

    PIN_REACH = 14.0  # how far from a wire end to look for a lead line (across a connector)

    def __init__(self, cells, box, anchors, pins=()):
        pad = BOX_PAD
        x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad

        def inbox(px, py):
            return x0 <= px <= x1 and y0 <= py <= y1

        polylines, dots, bodies, housings = [], [], [], []  # body: (rect, outline polylines)
        for c in cells:
            if c.kind == "edge":
                if c.style.get("wire") == "1" or c.style.get("dashed") == "1" \
                        or not all(inbox(*q) for q in c.pts):
                    continue
                xs = [q[0] for q in c.pts]
                ys = [q[1] for q in c.pts]
                if len(c.pts) >= 4 and math.dist(c.pts[0], c.pts[-1]) < 0.5:
                    # closed outline: a block symbol if large, otherwise a connector
                    # housing or diode triangle that must not join its pins
                    r = (min(xs), min(ys), max(xs), max(ys))
                    if min(r[2] - r[0], r[3] - r[1]) > 20:
                        bodies.append((r, [c.pts]))
                    else:
                        housings.append(r)
                    continue
                polylines.append(c.pts)
            elif c.style.get("shape", "").startswith("stencil("):
                x, y, w, h = c.geo
                if not (inbox(x, y) and inbox(x + w, y + h)):
                    continue
                filled = c.style.get("fillColor", "none") != "none"
                if max(w, h) <= 14 and abs(w - h) < 2 and filled:
                    dots.append((x, y, x + w, y + h))      # junction dot / pin circle
                elif min(w, h) <= 12 and max(w, h) > 14 and filled:
                    housings.append((x, y, x + w, y + h))  # connector housing on a pin row
                else:
                    bodies.append(((x, y, x + w, y + h), stencil_outline(c)))

        # lines drawn inside a connector housing (e.g. its centre divider) carry no circuit
        polylines = [pts for pts in polylines
                     if not any(all(rect_dist(q, h) <= 1.0 for q in pts) for h in housings)]
        rects = [b[0] for b in bodies]
        # cut lead lines where they run through a component body (e.g. a bulb)
        self.lines = []
        for pts in polylines:
            for a, b in zip(pts, pts[1:]):
                self.lines.extend(clip_outside(a, b, rects))
        n = len(self.lines)
        uf = UnionFind(n + len(dots))
        self.grid = Grid(40)
        for i, pts in enumerate(self.lines):
            self.grid.add(i, pts[0], pts[1], pad=self.PIN_REACH)
        for i, pts in enumerate(self.lines):
            for q in pts:
                for j, a, b in self.grid.near(q):
                    if j != i and seg_dist(q, a, b)[0] <= TOL:
                        uf.union(i, j)
        for k, r in enumerate(dots):
            for i, pts in enumerate(self.lines):
                if any(rect_dist(q, r) <= TOL for q in pts):
                    uf.union(n + k, i)
        self.uf = uf

        # small symbol parts touching a neighbour belong to it (e.g. separate coil windings)
        buf = UnionFind(len(bodies))
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                ri, rj = rects[i], rects[j]
                small = min(max(r[2] - r[0], r[3] - r[1]) for r in (ri, rj)) < 20
                if small and rect_gap(ri, rj) <= TOL:
                    buf.union(i, j)
        groups = defaultdict(list)
        for i in range(len(bodies)):
            groups[buf.find(i)].append(i)
        groups = list(groups.values())

        # lead-line clusters adjacent to each symbol
        group_clusters = []
        for g in groups:
            cl = set()
            for i, pts in enumerate(self.lines):
                if any(rect_dist(pts[0], rects[k]) <= TOL or rect_dist(pts[-1], rects[k]) <= TOL for k in g):
                    cl.add(uf.find(i))
            group_clusters.append(cl)

        def dist_to_group(a, g):
            best = 1e18
            for k in g:
                for poly in bodies[k][1]:
                    best = min(best, poly_dist(a, poly)[0] if len(poly) > 1 else math.dist(a, poly[0]))
            return best

        # match callouts to wired symbols, closest pairs first, one symbol per callout
        # (by distance to the drawn outline); leftover callouts share their nearest symbol
        self.cluster_owner = defaultdict(set)
        def extent(g):
            xs = [v for k in g for v in (rects[k][0], rects[k][2])]
            ys = [v for k in g for v in (rects[k][1], rects[k][3])]
            return max(max(xs) - min(xs), max(ys) - min(ys))

        # clusters that reach a wire end (pin) of this box
        pin_clusters = set()
        for q in pins:
            for i, a, b in self.grid.near(q):
                if seg_dist(q, a, b)[0] <= self.PIN_REACH:
                    pin_clusters.add(uf.find(i))

        def reaches_pin(k):
            if group_clusters[k] & pin_clusters:
                return True
            return any(dist_to_group(q, groups[k]) <= self.PIN_REACH for q in pins)

        # candidate symbols: connected to a pin, and not a tiny mark such as an arrow head
        wired = [k for k in range(len(groups)) if extent(groups[k]) >= 14 and reaches_pin(k)]
        if not wired:
            return
        pairs = sorted((dist_to_group(a, groups[k]), lid, k) for lid, a in anchors.items() for k in wired)
        owner, taken = {}, set()
        for d, lid, k in pairs:
            if lid not in owner and k not in taken:
                owner[lid] = k
                taken.add(k)
        for d, lid, k in pairs:
            owner.setdefault(lid, k)
        for lid, k in owner.items():
            for cl in group_clusters[k]:
                self.cluster_owner[cl].add(lid)
        # symbols whose own outline runs to the pin (leads drawn as part of the symbol)
        self.symbol_owner = [(groups[k], set()) for k in range(len(groups))]
        for lid, k in owner.items():
            self.symbol_owner[k][1].add(lid)
        self._dist_to_group = dist_to_group

    def components_at(self, p):
        """Callouts reached from a wire end; looks across a connector to the nearest lead line."""
        hits = []
        for i, a, b in self.grid.near(p):
            d = seg_dist(p, a, b)[0]
            if d <= self.PIN_REACH:
                hits.append((d, self.cluster_owner.get(self.uf.find(i), set())))
        for g, lids in getattr(self, "symbol_owner", ()):
            if lids:
                d = self._dist_to_group(p, g)
                if d <= self.PIN_REACH:
                    hits.append((d, lids))
        if not hits:
            return []
        dmin = min(d for d, _ in hits)
        found = set()
        for d, lids in hits:
            if d <= dmin + 1.0:
                found |= lids
        return sorted(found)


def assign_components(cells, segments, seg_net, ends):
    labels = [c for c in cells if "ref" in c.attrs]
    boxes = []
    for c in cells:
        if c.kind == "edge" and c.style.get("dashed") == "1" and c.style.get("wire") != "1" \
                and len(c.pts) == 5 and math.dist(c.pts[0], c.pts[-1]) < 1:
            xs = [p[0] for p in c.pts]
            ys = [p[1] for p in c.pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def inside(b, p, pad=0.0):
        return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad

    def box_contains(outer, inner):
        return outer != inner and outer[0] <= inner[0] and outer[1] <= inner[1] \
            and outer[2] >= inner[2] and outer[3] >= inner[3]

    comps = {}  # label id -> info
    for l in labels:
        x, y, w, h = l.geo
        anchor = (float(l.attrs.get("anchorX", x + w / 2)), float(l.attrs.get("anchorY", y + h / 2)))
        rect = (x, y, x + w, y + h)
        cont = [b for b in boxes if inside(b, anchor)]
        comps[l.id] = {"ref": int(l.attrs["ref"]), "anchor": anchor, "rect": rect,
                       "box": min(cont, key=area) if cont else None, "nets": set()}

    # Assembly labels sit outside a dashed box: adopt the box they touch
    assembly = defaultdict(list)  # box -> label ids
    for lid, c in comps.items():
        if c["box"] is None:
            touching = [b for b in boxes if rect_dist(c["anchor"], b) <= 30 or
                        any(inside(b, (px, py)) for px in (c["rect"][0], c["rect"][2])
                            for py in (c["rect"][1], c["rect"][3]))]
            if touching:
                b = max(touching, key=area)
                c["assembly"] = b
                assembly[b].append(lid)

    by_box = defaultdict(list)
    for lid, c in comps.items():
        if c["box"] is not None:
            by_box[c["box"]].append(lid)

    internal = {}
    for i, w, p, loose in ends:
        net = seg_net[i]
        cont = [b for b in boxes if inside(b, p, BOX_PAD)]
        if not loose and not cont:
            continue  # a junction between wires out in the harness
        cand = []
        if cont:
            # smallest enclosing box that actually holds a callout
            for b in sorted(cont, key=area):
                if by_box.get(b):
                    cand = by_box[b]
                    break
            # assemblies whose box encloses this end point
            for b in cont:
                for lid in assembly.get(b, ()):
                    comps[lid]["nets"].add(net)
        if not cand:
            free = [lid for lid, c in comps.items() if c["box"] is None]
            cand = [lid for lid in free if math.dist(comps[lid]["anchor"], p) <= NEAR_MAX]
            if not cand and not cont:
                continue
        if not cand:
            continue
        chosen = []
        if len(cand) > 1 and comps[cand[0]]["box"] is not None:
            b = comps[cand[0]]["box"]
            if b not in internal:
                pins = [q for _, _, q, _ in ends if inside(b, q, BOX_PAD)]
                internal[b] = BoxCircuit(cells, b, {lid: comps[lid]["anchor"] for lid in by_box[b]}, pins)
            chosen = internal[b].components_at(p)
        if not chosen:
            chosen = [min(cand, key=lambda lid: math.dist(comps[lid]["anchor"], p))]
        for lid in chosen:
            comps[lid]["nets"].add(net)

    # an assembly also owns everything that belongs to the callouts inside its box
    for b, lids in assembly.items():
        inner = [lid for lid, c in comps.items() if c["box"] and (c["box"] == b or box_contains(b, c["box"]))]
        for lid in lids:
            for o in inner:
                comps[lid]["nets"] |= comps[o]["nets"]
    return comps


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

WIRE_HEX = {
    "B": "#231F20", "Br": "#B7705B", "Ch": "#8B0204", "Dg": "#00652E", "G": "#0BB14B",
    "Gy": "#939598", "L": "#00AEEF", "Lg": "#8DC63F", "O": "#F36F21", "P": "#F49AC1",
    "R": "#ED1C24", "Sb": "#6DCFF6", "W": "#FFFFFF", "Y": "#FFF200",
}


def f1(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def pts_attr(pts) -> str:
    return " ".join(f"{f1(x)},{f1(y)}" for x, y in pts)


def stencil_d(cell) -> tuple[str, str]:
    x0, y0 = cell.geo[0], cell.geo[1]
    root = ET.fromstring(decompress_stencil(cell.style["shape"][8:-1]))
    d, op = [], "stroke"
    for el in root.iter():
        t = el.tag
        if t == "move":
            d.append(f"M{f1(x0 + float(el.get('x')))} {f1(y0 + float(el.get('y')))}")
        elif t == "line":
            d.append(f"L{f1(x0 + float(el.get('x')))} {f1(y0 + float(el.get('y')))}")
        elif t == "curve":
            d.append("C" + " ".join(f1((x0 if k[0] == "x" else y0) + float(el.get(k)))
                                    for k in ("x1", "y1", "x2", "y2", "x3", "y3")))
        elif t == "close":
            d.append("Z")
        elif t in ("fill", "stroke", "fillstroke"):
            op = t
    return "".join(d), op


def stroke_attrs(style) -> str:
    out = f' stroke="{style.get("strokeColor", "#000")}" stroke-width="{style.get("strokeWidth", "1")}"'
    if style.get("dashed") == "1" and style.get("dashPattern"):
        out += f' stroke-dasharray="{style["dashPattern"]}"'
    return out


def render_svg(cells, width, height, seg_of_cell, comp_index):
    art, wires, hits, labels = [], [], [], []
    hit_done = set()
    for c in cells:
        s = c.style
        if c.kind == "edge":
            seg = seg_of_cell.get(c.id)
            if seg is not None:
                wires.append(f'<polyline class="w" data-s="{seg}" points="{pts_attr(c.pts)}" fill="none"'
                             f'{stroke_attrs(s)}/>')
                if seg not in hit_done:
                    hit_done.add(seg)
                    hits.append(f'<polyline data-s="{seg}" points="{pts_attr(c.pts)}"/>')
            else:
                art.append(f'<polyline points="{pts_attr(c.pts)}" fill="none"{stroke_attrs(s)}/>')
            continue

        x, y, w, h = c.geo
        if s.get("shape", "").startswith("stencil("):
            d, op = stencil_d(c)
            fill = s.get("fillColor", "none") if op in ("fill", "fillstroke") else "none"
            stroke = stroke_attrs(s) if op in ("stroke", "fillstroke") else ' stroke="none"'
            art.append(f'<path d="{d}" fill="{fill}"{stroke}/>')
        elif c.raw.startswith("text;"):
            weight = ' font-weight="bold"' if s.get("fontStyle") == "1" else ""
            family = "Times New Roman,serif" if s.get("fontFamily", "").startswith("Times") else "Helvetica,Arial,sans-serif"
            art.append(f'<text x="{f1(x + w / 2)}" y="{f1(y + h / 2)}" font-size="{s.get("fontSize", "10")}" '
                       f'font-family="{family}" fill="{s.get("fontColor", "#000")}"{weight} '
                       f'text-anchor="middle" dominant-baseline="central">{html.escape(c.value)}</text>')
        elif c.id in comp_index:
            k = comp_index[c.id]
            lines = c.value.split("<br>")
            fs = float(s.get("fontSize", 9))
            parts = [f'<g class="lbl" data-c="{k}"><rect x="{f1(x)}" y="{f1(y)}" width="{f1(w)}" '
                     f'height="{f1(h)}" rx="3"/>']
            for i, line in enumerate(lines):
                ly = y + h / 2 + (i - (len(lines) - 1) / 2) * fs * 1.2
                parts.append(f'<text x="{f1(x + w / 2)}" y="{f1(ly)}">{html.escape(html.unescape(line))}</text>')
            parts.append("</g>")
            labels.append("".join(parts))
    return (
        f'<svg id="dia" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f1(width)} {f1(height)}">'
        f'<rect width="{f1(width)}" height="{f1(height)}" fill="#fff"/>'
        f'<g id="art">{"".join(art)}</g><g id="halo"></g><g id="wires">{"".join(wires)}</g>'
        f'<g id="hits">{"".join(hits)}</g><g id="labels">{"".join(labels)}</g></svg>'
    )


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f4f4f2; --panel: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #dcdcd8;
  --accent: #d9480f; --accent-soft: #fff1e6; --halo: rgba(255, 170, 0, .55);
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #18181a; --panel: #222225; --ink: #ececec; --muted: #a0a0a6; --line: #38383c;
          --accent: #ff8a4c; --accent-soft: #3a2a20; }
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
#app { display: grid; grid-template-columns: 1fr 340px; grid-template-rows: auto 1fr; height: 100%; }
header { grid-column: 1 / -1; display: flex; align-items: center; gap: 12px; padding: 8px 16px;
  background: var(--panel); border-bottom: 1px solid var(--line); }
header h1 { font-size: 16px; margin: 0; white-space: nowrap; }
header .hint { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
header .spacer { flex: 1; }
button { font: inherit; border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover { border-color: var(--accent); }
#stage { position: relative; overflow: hidden; background: #fff; cursor: grab; touch-action: none; }
#stage.panning { cursor: grabbing; }
#dia { position: absolute; inset: 0; width: 100%; height: 100%; user-select: none; }
#hits polyline { fill: none; stroke: transparent; stroke-width: 14; pointer-events: stroke; cursor: pointer; }
#wires, #art { pointer-events: none; }
.lbl { cursor: pointer; }
.lbl rect { fill: #fff; stroke: #231F20; stroke-width: 1.3; }
.lbl text { font: 9px Helvetica, Arial, sans-serif; fill: #000; text-anchor: middle; dominant-baseline: central; }
.lbl:hover rect { stroke: var(--accent); stroke-width: 2; }
#halo polyline { fill: none; stroke: var(--halo); stroke-width: 18; stroke-linejoin: round; stroke-linecap: round; }
#dia.dim #art { opacity: .22; }
#dia.dim .w { opacity: .12; }
#dia.dim .w.on { opacity: 1; }
#dia.dim .lbl { opacity: .35; }
#dia.dim .lbl.on { opacity: 1; }
.lbl.on rect { stroke: #d9480f; stroke-width: 2.5; fill: #fff4e6; }
.lbl.sel rect { fill: #ffd8a8; }
#zoom { position: absolute; right: 12px; bottom: 12px; display: flex; flex-direction: column; gap: 4px; }
#zoom button { width: 34px; height: 34px; padding: 0; font-size: 18px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }
#tip { position: absolute; pointer-events: none; background: rgba(20,20,20,.88); color: #fff; font-size: 12px;
  padding: 3px 7px; border-radius: 4px; display: none; white-space: nowrap; z-index: 5; }
aside { background: var(--panel); border-left: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
#search { margin: 12px; padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px; font: inherit;
  background: var(--bg); color: var(--ink); }
#info { padding: 0 14px 12px; border-bottom: 1px solid var(--line); max-height: 55%; overflow: auto; }
#info h2 { font-size: 15px; margin: 4px 0 6px; }
#info .sub { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
#info .empty { color: var(--muted); padding: 8px 0; }
.circuit { border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; margin: 6px 0; cursor: pointer; }
.circuit:hover { border-color: var(--accent); }
.circuit.cur { border-color: var(--accent); background: var(--accent-soft); }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.chip { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; }
.sw { display: inline-block; width: 26px; height: 9px; border-radius: 2px; border: 1px solid #888; }
.links a { color: var(--accent); cursor: pointer; text-decoration: none; font-size: 13px; }
.links a:hover { text-decoration: underline; }
.links { display: flex; flex-wrap: wrap; gap: 2px 10px; }
#list { overflow: auto; flex: 1; padding: 6px 0; }
#list div { padding: 4px 14px; cursor: pointer; display: flex; gap: 8px; }
#list div:hover, #list div.cur { background: var(--accent-soft); }
#list .n { color: var(--muted); width: 26px; text-align: right; flex: none; }
@media (max-width: 760px) {
  #app { grid-template-columns: 1fr; grid-template-rows: auto 1fr 40%; }
  aside { border-left: 0; border-top: 1px solid var(--line); }
  header .hint { display: none; }
}
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>__TITLE__</h1>
    <span class="hint">Click a wire to highlight its circuit, or a component to highlight its wires. Drag to pan, scroll to zoom, Esc to clear.</span>
    <span class="spacer"></span>
    <button id="clear">Clear</button>
  </header>
  <div id="stage">
    __SVG__
    <div id="zoom"><button id="zin" title="Zoom in">+</button><button id="zout" title="Zoom out">&minus;</button><button id="zfit" title="Fit">&#x2922;</button></div>
    <div id="tip"></div>
  </div>
  <aside>
    <input id="search" type="search" placeholder="Search components..." autocomplete="off">
    <div id="info"><div class="empty">Nothing selected.</div></div>
    <div id="list"></div>
  </aside>
</div>
<script>
const D = __DATA__;
const WIRE_HEX = __WIRE_HEX__;
const svg = document.getElementById('dia');
const stage = document.getElementById('stage');
const halo = document.getElementById('halo');
const info = document.getElementById('info');
const tip = document.getElementById('tip');

// ---- element lookup tables ----
const segEls = D.segs.map(() => []);
document.querySelectorAll('#wires .w').forEach(el => segEls[+el.dataset.s].push(el));
const segHit = [];
document.querySelectorAll('#hits polyline').forEach(el => segHit[+el.dataset.s] = el);
const lblEls = [];
document.querySelectorAll('#labels .lbl').forEach(el => lblEls[+el.dataset.c] = el);
const netComps = D.nets.map(() => []);
D.comps.forEach((c, i) => c.nets.forEach(n => netComps[n].push(i)));

// ---- view (pan / zoom) ----
const full = svg.viewBox.baseVal;
let vb = {x: full.x, y: full.y, w: full.width, h: full.height};
function applyView() { svg.setAttribute('viewBox', `${vb.x} ${vb.y} ${vb.w} ${vb.h}`); }
function fit() {
  const r = stage.getBoundingClientRect();
  const s = Math.max(full.width / r.width, full.height / r.height);
  vb = {w: r.width * s, h: r.height * s};
  vb.x = full.x + (full.width - vb.w) / 2; vb.y = full.y + (full.height - vb.h) / 2;
  applyView();
}
function toSvg(cx, cy) {
  const r = stage.getBoundingClientRect();
  return {x: vb.x + (cx - r.left) / r.width * vb.w, y: vb.y + (cy - r.top) / r.height * vb.h};
}
function zoomAt(cx, cy, f) {
  const p = toSvg(cx, cy);
  const nw = Math.min(full.width * 3, Math.max(60, vb.w * f));
  const k = nw / vb.w;
  vb = {x: p.x - (p.x - vb.x) * k, y: p.y - (p.y - vb.y) * k, w: vb.w * k, h: vb.h * k};
  applyView();
}
function zoomCenter(f) { const r = stage.getBoundingClientRect(); zoomAt(r.left + r.width / 2, r.top + r.height / 2, f); }
function focusBox(x0, y0, x1, y1) {
  const r = stage.getBoundingClientRect();
  const pad = 80, w = Math.max(x1 - x0 + 2 * pad, 500), h = Math.max(y1 - y0 + 2 * pad, 500);
  const s = Math.max(w / r.width, h / r.height);
  vb = {w: r.width * s, h: r.height * s};
  vb.x = (x0 + x1) / 2 - vb.w / 2; vb.y = (y0 + y1) / 2 - vb.h / 2;
  applyView();
}
stage.addEventListener('wheel', e => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0015)); }, {passive: false});
document.getElementById('zin').onclick = () => zoomCenter(1 / 1.4);
document.getElementById('zout').onclick = () => zoomCenter(1.4);
document.getElementById('zfit').onclick = fit;

const pointers = new Map();
let drag = null, moved = false, pinch = null;
stage.addEventListener('pointerdown', e => {
  if (e.target.closest('#zoom')) return;
  pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
  stage.setPointerCapture(e.pointerId);
  if (pointers.size === 1) { drag = {x: e.clientX, y: e.clientY, vx: vb.x, vy: vb.y}; moved = false; }
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinch = {d: Math.hypot(a.x - b.x, a.y - b.y), w: vb.w}; drag = null; moved = true;
  }
});
stage.addEventListener('pointermove', e => {
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, (pinch.w * pinch.d / d) / vb.w);
    return;
  }
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { moved = true; stage.classList.add('panning'); }
    if (moved) {
      const r = stage.getBoundingClientRect();
      vb.x = drag.vx - dx / r.width * vb.w; vb.y = drag.vy - dy / r.height * vb.h; applyView();
    }
    return;
  }
  showTip(e);
});
function endPointer(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (pointers.size === 0) {
    stage.classList.remove('panning');
    if (drag && !moved) handleClick(e);
    drag = null;
  }
}
stage.addEventListener('pointerup', endPointer);
stage.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); drag = null; pinch = null; });
stage.addEventListener('pointerleave', () => tip.style.display = 'none');

function hitAt(e) { return document.elementFromPoint(e.clientX, e.clientY); }
function showTip(e) {
  const el = hitAt(e);
  let text = '';
  if (el && el.parentNode && el.parentNode.id === 'hits') {
    const s = +el.dataset.s;
    text = 'Wire ' + D.segs[s][1] + ' - click to highlight circuit';
  } else if (el && el.closest && el.closest('.lbl')) {
    const c = D.comps[+el.closest('.lbl').dataset.c];
    text = '#' + c.ref + ' ' + c.name + ' (' + c.nets.length + ' circuits)';
  }
  if (!text) { tip.style.display = 'none'; return; }
  const r = stage.getBoundingClientRect();
  tip.textContent = text; tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 4) + 'px';
  tip.style.top = (e.clientY - r.top + 14) + 'px';
}
function handleClick(e) {
  const el = hitAt(e);
  if (el && el.parentNode && el.parentNode.id === 'hits') selectNet(D.segs[+el.dataset.s][0]);
  else if (el && el.closest && el.closest('.lbl')) selectComp(+el.closest('.lbl').dataset.c);
  else clearSel();
}

// ---- highlighting ----
let cur = null;
function reset() {
  svg.classList.remove('dim');
  halo.textContent = '';
  document.querySelectorAll('#wires .w.on').forEach(el => el.classList.remove('on'));
  document.querySelectorAll('.lbl.on, .lbl.sel').forEach(el => el.classList.remove('on', 'sel'));
  document.querySelectorAll('#list div.cur').forEach(el => el.classList.remove('cur'));
}
function lightNets(nets) {
  svg.classList.add('dim');
  const frag = document.createDocumentFragment();
  for (const n of nets) for (const s of D.nets[n].segs) {
    segEls[s].forEach(el => el.classList.add('on'));
    const h = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    h.setAttribute('points', segHit[s].getAttribute('points'));
    frag.appendChild(h);
  }
  halo.appendChild(frag);
}
function lightComps(ids, selected) {
  ids.forEach(i => lblEls[i] && lblEls[i].classList.add('on'));
  if (selected != null && lblEls[selected]) lblEls[selected].classList.add('sel');
}
function chip(code) {
  const [b, s] = code.split('/');
  const bh = WIRE_HEX[b] || '#ccc', sh = s ? (WIRE_HEX[s] || '#ccc') : null;
  const bg = sh ? `repeating-linear-gradient(90deg, ${bh} 0 7px, ${sh} 7px 10px)` : bh;
  return `<span class="chip"><span class="sw" style="background:${bg}"></span>${code}</span>`;
}
function netColors(n) { return [...new Set(D.nets[n].segs.map(s => D.segs[s][1]))].sort(); }
function compLink(i) {
  const c = D.comps[i];
  return `<a data-c="${i}">${c.ref}. ${esc(c.name)}</a>`;
}
function esc(s) { return s.replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch])); }

function selectNet(n, keepComp) {
  reset();
  cur = {net: n};
  lightNets([n]);
  lightComps(netComps[n]);
  const comps = netComps[n];
  info.innerHTML = `<h2>Circuit ${n + 1}</h2>
    <div class="sub">${D.nets[n].segs.length} wire segment(s)</div>
    <div class="chips">${netColors(n).map(chip).join('')}</div>
    <div class="sub" style="margin-top:8px">Connected components (${comps.length})</div>
    <div class="links">${comps.map(compLink).join('') || '<span class="empty">None detected</span>'}</div>`;
}
function selectComp(i, zoom) {
  reset();
  const c = D.comps[i];
  cur = {comp: i};
  lightNets(c.nets);
  const others = new Set();
  c.nets.forEach(n => netComps[n].forEach(j => others.add(j)));
  lightComps([...others], i);
  const rows = c.nets.map(n => {
    const o = netComps[n].filter(j => j !== i);
    return `<div class="circuit" data-n="${n}"><div class="chips">${netColors(n).map(chip).join('')}</div>
      <div class="links">${o.map(compLink).join('') || '<span class="sub">no other component</span>'}</div></div>`;
  }).join('');
  info.innerHTML = `<h2>${c.ref}. ${esc(c.name)}</h2>
    <div class="sub">${c.nets.length} connected circuit(s) - click one to show it alone</div>${rows ||
    '<div class="empty">No wires detected for this component.</div>'}`;
  const li = document.querySelector(`#list div[data-c="${i}"]`);
  if (li) li.classList.add('cur');
  if (zoom) {
    let [x0, y0, x1, y1] = c.box;
    for (const n of c.nets) for (const s of D.nets[n].segs) {
      const b = D.segs[s][2];
      x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
    }
    focusBox(x0, y0, x1, y1);
  }
}
function clearSel() {
  reset(); cur = null;
  info.innerHTML = '<div class="empty">Nothing selected.</div>';
}
info.addEventListener('click', e => {
  const a = e.target.closest('a[data-c]');
  if (a) { selectComp(+a.dataset.c, true); return; }
  const row = e.target.closest('.circuit');
  if (row && cur && cur.comp != null) {
    const i = cur.comp, n = +row.dataset.n;
    reset();
    lightNets([n]);
    lightComps(netComps[n], i);
    info.querySelectorAll('.circuit').forEach(r => r.classList.toggle('cur', r === row));
  }
});
document.getElementById('clear').onclick = clearSel;
document.addEventListener('keydown', e => { if (e.key === 'Escape') clearSel(); });

// ---- component list / search ----
const list = document.getElementById('list');
const order = D.comps.map((c, i) => i).sort((a, b) => D.comps[a].ref - D.comps[b].ref || a - b);
const seen = {};
list.innerHTML = order.map(i => {
  const c = D.comps[i];
  seen[c.ref] = (seen[c.ref] || 0) + 1;
  const dup = D.comps.filter(o => o.ref === c.ref).length > 1 ? ` (${seen[c.ref]})` : '';
  return `<div data-c="${i}" data-q="${esc((c.ref + ' ' + c.name).toLowerCase())}"><span class="n">${c.ref}</span><span>${esc(c.name)}${dup}</span></div>`;
}).join('');
list.addEventListener('click', e => { const d = e.target.closest('div[data-c]'); if (d) selectComp(+d.dataset.c, true); });
document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase();
  list.querySelectorAll('div[data-c]').forEach(d => d.style.display = !q || d.dataset.q.includes(q) ? '' : 'none');
});

window.addEventListener('resize', fit);
fit();
</script>
</body>
</html>
"""


def build(drawio_path: str, page: int = 0) -> tuple[str, dict]:
    cells, width, height, name = load_drawio(drawio_path, page)
    segments, seg_net, n_nets, ends = analyze(cells)
    comps = assign_components(cells, segments, seg_net, ends)

    seg_of_cell = {cid: i for i, s in enumerate(segments) for cid in s["cells"]}
    comp_ids = list(comps)
    comp_index = {lid: k for k, lid in enumerate(comp_ids)}
    labels = {c.id: c for c in cells if c.id in comps}

    def bbox(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]

    nets = [{"segs": []} for _ in range(n_nets)]
    for i, n in enumerate(seg_net):
        nets[n]["segs"].append(i)
    data = {
        "segs": [[seg_net[i], s["color"], bbox(s["pts"])] for i, s in enumerate(segments)],
        "nets": nets,
        "comps": [],
    }
    for lid in comp_ids:
        c = comps[lid]
        tip = labels[lid].attrs.get("tooltip", "")
        cname = re.sub(r"^#\d+\s*", "", tip) or html.unescape(labels[lid].value.replace("<br>", " "))
        data["comps"].append({"ref": c["ref"], "name": cname, "nets": sorted(c["nets"]),
                              "box": [round(v) for v in c["rect"]]})

    svg = render_svg(cells, width, height, seg_of_cell, comp_index)
    title = name or "Wiring diagram"
    page_html = (PAGE.replace("__TITLE__", html.escape(title))
                 .replace("__SVG__", svg)
                 .replace("__WIRE_HEX__", json.dumps(WIRE_HEX))
                 .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    stats = {"segments": len(segments), "circuits": n_nets, "components": len(comps),
             "components_without_wires": sorted(c["ref"] for c in comps.values() if not c["nets"])}
    return page_html, stats


def main():
    ap = argparse.ArgumentParser(description="Build an interactive HTML viewer from a wiring diagram .drawio file")
    ap.add_argument("drawio", help="input .drawio file (from wiring_to_drawio.py)")
    ap.add_argument("-o", "--output", default=None, help="output .html file")
    ap.add_argument("--page", type=int, default=1, help="1-based page of the .drawio file (default 1)")
    args = ap.parse_args()

    out = args.output or re.sub(r"\.drawio$", "", args.drawio, flags=re.I) + ".html"
    page_html, stats = build(args.drawio, args.page - 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"{stats['segments']} wire segments, {stats['circuits']} circuits, {stats['components']} component "
          f"labels; without wires: {stats['components_without_wires'] or 'none'}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
