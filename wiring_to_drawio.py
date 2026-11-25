#!/usr/bin/env python3
"""
Convert the Yamaha MT07 wiring diagram PDF into a draw.io (.drawio) file.

- The component list on the legend page ("1. Main switch", ...) is parsed into a
  number -> name lookup table.
- All vector paths and text on the diagram page (the color version by default)
  are converted:
    * stroke-only paths made of straight lines -> draw.io edges (editable);
      wires get a tooltip with their color code (e.g. R/W)
    * paths with curves or fills -> custom draw.io stencil shapes
    * text -> draw.io text cells
- Numbered callout circles are removed and replaced by a label with the
  component name.
- The diagram is stored rotated by 90 degrees in the PDF; it is rotated back to
  landscape on output.
- Power supplies, grounds and voltages are marked by power_rails.py.
- A second "Legend" page lists components, wire harnesses and wire color codes.

Usage:
    python wiring_to_drawio.py MT07-wiring.pdf -o MT07-wiring.drawio
    python wiring_to_drawio.py MT07-wiring.pdf --page 5 --with-number

Requires: pip install pymupdf
"""

from __future__ import annotations

import argparse
import base64
import math
import re
import sys
import textwrap
import urllib.parse
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

import pymupdf

# ---------------------------------------------------------------------------
# Wire colors
# ---------------------------------------------------------------------------

# Colors actually used in the color diagram -> wire color code
WIRE_COLORS = {
    "#231F20": "B", "#B7705B": "Br", "#8B0204": "Ch", "#00652E": "Dg", "#0BB14B": "G",
    "#939598": "Gy", "#00AEEF": "L", "#8DC63F": "Lg", "#F36F21": "O", "#F49AC1": "P",
    "#ED1C24": "R", "#6DCFF6": "Sb", "#FFFFFF": "W", "#FFF200": "Y",
}
CODE_TO_HEX = {v: k for k, v in WIRE_COLORS.items()}

LABEL_OPACITY = 35   # component label fill/border opacity (percent)
WIRE_WIDTH = 1.47    # base color stroke width of a wire (pt)
STRIPE_WIDTH = 1.13  # stripe (dashed) stroke width of a wire (pt)
CORE_WIDTH = 0.80    # white core of an outlined white wire, and the stripes on it (pt)


# ---------------------------------------------------------------------------
# Legend parsing
# ---------------------------------------------------------------------------

@dataclass
class Legend:
    components: dict[int, str] = field(default_factory=dict)
    harnesses: dict[str, str] = field(default_factory=dict)
    colors: list[tuple[str, str]] = field(default_factory=list)  # (code, name)
    title: str = "WIRING DIAGRAM"


def parse_legend(doc: pymupdf.Document) -> Legend:
    """Parse component numbers, wire harnesses and color codes from the page containing 'COLOR CODE'."""
    page = next((p for p in doc if "COLOR CODE" in p.get_text()), doc[0])
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    leg = Legend()

    cur_key = None
    in_colors = False
    pending_code = None
    for i, line in enumerate(lines):
        if line == "WIRING DIAGRAM" and i + 1 < len(lines):
            leg.title = f"{lines[i + 1]} {line}"
            continue
        if line.startswith("EAS") and line[3:].isdigit():
            continue
        if line == "COLOR CODE":
            in_colors, cur_key = True, None
            continue
        if in_colors:
            m = re.match(r"^([A-Z][a-z]?(?:/[A-Z][a-z]?)?)\s+(.+)$", line)
            if m:
                leg.colors.append((m.group(1), m.group(2)))
            elif re.fullmatch(r"[A-Z][a-z]?(?:/[A-Z][a-z]?)?", line):
                pending_code = line
            elif pending_code:
                leg.colors.append((pending_code, line))
                pending_code = None
            continue
        m = re.match(r"^(\d{1,3})\.\s*(.*)$", line)
        if m:
            cur_key = int(m.group(1))
            leg.components[cur_key] = m.group(2)
            continue
        m = re.match(r"^([A-D])\.\s+(.*)$", line)
        if m:
            cur_key = m.group(1)
            leg.harnesses[cur_key] = m.group(2)
            continue
        if cur_key is not None:  # continuation of a wrapped line
            target = leg.components if isinstance(cur_key, int) else leg.harnesses
            prev = target[cur_key]
            target[cur_key] = prev[:-1] + line if prev.endswith("-") else f"{prev} {line}"

    # The subscript 2 in "O2" may be extracted as a separate character
    for k, v in leg.components.items():
        leg.components[k] = re.sub(r"\s+", " ", v).replace("O 2", "O2")
    return leg


# ---------------------------------------------------------------------------
# draw.io output helpers
# ---------------------------------------------------------------------------

def hexcolor(rgb) -> str:
    if rgb is None:
        return "none"
    return "#%02X%02X%02X" % tuple(max(0, min(255, round(v * 255))) for v in rgb[:3])


def drawio_compress(text: str) -> str:
    """Same as draw.io's Graph.compress: encodeURIComponent -> deflateRaw -> base64."""
    data = urllib.parse.quote(text, safe="~()*!.'").encode("ascii")
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(c.compress(data) + c.flush()).decode("ascii")


def fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def parse_dashes(dashes: str | None, scale: float) -> str | None:
    if not dashes:
        return None
    m = re.match(r"\[\s*(.*?)\s*\]", dashes)
    if not m or not m.group(1):
        return None
    vals = [float(v) * scale for v in m.group(1).split()]
    return " ".join(fmt(v) for v in vals)


class DrawioWriter:
    def __init__(self):
        self.cells: list[str] = []
        self._id = 1

    def nid(self) -> str:
        self._id += 1
        return f"c{self._id}"

    def vertex(self, x, y, w, h, style, value="", attrs=None):
        cid = self.nid()
        geom = f'<mxGeometry x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" as="geometry"/>'
        cell = f'<mxCell style={quoteattr(style)} vertex="1" parent="1">{geom}</mxCell>'
        self._append(cid, cell, value, attrs)

    def edge(self, pts, style, value="", attrs=None):
        cid = self.nid()
        (sx, sy), (tx, ty) = pts[0], pts[-1]
        mid = "".join(f'<mxPoint x="{fmt(x)}" y="{fmt(y)}"/>' for x, y in pts[1:-1])
        geom = (
            '<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{fmt(sx)}" y="{fmt(sy)}" as="sourcePoint"/>'
            f'<mxPoint x="{fmt(tx)}" y="{fmt(ty)}" as="targetPoint"/>'
            + (f'<Array as="points">{mid}</Array>' if mid else "")
            + "</mxGeometry>"
        )
        cell = f'<mxCell style={quoteattr(style)} edge="1" parent="1">{geom}</mxCell>'
        self._append(cid, cell, value, attrs)

    def _append(self, cid, cell, value, attrs):
        if attrs:
            extra = " ".join(f"{k}={quoteattr(str(v))}" for k, v in attrs.items())
            self.cells.append(
                f'<UserObject label={quoteattr(value)} {extra} id="{cid}">{cell}</UserObject>'
            )
        else:
            self.cells.append(cell.replace("<mxCell ", f'<mxCell id="{cid}" value={quoteattr(value)} ', 1))

    def model_xml(self, width, height, background="#FFFFFF") -> str:
        body = "".join(self.cells)
        return (
            f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" '
            f'connect="0" arrows="0" fold="1" page="1" pageScale="1" pageWidth="{int(width)}" '
            f'pageHeight="{int(height)}" background="{background}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel>'
        )


def mxfile(pages: list[tuple[str, str]]) -> str:
    diagrams = "".join(
        f'<diagram id="p{i}" name={quoteattr(name)}>{xml}</diagram>' for i, (name, xml) in enumerate(pages)
    )
    return f'<mxfile host="wiring_to_drawio.py" type="device">{diagrams}</mxfile>\n'


# ---------------------------------------------------------------------------
# Geometry transform (rotate + scale)
# ---------------------------------------------------------------------------

class Transform:
    """PDF coordinates -> draw.io coordinates. With rotate=True a diagram rotated 90 degrees clockwise is turned back."""

    def __init__(self, bbox: pymupdf.Rect, scale: float, rotate: bool, margin: float = 40):
        self.b, self.s, self.rot, self.m = bbox, scale, rotate, margin

    def pt(self, p) -> tuple[float, float]:
        x, y = p[0], p[1]
        if self.rot:
            return ((y - self.b.y0) * self.s + self.m, (self.b.x1 - x) * self.s + self.m)
        return ((x - self.b.x0) * self.s + self.m, (y - self.b.y0) * self.s + self.m)

    def rect(self, r) -> tuple[float, float, float, float]:
        (ax, ay), (bx, by) = self.pt((r[0], r[1])), self.pt((r[2], r[3]))
        return min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)

    @property
    def size(self) -> tuple[float, float]:
        w, h = self.b.width, self.b.height
        if self.rot:
            w, h = h, w
        return w * self.s + 2 * self.m, h * self.s + 2 * self.m


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------

def subpaths(drawing) -> list[list[tuple]]:
    """Split PyMuPDF path items into subpaths; each segment is ('l', p0, p1) or ('c', p0, c1, c2, p1)."""
    out, cur, last = [], [], None

    def flush():
        nonlocal cur
        if cur:
            out.append(cur)
        cur = []

    for it in drawing["items"]:
        kind = it[0]
        if kind == "re":
            r = it[1]
            flush()
            q = [r.tl, r.tr, r.br, r.bl, r.tl]
            out.append([("l", q[i], q[i + 1]) for i in range(4)])
            last = None
            continue
        if kind == "qu":
            q = it[1]
            flush()
            pts = [q.ul, q.ur, q.lr, q.ll, q.ul]
            out.append([("l", pts[i], pts[i + 1]) for i in range(4)])
            last = None
            continue
        start = it[1]
        if last is None or abs(start.x - last.x) > 1e-3 or abs(start.y - last.y) > 1e-3:
            flush()
        cur.append(it)
        last = it[-1]
    flush()
    if drawing.get("closePath"):
        for sp in out:
            a, b = sp[0][1], sp[-1][-1]
            if abs(a.x - b.x) > 1e-3 or abs(a.y - b.y) > 1e-3:
                sp.append(("l", b, a))
    return out


def is_lines_only(drawing) -> bool:
    return all(it[0] in ("l", "re", "qu") for it in drawing["items"])


def stencil_xml(sps, T: Transform, ox, oy, w, h, op: str) -> str:
    parts = [f'<shape w="{fmt(w)}" h="{fmt(h)}" aspect="variable" strokewidth="inherit"><foreground><path>']
    for sp in sps:
        x, y = T.pt(sp[0][1])
        parts.append(f'<move x="{fmt(x - ox)}" y="{fmt(y - oy)}"/>')
        for seg in sp:
            if seg[0] == "l":
                x, y = T.pt(seg[2])
                parts.append(f'<line x="{fmt(x - ox)}" y="{fmt(y - oy)}"/>')
            else:
                (x1, y1), (x2, y2), (x3, y3) = T.pt(seg[2]), T.pt(seg[3]), T.pt(seg[4])
                parts.append(
                    f'<curve x1="{fmt(x1 - ox)}" y1="{fmt(y1 - oy)}" x2="{fmt(x2 - ox)}" '
                    f'y2="{fmt(y2 - oy)}" x3="{fmt(x3 - ox)}" y3="{fmt(y3 - oy)}"/>'
                )
        a, b = sp[0][1], sp[-1][-1]
        if abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) < 1e-3:
            parts.append("<close/>")
    parts.append(f"</path><{op}/></foreground></shape>")
    return "".join(parts)


def wire_kind(d):
    """'base', 'stripe' or 'core' for the stroke-only paths that make up a wire, else None."""
    if d["type"] != "s" or not is_lines_only(d):
        return None
    w = d.get("width") or 0
    if abs(w - WIRE_WIDTH) < 0.05:
        return "base"
    if abs(w - STRIPE_WIDTH) < 0.05:
        return "stripe"
    if abs(w - CORE_WIDTH) < 0.02:
        return "core"
    return None


def sample_path(drawing, step=2.0):
    pts = []
    for sp in subpaths(drawing):
        for seg in sp:
            a, b = seg[1], seg[-1]
            n = max(1, int(math.dist((a.x, a.y), (b.x, b.y)) / step))
            pts += [(a.x + (b.x - a.x) * (k + 0.5) / n, a.y + (b.y - a.y) * (k + 0.5) / n) for k in range(n)]
    return pts


def path_distance(p, drawing):
    best = 1e9
    for sp in subpaths(drawing):
        for seg in sp:
            a, b = seg[1], seg[-1]
            dx, dy = b.x - a.x, b.y - a.y
            L = dx * dx + dy * dy
            t = 0 if L == 0 else max(0, min(1, ((p[0] - a.x) * dx + (p[1] - a.y) * dy) / L))
            best = min(best, math.hypot(p[0] - a.x - t * dx, p[1] - a.y - t * dy))
    return best


def classify_wires(drawings):
    """Return ({base index: color code}, {overlay index: base index}).

    Stripes are matched to the wire they run along by overlap, because a stripe path is
    not always drawn with exactly the same end points as its base. White wires are drawn
    as a dark or colored outline with a white core on top.
    """
    kinds = {i: k for i, d in enumerate(drawings) if (k := wire_kind(d))}
    bases = [i for i, k in kinds.items() if k == "base"]
    overlays = defaultdict(list)
    overlay_of = {}
    for i, k in kinds.items():
        if k == "base":
            continue
        pts = sample_path(drawings[i])
        r = drawings[i]["rect"] + (-1, -1, 1, 1)
        best, best_i = 0.0, None
        for b in bases:
            if not r.intersects(drawings[b]["rect"] + (-0.5, -0.5, 0.5, 0.5)):
                continue
            cover = sum(path_distance(p, drawings[b]) < 0.3 for p in pts) / max(1, len(pts))
            # on a tie prefer the base drawn last before this overlay (paint order)
            if cover > best + 0.02 or (abs(cover - best) <= 0.02 and best_i is not None and best_i < b < i):
                best, best_i = cover, b
        if best >= 0.6:
            overlay_of[i] = best_i
            overlays[best_i].append(i)
    codes = {}
    for b in bases:
        base = WIRE_COLORS.get(hexcolor(drawings[b]["color"]), "?")
        stripes = [WIRE_COLORS.get(hexcolor(drawings[i]["color"]), "?") for i in overlays[b]
                   if kinds[i] == "stripe" or drawings[i].get("dashes") not in (None, "", "[] 0")]
        white_core = any(kinds[i] == "core" and hexcolor(drawings[i]["color"]) == "#FFFFFF" for i in overlays[b])
        if white_core:
            stripe = next((c for c in reversed(stripes) if c not in ("B", "W")), None)
            codes[b] = f"W/{stripe}" if stripe else "W"
        else:
            codes[b] = f"{base}/{stripes[-1]}" if stripes else base
    return codes, overlay_of


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def pick_diagram_page(doc) -> int:
    """Pick the color wiring diagram page: most vector paths, weighted by colored strokes."""
    best, best_score = 0, -1
    for i, p in enumerate(doc):
        dr = p.get_drawings()
        colored = sum(1 for d in dr if d.get("color") and hexcolor(d["color"]) not in ("#231F20", "#FFFFFF"))
        score = len(dr) + colored * 10
        if score > best_score:
            best, best_score = i, score
    return best


def is_circle(d, size_range=(4.5, 9.5)) -> bool:
    r = d["rect"]
    return (d["type"] == "s" and len(d["items"]) == 4 and all(it[0] == "c" for it in d["items"])
            and size_range[0] <= r.width <= size_range[1] and abs(r.width - r.height) < 0.6)


def spread_labels(labels, max_shift: float, gap: float = 2, iterations: int = 200):
    """Simple collision removal: push overlapping labels apart along the axis of least overlap,
    never moving a label more than max_shift from its original position."""
    for _ in range(iterations):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                ox = (a[2] + b[2]) / 2 + gap - abs(a[0] - b[0])
                oy = (a[3] + b[3]) / 2 + gap - abs(a[1] - b[1])
                if ox <= 0 or oy <= 0:
                    continue
                moved = True
                if oy <= ox:
                    d = oy / 2 * (1 if a[1] >= b[1] else -1)
                    a[1] += d; b[1] -= d
                else:
                    d = ox / 2 * (1 if a[0] >= b[0] else -1)
                    a[0] += d; b[0] -= d
        for l in labels:  # clamp the distance from the original position
            for k, k0 in ((0, 6), (1, 7)):
                l[k] = max(l[k0] - max_shift, min(l[k0] + max_shift, l[k]))
        if not moved:
            break


def convert(pdf_path: str, page_no: int | None, scale: float, with_number: bool,
            label_font: float, rotate: bool | None) -> str:
    doc = pymupdf.open(pdf_path)
    legend = parse_legend(doc)
    missing = sorted(set(range(1, max(legend.components, default=0) + 1)) - set(legend.components))
    if missing:
        print(f"warning: component list is missing numbers {missing}", file=sys.stderr)
    names = legend.components

    if page_no is None:
        page_no = pick_diagram_page(doc)
    page = doc[page_no]
    page.set_rotation(0)  # a viewer may have saved a page rotation; work in the original frame
    drawings = page.get_drawings()
    tdict = page.get_text("dict")

    spans = []
    for b in tdict["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    spans.append((s, l["dir"]))

    if rotate is None:  # mostly vertical text -> the diagram is rotated
        vertical = sum(1 for _, d in spans if abs(d[0]) < 0.5)
        rotate = vertical > len(spans) / 2

    # Content bounds
    bbox = pymupdf.Rect()
    for d in drawings:
        bbox |= d["rect"]
    for s, _ in spans:
        bbox |= pymupdf.Rect(s["bbox"])
    T = Transform(bbox, scale, rotate)

    # Find callout numbers and the circles around them
    callouts = []  # (number, circle rect)
    circle_idx = [i for i, d in enumerate(drawings) if is_circle(d)]
    used_circles = set()
    num_spans = set()
    for si, (s, _) in enumerate(spans):
        t = s["text"].strip()
        if not (t.isdigit() and int(t) in names):
            continue
        sb = pymupdf.Rect(s["bbox"])
        c = pymupdf.Point((sb.x0 + sb.x1) / 2, (sb.y0 + sb.y1) / 2)
        # smallest enclosing circle: a symbol may draw a larger circle around its callout
        hit = min((ci for ci in circle_idx if ci not in used_circles and drawings[ci]["rect"].contains(c)),
                  key=lambda ci: drawings[ci]["rect"].width, default=None)
        if hit is None:  # a number without a circle is not a callout
            continue
        used_circles.add(hit)
        num_spans.add(si)
        callouts.append((int(t), drawings[hit]["rect"]))

    code_of, overlay_of = classify_wires(drawings)

    W = DrawioWriter()
    stats = {"edges": 0, "shapes": 0, "texts": 0, "labels": 0}

    # ---- Vector paths ----
    for i, d in enumerate(drawings):
        if i in used_circles:
            continue
        typ = d["type"]
        stroke = hexcolor(d.get("color")) if typ in ("s", "fs") else "none"
        fill = hexcolor(d.get("fill")) if typ in ("f", "fs") else "none"
        sw = (d.get("width") or 0) * scale
        dash = parse_dashes(d.get("dashes"), scale) if typ in ("s", "fs") else None
        sps = subpaths(d)
        if not sps:
            continue

        if typ == "s" and is_lines_only(d):
            code = code_of.get(i)
            is_overlay = i in overlay_of
            if is_overlay:
                code = code_of[overlay_of[i]]
            style = (f"endArrow=none;startArrow=none;html=1;rounded=0;edgeStyle=none;jumpStyle=none;"
                     f"strokeColor={stroke};strokeWidth={fmt(sw)};")
            if dash:
                style += f"dashed=1;fixDash=1;dashPattern={dash};"
            if code:
                style += "wire=1;"
            if is_overlay:
                style += "overlay=1;"
            attrs = None
            if code:
                attrs = {"tooltip": f"Wire {code}", "wireColor": code,
                         "wireId": f"w{overlay_of.get(i, i)}"}
            for sp in sps:
                pts = [T.pt(sp[0][1])] + [T.pt(seg[2]) for seg in sp]
                W.edge(pts, style, attrs=attrs)
                stats["edges"] += 1
            continue

        # Curves / fills -> stencil
        x, y, w, h = T.rect(d["rect"])
        pad = max(sw, 0.2)
        x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
        op = {"s": "stroke", "f": "fill", "fs": "fillstroke"}[typ]
        st = stencil_xml(sps, T, x, y, w, h, op)
        style = (f"shape=stencil({drawio_compress(st)});html=1;whiteSpace=wrap;"
                 f"fillColor={fill};strokeColor={stroke};strokeWidth={fmt(sw)};")
        if dash:
            style += f"dashed=1;fixDash=1;dashPattern={dash};"
        W.vertex(x, y, w, h, style)
        stats["shapes"] += 1

    # ---- Plain text ----
    for si, (s, _) in enumerate(spans):
        if si in num_spans:
            continue
        x, y, w, h = T.rect(s["bbox"])
        bold = "Bold" in s["font"] or (s["flags"] & 16)
        family = "Times New Roman" if s["font"].startswith("Times") else "Helvetica"
        style = (f"text;html=1;align=center;verticalAlign=middle;spacing=0;resizable=1;"
                 f"fontFamily={family};fontSize={fmt(s['size'] * scale)};"
                 f"fontColor=#{s['color']:06X};" + ("fontStyle=1;" if bold else ""))
        W.vertex(x, y, w + 2, h, style, value=escape(s["text"].strip()))
        stats["texts"] += 1

    # ---- Component name labels (replacing callout numbers) ----
    fs = label_font
    labels = []  # [cx, cy, w, h, num, lines, cx0, cy0]
    for num, r in callouts:
        text = f"{num}. {names[num]}" if with_number else names[num]
        lines = textwrap.wrap(text, 36) or [text]
        lw = max(len(l) for l in lines) * fs * 0.56 + 8
        lh = len(lines) * fs * 1.25 + 4
        cx, cy = T.pt(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        labels.append([cx, cy, lw, lh, num, lines, cx, cy])
    spread_labels(labels, max_shift=6 * scale)

    # see-through fill and border so the drawing underneath stays readable; solid text
    style = ("rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#231F20;"
             f"strokeWidth={fmt(0.2 * scale)};fontSize={fmt(fs)};fontFamily=Helvetica;fontStyle=1;"
             f"fontColor=#0B3D91;spacing=1;align=center;verticalAlign=middle;fillOpacity={LABEL_OPACITY};"
             f"strokeOpacity={LABEL_OPACITY};textOpacity=100;")
    for cx, cy, lw, lh, num, lines, cx0, cy0 in labels:
        # anchorX/anchorY keep the original callout center (labels may have been nudged apart)
        W.vertex(cx - lw / 2, cy - lh / 2, lw, lh, style, value="<br>".join(escape(l) for l in lines),
                 attrs={"ref": num, "tooltip": f"#{num} {names[num]}",
                        "anchorX": fmt(cx0), "anchorY": fmt(cy0)})
        stats["labels"] += 1

    pw, ph = T.size
    diagram = W.model_xml(pw, ph)
    print(f"page {page_no + 1}: {stats['edges']} lines, {stats['shapes']} shapes, "
          f"{stats['texts']} texts, {stats['labels']} component labels (rotated={rotate})", file=sys.stderr)
    return mxfile([(legend.title, diagram), ("Legend", build_legend_page(legend))])


def build_legend_page(legend: Legend) -> str:
    W = DrawioWriter()
    fs, rh = 12, 18
    hdr = "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;spacing=0;"
    cell = f"text;html=1;align=left;verticalAlign=middle;fontSize={fs};spacing=0;whiteSpace=wrap;"

    W.vertex(20, 10, 600, 30, hdr.replace("fontSize=16", "fontSize=20"), escape(legend.title))
    W.vertex(20, 50, 300, 24, hdr, "Components")
    keys = sorted(legend.components)
    per_col = math.ceil(len(keys) / 2)
    for i, k in enumerate(keys):
        col, row = divmod(i, per_col)
        x, y = 20 + col * 420, 80 + row * rh
        W.vertex(x, y, 30, rh, cell + "align=right;", f"{k}.")
        W.vertex(x + 36, y, 380, rh, cell, escape(legend.components[k]))

    y0 = 80 + per_col * rh + 20
    W.vertex(20, y0, 300, 24, hdr, "Wire harness")
    for i, (k, v) in enumerate(sorted(legend.harnesses.items())):
        W.vertex(20, y0 + 30 + i * rh, 30, rh, cell + "align=right;", f"{k}.")
        W.vertex(56, y0 + 30 + i * rh, 780, rh, cell, escape(v))

    # Wire color codes
    x0 = 900
    W.vertex(x0, 50, 300, 24, hdr, "Color code")
    for i, (code, cname) in enumerate(legend.colors):
        y = 80 + i * rh
        parts = code.split("/")
        base = CODE_TO_HEX.get(parts[0])
        stripe = CODE_TO_HEX.get(parts[1]) if len(parts) > 1 else None
        if base:
            pts = [(x0, y + rh / 2), (x0 + 50, y + rh / 2)]
            W.edge(pts, f"endArrow=none;html=1;rounded=0;strokeColor={base};strokeWidth=6;")
            if stripe:
                W.edge(pts, f"endArrow=none;html=1;rounded=0;strokeColor={stripe};strokeWidth=4.5;"
                            "dashed=1;fixDash=1;dashPattern=4 4 4 15;")
            if base == "#FFFFFF":  # outline white wires so they stay visible
                for dy in (-3.5, 3.5):
                    W.edge([(x0, y + rh / 2 + dy), (x0 + 50, y + rh / 2 + dy)],
                           "endArrow=none;html=1;rounded=0;strokeColor=#231F20;strokeWidth=0.5;")
        W.vertex(x0 + 60, y, 50, rh, cell + "fontStyle=1;", escape(code))
        W.vertex(x0 + 110, y, 240, rh, cell, escape(cname))
    return W.model_xml(1300, max(y0 + 120, 80 + len(legend.colors) * rh + 40))


def main():
    ap = argparse.ArgumentParser(
        description="Convert a wiring diagram PDF to draw.io, replacing callout numbers with component names")
    ap.add_argument("pdf", help="input PDF (e.g. MT07-wiring.pdf)")
    ap.add_argument("-o", "--output", default=None, help="output .drawio file")
    ap.add_argument("--page", type=int, default=None,
                    help="1-based page number of the diagram (default: auto-detect the color diagram)")
    ap.add_argument("--scale", type=float, default=4.0, help="coordinate scale factor (default 4)")
    ap.add_argument("--with-number", action="store_true", help="keep the number in labels, e.g. '1. Main switch'")
    ap.add_argument("--label-font", type=float, default=9, help="component label font size (default 9)")
    ap.add_argument("--no-power", action="store_true", help="do not mark power, ground and voltages")
    ap.add_argument("--no-rotate", action="store_true", help="do not rotate the diagram upright")
    args = ap.parse_args()

    out = args.output or re.sub(r"\.pdf$", "", args.pdf, flags=re.I) + ".drawio"
    xml = convert(args.pdf, None if args.page is None else args.page - 1, args.scale,
                  args.with_number, args.label_font, False if args.no_rotate else None)
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    if not args.no_power:
        import power_rails
        power_rails.annotate(out)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
