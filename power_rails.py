#!/usr/bin/env python3
"""
Mark power supplies, grounds and voltages on the wiring diagram .drawio file.

Wire color alone does not tell what a wire carries (the black starter motor lead is
+12 V while cranking; the black clutch and sidestand leads are not plain grounds), so
every rail is seeded from a component terminal in RULES and then followed through the
wiring, the joint couplers and the engine ground point.

A wire that would get two different rails, or a rule that matches no wire, is an error.
Wires that no rule reaches (control and data lines) are left unmarked.

Voltages come from the service manual (MT07 1XB): system 12 V, regulated 14.1-14.9 V,
sensor output voltages. The one exception is the sensor supply: the manual gives no value
for it, so it is marked "5V*" (nominal, assumed) everywhere.

Usage:
    python power_rails.py MT07-wiring.drawio        (wiring_to_drawio.py runs this too)
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import build_viewer as bv

# kind -> (badge fill, badge text color, description)
KINDS = {
    "BATT": ("#B00020", "#FFFFFF", "Battery +12 V, always live (14.1-14.9 V regulated, engine running)"),
    "IGN": ("#E8590C", "#FFFFFF", "+12 V with main switch ON"),
    "PARK": ("#862E9C", "#FFFFFF", "+12 V with main switch ON or P (parking)"),
    "SW": ("#FFD43B", "#000000", "+12 V only while its switch or relay is on"),
    "SENS": ("#1864AB", "#FFFFFF", "Sensor supply from ECU. *5 V nominal is assumed; the manual gives no value"),
    "SIG": ("#2B8A3E", "#FFFFFF", "Sensor output voltage from the manual"),
    "AC": ("#0B7285", "#FFFFFF", "Stator AC, three phase"),
    "GND": ("#000000", "#FFFFFF", "Chassis / battery ground, 0 V"),
    "SGND": ("#5F6B76", "#FFFFFF", "Sensor ground returning to ECU, 0 V"),
}

# (component number, wire color at its terminal, kind, badge text, note)
RULES = [
    (15, "B", "GND", "GND 0V", "Battery negative / chassis ground"),
    (37, "B/L", "SGND", "SGND 0V", "Sensor ground"),
    (15, "R", "BATT", "+12V BATT", "Battery positive, before main fuse"),
    (18, "R", "BATT", "+12V BATT", "Battery / main fuse, always live"),
    (2, "R/W", "BATT", "+12V BATT", "ABS solenoid fuse output"),
    (3, "R/L", "BATT", "+12V BATT", "ABS motor fuse output"),
    (10, "R/B", "BATT", "+12V BATT", "Fuel injection system fuse output"),
    (11, "R/G", "BATT", "+12V BATT", "Backup fuse output"),
    (12, "R", "BATT", "+12V BATT", "Radiator fan motor fuse output"),
    (1, "Br/L", "IGN", "+12V IGN", "Main switch output, ON"),
    (5, "Br/W", "IGN", "+12V IGN", "ABS control unit fuse output"),
    (6, "R/Y", "IGN", "+12V IGN", "Auxiliary fuse output"),
    (7, "R/W", "IGN", "+12V IGN", "Ignition fuse output"),
    (8, "L/R", "IGN", "+12V IGN", "Signaling system fuse output"),
    (9, "R/Y", "IGN", "+12V IGN", "Headlight fuse output"),
    (1, "Br/R", "PARK", "+12V PARK", "Main switch output, ON or P"),
    (4, "L/R", "PARK", "+12V PARK", "Parking lighting fuse output"),
    (67, "R/B", "SW", "+12V RUN", "Engine stop switch in RUN"),
    (24, "R/L", "SW", "+12V FP", "Fuel pump relay on"),
    (86, "L", "SW", "+12V FAN", "Radiator fan motor relay on"),
    (87, "G/L", "SW", "+12V HL", "Headlight relay on"),
    (58, "Y", "SW", "+12V HI", "High beam or pass switch on"),
    (71, "G", "SW", "+12V LO", "Low beam on"),
    (68, "Br/W", "SW", "+12V FLASH", "Turn signal/hazard relay output, pulsed"),
    (75, "Dg", "SW", "+12V RIGHT", "Right turn signal, pulsed"),
    (77, "Ch", "SW", "+12V LEFT", "Left turn signal, pulsed"),
    (84, "Y", "SW", "+12V BRAKE", "Front or rear brake light switch on"),
    (62, "P", "SW", "+12V HORN", "Horn switch pressed"),
    (20, "B", "SW", "+12V START", "Starter relay on (cranking); black but not a ground"),
    (37, "L", "SENS", "5V*", "Sensor supply; value assumed, not in the manual"),
    (29, "W", "SIG", "0.63-0.73V", "Throttle position sensor output at idle"),
    (40, "P/W", "SIG", "3.57-3.71V", "Intake air pressure sensor output at 101.3 kPa"),
    (41, "Y/G", "SIG", "0.4-1.4V", "Lean angle sensor output below 65 deg (3.7-4.4 V above)"),
    (13, "W", "AC", "AC", "AC magneto phase (standard output 14.0 V, 29.3 A at 5000 r/min)"),
]

PASS_THROUGH = {25, 16}  # joint couplers and engine ground: plain internal connections
BADGE_H, BADGE_FONT = 9.0, 7.0
LAYER = "power_layer"
PIN_REACH = 14.0


def internal_links(cells, box, pins):
    """Groups of pins joined by the plain lines drawn inside a pass-through component."""
    pad = bv.BOX_PAD
    lines = []
    for c in cells:
        if c.kind == "edge" and c.style.get("wire") != "1" and c.style.get("dashed") != "1" \
                and all(box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad for x, y in c.pts):
            xs, ys = [p[0] for p in c.pts], [p[1] for p in c.pts]
            closed = len(c.pts) >= 4 and math.dist(c.pts[0], c.pts[-1]) < 0.5
            if not closed and max(max(xs) - min(xs), max(ys) - min(ys)) > 0:
                lines.append(c.pts)
    # drop the divider lines of connector symbols: they cross the lead without ending on it
    lines = [p for p in lines if not any(_in_housing(p, c) for c in cells)]
    uf = bv.UnionFind(len(lines))
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if i != j and any(bv.poly_dist(q, b)[0] <= 2.0 for q in (a[0], a[-1])):
                uf.union(i, j)
    for center, dia, filled in bv.round_marks(cells):  # junction dots
        if filled and dia <= 6.0:
            on = [i for i, l in enumerate(lines) if bv.poly_dist(center, l)[0] <= 1.5]
            for i in on[1:]:
                uf.union(on[0], i)
    groups = defaultdict(list)
    for pin in pins:
        d, i = min(((bv.poly_dist(pin[2], l)[0], i) for i, l in enumerate(lines)), default=(1e9, None))
        if d <= PIN_REACH:
            groups[uf.find(i)].append(pin)
    return list(groups.values())


def _in_housing(pts, c):
    if c.kind != "edge" or len(c.pts) != 5 or c.style.get("dashed") == "1" or math.dist(c.pts[0], c.pts[-1]) > 0.5:
        return False
    xs, ys = [p[0] for p in c.pts], [p[1] for p in c.pts]
    if min(max(xs) - min(xs), max(ys) - min(ys)) > 15:
        return False
    return all(min(xs) - 1 <= x <= max(xs) + 1 and min(ys) - 1 <= y <= max(ys) + 1 for x, y in pts)


def classify(cells):
    """Return (segments, seg_net, ends, {net: rule})."""
    segments, seg_net, n_nets, ends = bv.analyze(cells)
    comps = bv.assign_components(cells, segments, seg_net, ends)
    uf = bv.UnionFind(n_nets)
    for c in comps.values():
        if c["ref"] in PASS_THROUGH and c["box"]:
            pins = [(i, w, segments[i]["pts"][0 if w == 0 else -1]) for i, w in c["pins"]]
            for g in internal_links(cells, c["box"], pins):
                for i, _, _ in g[1:]:
                    uf.union(seg_net[g[0][0]], seg_net[i])
    rails, errors = {}, []
    for rule in RULES:
        ref, color, kind = rule[:3]
        hit = {seg_net[i] for c in comps.values() if c["ref"] == ref for i, _ in c["pins"]
               if segments[i]["color"] == color}
        if not hit:
            errors.append(f"rule {rule[:3]} matches no wire")
        for n in hit:
            r = uf.find(n)
            if r in rails and rails[r][2] != kind:
                errors.append(f"conflict on {color}: {rails[r][:3]} vs {rule[:3]}")
            rails.setdefault(r, rule)
    if errors:
        raise SystemExit("power rails:\n  " + "\n  ".join(errors))
    return segments, seg_net, ends, {n: rails[uf.find(n)] for n in range(n_nets) if uf.find(n) in rails}


# ---------------------------------------------------------------------------
# Badge placement
# ---------------------------------------------------------------------------

def walk(pts, which, s):
    """Point and direction at distance s along the polyline from the given end."""
    seq = pts if which == 0 else pts[::-1]
    for a, b in zip(seq, seq[1:]):
        L = math.dist(a, b)
        if L and s <= L:
            return (a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L), (a, b), s, L - s
        s -= L
    return None


def place_badges(cells, segments, seg_net, ends, rails):
    labels = [(x, y, x + w, y + h) for c in cells if "ref" in c.attrs for x, y, w, h in [c.geo]]
    housings = []
    for c in cells:
        if c.kind == "edge" and len(c.pts) == 5 and c.style.get("wire") != "1" and c.style.get("dashed") != "1" \
                and math.dist(c.pts[0], c.pts[-1]) < 0.5:
            xs, ys = [p[0] for p in c.pts], [p[1] for p in c.pts]
            if min(max(xs) - min(xs), max(ys) - min(ys)) <= 15:
                housings.append((min(xs), min(ys), max(xs), max(ys)))
    runs = [(i, a, b) for i, s in enumerate(segments) for a, b in zip(s["pts"], s["pts"][1:])]
    placed, out = [], []

    def hits(rect, own):
        bad = 6 * sum(bv.rect_gap(rect, r) < 1.0 for r in placed)
        bad += 3 * sum(bv.rect_gap(rect, r) < 0.5 for r in labels)
        bad += 2 * sum(bv.rect_gap(rect, r) == 0 for r in housings)
        for i, a, b in runs:
            if i != own:
                r = (min(a[0], b[0]) - 3, min(a[1], b[1]) - 3, max(a[0], b[0]) + 3, max(a[1], b[1]) + 3)
                bad += bv.rect_gap(rect, r) == 0
        return bad

    for i, w, p, loose in ends:
        if not loose or seg_net[i] not in rails:
            continue
        rule = rails[seg_net[i]]
        kind, text = rule[2], rule[3]
        bw = len(text) * BADGE_FONT * 0.6 + 4
        best = None
        total = sum(math.dist(a, b) for a, b in zip(segments[i]["pts"], segments[i]["pts"][1:]))
        if w == 1 and total < 2 * bw + 40 and any(o[7] == i for o in out):
            continue  # one badge is enough on a short wire
        s = 8.0 + bw / 2
        while s <= min(total, 320.0):
            at = walk(segments[i]["pts"], w, s)
            if at:
                (x, y), (a, b), before, after = at
                vertical = abs(a[0] - b[0]) < abs(a[1] - b[1])
                hw, hh = (BADGE_H / 2, bw / 2) if vertical else (bw / 2, BADGE_H / 2)
                rect = (x - hw, y - hh, x + hw, y + hh)
                over = max(0.0, bw / 2 - before) + max(0.0, bw / 2 - after)  # sticks out past a bend
                score = hits(rect, i) + over / 4 + s / 200
                if best is None or score < best[0]:
                    best = (score, x, y, vertical, rect)
            s += 4.0
        if best is None:  # wire shorter than the badge: center it
            at = walk(segments[i]["pts"], w, total / 2)
            (x, y), (a, b), _, _ = at
            vertical = abs(a[0] - b[0]) < abs(a[1] - b[1])
            hw, hh = (BADGE_H / 2, bw / 2) if vertical else (bw / 2, BADGE_H / 2)
            best = (0, x, y, vertical, (x - hw, y - hh, x + hw, y + hh))
        if any(bv.rect_gap(best[4], r) == 0 and t == text for r, t in zip(placed, (o[4] for o in out))):
            continue  # the same text is already on this very spot
        placed.append(best[4])
        out.append((best[1], best[2], bw, best[3], text, kind, seg_net[i], i))
    return out


def ground_symbols(cells):
    """Bottom center of each chassis ground symbol (three stacked, shrinking bars)."""
    bars = [(min(a[0], b[0]), max(a[0], b[0]), a[1]) for c in cells
            if c.kind == "edge" and len(c.pts) == 2 and c.style.get("wire") != "1" for a, b in [c.pts]
            if abs(a[1] - b[1]) < 0.3 and 1.5 < abs(a[0] - b[0]) < 22]
    out = []
    for x0, x1, y in bars:
        chain, cur = [(x0, x1, y)], (x0, x1, y)
        while True:
            nxt = [b for b in bars if 1 < b[2] - cur[2] < 6 and abs((b[0] + b[1]) - (cur[0] + cur[1])) < 1.5
                   and b[1] - b[0] < cur[1] - cur[0] - 0.8]
            if not nxt:
                break
            cur = min(nxt, key=lambda b: b[2])
            chain.append(cur)
        longer = [b for b in bars if 1 < y - b[2] < 6 and abs((b[0] + b[1]) - (x0 + x1)) < 1.5
                  and b[1] - b[0] > x1 - x0 + 0.8]
        if len(chain) >= 3 and not longer:
            out.append(((x0 + x1) / 2, chain[-1][2]))
    return out


# ---------------------------------------------------------------------------
# Writing back to the .drawio file
# ---------------------------------------------------------------------------

def badge_style(kind, rotated=False):
    fill, font, _ = KINDS[kind]
    return (f"rounded=1;arcSize=40;whiteSpace=nowrap;html=1;fillColor={fill};strokeColor=#FFFFFF;strokeWidth=0.6;"
            f"fontColor={font};fontSize={bv.f1(BADGE_FONT)};fontStyle=1;fontFamily=Helvetica;spacing=0;"
            "align=center;verticalAlign=middle;" + ("rotation=-90;" if rotated else ""))


def annotate(path):
    cells, _, _, _ = bv.load_drawio(path)
    cells = [c for c in cells if not c.attrs.get("badge")]
    segments, seg_net, ends, rails = classify(cells)
    badges = place_badges(cells, segments, seg_net, ends, rails)
    net_of_cell = {cid: seg_net[i] for i, s in enumerate(segments) for cid in s["cells"]}

    tree = ET.parse(path)
    model = tree.getroot().find("diagram").find("mxGraphModel")
    root = model.find("root")
    for el in [e for e in root if e.get("badge") or e.get("id") == LAYER]:  # re-run safe
        root.remove(el)
    ET.SubElement(root, "mxCell", id=LAYER, value="Power / ground", parent="0")

    for el in root:
        for k in ("power", "voltage", "powerNote"):
            el.attrib.pop(k, None)
        if el.get("wireColor"):
            el.set("tooltip", f"Wire {el.get('wireColor')}")
            rule = rails.get(net_of_cell.get(el.get("id")))
            if rule:
                el.set("power", rule[2])
                el.set("voltage", rule[3])
                el.set("powerNote", rule[4])
                el.set("tooltip", f"Wire {el.get('wireColor')}: {rule[3]} - {rule[4]}")

    def add(cid, text, style, x, y, w, h, **attrs):
        obj = ET.SubElement(root, "UserObject", {"label": text, "id": cid, "badge": "1", **attrs})
        cell = ET.SubElement(obj, "mxCell", style=style, vertex="1", parent=LAYER)
        ET.SubElement(cell, "mxGeometry", {"x": bv.f1(x), "y": bv.f1(y), "width": bv.f1(w), "height": bv.f1(h),
                                           "as": "geometry"})

    for k, (x, y, bw, vertical, text, kind, net, _) in enumerate(badges):
        add(f"pw{k}", text, badge_style(kind, vertical), x - bw / 2, y - BADGE_H / 2, bw, BADGE_H,
            power=kind, tooltip=f"{text} - {rails[net][4]}")

    for k, (x, y) in enumerate(ground_symbols(cells)):
        add(f"pws{k}", "GND", badge_style("GND"), x - 9, y + 2, 18, BADGE_H, power="GND", symbol="1",
            tooltip="GND - grounded through the component body / frame")

    # key below the diagram
    used = {r[2] for r in rails.values()}
    content = max(max(p[1] for p in c.pts) if c.kind == "edge" else c.geo[1] + c.geo[3] for c in cells)
    y0, x0, per_row = content + 24, 40, 3
    shown = [k for k in KINDS if k in used]
    add("pwk_t", "Power / ground key", "text;html=1;align=left;verticalAlign=middle;fontSize=14;fontStyle=1;spacing=0;",
        x0, y0, 300, 20, key="1")
    sample = {}
    for r in RULES:
        sample.setdefault(r[2], r[3] if r[2] != "SW" else "+12V SW")
    sample["SIG"] = "x.xx V"
    for n, kind in enumerate(shown):
        x, y = x0 + (n % per_row) * 640, y0 + 28 + (n // per_row) * 22
        add(f"pwk{n}", sample[kind], badge_style(kind).replace(f"fontSize={bv.f1(BADGE_FONT)}", "fontSize=10"),
            x, y, 70, 14, key="1", power=kind)
        add(f"pwk{n}d", KINDS[kind][2], "text;html=1;align=left;verticalAlign=middle;fontSize=11;spacing=0;",
            x + 78, y - 2, 550, 18, key="1")
    height = y0 + 28 + math.ceil(len(shown) / per_row) * 22 + 30
    model.set("pageHeight", str(int(max(float(model.get("pageHeight")), height))))
    tree.write(path, encoding="utf-8")

    unmarked = sorted({segments[i]["color"] + f" (circuit {n + 1})" for i, n in enumerate(seg_net) if n not in rails})
    count = defaultdict(int)
    for n, r in rails.items():
        count[r[2]] += 1
    print("power: " + ", ".join(f"{k} {count[k]}" for k in KINDS if count[k])
          + f" circuits; {len(set(seg_net)) - len(rails)} unmarked (control/data); {len(badges)} badges", file=sys.stderr)
    return unmarked


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python power_rails.py FILE.drawio")
    annotate(sys.argv[1])
