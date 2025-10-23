import sys
import math
from dataclasses import dataclass
from typing import Sequence, Tuple, List

from KicadModTree import Footprint, Line, Arc, Pad, Text, KicadFileHandler

Vec2D = tuple[float, float]


class Cable:
    """Cross-section spec (edge_gap, trace, gap, trace, ... , edge_gap)"""

    def __init__(self, *specs: float):
        if (
            not specs
            or len(specs) % 2 == 0
            or not all(isinstance(s, (int, float)) and s > 0 for s in specs)
        ):
            raise ValueError(
                "Cable requires odd number of positive specs (edge, trace, gap, ..., edge)."
            )
        self.specs = tuple(float(s) for s in specs)
        self.trace_count = (len(self.specs) - 1) // 2

    def __repr__(self):
        return f"Cable{self.specs}"

    @property
    def traces(self) -> Tuple[float, ...]:
        return self.specs[1::2]

    @property
    def gaps(self) -> Tuple[float, ...]:
        return self.specs[0::2]

    @property
    def total_width(self) -> float:
        return sum(self.specs)

    def trace_details(self) -> Tuple[List[float], List[float]]:
        """Return (offsets, widths) for traces relative to centerline."""
        total, cur = self.total_width, -self.total_width / 2.0
        cur += self.gaps[0]
        offsets, widths = [], []
        for i, w in enumerate(self.traces):
            offsets.append(cur + w / 2.0)
            widths.append(w)
            cur += w
            if i < self.trace_count:
                cur += self.gaps[i + 1]
        return offsets, widths

    def edge_details(self) -> Tuple[List[float], float]:
        """Return two edge offsets and an Edge.Cuts width."""
        return [-self.total_width / 2.0, self.total_width / 2.0], 0.1

    def arc_primitives(
        self,
        center_start: Vec2D,
        radius: float,
        from_dir: Vec2D,
        to_dir: Vec2D,
        offsets,
        widths,
        layer: str,
    ):
        """
        Unified arc geometry for both Edge.Cuts and F.Cu traces.
        offsets: list of offsets (floats)
        widths: either single float (for edges) or list of floats (for traces)
        returns (primitives_list, center_end)
        """
        # cw detection via 2D cross product
        cross = from_dir[0] * to_dir[1] - from_dir[1] * to_dir[0]
        is_cw = cross > 0
        perp_from = get_perp_vec(from_dir)

        # compute center-line arc_center and end like original logic (kept compact)
        if is_cw:
            arc_center = vec_add(
                center_start, (to_dir[0] * radius, from_dir[0] * radius)
            )
            center_end = vec_add(arc_center, (to_dir[1] * radius, -to_dir[0] * radius))
        else:
            arc_center = vec_add(
                center_start, (from_dir[1] * radius, to_dir[1] * radius)
            )
            center_end = vec_add(arc_center, (-to_dir[1] * radius, to_dir[0] * radius))

        def _arc_for_offset(off, w):
            start_pos = vec_add(center_start, vec_scale(perp_from, off))
            if is_cw:
                r_trace = radius - off
                arc_c = vec_add(start_pos, (to_dir[0] * r_trace, from_dir[0] * r_trace))
                end_pos = vec_add(arc_c, (to_dir[1] * r_trace, -to_dir[0] * r_trace))
            else:
                r_trace = radius + off
                arc_c = vec_add(start_pos, (from_dir[1] * r_trace, to_dir[1] * r_trace))
                end_pos = vec_add(arc_c, (-to_dir[1] * r_trace, to_dir[0] * r_trace))
            return Arc(center=arc_c, start=start_pos, end=end_pos, width=w, layer=layer)

        # widths may be scalar or list
        if isinstance(widths, (int, float)):
            primitives = [_arc_for_offset(off, widths) for off in offsets]
        else:
            primitives = [
                _arc_for_offset(off, widths[i]) for i, off in enumerate(offsets)
            ]

        return primitives, center_end


@dataclass(frozen=True)
class Segment:
    length: float
    vec: tuple[float, float]


@dataclass(frozen=True)
class Curve:
    r: float

def Right(length: float):
    return Segment(length, (1.0, 0.0))

def Left(length: float):
    return Segment(length, (-1.0, 0.0))

def Down(length: float):
    return Segment(length, (0.0, 1.0))

def Up(length: float):
    return Segment(length, (0.0, -1.0))

def Angle(length: float, angle: float):
    rad = math.radians(angle)
    return Segment(length, (math.cos(rad), math.sin(rad)))


def get_perp_vec(direction: Vec2D) -> Vec2D:
    return (-direction[1], direction[0])


def vec_add(a: Vec2D, b: Vec2D) -> Vec2D:
    return (a[0] + b[0], a[1] + b[1])


def vec_scale(v: Vec2D, s: float) -> Vec2D:
    return (v[0] * s, v[1] * s)


def generate_cable(filename: str, cable_top: Cable | None, cable_bot: Cable | None, sections: Sequence):
    print(f"--- Generating KiCad Mod --- Target: {filename}")
    try:
        fp_name = filename.split("/")[-1].split("\\")[-1].replace(".kicad_mod", "")
    except Exception:
        fp_name = "fpc_cable"

    fp = Footprint(fp_name)
    fp.setDescription(
        f"Generated dual-layer FPC Cable: top={cable_top}, bottom={cable_bot} with {len(sections)} sections"
    )
    fp.setTags("fpc flexible cable generated dual-layer")

    if cable_top is None and cable_bot is None:
        raise ValueError('At least one of cable_top and cable_bot must be specified')
    cable_top = cable_top or Cable(cable_bot.total_width)
    cable_bot = cable_bot or Cable(cable_top.total_width)

    if abs(cable_top.total_width - cable_bot.total_width) > 1e-3:
        raise ValueError('Top and bottom cables must have equal widths')

    # cross-section details
    trace_offsets_top, trace_widths_top = cable_top.trace_details()
    trace_offsets_bot, trace_widths_bot = cable_bot.trace_details()
    edge_offsets, edge_width = cable_top.edge_details()

    trace_prims_top: List[List] = [[] for _ in range(cable_top.trace_count)]
    trace_prims_bot: List[List] = [[] for _ in range(cable_bot.trace_count)]

    center_pos: Vec2D = (0.0, 0.0)
    current_dir = None

    i = 0
    while i < len(sections):
        seg = sections[i]
        if isinstance(seg, Segment):
            d = seg.vec
            length = seg.length
            center_end = vec_add(center_pos, vec_scale(d, length))
            perp = get_perp_vec(d)

            # shared Edge.Cuts lines
            for off in edge_offsets:
                s = vec_add(center_pos, vec_scale(perp, off))
                e = vec_add(center_end, vec_scale(perp, off))
                fp.append(Line(start=s, end=e, width=edge_width, layer="Edge.Cuts"))

            # top traces
            for idx in range(len(trace_prims_top)):
                off, w = trace_offsets_top[idx], trace_widths_top[idx]
                s = vec_add(center_pos, vec_scale(perp, off))
                e = vec_add(center_end, vec_scale(perp, off))
                trace_prims_top[idx].append(Line(start=s, end=e, width=w))

            # bottom traces
            for idx in range(len(trace_prims_bot)):
                off, w = trace_offsets_bot[idx], trace_widths_bot[idx]
                s = vec_add(center_pos, vec_scale(perp, off))
                e = vec_add(center_end, vec_scale(perp, off))
                trace_prims_bot[idx].append(Line(start=s, end=e, width=w))

            center_pos, current_dir = center_end, d
            i += 1

        elif isinstance(seg, Curve):
            if current_dir is None or i + 1 >= len(sections):
                raise ValueError("Arc cannot be first or last segment.")
            next_seg = sections[i + 1]
            if not isinstance(next_seg, Segment):
                raise ValueError("Arc must be followed by a straight segment.")
            next_dir = next_seg.vec
            if next_dir == current_dir:
                raise ValueError("Arc connecting parallel segments is unsupported.")

            # shared edge arcs
            edge_arcs, center_end = cable_top.arc_primitives(
                center_pos, seg.r, current_dir, next_dir, edge_offsets, edge_width, "Edge.Cuts"
            )
            for a in edge_arcs:
                fp.append(a)

            # top traces
            trace_arcs_top, _ = cable_top.arc_primitives(
                center_pos, seg.r, current_dir, next_dir, trace_offsets_top, trace_widths_top, "F.Cu"
            )
            for idx, a in enumerate(trace_arcs_top):
                trace_prims_top[idx].append(a)

            # bottom traces
            trace_arcs_bot, _ = cable_bot.arc_primitives(
                center_pos, seg.r, current_dir, next_dir, trace_offsets_bot, trace_widths_bot, "B.Cu"
            )
            for idx, a in enumerate(trace_arcs_bot):
                trace_prims_bot[idx].append(a)

            center_pos, current_dir = center_end, next_dir
            i += 1

        else:
            raise TypeError(f"Unknown segment type: {seg}")

    # assemble top pads
    for idx, prims in enumerate(trace_prims_top):
        if not prims:
            continue
        start_pos = (prims[0].start_pos[0], prims[0].start_pos[1]) # make a copy
        trace_w = trace_widths_top[idx]
        for p in prims:
            p.translate((-start_pos[0], -start_pos[1]))
        pad = Pad(
            number=idx + 1,
            type=Pad.TYPE_SMT,
            shape=Pad.SHAPE_CUSTOM,
            at=start_pos,
            size=[trace_w, trace_w],
            layers=["F.Cu"],
            primitives=prims,
        )
        fp.append(pad)

    # assemble bottom pads
    for idx, prims in enumerate(trace_prims_bot):
        if not prims:
            continue
        start_pos = (prims[0].start_pos[0], prims[0].start_pos[1]) # make a copy
        trace_w = trace_widths_bot[idx]
        for p in prims:
            p.translate((-start_pos[0], -start_pos[1]))
        pad = Pad(
            number=idx + 1 + len(trace_prims_top),
            type=Pad.TYPE_SMT,
            shape=Pad.SHAPE_CUSTOM,
            at=start_pos,
            size=[trace_w, trace_w],
            layers=["B.Cu"],
            primitives=prims,
        )
        fp.append(pad)

    # texts
    fp.append(
        Text(
            type="reference",
            text="REF**",
            at=[0, -max(cable_top.total_width, cable_bot.total_width)],
            layer="F.SilkS",
            effects={"font": {"size": [1, 1], "thickness": 0.15}},
        )
    )
    fp.append(
        Text(
            type="value",
            text=fp_name,
            at=[0, max(cable_top.total_width, cable_bot.total_width) + 1],
            layer="F.Fab",
            effects={"font": {"size": [1, 1], "thickness": 0.15}},
        )
    )

    try:
        KicadFileHandler(fp).writeFile(filename)
        print(f"--- Done: '{filename}' ---")
    except (IOError, PermissionError) as e:
        print(f"Error saving file '{filename}': {e}", file=sys.stderr)
