"""A line chart as a PNG, with nothing but the standard library (zlib + struct).

Good enough for a phone notification: one or more series, restart segments in different shades, red markers at
incidents, a dashed line at each recovery, and y-axis labels drawn with a tiny built-in bitmap font.
"""

from __future__ import annotations

import math
import struct
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

Color = tuple[int, int, int]

BG: Color = (17, 17, 21)
GRID: Color = (44, 44, 54)
AXIS_TEXT: Color = (150, 150, 165)
SERIES: list[Color] = [(122, 162, 247), (158, 206, 106), (224, 175, 104), (187, 154, 247)]
ATTEMPT_SHADES: list[Color] = [(122, 162, 247), (125, 207, 255), (115, 218, 202), (158, 206, 106)]
INCIDENT: Color = (247, 118, 142)
RECOVERY: Color = (224, 175, 104)

# 3x5 glyphs, rows top to bottom, 3 bits per row
_FONT: dict[str, tuple[int, ...]] = {
    "0": (7, 5, 5, 5, 7), "1": (2, 6, 2, 2, 7), "2": (7, 1, 7, 4, 7), "3": (7, 1, 7, 1, 7),
    "4": (5, 5, 7, 1, 1), "5": (7, 4, 7, 1, 7), "6": (7, 4, 7, 5, 7), "7": (7, 1, 1, 1, 1),
    "8": (7, 5, 7, 5, 7), "9": (7, 5, 7, 1, 7), ".": (0, 0, 0, 0, 2), "-": (0, 0, 7, 0, 0),
    "e": (0, 7, 7, 4, 7), "+": (0, 2, 7, 2, 0), " ": (0, 0, 0, 0, 0), "k": (4, 5, 6, 5, 5),
    "n": (0, 6, 5, 5, 5), "a": (0, 7, 1, 7, 7), "i": (2, 0, 2, 2, 2), "f": (3, 2, 7, 2, 2),
    "s": (0, 7, 6, 1, 7), "t": (2, 7, 2, 2, 3), "p": (0, 7, 5, 7, 4), ":": (0, 2, 0, 2, 0),
}  # fmt: skip


@dataclass
class Series:
    points: list[tuple[float, float]]  # (x, y); y may be NaN/Inf
    color: Color | None = None
    attempts: list[int] = field(default_factory=list)  # same length as points, optional


class Canvas:
    def __init__(self, w: int, h: int, bg: Color = BG) -> None:
        self.w, self.h = w, h
        self.px = bytearray(bytes(bg) * (w * h))

    def set(self, x: int, y: int, c: Color) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i : i + 3] = bytes(c)

    def rect(self, x0: int, y0: int, x1: int, y1: int, c: Color) -> None:
        for y in range(max(0, y0), min(self.h, y1 + 1)):
            for x in range(max(0, x0), min(self.w, x1 + 1)):
                self.set(x, y, c)

    def line(self, x0: float, y0: float, x1: float, y1: float, c: Color, width: int = 2) -> None:
        dx, dy = x1 - x0, y1 - y0
        steps = int(max(abs(dx), abs(dy), 1))
        r = width // 2
        for s in range(steps + 1):
            x = round(x0 + dx * s / steps)
            y = round(y0 + dy * s / steps)
            for ox in range(-r, width - r):
                for oy in range(-r, width - r):
                    self.set(x + ox, y + oy, c)

    def vdash(self, x: int, y0: int, y1: int, c: Color, dash: int = 4) -> None:
        for y in range(y0, y1):
            if (y // dash) % 2 == 0:
                self.set(x, y, c)
                self.set(x + 1, y, c)

    def text(self, x: int, y: int, s: str, c: Color, scale: int = 2) -> None:
        cx = x
        for ch in s.lower():
            glyph = _FONT.get(ch, _FONT[" "])
            for row, bits in enumerate(glyph):
                for col in range(3):
                    if bits & (4 >> col):
                        self.rect(cx + col * scale, y + row * scale, cx + col * scale + scale - 1,
                                  y + row * scale + scale - 1, c)  # fmt: skip
            cx += 4 * scale

    def png(self) -> bytes:
        raw = b"".join(b"\x00" + bytes(self.px[y * self.w * 3 : (y + 1) * self.w * 3]) for y in range(self.h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b"")
        )


def _label(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.1e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:.3g}"


def render(
    series: Sequence[Series],
    *,
    width: int = 800,
    height: int = 400,
    incidents_x: Sequence[float] = (),
    recoveries_x: Sequence[float] = (),
    log_y: bool = False,
    x_is_time: bool = True,
) -> bytes:
    c = Canvas(width, height)
    left, right, top, bottom = 70, width - 16, 16, height - 28
    finite = [(x, y) for s in series for x, y in s.points if math.isfinite(y) and (not log_y or y > 0)]
    xs = [x for s in series for x, _ in s.points] + list(incidents_x) + list(recoveries_x)
    if not xs:
        return c.png()
    x_lo, x_hi = min(xs), max(xs)
    if x_hi == x_lo:
        x_hi = x_lo + 1
    if finite:
        raw = sorted(y for _, y in finite)
        cap = raw[-1]
        if len(raw) > 10 and cap > 3 * raw[int(0.9 * (len(raw) - 1))]:
            cap = (
                raw[int(0.9 * (len(raw) - 1))] * 1.6
            )  # a blow-up runs off the top instead of flattening the rest
        ys = [math.log10(y) if log_y else y for y in raw if y <= cap]
        y_lo, y_hi = min(ys), max(ys)
    else:
        y_lo, y_hi = 0.0, 1.0
    if y_hi == y_lo:
        y_hi = y_lo + 1
    pad = (y_hi - y_lo) * 0.06
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def sx(x: float) -> float:
        return left + (x - x_lo) / (x_hi - x_lo) * (right - left)

    def sy(y: float) -> float:
        yy = math.log10(y) if log_y else y
        return max(top - 2, bottom - (yy - y_lo) / (y_hi - y_lo) * (bottom - top))

    for k in range(5):  # grid + y labels
        gy = top + k * (bottom - top) / 4
        c.line(left, gy, right, gy, GRID, width=1)
        val = y_hi - k * (y_hi - y_lo) / 4
        c.text(4, int(gy) - 5, _label(10**val if log_y else val), AXIS_TEXT)
    for k in range(3):  # x labels: start, middle, end
        xv = x_lo + k * (x_hi - x_lo) / 2
        if x_is_time:
            lab = time.strftime("%H:%M", time.localtime(xv))
        else:
            lab = str(round(xv)) if abs(xv) < 1e7 else _label(xv)  # steps read as plain integers
        lx = left + k * (right - left) / 2 - (len(lab) * 8 if k == 2 else len(lab) * 4 if k == 1 else 0)
        c.text(int(lx), height - 18, lab, AXIS_TEXT)

    for x in recoveries_x:
        c.vdash(int(sx(x)), top, bottom, RECOVERY)

    for idx, s in enumerate(series):
        prev: tuple[float, float] | None = None
        prev_attempt: int | None = None
        for j, (x, y) in enumerate(s.points):
            attempt = s.attempts[j] if j < len(s.attempts) else 0
            color = s.color or (
                ATTEMPT_SHADES[attempt % len(ATTEMPT_SHADES)] if s.attempts else SERIES[idx % 4]
            )
            if not math.isfinite(y) or (log_y and y <= 0):
                c.rect(
                    int(sx(x)) - 3, top, int(sx(x)) + 3, top + 6, INCIDENT
                )  # NaN/Inf: a red tick at the top
                prev = None
                continue
            cur = (sx(x), sy(y))
            if prev is not None and attempt == prev_attempt:
                c.line(prev[0], prev[1], cur[0], cur[1], color)
            prev, prev_attempt = cur, attempt

    for x in incidents_x:
        px = int(sx(x))
        c.line(px, top, px, bottom, INCIDENT, width=2)
    return c.png()
