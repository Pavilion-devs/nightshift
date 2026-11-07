"""Scalars from TensorBoard event files, with no tensorboard or protobuf dependency.

An events file is a sequence of TFRecords: uint64 length | uint32 masked_crc32c(length) | data |
uint32 masked_crc32c(data). ``data`` is a serialized ``tensorflow.Event``; we decode only what scalars need:

  Event:        wall_time = 1 (double), step = 2 (int64), summary = 5 (Summary)
  Summary:      value = 1 (repeated Value)
  Value:        tag = 1 (string), simple_value = 2 (float), tensor = 8 (TensorProto)
  TensorProto:  dtype = 1, tensor_content = 4 (bytes), float_val = 5, double_val = 6, int64_val = 10

``simple_value`` covers torch.utils.tensorboard / tensorboardX ``add_scalar``; ``tensor`` covers TF2
``tf.summary``.
A record that is still being written (short read) is left for the next poll; a corrupt record is skipped.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from pathlib import Path

from ..models import MetricPoint, normalize_name

# --- crc32c (Castagnoli), table driven ---------------------------------------------------------------------

_CRC_TABLE: list[int] = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --- minimal protobuf wire decoding -----------------------------------------------------------------------


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for a serialized message."""
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = _varint(buf, pos)
        elif wt == 1:
            if pos + 8 > n:
                raise ValueError("truncated fixed64")
            val, pos = buf[pos : pos + 8], pos + 8
        elif wt == 2:
            length, pos = _varint(buf, pos)
            if pos + length > n:
                raise ValueError("truncated bytes")
            val, pos = buf[pos : pos + length], pos + length
        elif wt == 5:
            if pos + 4 > n:
                raise ValueError("truncated fixed32")
            val, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, val


def _tensor_scalar(buf: bytes) -> float | None:
    dtype = 0
    content = b""
    for num, wt, val in _fields(buf):
        if num == 1 and wt == 0:
            dtype = int(val)  # type: ignore[arg-type]
        elif num == 4 and wt == 2:
            content = val  # type: ignore[assignment]
        elif num == 5:  # float_val: packed (wt 2) or a single fixed32 (wt 5); both start with the value
            if len(val) >= 4:  # type: ignore[arg-type]
                return struct.unpack("<f", val[:4])[0]  # type: ignore[index]
        elif num == 6:
            if len(val) >= 8:  # type: ignore[arg-type]
                return struct.unpack("<d", val[:8])[0]  # type: ignore[index]
        elif num == 10 and wt == 0:
            return float(val)  # type: ignore[arg-type]
    if content:
        if dtype == 1 and len(content) >= 4:  # DT_FLOAT
            return struct.unpack("<f", content[:4])[0]
        if dtype == 2 and len(content) >= 8:  # DT_DOUBLE
            return struct.unpack("<d", content[:8])[0]
        if dtype == 9 and len(content) >= 8:  # DT_INT64
            return float(struct.unpack("<q", content[:8])[0])
    return None


def decode_event(data: bytes) -> list[tuple[float, int, str, float]]:
    """Return [(wall_time, step, tag, value)] for every scalar in one serialized Event."""
    wall, step = 0.0, 0
    summaries: list[bytes] = []
    for num, wt, val in _fields(data):
        if num == 1 and wt == 1:
            wall = struct.unpack("<d", val)[0]  # type: ignore[arg-type]
        elif num == 2 and wt == 0:
            step = int(val)  # type: ignore[arg-type]
            if step >= 1 << 63:
                step -= 1 << 64
        elif num == 5 and wt == 2:
            summaries.append(val)  # type: ignore[arg-type]
    out: list[tuple[float, int, str, float]] = []
    for summary in summaries:
        for snum, swt, sval in _fields(summary):
            if snum != 1 or swt != 2:
                continue
            tag, value = "", None
            for vnum, vwt, vval in _fields(sval):  # type: ignore[arg-type]
                if vnum == 1 and vwt == 2:
                    tag = vval.decode("utf-8", errors="replace")  # type: ignore[union-attr]
                elif vnum == 2 and vwt == 5:
                    value = struct.unpack("<f", vval)[0]  # type: ignore[arg-type]
                elif vnum == 8 and vwt == 2 and value is None:
                    value = _tensor_scalar(vval)  # type: ignore[arg-type]
            if tag and value is not None:
                out.append((wall, step, tag, float(value)))
    return out


# --- record framing ---------------------------------------------------------------------------------------


class _EventFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.corrupt = 0

    def read(self) -> list[tuple[float, int, str, float]]:
        out: list[tuple[float, int, str, float]] = []
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                buf = fh.read()
        except FileNotFoundError:
            return out
        pos = 0
        while pos + 12 <= len(buf):
            header = buf[pos : pos + 8]
            (length,) = struct.unpack("<Q", header)
            (len_crc,) = struct.unpack("<I", buf[pos + 8 : pos + 12])
            if masked_crc(header) != len_crc:
                # Not a record boundary: the file is corrupt from here; stop rather than misread garbage.
                self.corrupt += 1
                break
            end = pos + 12 + length + 4
            if end > len(buf):
                break  # record still being written
            data = buf[pos + 12 : pos + 12 + length]
            (data_crc,) = struct.unpack("<I", buf[pos + 12 + length : end])
            if masked_crc(data) == data_crc:
                try:
                    out.extend(decode_event(data))
                except ValueError:
                    self.corrupt += 1
            else:
                self.corrupt += 1
            pos = end
        self.offset += pos
        return out


class TfEventsSource:
    """Follow every ``events.out.tfevents.*`` file under a log dir (or one file)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.root = Path(path)
        self._files: dict[Path, _EventFile] = {}

    def _discover(self) -> None:
        if self.root.is_file():
            candidates = [self.root]
        elif self.root.is_dir():
            candidates = sorted(self.root.rglob("events.out.tfevents.*"))
        else:
            candidates = []
        for p in candidates:
            if p not in self._files:
                self._files[p] = _EventFile(p)

    def poll(self) -> list[MetricPoint]:
        self._discover()
        points: list[MetricPoint] = []
        for ef in self._files.values():
            for wall, step, tag, value in ef.read():
                points.append(MetricPoint(step, normalize_name(tag), value, wall))
        points.sort(key=lambda p: (p.step, p.wall))
        return points


# --- writer, used by the synthetic trainer and tests (format-compatible with TensorBoard) -----------------


def _enc_varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _enc_field(num: int, wt: int, payload: bytes) -> bytes:
    key = _enc_varint((num << 3) | wt)
    if wt == 2:
        return key + _enc_varint(len(payload)) + payload
    return key + payload


def encode_scalar_event(wall: float, step: int, tag: str, value: float) -> bytes:
    val = _enc_field(1, 2, tag.encode()) + _enc_field(2, 5, struct.pack("<f", value))
    summary = _enc_field(1, 2, val)
    return (
        _enc_field(1, 1, struct.pack("<d", wall))
        + _enc_field(2, 0, _enc_varint(step))
        + _enc_field(5, 2, summary)
    )


def frame_record(data: bytes) -> bytes:
    header = struct.pack("<Q", len(data))
    return header + struct.pack("<I", masked_crc(header)) + data + struct.pack("<I", masked_crc(data))
