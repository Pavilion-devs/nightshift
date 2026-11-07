from __future__ import annotations

import math
import os
import shutil
import struct
from pathlib import Path

import pytest

from nightshift.core.clock import FakeClock
from nightshift.core.sources import (
    CsvSource,
    FileFollower,
    JsonlSource,
    LogTailSource,
    TfEventsSource,
    make_source,
    parse_line,
    parse_num,
)
from nightshift.core.sources.tfevents import (
    crc32c,
    decode_event,
    encode_scalar_event,
    frame_record,
    masked_crc,
)

FIX = Path(__file__).parent / "fixtures" / "tfevents"


# --- line parsing -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "step", "metrics"),
    [
        ("iter 100: loss 2.3456, time 12.34ms, mfu 3.21%", 100, {"loss": 2.3456}),
        ("step 200: train loss 2.1000, val loss 2.2000", 200, {"train_loss": 2.1, "val_loss": 2.2}),
        (
            "{'loss': 2.31, 'grad_norm': 1.2, 'learning_rate': 5e-05, 'epoch': 0.1}",
            None,
            {"loss": 2.31, "grad_norm": 1.2, "learning_rate": 5e-05, "epoch": 0.1},
        ),
        ('{"step": 10, "loss": 1.5, "tag": "x"}', 10, {"loss": 1.5}),
        ("Epoch 1 | step: 10 | loss: 2.31 | lr: 3.0e-04", 10, {"loss": 2.31, "lr": 3e-4}),
        ("global_step=7 train/loss=0.5 eval/loss=0.6", 7, {"train/loss": 0.5, "eval/loss": 0.6}),
        (
            "Epoch 0:  45%|####      | 450/1000 [00:12<00:15, 36.1it/s, loss=2.31, v_num=0]",
            450,
            {"loss": 2.31},
        ),
        ("Training loss 1.25", None, {"train_loss": 1.25}),
        ("INFO: saving checkpoint to out/ckpt_500.pt", None, {}),
        ("{not valid json or python}", None, {}),
    ],
)
def test_parse_line(line: str, step: int | None, metrics: dict[str, float]) -> None:
    got_step, got = parse_line(line)
    assert got_step == step
    assert got == metrics


def test_parse_line_nan_and_inf() -> None:
    step, m = parse_line("step=10 loss=nan grad_norm=inf")
    assert step == 10 and math.isnan(m["loss"]) and math.isinf(m["grad_norm"])
    _, m = parse_line('{"step": 3, "loss": NaN}')
    assert math.isnan(m["loss"])


def test_custom_pattern() -> None:
    import re

    pat = re.compile(r"it=(?P<step>\d+) obj=(?P<objective>\S+)")
    assert parse_line("it=5 obj=0.25", pat) == (5, {"objective": 0.25})
    assert parse_line("nothing here", pat) == (None, {})


def test_parse_num() -> None:
    assert parse_num("3.5%") == 3.5
    assert parse_num("1e-3") == 1e-3
    assert parse_num("abc") is None
    assert math.isinf(parse_num("-inf") or 0)


# --- following files --------------------------------------------------------------------------------------


def test_follower_holds_partial_lines_and_handles_truncation(tmp_path: Path) -> None:
    f = tmp_path / "log.txt"
    ff = FileFollower(f)
    assert ff.read_lines() == []  # file does not exist yet
    f.write_text("a\nb")
    assert ff.read_lines() == ["a"]
    with f.open("a") as fh:
        fh.write("c\nprogress 1\rprogress 2\n")
    assert ff.read_lines() == ["bc", "progress 1", "progress 2"]
    f.write_text("x\n")  # truncated + rewritten
    assert ff.read_lines() == ["x"]
    assert ff.read_lines() == []


def test_follower_handles_rotation(tmp_path: Path) -> None:
    f = tmp_path / "log.txt"
    f.write_text("one\n")
    ff = FileFollower(f)
    assert ff.read_lines() == ["one"]
    os.rename(f, tmp_path / "log.txt.1")
    f.write_text("fresh line that is longer\n")
    assert ff.read_lines() == ["fresh line that is longer"]


def test_log_tail_source_steps_and_tail(tmp_path: Path) -> None:
    f = tmp_path / "train.log"
    clock = FakeClock()
    src = LogTailSource(f, clock=clock)
    f.write_text("{'loss': 3.0}\n{'loss': 2.9}\nstep 50: loss 2.5\n{'loss': 2.4}\nno metrics here\n")
    pts = src.poll()
    assert [(p.step, p.value) for p in pts] == [(1, 3.0), (2, 2.9), (50, 2.5), (50, 2.4)]
    assert all(p.wall == clock.now() for p in pts)
    assert src.tail[-1] == "no metrics here" and src.lines_seen == 5


def test_log_tail_keeps_bounded_tail(tmp_path: Path) -> None:
    f = tmp_path / "train.log"
    f.write_text("".join(f"line {i}\n" for i in range(500)))
    src = LogTailSource(f)
    src.poll()
    assert len(src.tail) == 200 and src.tail[-1] == "line 499"


def test_jsonl_source(tmp_path: Path) -> None:
    f = tmp_path / "m.jsonl"
    f.write_text(
        '{"step": 1, "loss": 2.0, "time": 1700000000.5}\n[1,2]\nnot json\n'
        '{"loss": 1.9, "note": "x", "ok": true}\n'
    )
    pts = JsonlSource(f, clock=FakeClock()).poll()
    assert [(p.step, p.name, p.value) for p in pts] == [(1, "loss", 2.0), (2, "loss", 1.9)]
    assert pts[0].wall == 1700000000.5


def test_csv_source(tmp_path: Path) -> None:
    f = tmp_path / "metrics.csv"
    f.write_text("step,train_loss,val_loss\n10,2.0,\n20,1.8,1.9\nstep,train_loss,val_loss\n30,1.7,\n")
    pts = CsvSource(f, clock=FakeClock()).poll()
    assert [(p.step, p.name, p.value) for p in pts] == [
        (10, "train_loss", 2.0),
        (20, "train_loss", 1.8),
        (20, "val_loss", 1.9),
        (30, "train_loss", 1.7),
    ]


def test_csv_source_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "metrics.csv"
    f.write_text("")
    assert CsvSource(f).poll() == []


def test_make_source(tmp_path: Path) -> None:
    assert isinstance(make_source({"type": "log", "path": str(tmp_path / "a")}), LogTailSource)
    assert isinstance(make_source({"type": "jsonl", "path": str(tmp_path / "a")}), JsonlSource)
    assert isinstance(make_source({"type": "csv", "path": str(tmp_path / "a")}), CsvSource)
    assert isinstance(make_source({"type": "tfevents", "path": str(tmp_path)}), TfEventsSource)
    with pytest.raises(ValueError):
        make_source({"type": "wandb", "path": "x"})
    with pytest.raises(ValueError):
        make_source({"type": "log"})


# --- TensorBoard ------------------------------------------------------------------------------------------


def test_crc32c_known_vector() -> None:
    assert crc32c(b"123456789") == 0xE3069283  # standard Castagnoli check value


def test_real_tensorboardx_classic_scalars_including_nan() -> None:
    pts = TfEventsSource(FIX / "classic.tfevents").poll()
    train = [p for p in pts if p.name == "train/loss"]
    assert [p.step for p in train] == list(range(0, 101, 10))
    assert train[0].value == pytest.approx(4.0)
    assert math.isnan(train[-1].value)
    assert {p.name for p in pts} == {"train/loss", "val/loss"}
    assert all(p.wall > 1.6e9 for p in pts)


def test_real_tensor_style_scalars() -> None:
    pts = TfEventsSource(FIX / "newstyle.tfevents").poll()
    loss = [(p.step, p.value) for p in pts if p.name == "loss"]
    lr = [(p.step, p.value) for p in pts if p.name == "lr"]
    assert loss == [(i, 1.5 + i) for i in range(5)]  # float_val
    assert lr == [(i, 10.0 + i) for i in range(5)]  # double tensor_content


def test_tfevents_incremental_partial_and_corrupt_records(tmp_path: Path) -> None:
    d = tmp_path / "logs"
    d.mkdir()
    f = d / "events.out.tfevents.1.host"
    rec1 = frame_record(encode_scalar_event(1.7e9, 1, "loss", 2.0))
    rec2 = frame_record(encode_scalar_event(1.7e9, 2, "loss", 1.9))
    f.write_bytes(rec1 + rec2[:10])  # second record still being written
    src = TfEventsSource(d)
    assert [(p.step, p.value) for p in src.poll()] == [(1, 2.0)]
    with f.open("ab") as fh:
        fh.write(rec2[10:])
    assert [(p.step, p.value) for p in src.poll()] == [(2, pytest.approx(1.9))]
    bad = bytearray(frame_record(encode_scalar_event(1.7e9, 3, "loss", 1.8)))
    bad[-6] ^= 0xFF  # corrupt the payload, keep framing intact
    with f.open("ab") as fh:
        fh.write(bytes(bad) + frame_record(encode_scalar_event(1.7e9, 4, "loss", 1.7)))
    assert [p.step for p in src.poll()] == [4]
    assert src._files[f].corrupt == 1


def test_tfevents_bad_length_crc_stops_reading(tmp_path: Path) -> None:
    f = tmp_path / "events.out.tfevents.x"
    f.write_bytes(struct.pack("<Q", 5) + b"\x00\x00\x00\x00" + b"hello" + b"\x00" * 4)
    src = TfEventsSource(f)
    assert src.poll() == []
    assert src._files[f].corrupt == 1


def test_tfevents_missing_path_and_new_files(tmp_path: Path) -> None:
    src = TfEventsSource(tmp_path / "nope")
    assert src.poll() == []
    d = tmp_path / "logs"
    d.mkdir()
    src = TfEventsSource(d)
    assert src.poll() == []
    shutil.copy(FIX / "classic.tfevents", d / "events.out.tfevents.2.host")
    assert len(src.poll()) == 21


def test_decode_event_negative_step_and_non_scalar() -> None:
    data = encode_scalar_event(1.0, -1, "x", 1.0)
    assert decode_event(data)[0][1] == -1
    assert masked_crc(b"") != 0
