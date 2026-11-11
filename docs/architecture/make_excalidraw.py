"""Generate docs/architecture/nightshift.excalidraw: the architecture diagram, in Excalidraw's own format.

Open it at https://excalidraw.com (Menu → Open, or drag the file onto the canvas) to view or edit it.
    python docs/architecture/make_excalidraw.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).with_name("nightshift.excalidraw")
INK, GREY, RED, BLUE, GREEN = "#1e1e1e", "#495057", "#e03131", "#1971c2", "#2f9e44"
FONT = 5  # Excalifont
rnd = random.Random(7)
elements: list[dict] = []


def _base(kind: str, x: float, y: float, w: float, h: float, **kw) -> dict:
    e = {
        "id": f"{kind[:2]}{len(elements):03d}{rnd.randrange(16**6):06x}",
        "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": kw.pop("stroke", INK), "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": kw.pop("sw", 1), "strokeStyle": kw.pop("style", "solid"), "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None, "seed": rnd.randrange(2**31),
        "version": 1, "versionNonce": rnd.randrange(2**31), "isDeleted": False, "boundElements": None,
        "updated": 1794300000000, "link": None, "locked": False,
    }  # fmt: skip
    e.update(kw)
    elements.append(e)
    return e


def _text_size(text: str, size: float) -> tuple[float, float]:
    lines = text.split("\n")
    return max(len(ln) for ln in lines) * size * 0.56, len(lines) * size * 1.25


def text(x: float, y: float, s: str, size: float = 16, color: str = INK, align: str = "left") -> dict:
    w, h = _text_size(s, size)
    return _base("text", x, y, w, h, stroke=color, text=s, originalText=s, fontSize=size, fontFamily=FONT,
                 textAlign=align, verticalAlign="top", containerId=None, autoResize=True, lineHeight=1.25)  # fmt: skip


def box(x, y, w, h, s, size=16, color=INK, style="solid", sw=1, text_color=None) -> dict:
    r = _base("rectangle", x, y, w, h, stroke=color, style=style, sw=sw)
    tw, th = _text_size(s, size)
    t = _base("text", x + (w - tw) / 2, y + (h - th) / 2, tw, th, stroke=text_color or color, text=s, originalText=s,
              fontSize=size, fontFamily=FONT, textAlign="center", verticalAlign="middle", containerId=r["id"],
              autoResize=True, lineHeight=1.25)  # fmt: skip
    r["boundElements"] = [{"id": t["id"], "type": "text"}]
    return r


def frame(x, y, w, h, title, size=17) -> dict:
    r = _base("rectangle", x, y, w, h, sw=1)
    tw, _ = _text_size(title, size)
    text(x + (w - tw) / 2, y + 12, title, size)
    return r


def arrow(x1, y1, x2, y2, color=INK, style="solid", label: str | None = None, lx=8, ly=-4) -> dict:
    a = _base("arrow", x1, y1, x2 - x1, y2 - y1, stroke=color, style=style, sw=1,
              points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None, startBinding=None, endBinding=None,
              startArrowhead=None, endArrowhead="arrow", elbowed=False)  # fmt: skip
    if label:
        text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, 13, GREY)
    return a


# --- title ---------------------------------------------------------------------------------------------------
text(80, 100, "NIGHTSHIFT\nARCHITECTURE", 44)
text(82, 216, "the agent proposes · the rails act · you sleep", 16, GREY)

# --- inputs --------------------------------------------------------------------------------------------------
frame(480, 80, 760, 160, "Inputs")
box(500, 118, 230, 104, "Training run\nyour train.py · any trainer\nstdout · JSONL · CSV\nTensorBoard events", 14)
box(745, 118, 230, 104, "Playbook you approve\nlr ×0.5 · batch ÷2\nbudget 3 · cooldown 10 min\nhumans own it", 14)
box(990, 118, 230, 104, "Checkpoints\nckpt_*.pt · checkpoint-*\nthe rollback points\non your disk", 14)
arrow(860, 240, 860, 300, label="command + playbook · per run")

# --- agent layer ---------------------------------------------------------------------------------------------
frame(80, 300, 1160, 270, "Agent · Hermes + your model · 6 tools + a skill · no tool can approve, stop or restart a run")
box(100, 340, 260, 104, "Propose\nnightshift_launch\nreads train.py · writes the\ncommand with {lr} {resume}", 14)
box(380, 340, 260, 104, "Approval card\nyou approve the exact command\n/nightshift approve\nor the Approve button", 14)
box(660, 340, 260, 104, "Diagnose\nctx.llm · critical only\ncapped 3 per run\nafter the run is safe", 14)
box(940, 340, 280, 104, "Explain\nnightshift_run · report\n\"why did it diverge?\"\npre_llm_call: once per incident", 14)
for a, b in [(360, 380), (640, 660), (920, 940)]:
    arrow(a + 2, 392, b - 2, 392)
box(100, 470, 260, 84, "Skill + tools as code\nbundled SKILL.md\nplugin.yaml = registrations", 14, style="dashed")
box(380, 470, 840, 84, "Fail-safe\nno approval → nothing runs · unattended contexts can't approve\nthe agent only reads and explains · the supervisor owns every signal", 14)
box(1300, 345, 230, 90, "Your model\nNous Portal · via ctx.llm\naudit-logged by Hermes", 14)
arrow(1222, 392, 1298, 392, style="dashed")
text(1226, 400, "diagnose,\nnever act", 12, GREY)

arrow(640, 570, 640, 632, label="approved run · its own process group")

# --- rails ---------------------------------------------------------------------------------------------------
frame(80, 632, 1160, 280, "Rails · Supervisor thread · deterministic Python · zero runtime deps · no LLM while healthy")
box(100, 672, 260, 104, "Watch\nevery new point, every poll\nlog lines · JSONL · CSV\nTFRecord (own reader)", 14)
box(380, 672, 260, 104, "Detect\nNaN · spike · divergence\nOOM · crash · stall\noverfit · plateau", 14)
box(660, 672, 260, 104, "Decide\nplaybook → action\nbudget · cooldown\nonset refinement", 14)
box(940, 672, 280, 104, "Act\nroll back to a pre-onset checkpoint\nlr ×0.5 · or batch ÷2, accum ×2\nsignals only its own group", 14)
for a, b in [(360, 380), (640, 660), (920, 940)]:
    arrow(a + 2, 724, b - 2, 724)
box(100, 800, 540, 84, "remediation: the restored checkpoint always predates the anomaly\nspike at 610 → restore step 600 → lr 0.003 → 0.0015", 14, style="dashed")
box(660, 800, 560, 84, "hard caps: 3 interventions · 10-min cooldown · lr floor 1e-7\nrelapse or no checkpoint → STOP + page you", 14, color=RED, style="dashed")
box(1300, 700, 230, 120, "SQLite ledger\nplugin-data · WAL\nruns · metrics · incidents\nactions · leases:\none supervisor per run", 14)
arrow(1298, 760, 1222, 760)
text(1236, 768, "state in,\nhistory out", 12, GREY)

arrow(640, 912, 640, 972)

# --- outcomes ------------------------------------------------------------------------------------------------
box(440, 972, 400, 64, "healthy · recovered · stopped", 24, sw=2)
box(80, 980, 340, 70, "Proven: 4 demo rounds in real Hermes\n440+ tests · clean-curve sweep\nblow-up caught at onset · rolled back", 13, style="dashed")
box(860, 980, 380, 70, "No telemetry · no stored keys\nalerts via Hermes's own `hermes send`\nPID-reuse guard before every signal", 13, style="dashed")
arrow(560, 1036, 260, 1112, label="every critical incident", lx=-190, ly=2)
arrow(640, 1036, 640, 1112, label="every run", lx=6, ly=-4)
arrow(720, 1036, 1010, 1112, label="every morning", lx=40, ly=2)

# --- outputs -------------------------------------------------------------------------------------------------
box(80, 1112, 380, 100, "Phone alert · Telegram / Discord\nchart PNG + the model's root cause\ncritical only · silent while healthy", 14)
box(480, 1112, 340, 100, "Desktop page + pane\nincidents board · loss chart\nrun log · approve · status chip", 14)
box(840, 1112, 400, 100, "Morning report · blueprint\n\"while you slept: 2 incidents,\n2 recovered automatically\"", 14)
box(1300, 1120, 230, 84, "You\nasleep · phone on\napprove · ack · stop", 14)
arrow(1298, 1162, 1242, 1162)

# --- slogan + footer -----------------------------------------------------------------------------------------
text(100, 1260, "SILENT WHILE HEALTHY\nAGENTS PROPOSE\nRAILS ACT\nEVERY RUN COMES BACK", 30)
text(820, 1286, "launch → watch → detect → decide → act → alert\nNaN · spike · divergence · OOM · stall · overfit\nHermes Agent v0.23 · zero runtime dependencies\ngithub.com/Pavilion-devs/nightshift", 13, GREY)

doc = {
    "type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": elements,
    "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"}, "files": {},
}  # fmt: skip
OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print(f"wrote {OUT} ({len(elements)} elements)")
