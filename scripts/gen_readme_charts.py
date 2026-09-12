"""
scripts/gen_readme_charts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate the SVG charts embedded in README.md.

Stdlib only — no matplotlib, no new dependency. Each chart is a theme-aware SVG
(a `prefers-color-scheme` block inside the file), so one asset reads correctly
on GitHub in both light and dark.

Where the numbers come from
---------------------------
Nothing here is hand-typed from the README. Each figure is either

  * **scored live** against the frozen corpora in ``tests/data/`` through the
    same ``RulesClassifier`` path ``scripts/classifier_accuracy.py`` uses
    (corpora A-E),
  * **derived live** from ``scripts/bench_synthetic.py``'s own ``_task_body``
    success logic (the routing demo), or
  * a **historical constant** that cannot be recomputed — corpus C's pre-tuning
    v1.0 score (``rules.py`` has since been tuned against it) and the
    ``LLMClassifier``/``HybridClassifier`` runs (they need a live model). Both
    are marked HISTORICAL below, with the source that recorded them.

So a ``rules.py`` change that moves a held-out number moves these charts on the
next run; they cannot silently drift from the README's tables.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_readme_charts.py
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import Step
from triage.trajectory import Trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
OUT_DIR = REPO_ROOT / "docs" / "assets" / "charts"

SELF_HEALING = ("external_fault", "timeout")
ROUTING_SENSITIVE = ("wrong_tool_called", "schema_mismatch")

# ── palette ───────────────────────────────────────────────────────────────────
# Categorical slots 1 and 2 of the validated default palette, plus the
# de-emphasis gray of the emphasis form (accent for the series carrying the
# story, gray for the context series). The blue/orange pair clears every
# adjacent CVD and contrast gate in both modes; the gray is a neutral text
# token, never a third categorical slot, and every chart using it also ships a
# legend and direct labels so identity never rests on color alone.

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

STYLE = """
  :root {
    --surface: #fcfcfb;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --s1: #2a78d6;
    --s2: #eb6834;
    --demph: #898781;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --s1: #3987e5;
      --s2: #d95926;
      --demph: #898781;
    }
  }
  .bg { fill: var(--surface); }
  text { font-family: FONT_STACK; }
  .title { fill: var(--ink); font-size: 16px; font-weight: 600; }
  .subtitle { fill: var(--ink-2); font-size: 12.5px; }
  .note { fill: var(--muted); font-size: 11px; }
  .axis-label { fill: var(--ink-2); font-size: 12px; }
  .axis-sub { fill: var(--muted); font-size: 10.5px; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .value { fill: var(--ink); font-size: 12px; font-weight: 600; }
  .value-2 { fill: var(--ink-2); font-size: 11px; }
  .in-bar { fill: #ffffff; font-size: 12px; font-weight: 600; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .s1-fill { fill: var(--s1); }
  .s2-fill { fill: var(--s2); }
  .demph-fill { fill: var(--demph); }
  .s1-stroke { stroke: var(--s1); fill: none; stroke-width: 2;
               stroke-linejoin: round; stroke-linecap: round; }
  .demph-stroke { stroke: var(--demph); fill: none; stroke-width: 2;
                  stroke-linejoin: round; stroke-linecap: round; }
  .ring { stroke: var(--surface); stroke-width: 2; }
  .ann { stroke: var(--muted); stroke-width: 1; fill: none; }
""".replace("FONT_STACK", FONT)

W = 880
PAD_L = 60
PAD_R = 26
PLOT_W = W - PAD_L - PAD_R
BAR_MAX = 24  # mark spec: bars never thicker than 24px
GAP = 2  # the surface gap that separates touching marks
RADIUS = 4  # rounded data-end


# ── tiny SVG toolkit ──────────────────────────────────────────────────────────


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Per-character widths as a fraction of font size, calibrated against Chromium's
# getComputedTextLength for the font stack above and deliberately biased to
# over-estimate by ~10%: an over-estimate costs a little air, an under-estimate
# collides two labels.
_W_NARROW = 0.34  # i l j t f r I . , : ; ' | ! ( ) [ ] - ·
_W_WIDE = 1.00  # m w M W @
_W_UPPER = 0.75
_W_DIGIT = 0.66
_W_SPACE = 0.32
_W_DEFAULT = 0.60
_SPECIAL = {"—": 1.05, "–": 0.66, "−": 0.66, "%": 1.00, "×": 0.85, "→": 1.00}
_NARROW = set("iljtfrI.,:;'|!()[]-·")
_WIDE = set("mwMW@")


def text_width(s: str, size: float, bold: bool = False) -> float:
    total = 0.0
    for ch in s:
        if ch in _SPECIAL:
            total += _SPECIAL[ch]
        elif ch in _NARROW:
            total += _W_NARROW
        elif ch in _WIDE:
            total += _W_WIDE
        elif ch == " ":
            total += _W_SPACE
        elif ch.isdigit():
            total += _W_DIGIT
        elif ch.isupper():
            total += _W_UPPER
        else:
            total += _W_DEFAULT
    return total * size * (1.15 if bold else 1.0)


def wrap(s: str, max_w: float, size: float, bold: bool = False) -> list[str]:
    """Greedy wrap, so a long title, subtitle or source note never runs off the canvas."""
    lines: list[str] = []
    current = ""
    for word in s.split(" "):
        candidate = f"{current} {word}".strip()
        if current and text_width(candidate, size, bold) > max_w:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def text(x: float, y: float, s: str, cls: str = "value", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "grid") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def column(x: float, y_base: float, y_top: float, w: float, cls: str) -> str:
    """Vertical bar: square at the baseline, 4px rounded at the data end."""
    h = y_base - y_top
    if h <= 0.5:
        return ""
    r = min(RADIUS, w / 2, h)
    return (
        f'<path class="{cls}" d="M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} '
        f"Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
        f"L{x + w - r:.1f},{y_top:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} "
        f'L{x + w:.1f},{y_base:.1f} Z"/>'
    )


def hbar(x0: float, x1: float, y: float, h: float, cls: str, rounded: bool = True) -> str:
    """Horizontal bar: square at the baseline, rounded only at a true data end."""
    if x1 - x0 <= 0.5:
        return ""
    if not rounded:
        return (
            f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="{h:.1f}"/>'
        )
    r = min(RADIUS, h / 2, x1 - x0)
    return (
        f'<path class="{cls}" d="M{x0:.1f},{y:.1f} L{x1 - r:.1f},{y:.1f} '
        f"Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} L{x1:.1f},{y + h - r:.1f} "
        f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} L{x0:.1f},{y + h:.1f} Z"/>'
    )


def dot(cx: float, cy: float, cls: str, r: float = 4.5) -> str:
    """Marker with the 2px surface ring that keeps it legible over other marks."""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{cls} ring"/>'


@dataclass
class Canvas:
    """Vertical layout: title → subtitle → legend → plot → x-axis band → source.

    Every band is measured rather than guessed, so no chart can grow a label
    into its neighbour.
    """

    title: str
    subtitle: str
    note: str
    legend_items: list[tuple[str, str]]
    legend_kind: str = "swatch"
    plot_h: float = 200
    axis_h: float = 46  # room under the baseline for tick labels and annotations
    head_h: float = 30  # room above the plot for direct labels on the tallest mark
    plot_left: float = PAD_L
    parts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.title_lines = wrap(self.title, PLOT_W, 16, bold=True)
        self.sub_lines = wrap(self.subtitle, PLOT_W, 12.5)
        self.note_lines = wrap(self.note, PLOT_W, 11)
        head = 30 + (len(self.title_lines) - 1) * 20
        self.legend_y = head + len(self.sub_lines) * 17 + 22
        self.px = self.plot_left
        self.pw = W - self.plot_left - PAD_R
        self.py = self.legend_y + 16 + self.head_h
        self.height = self.py + self.plot_h + self.axis_h + len(self.note_lines) * 14 + 16

    @property
    def base(self) -> float:
        """y of the plot baseline."""
        return self.py + self.plot_h

    def add(self, *svg: str) -> None:
        self.parts.extend(svg)

    def y_of(self, pct: float) -> float:
        return self.base - (pct / 100) * self.plot_h

    def check_gutter(self, label: str, size: float) -> str:
        """Fail loudly if a row label would run into the plot instead of clipping."""
        avail = self.px - PAD_L - 12
        if text_width(label, size) > avail:
            raise ValueError(
                f"row label {label!r} needs {text_width(label, size):.0f}px, "
                f"gutter holds {avail:.0f}px — shorten it or widen plot_left"
            )
        return label

    def pct_axis(self) -> None:
        """0-100% scale: hairline solid gridlines, ticks on clean numbers."""
        for v in (0, 25, 50, 75, 100):
            y = self.y_of(v)
            self.add(line(self.px, y, self.px + self.pw, y, "axis" if v == 0 else "grid"))
            self.add(text(self.px - 10, y + 4, f"{v}%", "tick", "end"))

    def legend(self) -> None:
        cx = PAD_L
        y = self.legend_y
        for cls, label in self.legend_items:
            if self.legend_kind == "line":
                stroke = "s1-stroke" if "s1" in cls else "demph-stroke"
                self.add(line(cx, y, cx + 16, y, stroke), dot(cx + 8, y, cls, 4))
                cx += 22
            else:
                self.add(
                    f'<rect class="{cls}" x="{cx:.1f}" y="{y - 5:.1f}" '
                    f'width="10" height="10" rx="2"/>'
                )
                cx += 16
            self.add(text(cx, y + 4, label, "value-2"))
            cx += text_width(label, 11) + 24

    def render(self) -> str:
        head = [text(PAD_L, 30 + i * 20, s, "title") for i, s in enumerate(self.title_lines)]
        sub_top = 30 + len(self.title_lines) * 20
        head += [text(PAD_L, sub_top + i * 17, s, "subtitle") for i, s in enumerate(self.sub_lines)]
        first_note_y = self.height - 16 - (len(self.note_lines) - 1) * 14
        foot = [
            text(PAD_L, first_note_y + i * 14, s, "note") for i, s in enumerate(self.note_lines)
        ]
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.height:.0f}" '
            f'viewBox="0 0 {W} {self.height:.0f}" role="img" aria-label="{esc(self.title)}">'
            f"<style>{STYLE}</style>"
            f'<rect class="bg" width="{W}" height="{self.height:.0f}"/>'
            + "".join(head)
            + "".join(self.parts)
            + "".join(foot)
            + "</svg>"
        )


# ── measured data ─────────────────────────────────────────────────────────────


def _classify(entry: dict) -> str:
    t = Trajectory()
    t.append(
        Step(
            index=0,
            action="test",
            error=entry["error"],
            exception_type=entry.get("exception_type"),
            metadata=entry.get("metadata") or {},
        )
    )
    return RulesClassifier().classify(t, "task").value


def _corpus(name: str) -> list[dict]:
    return json.loads((DATA_DIR / f"error_corpus_{name}.json").read_text())


@dataclass
class CorpusScore:
    hits: int
    total: int
    by_type: dict[str, tuple[int, int]]

    @property
    def pct(self) -> float:
        return 100.0 * self.hits / self.total

    def group(self, labels: tuple[str, ...]) -> tuple[int, int]:
        h = sum(self.by_type.get(x, (0, 0))[0] for x in labels)
        t = sum(self.by_type.get(x, (0, 0))[1] for x in labels)
        return h, t


def score(name: str) -> CorpusScore:
    hits: Counter[str] = Counter()
    total: Counter[str] = Counter()
    ok = 0
    entries = _corpus(name)
    for e in entries:
        total[e["label"]] += 1
        if _classify(e) == e["label"]:
            ok += 1
            hits[e["label"]] += 1
    return CorpusScore(ok, len(entries), {k: (hits[k], total[k]) for k in total})


# HISTORICAL — corpus C scored once at v1.0, before rules.py was tuned against
# its 13 misses. Re-scoring C today returns 100% (it is training data now), so
# this measurement cannot be recomputed. Source: docs/known-limitations.md,
# "Accuracy is corpus-dependent" and the per-type table beneath it.
C_V10_OVERALL = (14, 27)
C_V10_SELF_HEALING = (12, 14)
C_V10_ROUTING = (1, 12)

# HISTORICAL — LLMClassifier / HybridClassifier over corpus D with
# gpt-oss:120b-cloud (Ollama Cloud). Needs a live model, so it is not recomputed
# here. Source: scripts/llm_classifier_accuracy.py runs, recorded in
# docs/known-limitations.md, "LLMClassifier/HybridClassifier close the recall
# gap, but not the precision gap".
LLM_ROUTING_RANGE = (9, 10, 12)  # low, high, total — the two runs disagreed
LLM_MISROUTES = (4, 20)
HYBRID_ROUTING = (10, 12)
HYBRID_MISROUTES = (3, 20)


def routing_demo() -> dict[str, tuple[int, int, int]]:
    """{failure kind: (runs, baseline successes, triage successes)}.

    Derived by calling scripts/bench_synthetic.py's own ``_task_body`` — the
    shared success logic both benchmark arms run — rather than transcribing the
    benchmark's printed output.
    """
    spec = importlib.util.spec_from_file_location(
        "_bench_synthetic", REPO_ROOT / "scripts" / "bench_synthetic.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The hint each arm carries into the recovery attempt: the baseline retries
    # with no hint at all, triage with the hint its policy routed to.
    triage_hints = {
        "external_fault": "External fault. Retry with backoff.",
        "wrong_tool": "Available tools (manifest): ...",
        "schema_mismatch": "Use strict JSON schema validation",
    }
    out: dict[str, list[int]] = {}
    for task in mod.TASKS:
        kind = task.split(":")[0]
        row = out.setdefault(kind, [0, 0, 0])
        row[0] += 1
        for idx, hint in ((1, ""), (2, triage_hints[kind])):
            try:
                mod._task_body(task, 1, hint)
            except RuntimeError:
                continue
            row[idx] += 1
    return {k: (v[0], v[1], v[2]) for k, v in out.items()}


# ── chart 1: the synthetic routing demo ───────────────────────────────────────


def chart_routing_demo() -> str:
    demo = routing_demo()
    c = Canvas(
        title="The gap is exactly the two types where the hint changes the outcome",
        subtitle="Synthetic routing demo, 6 runs. Both arms call the same task body, so "
        "classification and routing are the only variables between them",
        note="Source: derived from scripts/bench_synthetic.py's own success logic · these are "
        "constructed scenarios that assume correct classification, so the chart measures "
        "routing, not detection — for detection on unseen error strings, see the held-out "
        "corpora below.",
        legend_items=[
            ("demph-fill", "no-recovery baseline — retries 3× with no hint"),
            ("s1-fill", "triage — classified, then routed to the matching hint"),
        ],
        plot_h=200,
        axis_h=48,
        head_h=34,
    )
    order = [
        ("external_fault", "transient 503 · 3 runs"),
        ("wrong_tool", "wrong_tool_called · 2 runs"),
        ("schema_mismatch", "schema_mismatch · 1 run"),
    ]
    groups = [(key, sub, *demo[key]) for key, sub in order]
    groups.append(
        (
            "all six runs",
            "the benchmark's headline",
            sum(g[2] for g in groups),
            sum(g[3] for g in groups),
            sum(g[4] for g in groups),
        )
    )
    band = c.pw / len(groups)
    bw = min(BAR_MAX, (band - 60) / 2)
    # Both arms hit 100% on external_fault, so the two cap labels would sit on
    # top of each other at a 2px gap. Separate the pair by a label's width — the
    # bars no longer touch, so the surface-gap rule doesn't apply.
    pair_gap = 22

    c.pct_axis()
    for i, (name, sub, runs, base, tri) in enumerate(groups):
        cx = c.px + band * (i + 0.5)
        for j, wins in enumerate((base, tri)):
            bx = cx - bw - pair_gap / 2 + j * (bw + pair_gap)
            pct = 100 * wins / runs
            y_top = c.y_of(pct)
            c.add(column(bx, c.base, y_top, bw, "demph-fill" if j == 0 else "s1-fill"))
            c.add(text(bx + bw / 2, y_top - 20, f"{pct:.0f}%", "value", "middle"))
            c.add(text(bx + bw / 2, y_top - 7, f"{wins}/{runs}", "value-2", "middle"))
        c.add(text(cx, c.base + 22, name, "axis-label", "middle"))
        c.add(text(cx, c.base + 38, sub, "axis-sub", "middle"))

    c.add(line(c.px + band * 3, c.py - 24, c.px + band * 3, c.base + 42, "grid"))
    c.legend()
    return c.render()


# ── chart 2: training vs held-out, corpus by corpus ───────────────────────────


def chart_corpus_scores(
    a: CorpusScore, b: CorpusScore, c_now: CorpusScore, d: CorpusScore, e: CorpusScore
) -> str:
    c = Canvas(
        title="Every corpus reaches 100% once it becomes training data",
        subtitle="RulesClassifier overall accuracy per corpus, in the order each was scored",
        note="Source: scripts/classifier_accuracy.py blocks 3-9, scored live · corpus C "
        "appears twice: the held-out score it earned at v1.0, and the score it returns "
        "today, now that rules.py has been tuned against it.",
        legend_items=[
            ("s1-fill", "held-out when scored — the number that means something"),
            ("demph-fill", "training data — tuned against, so 100% proves nothing"),
        ],
        plot_h=200,
        axis_h=104,
        head_h=34,
    )
    bars = [
        ("Corpus A", "v0.25 · training", a.hits, a.total, False),
        ("Corpus B", "v0.26 · training", b.hits, b.total, False),
        ("Corpus C", "v1.0 · held-out", *C_V10_OVERALL, True),
        ("Corpus C", "v1.1 · training", c_now.hits, c_now.total, False),
        ("Corpus D", "v1.1 · held-out", d.hits, d.total, True),
        ("Corpus E", "v1.2 · held-out", e.hits, e.total, True),
    ]
    band = c.pw / len(bars)
    bw = min(BAR_MAX, band - 26)

    c.pct_axis()
    for i, (name, sub, hits, total, heldout) in enumerate(bars):
        cx = c.px + band * (i + 0.5)
        pct = 100 * hits / total
        y_top = c.y_of(pct)
        c.add(column(cx - bw / 2, c.base, y_top, bw, "s1-fill" if heldout else "demph-fill"))
        c.add(text(cx, y_top - 20, f"{pct:.0f}%", "value", "middle"))
        c.add(text(cx, y_top - 7, f"{hits}/{total}", "value-2", "middle"))
        c.add(text(cx, c.base + 22, name, "axis-label", "middle"))
        c.add(text(cx, c.base + 38, sub, "axis-sub", "middle"))

    # Corpus C twice: the same 27 strings, before and after being tuned against.
    x_from, x_to = c.px + band * 2.5, c.px + band * 3.5
    arc_y = c.base + 52
    c.add(
        f'<path class="ann" d="M{x_from:.1f},{arc_y:.1f} L{x_from:.1f},{arc_y + 9:.1f} '
        f'L{x_to:.1f},{arc_y + 9:.1f} L{x_to:.1f},{arc_y:.1f}"/>'
    )
    c.add(
        text(
            (x_from + x_to) / 2,
            arc_y + 26,
            "the same 27 strings, after rules.py was tuned against its 13 misses",
            "axis-sub",
            "middle",
        )
    )
    c.legend()
    return c.render()


# ── chart 3: held-out recall by group, across successive corpora ──────────────


def chart_heldout_recall(d: CorpusScore, e: CorpusScore) -> str:
    c = Canvas(
        title="One tuning cycle moved nothing; a structural change moved the number",
        subtitle="RulesClassifier recall on three successive held-out corpora, each scored "
        "once, split by whether classification changes the recovery outcome",
        note="Source: scripts/classifier_accuracy.py blocks 8 and 10, corpora D and E scored "
        "live · corpus C is its pre-tuning v1.0 measurement, recorded in "
        "docs/known-limitations.md · three different corpora scored once each, not one "
        "benchmark tracked over time.",
        legend_items=[
            ("s1-fill", "routing-sensitive types — recovery needs the right hint"),
            ("demph-fill", "self-healing types — any retry heals these"),
        ],
        legend_kind="line",
        plot_h=210,
        axis_h=116,
        head_h=24,
    )
    series = [
        (
            "s1-fill",
            "s1-stroke",
            [C_V10_ROUTING, d.group(ROUTING_SENSITIVE), e.group(ROUTING_SENSITIVE)],
        ),
        (
            "demph-fill",
            "demph-stroke",
            [C_V10_SELF_HEALING, d.group(SELF_HEALING), e.group(SELF_HEALING)],
        ),
    ]
    cols = [
        ("Corpus C", "v1.0 — scored before tuning"),
        ("Corpus D", "v1.1 — after pattern tuning"),
        ("Corpus E", "v1.2 — after structured codes"),
    ]
    xs = [c.px + c.pw * (i + 0.5) / 3 for i in range(3)]

    c.pct_axis()
    for fill, stroke, pts in series:
        coords = [(x, c.y_of(100 * h / t)) for x, (h, t) in zip(xs, pts, strict=True)]
        c.add(
            f'<polyline class="{stroke}" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            + '"/>'
        )
        for (x, y), (hits, total) in zip(coords, pts, strict=True):
            pct = 100 * hits / total
            c.add(dot(x, y, fill))
            c.add(
                text(
                    x,
                    y - 13 if pct < 92 else y + 22,
                    f"{pct:.0f}%  ({hits}/{total})",
                    "value",
                    "middle",
                )
            )

    for x, (name, sub) in zip(xs, cols, strict=True):
        c.add(text(x, c.base + 22, name, "axis-label", "middle"))
        c.add(text(x, c.base + 38, sub, "axis-sub", "middle"))

    # What changed between one corpus and the next.
    for i, note in enumerate(
        (
            "v1.1: ~15 new regex patterns,\nreverse-engineered from corpus C's misses",
            "v1.2: Step.metadata\nstructured error-code matching",
        )
    ):
        mx = (xs[i] + xs[i + 1]) / 2
        c.add(line(mx - 60, c.base + 60, mx + 60, c.base + 60, "ann"))
        for j, part in enumerate(note.split("\n")):
            c.add(text(mx, c.base + 78 + j * 14, part, "axis-sub", "middle"))

    c.legend()
    return c.render()


# ── chart 4: rules vs LLM vs hybrid ───────────────────────────────────────────


def chart_classifiers(d: CorpusScore) -> str:
    c = Canvas(
        title="Semantic classification buys recall and spends precision",
        subtitle="Corpus D scored three ways. RulesClassifier misses safely — every miss is "
        "UNKNOWN — while the LLM paths answer confidently and are sometimes wrong",
        note="Source: RulesClassifier scored live · LLM and Hybrid rows from "
        "scripts/llm_classifier_accuracy.py with gpt-oss:120b-cloud — representative runs, "
        "not a frozen benchmark: reasoning-model sampling varies run to run, which is why the "
        "LLM recall bar carries a range.",
        legend_items=[
            ("s1-fill", "routing-sensitive recall (higher is better)"),
            ("s2-fill", "misroutes — a confident wrong answer (lower is better)"),
        ],
        plot_h=210,
        axis_h=44,
        head_h=0,
        plot_left=250,
    )
    r_hits, r_total = d.group(ROUTING_SENSITIVE)
    lo, hi, tot = LLM_ROUTING_RANGE
    groups = [
        (
            "RulesClassifier",
            "zero API calls",
            (100 * r_hits / r_total, f"{r_hits}/{r_total}", None),
            (0.0, "0/20", None),
        ),
        (
            "LLMClassifier",
            "every failure asked",
            (100 * hi / tot, f"{lo}–{hi}/{tot}", (100 * lo / tot, 100 * hi / tot)),
            (100 * LLM_MISROUTES[0] / LLM_MISROUTES[1], f"{LLM_MISROUTES[0]}/20", None),
        ),
        (
            "HybridClassifier",
            "rules first, LLM on UNKNOWN",
            (100 * HYBRID_ROUTING[0] / HYBRID_ROUTING[1], f"{HYBRID_ROUTING[0]}/12", None),
            (100 * HYBRID_MISROUTES[0] / HYBRID_MISROUTES[1], f"{HYBRID_MISROUTES[0]}/20", None),
        ),
    ]
    # Horizontal bars: the value labels sit at the tip with room to breathe, which
    # a 0% column cannot offer without colliding with its neighbour.
    label_w = 96
    axis_w = c.pw - label_w
    band = c.plot_h / len(groups)

    for tick in (0, 25, 50, 75, 100):
        x = c.px + axis_w * tick / 100
        c.add(line(x, c.py - 6, x, c.py + c.plot_h, "axis" if tick == 0 else "grid"))
        c.add(text(x, c.py + c.plot_h + 18, f"{tick}%", "tick", "middle"))

    for i, (name, sub, recall, misroute) in enumerate(groups):
        top = c.py + band * i + (band - 2 * BAR_MAX - GAP) / 2
        c.add(text(PAD_L, top + 14, c.check_gutter(name, 12), "axis-label"))
        c.add(text(PAD_L, top + 30, c.check_gutter(sub, 10.5), "axis-sub"))
        for j, (val, label, rng) in enumerate((recall, misroute)):
            y = top + j * (BAR_MAX + GAP)
            x1 = c.px + axis_w * val / 100
            c.add(hbar(c.px, x1, y, BAR_MAX, "s1-fill" if j == 0 else "s2-fill"))
            tip = x1
            if rng:  # the two runs disagreed — show the range rather than pick one
                x_lo, x_hi = (c.px + axis_w * v / 100 for v in rng)
                mid = y + BAR_MAX / 2
                c.add(line(x_lo, mid, x_hi, mid, "ann"))
                c.add(line(x_lo, mid - 5, x_lo, mid + 5, "ann"))
                c.add(line(x_hi, mid - 5, x_hi, mid + 5, "ann"))
                tip = max(x1, x_hi)
            c.add(text(tip + 10, y + BAR_MAX / 2 + 4, f"{val:.0f}%", "value"))
            c.add(
                text(
                    tip + 12 + text_width(f"{val:.0f}%", 12, bold=True),
                    y + BAR_MAX / 2 + 4,
                    label,
                    "value-2",
                )
            )

    c.legend()
    return c.render()


# ── chart 5: which signal caught corpus E's routing-sensitive failures ────────


def chart_corpus_e_signal() -> str:
    """Split corpus E's routing-sensitive entries by the code family they carry.

    An entry counts as caught *by the structured code* only when removing
    Step.metadata changes the answer — otherwise the message text alone already
    matched and the code added nothing.
    """
    rows: dict[str, list[int]] = {"json_rpc": [0, 0, 0], "http": [0, 0, 0]}
    for entry in _corpus("e"):
        if entry["label"] not in ROUTING_SENSITIVE:
            continue
        family = "json_rpc" if "json_rpc_code" in (entry.get("metadata") or {}) else "http"
        hit = _classify(entry) == entry["label"]
        if hit and _classify(dict(entry, metadata={})) != entry["label"]:
            rows[family][0] += 1  # only the structured code caught it
        elif hit:
            rows[family][1] += 1  # the message text would have caught it anyway
        else:
            rows[family][2] += 1

    c = Canvas(
        title="The structural signal works where a protocol spec mandates the code",
        subtitle="Corpus E's routing-sensitive failures, split by the error code the vendor "
        "actually returns",
        note="Source: scripts/classifier_accuracy.py block 10, scored live · an entry counts "
        "as caught by the code only when removing Step.metadata changes the answer · HTTP "
        "404 / 400 / 422 are left out of the mapping on purpose: one code, several unrelated "
        "failures, so routing on it would misroute.",
        legend_items=[
            ("s1-fill", "caught by the structured code"),
            ("s2-fill", "caught by message text"),
            ("demph-fill", "missed — falls through to UNKNOWN"),
        ],
        plot_h=136,
        axis_h=48,
        head_h=0,
        plot_left=330,
    )
    n_max = max(sum(v) for v in rows.values())
    caught_w = 104  # room at the tip for the "n of m caught" label
    scale = (c.pw - caught_w) / n_max
    row_h = 68
    labels = {
        "json_rpc": (
            "JSON-RPC code (MCP)",
            "−32601 / −32700 — mandated by the",
            "spec, so every server returns them",
        ),
        "http": (
            "HTTP status only",
            "Together AI, Fireworks, Replicate ×2,",
            "Cerebras, Perplexity — all 404 / 400 / 422",
        ),
    }
    seg_cls = ("s1-fill", "s2-fill", "demph-fill")

    for tick in range(n_max + 1):
        x = c.px + tick * scale
        c.add(line(x, c.py, x, c.py + c.plot_h, "axis" if tick == 0 else "grid"))
        c.add(text(x, c.py + c.plot_h + 18, str(tick), "tick", "middle"))
    c.add(
        text(
            c.px + (c.pw - caught_w) / 2,
            c.py + c.plot_h + 36,
            "corpus E entries",
            "axis-sub",
            "middle",
        )
    )

    for i, (key, counts) in enumerate(rows.items()):
        y = c.py + i * row_h + 16
        row_title, sub1, sub2 = labels[key]
        c.add(text(PAD_L, y + 4, c.check_gutter(row_title, 12), "axis-label"))
        c.add(text(PAD_L, y + 19, c.check_gutter(sub1, 10.5), "axis-sub"))
        c.add(text(PAD_L, y + 32, c.check_gutter(sub2, 10.5), "axis-sub"))
        cursor = c.px
        last = max(idx for idx, n in enumerate(counts) if n)
        for idx, n in enumerate(counts):
            if not n:
                continue
            x1 = cursor + n * scale
            end = idx == last
            c.add(hbar(cursor, x1 - (0 if end else GAP), y, BAR_MAX, seg_cls[idx], rounded=end))
            if text_width(str(n), 12, bold=True) + 18 < (x1 - cursor):
                c.add(text((cursor + x1) / 2, y + BAR_MAX / 2 + 4, str(n), "in-bar", "middle"))
            cursor = x1
        caught = counts[0] + counts[1]
        c.add(text(cursor + 12, y + BAR_MAX / 2 + 4, f"{caught} of {sum(counts)} caught", "value"))

    c.legend()
    return c.render()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    a, b, c, d, e = (score(x) for x in "abcde")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "routing-demo.svg": chart_routing_demo(),
        "corpus-scores.svg": chart_corpus_scores(a, b, c, d, e),
        "heldout-recall-by-group.svg": chart_heldout_recall(d, e),
        "classifier-comparison.svg": chart_classifiers(d),
        "corpus-e-signal.svg": chart_corpus_e_signal(),
    }
    for name, svg in charts.items():
        (OUT_DIR / name).write_text(svg + "\n")
        print(f"  wrote {OUT_DIR.relative_to(REPO_ROOT) / name}  ({len(svg) / 1024:.1f} KB)")

    print("\nScored live (must match scripts/classifier_accuracy.py):")
    for name, s in (("A", a), ("B", b), ("C", c), ("D", d), ("E", e)):
        rs_h, rs_t = s.group(ROUTING_SENSITIVE)
        sh_h, sh_t = s.group(SELF_HEALING)
        print(
            f"  corpus {name}: {s.hits}/{s.total} = {s.pct:3.0f}%   "
            f"routing-sensitive {rs_h}/{rs_t}   self-healing {sh_h}/{sh_t}"
        )


if __name__ == "__main__":
    main()
