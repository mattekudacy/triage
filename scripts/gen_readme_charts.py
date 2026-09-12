"""
scripts/gen_readme_charts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate the PNG charts embedded in README.md, using matplotlib/seaborn.

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

Requires the ``dev`` extra (``matplotlib``, ``seaborn`` — not runtime deps of
``triage`` itself, see pyproject.toml's comment on why they live there).

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_readme_charts.py
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import Step
from triage.trajectory import Trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
OUT_DIR = REPO_ROOT / "docs" / "assets" / "charts"

SELF_HEALING = ("external_fault", "timeout")
ROUTING_SENSITIVE = ("wrong_tool_called", "schema_mismatch")

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
PALETTE = sns.color_palette("deep")
BLUE, ORANGE, GREEN, GRAY = PALETTE[0], PALETTE[1], PALETTE[2], "#8c8c8c"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#444444",
        "grid.color": "#dddddd",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
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


# ── shared chart chrome ─────────────────────────────────────────────────────


def _pct_axis(ax, top: int = 118) -> None:
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def _bar_labels(ax, bars, fmt_fn) -> None:
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.annotate(
            fmt_fn(i),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")


# ── chart 1: the synthetic routing demo ───────────────────────────────────────


def chart_routing_demo() -> None:
    demo = routing_demo()
    order = [
        ("external_fault", "external_fault\n(transient 503)\nn=3"),
        ("wrong_tool", "wrong_tool_called\nn=2"),
        ("schema_mismatch", "schema_mismatch\nn=1"),
    ]
    labels = [lbl for _, lbl in order] + ["all six\nruns"]
    baseline = [100 * demo[k][1] / demo[k][0] for k, _ in order]
    triage = [100 * demo[k][2] / demo[k][0] for k, _ in order]
    baseline.append(100 * sum(demo[k][1] for k, _ in order) / sum(demo[k][0] for k, _ in order))
    triage.append(100 * sum(demo[k][2] for k, _ in order) / sum(demo[k][0] for k, _ in order))
    n_base = [demo[k][1] for k, _ in order] + [sum(demo[k][1] for k, _ in order)]
    n_tri = [demo[k][2] for k, _ in order] + [sum(demo[k][2] for k, _ in order)]
    n_runs = [demo[k][0] for k, _ in order] + [sum(demo[k][0] for k, _ in order)]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], baseline, width=w, color=GRAY, label="no-recovery baseline")
    b2 = ax.bar([i + w / 2 for i in x], triage, width=w, color=BLUE, label="triage")
    _bar_labels(ax, b1, lambda i: f"{n_base[i]}/{n_runs[i]}")
    _bar_labels(ax, b2, lambda i: f"{n_tri[i]}/{n_runs[i]}")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("success rate")
    ax.axvline(2.5, color="#bbbbbb", lw=1, ls="--")
    _pct_axis(ax)
    ax.set_title("Routing demo: the gap is exactly the two types where the hint matters")
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)
    fig.text(
        0.01,
        -0.02,
        "Synthetic, 6 runs — both arms call the same task body, so classification/routing "
        "are the only variables. Source: derived from scripts/bench_synthetic.py.",
        fontsize=7.5,
        color="#666666",
    )
    fig.tight_layout()
    _save(fig, "routing-demo.png")


# ── chart 2: training vs held-out, corpus by corpus ───────────────────────────


def chart_corpus_scores(a, b, c_now, d, e) -> None:
    bars = [
        ("Corpus A\nv0.25\ntraining", a.hits, a.total, False),
        ("Corpus B\nv0.26\ntraining", b.hits, b.total, False),
        ("Corpus C\nv1.0\nheld-out", *C_V10_OVERALL, True),
        ("Corpus C\nv1.1\ntraining", c_now.hits, c_now.total, False),
        ("Corpus D\nv1.1\nheld-out", d.hits, d.total, True),
        ("Corpus E\nv1.2\nheld-out", e.hits, e.total, True),
    ]
    labels = [b[0] for b in bars]
    pcts = [100 * b[1] / b[2] for b in bars]
    colors = [BLUE if b[3] else GRAY for b in bars]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars_obj = ax.bar(range(len(bars)), pcts, color=colors, width=0.6)
    _bar_labels(ax, bars_obj, lambda i: f"{pcts[i]:.0f}%\n{bars[i][1]}/{bars[i][2]}")

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy")
    _pct_axis(ax)
    ax.set_title("Every corpus reaches 100% once it becomes training data", pad=14)

    # Bracket under corpus C's two bars — the same 27 strings, before and after
    # rules.py was tuned against them. Plain data coordinates with clip_on=False,
    # same technique as chart_heldout_recall's below-axis notes; the 3-line
    # x-tick labels (vs. that chart's 2-line ones) need it pushed further down.
    ax.plot([2, 2, 3, 3], [-34, -40, -40, -34], color="#888888", lw=1, clip_on=False)
    ax.text(
        2.5,
        -46,
        "same 27 strings, before/after rules.py was tuned\nagainst its 13 misses",
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
        clip_on=False,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
    ]
    ax.legend(
        handles,
        ["held-out when scored", "training data (tuned against)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.62),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.42)
    _save(fig, "corpus-scores.png")


# ── chart 3: held-out recall by group, across successive corpora ──────────────


def chart_heldout_recall(d: CorpusScore, e: CorpusScore) -> None:
    cols = ["Corpus C\nv1.0", "Corpus D\nv1.1", "Corpus E\nv1.2"]
    routing = [C_V10_ROUTING, d.group(ROUTING_SENSITIVE), e.group(ROUTING_SENSITIVE)]
    healing = [C_V10_SELF_HEALING, d.group(SELF_HEALING), e.group(SELF_HEALING)]
    r_pct = [100 * h / t for h, t in routing]
    h_pct = [100 * h / t for h, t in healing]
    x = range(3)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        x,
        h_pct,
        marker="o",
        ms=9,
        lw=2.5,
        color=GRAY,
        label="self-healing types (any retry heals these)",
    )
    ax.plot(
        x,
        r_pct,
        marker="o",
        ms=9,
        lw=2.5,
        color=BLUE,
        label="routing-sensitive types (need the right hint)",
    )

    # Both series only ever move apart or hold flat — every point can be labeled
    # above its marker without colliding with the other series or the x-axis.
    for xi, (h, t), pct in zip(x, healing, h_pct, strict=True):
        ax.annotate(
            f"{pct:.0f}% ({h}/{t})",
            (xi, pct),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    for xi, (h, t), pct in zip(x, routing, r_pct, strict=True):
        ax.annotate(
            f"{pct:.0f}% ({h}/{t})",
            (xi, pct),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(cols)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("recall")
    _pct_axis(ax)
    ax.set_title("One tuning cycle moved nothing; a structural change moved the number")
    ax.legend(loc="center left", frameon=True, fontsize=9.5)

    # What changed between one corpus and the next, in data coordinates just under
    # the x-axis so bbox_inches='tight' sizes the figure to fit it exactly.
    for xi, note in (
        (0.5, "v1.1: ~15 regex patterns\nreverse-engineered from C's misses"),
        (1.5, "v1.2: Step.metadata\nstructured error-code matching"),
    ):
        ax.text(xi, -16, note, ha="center", va="top", fontsize=8, color="#666666", clip_on=False)

    fig.text(
        0.01,
        0.01,
        "Three different corpora, each scored once — not one benchmark tracked over time. "
        "Source: classifier_accuracy.py blocks 8/10 (D, E scored live); corpus C is its "
        "v1.0 pre-tuning score, recorded in docs/known-limitations.md.",
        fontsize=7.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, "heldout-recall-by-group.png")


# ── chart 4: rules vs LLM vs hybrid ───────────────────────────────────────────


def chart_classifiers(d: CorpusScore) -> None:
    r_hits, r_total = d.group(ROUTING_SENSITIVE)
    lo, hi, tot = LLM_ROUTING_RANGE
    names = [
        "RulesClassifier\n(zero API calls)",
        "LLMClassifier\n(every failure asked)",
        "HybridClassifier\n(rules first, LLM on UNKNOWN)",
    ]
    recall = [100 * r_hits / r_total, 100 * hi / tot, 100 * HYBRID_ROUTING[0] / HYBRID_ROUTING[1]]
    recall_lo = [None, 100 * lo / tot, None]
    recall_lbl = [
        f"{r_hits}/{r_total}",
        f"{lo}–{hi}/{tot}",
        f"{HYBRID_ROUTING[0]}/{HYBRID_ROUTING[1]}",
    ]
    misroute = [
        0.0,
        100 * LLM_MISROUTES[0] / LLM_MISROUTES[1],
        100 * HYBRID_MISROUTES[0] / HYBRID_MISROUTES[1],
    ]
    misroute_lbl = ["0/20", f"{LLM_MISROUTES[0]}/20", f"{HYBRID_MISROUTES[0]}/20"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(3)
    w = 0.36
    b1 = ax.bar(
        [i - w / 2 for i in x],
        recall,
        width=w,
        color=BLUE,
        label="routing-sensitive recall (higher is better)",
    )
    b2 = ax.bar(
        [i + w / 2 for i in x], misroute, width=w, color=ORANGE, label="misroutes (lower is better)"
    )
    _bar_labels(ax, b1, lambda i: f"{recall[i]:.0f}%\n{recall_lbl[i]}")
    _bar_labels(ax, b2, lambda i: f"{misroute[i]:.0f}%\n{misroute_lbl[i]}")

    for i, lo_v in enumerate(recall_lo):
        if lo_v is not None:
            xpos = i - w / 2
            ax.plot([xpos, xpos], [lo_v, recall[i]], color="#444444", lw=1.4)
            ax.plot([xpos - 0.06, xpos + 0.06], [lo_v, lo_v], color="#444444", lw=1.4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("rate")
    _pct_axis(ax)
    ax.set_title("Semantic classification buys recall and spends precision")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9.5)
    fig.text(
        0.01,
        0.01,
        "Corpus D scored three ways. RulesClassifier misses safely (UNKNOWN); LLM paths guess "
        "confidently and are sometimes wrong. LLM/Hybrid: scripts/llm_classifier_accuracy.py "
        "with gpt-oss:120b-cloud, representative runs (sampling varies run to run).",
        fontsize=7.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "classifier-comparison.png")


# ── chart 5: which signal caught corpus E's routing-sensitive failures ────────


def chart_corpus_e_signal() -> None:
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

    labels = [
        "JSON-RPC code (MCP)\n−32601 / −32700 — spec-mandated",
        "HTTP status only\nTogether/Fireworks/Replicate×2/Cerebras/Perplexity",
    ]
    seg_labels = ["caught by structured code", "caught by message text", "missed → UNKNOWN"]
    colors = [BLUE, ORANGE, GRAY]
    keys = ["json_rpc", "http"]
    data = [rows[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = range(len(labels))
    left = [0.0, 0.0]
    for seg_idx in range(3):
        vals = [data[row][seg_idx] for row in range(2)]
        ax.barh(
            list(y), vals, left=left, height=0.55, color=colors[seg_idx], label=seg_labels[seg_idx]
        )
        for row, (v, lft) in enumerate(zip(vals, left, strict=True)):
            if v:
                ax.text(
                    lft + v / 2,
                    row,
                    str(v),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )
        left = [prev + v for prev, v in zip(left, vals, strict=True)]

    for row in range(2):
        caught = data[row][0] + data[row][1]
        total = sum(data[row])
        ax.text(
            total + 0.15,
            row,
            f"{caught} of {total} caught",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("corpus E entries")
    ax.set_xlim(0, max(sum(r) for r in data) + 1.6)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax.set_title("The structural signal works where a protocol spec mandates the code")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.01,
        -0.02,
        "An entry counts as caught by the code only if removing Step.metadata changes the "
        "answer. HTTP 404/400/422 are excluded from the mapping on purpose (one code, several "
        "unrelated failures).",
        fontsize=7.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "corpus-e-signal.png")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    a, b, c, d, e = (score(x) for x in "abcde")

    chart_routing_demo()
    chart_corpus_scores(a, b, c, d, e)
    chart_heldout_recall(d, e)
    chart_classifiers(d)
    chart_corpus_e_signal()

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
