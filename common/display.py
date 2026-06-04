"""Reusable terminal rendering helpers shared across the experiments.

100% library-rendered, all in the terminal:
- `rich`      → layout, tables, bars, color, width/TTY detection.
- `matplotlib.colormaps` + `Normalize`/`LogNorm` → value->color mapping
  (the colormap *is* the heatmap; no hand-rolled color ramp).
- `plotext`   → in-terminal line charts (training curves).

No from-scratch rendering or color math remains — only glue that fills the
library widgets.

Typical use:

    from common import display as ui

    ui.banner("MY MODEL", "subtitle")
    ui.section("Counts")
    ui.matrix(N, itos)
"""
import torch
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from rich.bar import Bar
from rich.color import Color
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

try:
    import plotext as _plt
    _HAS_PLOTEXT = True
except ImportError:  # pragma: no cover - plotext is an optional nicety
    _HAS_PLOTEXT = False

# Single shared console: rich auto-detects width, TTY and NO_COLOR.
console = Console()

DEFAULT_CMAP = "magma"


# ─────────────────────────────────────────────────────────────────────────────
# Colormap helpers (matplotlib does the value -> color mapping)
# ─────────────────────────────────────────────────────────────────────────────
def _make_norm(vmax, log):
    """matplotlib normalization mapping [vmin, vmax] -> [0, 1]."""
    if log:
        return LogNorm(vmin=1, vmax=max(float(vmax), 1.0 + 1e-9))
    return Normalize(vmin=0.0, vmax=max(float(vmax), 1e-9))


def _rgb(cmap, t):
    r, g, b, _ = cmap(min(max(float(t), 0.0), 1.0))
    return r, g, b


def _cell_style(value, cmap, norm):
    """Rich style: background = colormap(value), text auto-contrasted by luminance."""
    t = 0.0 if value <= 0 else float(norm(value))
    r, g, b = _rgb(cmap, t)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = (15, 15, 15) if lum > 0.55 else (235, 235, 235)
    return Style(color=Color.from_rgb(*fg),
                 bgcolor=Color.from_rgb(r * 255, g * 255, b * 255))


def _bar_color(value, cmap, norm):
    r, g, b = _rgb(cmap, float(norm(value)))
    return Color.from_rgb(r * 255, g * 255, b * 255)


# ─────────────────────────────────────────────────────────────────────────────
# Layout blocks
# ─────────────────────────────────────────────────────────────────────────────
def banner(title, subtitle=""):
    """Print a boxed title with an optional dimmed subtitle."""
    body = Text(title, style="bold cyan", justify="center")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="dim cyan")
    print()
    console.print(Panel(body, border_style="cyan", padding=(0, 2)))


def section(label):
    """Print a section header followed by a rule."""
    console.print()
    head = Text("▸ ", style="bold yellow")
    head.append(label, style="bold")
    console.print(head)
    console.print(Rule(style="grey37"))


def kv(key, value):
    """Print one aligned `key : value` row inside a section."""
    line = Text("  ")
    line.append(f"{key:<18}", style="dim")
    line.append(" : ", style="grey37")
    line.append(str(value), style="bold cyan")
    console.print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Rich views
# ─────────────────────────────────────────────────────────────────────────────
def matrix(M, itos, fmt=None, log=True, cmap=DEFAULT_CMAP):
    """Render a square 2-D tensor as a colored heatmap table (cells + values).

    Background color per cell comes from a matplotlib colormap (`LogNorm` by
    default); the value is printed on top with auto-contrasted text. Integer
    tensors show counts, floats fall back to 2 decimals (override with `fmt`).
    """
    M = M.detach() if hasattr(M, "detach") else M
    n = M.shape[0]
    vals = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    vmax = max((v for row in vals for v in row), default=1.0)

    is_int = all(float(v).is_integer() for row in vals for v in row)
    if fmt is None:
        fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.2f}")
    maxd = max((len(fmt(v)) for row in vals for v in row), default=1)
    labels = [itos[i] for i in range(n)]

    cm = colormaps[cmap]
    norm = _make_norm(vmax, log)

    table = Table(box=None, padding=0, show_edge=False, pad_edge=False,
                  collapse_padding=True, header_style="dim cyan")
    table.add_column("", justify="right", no_wrap=True)
    for lab in labels:
        table.add_column(lab, justify="center", width=maxd + 2, no_wrap=True)

    for i in range(n):
        row = [Text(f" {labels[i]} ", style="bold cyan")]
        for j in range(n):
            v = vals[i][j]
            row.append(Text(f" {fmt(v):>{maxd}} ", style=_cell_style(v, cm, norm)))
        table.add_row(*row)
    console.print(table)
    _colorbar(cm, vmax, is_int, log)


def _colorbar(cmap, vmax, is_int, log, steps=28):
    """Render a colormap legend strip (library-colored, no hand-rolled ramp)."""
    bar = Text("  scale  ")
    bar.append("low ", style="grey37")
    for k in range(steps):
        r, g, b = _rgb(cmap, k / (steps - 1))
        bar.append(" ", style=Style(bgcolor=Color.from_rgb(r * 255, g * 255, b * 255)))
    maxs = f"{vmax:,.0f}" if is_int else f"{vmax:.3g}"
    bar.append(" high", style="grey37")
    bar.append(f"   ({'log' if log else 'linear'} · max {maxs})", style="dim")
    console.print()
    console.print(bar)


def distribution(P, itos, row_idx, top=12, width=34, cmap=DEFAULT_CMAP):
    """Horizontal bar chart of the most likely transitions from a given context."""
    probs = P[row_idx]
    ranked = sorted(range(P.shape[0]), key=lambda j: float(probs[j]), reverse=True)
    ranked = [j for j in ranked if float(probs[j]) > 0][:top]
    if not ranked:
        return
    pmax = float(probs[ranked[0]])
    cm = colormaps[cmap]
    norm = Normalize(vmin=0.0, vmax=pmax)
    ctx = itos[row_idx]
    ctx_label = "START" if ctx == "." else f"'{ctx}'"

    header = Text("  P( next | ")
    header.append(ctx_label, style="bold cyan")
    header.append(" )   ")
    header.append("top transitions", style="dim")
    console.print(header)

    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right")
    grid.add_column()
    grid.add_column(justify="right")
    for j in ranked:
        p = float(probs[j])
        bar = Bar(size=pmax, begin=0.0, end=p, width=width, color=_bar_color(p, cm, norm))
        label = "END" if itos[j] == "." else itos[j]
        grid.add_row(Text(label, style="bold"),
                     bar,
                     Text(f"{p * 100:5.2f}%", style="bold cyan"))
    console.print(Padding(grid, (0, 0, 0, 4)))


def trace(steps, cmap=DEFAULT_CMAP):
    """Print the step-by-step sampling decisions for one generated name.

    `steps` is a list of (from_char, to_char, probability) tuples, as returned
    by `sample_name`.
    """
    cm = colormaps[cmap]
    norm = Normalize(vmin=0.0, vmax=1.0)
    table = Table(box=None, padding=(0, 1), pad_edge=False)
    table.add_column("step", justify="right", style="dim")
    table.add_column("from", justify="center", style="bold cyan")
    table.add_column("", justify="center", style="grey37")
    table.add_column("to", justify="center", style="bold magenta")
    table.add_column("p(to|from)", justify="right")
    table.add_column("", justify="left")
    for k, (frm, to, p) in enumerate(steps, 1):
        frm_l = "START" if frm == "." else frm
        to_l = "END" if to == "." else to
        bar = Bar(size=1.0, begin=0.0, end=p, width=16, color=_bar_color(p, cm, norm))
        table.add_row(str(k), frm_l, "→", to_l, f"{p * 100:5.2f}%", bar)
    console.print(Padding(table, (0, 0, 0, 2)))


def loss_curve(values, title="Training loss", width=72, height=16):
    """Plot a training curve in the terminal with plotext."""
    if not _HAS_PLOTEXT:
        kv("final loss", f"{values[-1]:.4f}")
        return
    _plt.clf()
    _plt.plot(values, marker="braille")
    _plt.title(title)
    _plt.xlabel("step")
    _plt.ylabel("loss")
    _plt.plotsize(width, height)
    _plt.theme("clear")
    _plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Name helpers (sampling + styling)
# ─────────────────────────────────────────────────────────────────────────────
def sample_name(P, itos, generator):
    """Sample one name from a row-stochastic matrix P, returning (name, trace).

    Index 0 is assumed to be the boundary token '.', used as both start and stop.
    The returned trace is the list of (from, to, p) steps for `trace()`.
    """
    idx = 0
    out, steps = [], []
    while True:
        p = P[idx]
        nxt = torch.multinomial(p, num_samples=1, generator=generator).item()
        steps.append((itos[idx], itos[nxt], float(p[nxt])))
        if nxt == 0:
            break
        out.append(itos[nxt])
        idx = nxt
    return "".join(out), steps


def style_name(name):
    """Return a rich Text for a generated name (dim boundary dots, bright letters)."""
    text = Text(".", style="grey37")
    text.append(name, style="bold cyan")
    text.append(".", style="grey37")
    return text


def name_list(samples):
    """Print a styled bullet list of (name, trace) samples with their lengths."""
    for name, _steps in samples:
        line = Text("    ● ", style="green")
        line.append_text(style_name(name))
        line.append(f"  ({len(name)})", style="dim")
        console.print(line)
    console.print()
