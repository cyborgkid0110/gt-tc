r"""Shared matplotlib styling for the paper figures.

The paper (single column, A4, 2.5 cm margins) has ``\linewidth`` ~= 6.3 in and
places every result figure inside a ``0.48\linewidth`` subfigure, i.e. ~3.0 in
wide on the page. A figure rendered at 8 in wide is therefore downscaled by
~0.38x, shrinking a 10 pt tick label to ~4 pt on paper.

The fix is to render close to the final printed size and set the font sizes so
that, after the small residual downscale, text lands at ~9-10 pt on the page.

Use `apply_paper_style()` once at import time, and `PANEL_SIZE` / `WIDE_SIZE`
for `figsize`.
"""
import os

import matplotlib
import matplotlib.pyplot as plt

# Printed width of a 0.48\linewidth subfigure, in inches.
SUBFIG_WIDTH_IN = 3.0
# Render slightly larger than printed so the raster stays sharp; the residual
# downscale (~1.4x) is small enough that fonts stay legible.
PANEL_SIZE = (4.3, 3.1)      # figures placed at 0.48\linewidth (2 per row)
WIDE_SIZE = (7.2, 4.4)       # figures placed at full \linewidth

# Three-per-row panels sit in a 0.32\linewidth subfigure, i.e. ~2.1 in printed
# on the Elsevier cas-sc template (\textwidth = 468 pt = 6.5 in). Rendering at
# 3.0 in keeps the same ~1.4x downscale as PANEL_SIZE, so text prints at the
# same size; the panel is squarer because it has less width to spend.
NARROW_PANEL_SIZE = (3.0, 2.5)   # figures placed at 0.32\linewidth (3 per row)

# Base font size chosen so PANEL_SIZE text prints at ~9 pt:
#   13 pt * (3.0 / 4.3) ~= 9.1 pt
BASE_FONT = 13

DPI = 300


def scenario_panel_size():
    """`figsize` for the six tiled per-scenario panels.

    Two papers include these figures at different subfigure widths, so the
    number of panels per row is selected by the PANEL_COLS environment
    variable (2 = 0.48\\linewidth, the default; 3 = 0.32\\linewidth).
    """
    return NARROW_PANEL_SIZE if os.environ.get('PANEL_COLS') == '3' \
        else PANEL_SIZE


def thin_xticks(ax=None):
    """Limit the x axis to few ticks so labels do not run together.

    Round counts reach five digits, and three of them do not fit side by side
    across a 3-per-row panel. Only applied when PANEL_COLS selects the narrow
    layout; the wider panels have room for matplotlib's default ticking.
    """
    if os.environ.get('PANEL_COLS') != '3':
        return
    from matplotlib.ticker import MaxNLocator
    ax = ax or plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))


def apply_paper_style(base_font=BASE_FONT):
    """Set global rcParams for paper-ready figures."""
    matplotlib.rcParams.update({
        'font.size':          base_font,
        'axes.titlesize':     base_font + 1,
        'axes.labelsize':     base_font,
        'xtick.labelsize':    base_font - 1,
        'ytick.labelsize':    base_font - 1,
        'legend.fontsize':    base_font - 2,
        'legend.title_fontsize': base_font - 2,
        'figure.titlesize':   base_font + 1,
        'lines.linewidth':    1.6,
        'axes.linewidth':     0.9,
        'xtick.major.width':  0.9,
        'ytick.major.width':  0.9,
        'legend.frameon':     True,
        'legend.framealpha':  0.85,
        'legend.borderpad':   0.3,
        'legend.labelspacing': 0.25,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 0.9,
        'savefig.dpi':        DPI,
        'figure.dpi':         DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
    })


def colour_map(algos):
    """Distinct colour per algorithm.

    The benchmark has 11 algorithms but the default cycle only has 10 colours,
    so two lines would otherwise share one colour; extend with tab20b.
    """
    base = (list(plt.get_cmap('tab10').colors)
            + list(plt.get_cmap('tab20b').colors))
    return {a: base[i % len(base)] for i, a in enumerate(algos)}
