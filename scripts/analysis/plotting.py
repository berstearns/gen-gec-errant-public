"""Theme-neutral matplotlib helpers. Agg backend (no display), labelled axes,
a fixed source colour legend {LEARNER, AL, CONTROL} shared across every figure.
Every figure is re-derivable from a sibling CSV (written by the caller)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed, colour-blind-safe roles palette — identical across all analyses.
ROLE_COLOR = {
    "LEARNER": "#4477AA",  # blue
    "AL": "#EE6677",       # red
    "CONTROL": "#228833",  # green
}
ROLE_ORDER = ["LEARNER", "AL", "CONTROL"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def new_fig(w=7.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
