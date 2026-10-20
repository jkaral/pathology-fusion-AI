"""Static figures for the technical report (PNG, light surface).

Colours follow a validated categorical order (blue, orange, aqua) assigned by entity in a
fixed order; magnitude uses a single blue ramp. Thin marks, recessive grid, one y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc2ee", "#2a78d6", "#103d73"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, path: str | Path) -> None:
    """Row-normalised confusion matrix; cells show count and row %."""
    rows = cm.sum(1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(frac, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = "white" if frac[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n{100 * frac[i, j]:.0f}%", ha="center", va="center", fontsize=7, color=ink)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(classes)), classes, fontsize=8, color=INK_2)
    ax.set_xlabel("Predicted", color=INK, fontsize=9)
    ax.set_ylabel("True", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def plot_forest(rows: list[dict], metric: str, label: str, path: str | Path, highlight: str | None = None) -> None:
    """Point estimate + 95% bootstrap CI per configuration (one row per table line)."""
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.0))
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(rows[::-1]):
        m = r[metric]
        color = SERIES[1] if r["name"] == highlight else SERIES[0]
        ax.plot([m["lo"], m["hi"]], [i, i], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([m["point"]], [i], "o", color=color, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.text(m["hi"] + 0.01, i, f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]",
                va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]], fontsize=8, color=INK)
    ax.set_xlabel(label + " (test, seed-ensemble, 95% stratified bootstrap CI)", fontsize=8.5, color=INK)
    lo = min(r[metric]["lo"] for r in rows)
    hi = max(r[metric]["hi"] for r in rows)
    ax.set_xlim(max(0.0, lo - 0.05), min(1.0, hi + 0.25))
    _save(fig, path)


def plot_curves(histories: dict[str, list[list[dict]]], path: str | Path) -> None:
    """Train vs validation curves per configuration (mean over seeds, band = min..max)."""
    panels = [("loss", "train/loss", "val/loss", "Loss (train: weighted CE; val: log-loss)"),
              ("auroc", "train/macro_auroc", "val/macro_auroc", "Macro AUROC"),
              ("bacc", "train/balanced_accuracy", "val/balanced_accuracy", "Balanced accuracy")]
    names = [n for n, h in histories.items() if h and h[0]]
    fig, axes = plt.subplots(len(names), 3, figsize=(11, 2.3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        for c, (_, tk, vk, title) in enumerate(panels):
            ax = axes[r, c]
            _style(ax)
            for key, color, lab in ((tk, SERIES[0], "train"), (vk, SERIES[1], "validation")):
                runs = [[rec.get(key, np.nan) for rec in h] for h in histories[name] if h]
                if not runs or all(np.all(np.isnan(x)) for x in runs):
                    continue
                arr = np.array(runs, dtype=float)
                ep = np.arange(1, arr.shape[1] + 1)
                ax.fill_between(ep, np.nanmin(arr, 0), np.nanmax(arr, 0), color=color, alpha=0.15, linewidth=0)
                ax.plot(ep, np.nanmean(arr, 0), color=color, linewidth=2, label=lab)
            if r == 0:
                ax.set_title(title, fontsize=9, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(name, fontsize=8, color=INK)
            if r == len(names) - 1:
                ax.set_xlabel("epoch", fontsize=8, color=INK_2)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, path)


def plot_per_class_auroc(per_class: dict[str, np.ndarray], classes: list[str], path: str | Path) -> None:
    """Grouped bars: per-class OvR AUROC for up to three configurations."""
    names = list(per_class)[:3]
    x = np.arange(len(classes))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    _style(ax)
    ax.grid(axis="x", visible=False)
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, per_class[n], width=w - 0.03, color=SERIES[i], label=n,
               zorder=2)
    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x, classes, fontsize=8, color=INK)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("OvR AUROC (test)", fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=len(names), loc="upper left", bbox_to_anchor=(0, 1.15))
    _save(fig, path)


def plot_agent_trajectory(records: list[dict], path: str | Path) -> None:
    """CV score per trial in order, with the running incumbent (step line)."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    _style(ax)
    idx = [r["index"] for r in records]
    mean = np.array([r["cv_mean"] for r in records])
    sd = np.array([r["cv_sd"] for r in records])
    ref = [r["phase"] == "reference" for r in records]
    for i, m, s, isref, acc in zip(idx, mean, sd, ref, [r["accepted"] for r in records]):
        color = INK_2 if isref else (SERIES[1] if acc else SERIES[0])
        ax.plot([i, i], [m - s, m + s], color=color, linewidth=1.2, alpha=0.6)
        ax.plot([i], [m], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    best = np.maximum.accumulate(np.where(ref, -np.inf, mean))
    ax.step(idx, np.where(np.isfinite(best), best, np.nan), where="post", color=SERIES[2], linewidth=2,
            label="best fusion so far")
    ax.set_xlabel("trial (grey: single-model references, orange: accepted)", fontsize=8.5, color=INK)
    ax.set_ylabel("CV macro AUROC (mean ± fold sd)", fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, path)
