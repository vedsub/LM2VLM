"""
utils.py  –  Visualization utilities for the VLM project

Provides helper functions for plotting and inspecting model outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


DEFAULT_SAVE_PATH = (
    Path(__file__).resolve().parent.parent / "inference_results" / "similarity_grid.jpg"
)


def plot_similarity_grid(
    similarity_matrix: np.ndarray,
    image_paths: Sequence[str | Path],
    text_captions: Sequence[str],
    save_path: str | Path = DEFAULT_SAVE_PATH,
    max_caption_len: int = 40,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".2f",
) -> Path:
    """Plot a cosine-similarity matrix as a heatmap and save to disk.

    Args:
        similarity_matrix: (N_images, N_texts) array of similarity scores.
        image_paths:       Paths (or identifiers) for each image (Y-axis).
        text_captions:     Caption strings for each text (X-axis).
        save_path:         Where to save the figure.
        max_caption_len:   Truncate captions longer than this.
        figsize:           Matplotlib figure size; auto-calculated if *None*.
        cmap:              Colour-map name.
        annot:             Whether to annotate cells with values.
        fmt:               Number format string for annotations.

    Returns:
        Resolved save path.
    """
    sim = np.asarray(similarity_matrix)
    n_images, n_texts = sim.shape

    # ── Label preparation ─────────────────────────────────────────────
    y_labels = [
        Path(p).name if isinstance(p, (str, Path)) else str(p)
        for p in image_paths
    ]
    x_labels = [
        (c[:max_caption_len] + "…") if len(c) > max_caption_len else c
        for c in text_captions
    ]

    # ── Figure sizing ─────────────────────────────────────────────────
    if figsize is None:
        w = max(8, n_texts * 1.2)
        h = max(6, n_images * 0.8)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    # ── Heatmap ───────────────────────────────────────────────────────
    # Disable cell annotations for large matrices to stay readable.
    if n_images * n_texts > 400:
        annot = False

    if _HAS_SEABORN:
        sns.heatmap(
            sim,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(sim, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_texts))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

        ax.set_yticks(range(n_images))
        ax.set_yticklabels(y_labels, fontsize=8)

        if annot:
            for i in range(n_images):
                for j in range(n_texts):
                    ax.text(
                        j, i, f"{sim[i, j]:{fmt[1:]}}",
                        ha="center", va="center", fontsize=7,
                        color="white" if sim[i, j] < sim.mean() else "black",
                    )

    ax.set_xlabel("Text Captions", fontsize=12)
    ax.set_ylabel("Images", fontsize=12)
    ax.set_title("Image–Text Cosine Similarity", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Similarity grid saved to {save_path}")

    return save_path
