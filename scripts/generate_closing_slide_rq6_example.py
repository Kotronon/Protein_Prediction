from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT_DIR = Path("docs")
PNG_OUT = OUT_DIR / "closing_slide_rq6_example_readable.png"
PDF_OUT = OUT_DIR / "closing_slide_rq6_example_readable.pdf"


def add_box(ax, xy, width, height, face, edge="#d8dee9", lw=1.0, radius=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, size=20, weight="normal", color="#111827", ha="left", va="top"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        family="DejaVu Sans",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f8fafc"))

    # Top band
    ax.add_patch(Rectangle((0, 0.91), 1, 0.09, transform=ax.transAxes, color="#111827"))
    add_text(ax, 0.055, 0.965, "Final Take-Home", size=28, weight="bold", color="#ffffff", va="center")
    add_text(ax, 0.945, 0.965, "with RQ6", size=18, color="#cbd5e1", ha="right", va="center")

    # Main claim
    add_text(
        ax,
        0.055,
        0.835,
        "Disorder prediction depends on",
        size=34,
        weight="bold",
        color="#0f172a",
    )
    add_text(
        ax,
        0.055,
        0.780,
        "what the label measures.",
        size=34,
        weight="bold",
        color="#0f172a",
    )
    add_text(
        ax,
        0.057,
        0.725,
        "Our results show a path from model performance to biological disagreement.",
        size=20,
        color="#475569",
    )

    # Storyline cards: intentionally short for projector readability.
    cards = [
        (
            0.055,
            0.425,
            0.26,
            0.22,
            "#e0f2fe",
            "#0369a1",
            "1",
            "Model signal",
            "UdonPred beats\nsimple baselines",
        ),
        (
            0.370,
            0.425,
            0.26,
            0.22,
            "#ecfdf5",
            "#047857",
            "2",
            "Different labels",
            "Disorder, flexibility\nand confidence differ",
        ),
        (
            0.685,
            0.425,
            0.26,
            0.22,
            "#fff7ed",
            "#c2410c",
            "3",
            "Target-specific",
            "Validate improvements\nper target",
        ),
    ]

    for x, y, width, height, face, accent, number, title, body in cards:
        add_box(ax, (x, y), width, height, face, edge="#e2e8f0", lw=1.1, radius=0.020)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.023, y + height - 0.065),
                0.055,
                0.055,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=0,
                facecolor=accent,
                transform=ax.transAxes,
            )
        )
        add_text(ax, x + 0.0505, y + height - 0.037, number, size=18, weight="bold", color="#ffffff", ha="center", va="center")
        add_text(ax, x + 0.095, y + height - 0.020, title, size=23, weight="bold", color="#111827")
        add_text(ax, x + 0.026, y + 0.090, body, size=22, color="#334155")

    # RQ6 highlight
    add_box(ax, (0.055, 0.215), 0.89, 0.145, "#f3e8ff", edge="#c084fc", lw=1.6, radius=0.020)
    ax.add_patch(Rectangle((0.055, 0.215), 0.014, 0.145, transform=ax.transAxes, color="#7e22ce"))
    add_text(ax, 0.087, 0.325, "RQ6 adds the biological interpretation", size=23, weight="bold", color="#6b21a8")
    add_text(
        ax,
        0.087,
        0.280,
        "Contested regions show where disorder, flexibility",
        size=20,
        color="#111827",
    )
    add_text(
        ax,
        0.087,
        0.242,
        "and confidence-based signals diverge.",
        size=20,
        color="#111827",
    )

    # Closing line
    add_text(
        ax,
        0.5,
        0.120,
        "Disagreement is not just noise.",
        size=31,
        weight="bold",
        color="#0f172a",
        ha="center",
    )
    add_text(
        ax,
        0.5,
        0.067,
        "It tells us which biological concept the model is seeing.",
        size=22,
        color="#475569",
        ha="center",
    )

    # Connector arrows
    for x in [0.342, 0.657]:
        add_text(ax, x, 0.535, "→", size=34, weight="bold", color="#64748b", ha="center", va="center")

    fig.savefig(PNG_OUT, dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(PDF_OUT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
