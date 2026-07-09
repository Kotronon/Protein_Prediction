from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT_DIR = Path("docs")
PNG_OUT = OUT_DIR / "closing_slide_example.png"
PDF_OUT = OUT_DIR / "closing_slide_example.pdf"


def add_round_box(ax, xy, width, height, face, edge="#d8dee9", lw=1.0, radius=0.025):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    return box


def add_text(ax, x, y, text, size=24, weight="normal", color="#111827", ha="left", va="top"):
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

    # Background
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f8fafc"))
    ax.add_patch(Rectangle((0, 0.91), 1, 0.09, transform=ax.transAxes, color="#111827"))

    # Header
    add_text(ax, 0.055, 0.965, "Final Take-Home", size=24, weight="bold", color="#ffffff", va="center")
    add_text(
        ax,
        0.945,
        0.965,
        "Protein Disorder Prediction",
        size=14,
        color="#cbd5e1",
        ha="right",
        va="center",
    )

    # Main claim
    add_text(
        ax,
        0.055,
        0.835,
        "Intrinsic disorder is not one single label.",
        size=36,
        weight="bold",
        color="#0f172a",
    )
    add_text(
        ax,
        0.057,
        0.775,
        "Different annotations capture related but distinct biological signals.",
        size=18,
        color="#475569",
    )

    # Three evidence cards
    cards = [
        (
            0.055,
            "#e0f2fe",
            "#0369a1",
            "1",
            "UdonPred learns\nreal signal",
            "Beats simple sequence baselines;\nstrongest for DisProt and pLDDT.",
        ),
        (
            0.365,
            "#ecfdf5",
            "#047857",
            "2",
            "Labels are\nnot interchangeable",
            "PDBFlex reflects flexibility;\nDisProt reflects curated disorder.",
        ),
        (
            0.675,
            "#fff7ed",
            "#c2410c",
            "3",
            "Best strategy is\ntarget-specific",
            "Avoid naive averaging;\nvalidate ensembles per target.",
        ),
    ]

    for x, face, accent, number, title, body in cards:
        add_round_box(ax, (x, 0.435), 0.27, 0.25, face, edge="#e2e8f0", lw=1.2, radius=0.02)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.022, 0.61),
                0.045,
                0.05,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=0,
                facecolor=accent,
                transform=ax.transAxes,
            )
        )
        add_text(ax, x + 0.0445, 0.635, number, size=16, weight="bold", color="#ffffff", ha="center", va="center")
        add_text(ax, x + 0.085, 0.645, title, size=19, weight="bold", color="#111827")
        add_text(ax, x + 0.026, 0.535, body, size=15, color="#334155")

    # CAID4 recommendation box
    add_round_box(ax, (0.055, 0.205), 0.89, 0.145, "#ffffff", edge="#94a3b8", lw=1.5, radius=0.018)
    ax.add_patch(Rectangle((0.055, 0.205), 0.012, 0.145, transform=ax.transAxes, color="#2563eb"))
    add_text(ax, 0.085, 0.315, "CAID4 recommendation", size=18, weight="bold", color="#1d4ed8")
    add_text(
        ax,
        0.085,
        0.268,
        "Use same-head models as a robust base, then add validated per-target combinations and contested-region diagnostics.",
        size=18,
        color="#111827",
    )

    # Closing sentence
    add_text(
        ax,
        0.5,
        0.095,
        "Respect the annotation concept before optimizing the model.",
        size=25,
        weight="bold",
        color="#0f172a",
        ha="center",
    )
    add_text(
        ax,
        0.5,
        0.052,
        "This is the main lesson from the full project.",
        size=14,
        color="#64748b",
        ha="center",
    )

    fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
