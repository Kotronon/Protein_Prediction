#!/usr/bin/env python3
"""Render the DisProt-focused Markdown report as a PDF."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT = Path("docs/disprot_focus_udonpred_report.md")
OUTPUT = Path("docs/disprot_focus_udonpred_report.pdf")
PAGE_SIZE = (8.27, 11.69)
LEFT = 0.08
TOP = 0.94
LINE = 0.026


def image_path_from_markdown(line: str) -> tuple[str, Path] | None:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not match:
        return None
    title, rel_path = match.groups()
    return title, REPORT.parent / rel_path


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(LEFT, TOP, title, fontsize=16, fontweight="bold", ha="left", va="top")
    image = mpimg.imread(image_path)
    image_ax = fig.add_axes([0.08, 0.08, 0.84, 0.78])
    image_ax.imshow(image)
    image_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def flush_text_page(pdf: PdfPages, page_lines: list[tuple[str, float, str]]) -> None:
    if not page_lines:
        return
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = TOP
    for text, size, weight in page_lines:
        fig.text(
            LEFT,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            ha="left",
            va="top",
            family="monospace" if text.startswith("|") else "sans-serif",
        )
        y -= LINE * (size / 10.5)
    pdf.savefig(fig)
    plt.close(fig)


def append_wrapped(
    pages: list[list[tuple[str, float, str]]],
    text: str,
    *,
    size: float = 10.5,
    weight: str = "normal",
    width: int = 95,
) -> None:
    if not pages:
        pages.append([])
    wrapped = textwrap.wrap(text, width=width) or [""]
    for line in wrapped:
        if len(pages[-1]) >= 34:
            pages.append([])
        pages[-1].append((line, size, weight))


def markdown_to_pages(lines: list[str]) -> tuple[list[list[tuple[str, float, str]]], list[tuple[str, Path]]]:
    pages: list[list[tuple[str, float, str]]] = [[]]
    images: list[tuple[str, Path]] = []
    in_table = False

    for raw_line in lines:
        line = raw_line.rstrip()
        image = image_path_from_markdown(line)
        if image is not None:
            images.append(image)
            in_table = False
            continue

        if line.startswith("# "):
            append_wrapped(pages, line[2:], size=18, weight="bold", width=62)
            append_wrapped(pages, "", size=6)
            in_table = False
        elif line.startswith("## "):
            append_wrapped(pages, "", size=4)
            append_wrapped(pages, line[3:], size=14, weight="bold", width=75)
            in_table = False
        elif line.startswith("|"):
            in_table = True
            if len(pages[-1]) >= 31:
                pages.append([])
            pages[-1].append((line, 7.3, "normal"))
        elif in_table and not line:
            append_wrapped(pages, "", size=5)
            in_table = False
        elif line.startswith("- "):
            append_wrapped(pages, "- " + line[2:], size=10.5, width=90)
            in_table = False
        elif line.startswith("> "):
            append_wrapped(pages, line, size=10.5, weight="bold", width=90)
            in_table = False
        elif line:
            append_wrapped(pages, line, size=10.5, width=92)
            in_table = False
        else:
            append_wrapped(pages, "", size=5)
            in_table = False

    return pages, images


def main() -> None:
    lines = REPORT.read_text().splitlines()
    pages, images = markdown_to_pages(lines)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        for page in pages:
            flush_text_page(pdf, page)
        for title, image_path in images:
            add_image_page(pdf, title, image_path)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
