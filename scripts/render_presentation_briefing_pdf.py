from __future__ import annotations

import argparse
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (8.27, 11.69)  # A4 portrait in inches
LEFT = 0.08
RIGHT = 0.92
TOP = 0.94
BOTTOM = 0.06
BASE_FONT = 9.5
MONO_FONT = 8.0
LINE_HEIGHT = 0.022


def clean_inline_markdown(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text


def wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(
        clean_inline_markdown(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


class PdfWriter:
    def __init__(self, output: Path):
        self.output = output
        self.pdf = PdfPages(output)
        self.page_number = 0
        self.fig = None
        self.ax = None
        self.y = TOP

    def new_page(self) -> None:
        if self.fig is not None:
            self.footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_number += 1
        self.fig, self.ax = plt.subplots(figsize=PAGE_SIZE)
        self.ax.axis("off")
        self.y = TOP

    def footer(self) -> None:
        assert self.ax is not None
        self.ax.text(
            0.5,
            0.025,
            f"Protein Prediction Briefing - Seite {self.page_number}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#666666",
            transform=self.ax.transAxes,
        )

    def ensure_space(self, needed: float) -> None:
        if self.fig is None or self.y - needed < BOTTOM:
            self.new_page()

    def text(self, line: str, *, fontsize: float = BASE_FONT, weight: str = "normal", family: str = "DejaVu Sans", color: str = "#111111", indent: float = 0.0, line_height: float = LINE_HEIGHT) -> None:
        self.ensure_space(line_height * 1.4)
        assert self.ax is not None
        self.ax.text(
            LEFT + indent,
            self.y,
            line,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight=weight,
            family=family,
            color=color,
            transform=self.ax.transAxes,
        )
        self.y -= line_height

    def blank(self, amount: float = LINE_HEIGHT * 0.55) -> None:
        self.ensure_space(amount)
        self.y -= amount

    def close(self) -> None:
        if self.fig is not None:
            self.footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.pdf.close()


def render_markdown(source: Path, output: Path) -> None:
    writer = PdfWriter(output)
    writer.new_page()

    in_code = False
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            writer.blank(0.008)
            continue

        if in_code:
            for wrapped in wrap_text(line, 92):
                writer.text(wrapped, fontsize=MONO_FONT, family="DejaVu Sans Mono", color="#333333", line_height=0.019)
            continue

        if not line:
            writer.blank()
            continue

        if line.startswith("# "):
            writer.ensure_space(0.08)
            for wrapped in wrap_text(line[2:], 42):
                writer.text(wrapped, fontsize=19, weight="bold", color="#0f172a", line_height=0.035)
            writer.blank(0.012)
            continue

        if line.startswith("## "):
            writer.ensure_space(0.06)
            writer.blank(0.006)
            for wrapped in wrap_text(line[3:], 54):
                writer.text(wrapped, fontsize=13.2, weight="bold", color="#1f4e79", line_height=0.028)
            writer.blank(0.004)
            continue

        if line.startswith("### "):
            writer.ensure_space(0.045)
            for wrapped in wrap_text(line[4:], 64):
                writer.text(wrapped, fontsize=11, weight="bold", color="#334155", line_height=0.024)
            continue

        if line.startswith("|"):
            writer.ensure_space(0.025)
            if re.match(r"^\|\s*-+", line):
                continue
            table_line = clean_inline_markdown(line).strip("|").replace(" | ", "  |  ")
            for wrapped in wrap_text(table_line, 98):
                writer.text(wrapped, fontsize=7.4, family="DejaVu Sans Mono", color="#222222", line_height=0.017)
            continue

        bullet_match = re.match(r"^(\s*)[-*]\s+(.*)", line)
        numbered_match = re.match(r"^(\s*)(\d+)\.\s+(.*)", line)
        if bullet_match:
            content = bullet_match.group(2)
            wrapped = wrap_text(content, 86)
            writer.text(f"- {wrapped[0]}", indent=0.02, line_height=0.021)
            for extra in wrapped[1:]:
                writer.text(f"  {extra}", indent=0.02, line_height=0.021)
            continue
        if numbered_match:
            number = numbered_match.group(2)
            content = numbered_match.group(3)
            wrapped = wrap_text(content, 84)
            writer.text(f"{number}. {wrapped[0]}", indent=0.02, line_height=0.021)
            for extra in wrapped[1:]:
                writer.text(f"   {extra}", indent=0.02, line_height=0.021)
            continue

        for wrapped in wrap_text(line, 92):
            writer.text(wrapped)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("docs/presentation_question_briefing.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/presentation_question_briefing.pdf"))
    args = parser.parse_args()
    render_markdown(args.source, args.output)


if __name__ == "__main__":
    main()
