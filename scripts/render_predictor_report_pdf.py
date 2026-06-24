#!/usr/bin/env python3
"""Render the predictor comparison Markdown report as an illustrated PDF.

The renderer intentionally supports the Markdown constructs used by the report
and keeps the PDF build independent of pandoc. Wide figures are placed on
landscape pages so heatmap labels remain readable.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


IMAGE_RE = re.compile(r"^!\[([^]]*)\]\(([^)]+)\)$")
LINK_RE = re.compile(r"\[([^]]+)\]\((https?://[^)]+)\)")
INLINE_RE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^]]+\]\(https?://[^)]+\))")
TABLE_SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def escape_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def escape_url(value: str) -> str:
    return value.replace("%", r"\%").replace("#", r"\#")


def inline_latex(value: str) -> str:
    parts: list[str] = []
    position = 0
    for match in INLINE_RE.finditer(value):
        parts.append(escape_text(value[position : match.start()]))
        token = match.group(0)
        link = LINK_RE.fullmatch(token)
        if link:
            parts.append(
                rf"\href{{{escape_url(link.group(2))}}}{{{inline_latex(link.group(1))}}}"
            )
        elif token.startswith("**"):
            parts.append(rf"\textbf{{{inline_latex(token[2:-2])}}}")
        elif token.startswith("*"):
            parts.append(rf"\emph{{{inline_latex(token[1:-1])}}}")
        elif token.startswith("`"):
            parts.append(rf"\texttt{{{escape_text(token[1:-1])}}}")
        position = match.end()
    parts.append(escape_text(value[position:]))
    return "".join(parts)


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_table(lines: list[str]) -> str:
    rows = [table_cells(line) for line in lines if not TABLE_SEPARATOR_RE.match(line)]
    if not rows:
        return ""
    columns = max(len(row) for row in rows)
    column_spec = " ".join([r">{\raggedright\arraybackslash}X"] * columns)
    output = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\renewcommand{\arraystretch}{1.18}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{column_spec}}}",
        r"\toprule",
    ]
    for index, row in enumerate(rows):
        padded = row + [""] * (columns - len(row))
        cells = [inline_latex(cell) for cell in padded]
        if index == 0:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        output.append(" & ".join(cells) + r" \\")
        if index == 0:
            output.append(r"\midrule")
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(output)


def render_figure(image_path: Path, caption: str, alt_text: str) -> str:
    with Image.open(image_path) as image:
        aspect_ratio = image.width / image.height
    landscape = aspect_ratio > 1.4
    path = image_path.resolve().as_posix()
    figure = [r"\begin{figure}[H]", r"\centering"]
    if landscape:
        figure.insert(0, r"\begin{landscape}")
        figure.append(
            rf"\includegraphics[width=0.97\linewidth,height=0.76\textheight,keepaspectratio]{{\detokenize{{{path}}}}}"
        )
    else:
        figure.append(
            rf"\includegraphics[width=0.96\linewidth,height=0.68\textheight,keepaspectratio]{{\detokenize{{{path}}}}}"
        )
    figure.append(rf"\caption{{{inline_latex(caption or alt_text)}}}")
    figure.append(r"\end{figure}")
    if landscape:
        figure.append(r"\end{landscape}")
    return "\n".join(figure)


def report_body(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []

    # The first Markdown block is rendered as a dedicated title page.
    title = lines[0].removeprefix("# ").strip()
    subtitle = next(
        (line.removeprefix("## ").strip() for line in lines[1:] if line.startswith("## ")),
        "",
    )
    metadata = []
    first_rule = lines.index("---")
    for line in lines[1:first_rule]:
        if line.startswith("**") and ":**" in line:
            metadata.append(line)
    output.extend(
        [
            r"\begin{titlepage}",
            r"\centering",
            r"\vspace*{2.2cm}",
            rf"{{\Huge\bfseries\color{{reportblue}} {inline_latex(title)}\par}}",
            r"\vspace{1.0cm}",
            rf"{{\Large {inline_latex(subtitle)}\par}}",
            r"\vspace{2.0cm}",
        ]
    )
    for entry in metadata:
        output.append(rf"{{\large {inline_latex(entry)}\par\vspace{{0.5cm}}}}")
    output.extend(
        [
            r"\vfill",
            r"{\large Human-proteome predictor analysis\par}",
            r"\vspace{0.3cm}",
            r"{\large Reproducible report with embedded figures\par}",
            r"\end{titlepage}",
        ]
    )

    index = first_rule + 1
    paragraph: list[str] = []
    list_kind: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            output.append(inline_latex(" ".join(part.strip() for part in paragraph)))
            output.append("\n")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(rf"\end{{{list_kind}}}")
            list_kind = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped == "## Contents":
            flush_paragraph()
            close_list()
            output.extend([r"\tableofcontents", r"\clearpage"])
            index += 1
            while index < len(lines) and lines[index].strip() != "---":
                index += 1
            index += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph()
            close_list()
            image_path = (markdown_path.parent / image_match.group(2)).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Report image not found: {image_path}")
            caption = ""
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines):
                candidate = lines[lookahead].strip()
                if candidate.startswith("*Figure ") and candidate.endswith("*"):
                    caption = re.sub(r"^Figure\s+\d+\.\s*", "", candidate[1:-1])
                    index = lookahead
            output.append(render_figure(image_path, caption, image_match.group(1)))
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            close_list()
            table_lines = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                table_lines.append(candidate)
                index += 1
            output.append(render_table(table_lines))
            continue

        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            command = {2: "section", 3: "subsection", 4: "subsubsection"}[len(heading.group(1))]
            heading_text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading.group(2))
            if heading_text == "Abstract":
                output.append(r"\section*{Abstract}\addcontentsline{toc}{section}{Abstract}")
            else:
                output.append(rf"\{command}{{{inline_latex(heading_text)}}}")
            index += 1
            continue

        bullet = re.match(r"^-\s+(.+)$", stripped)
        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if bullet or numbered:
            flush_paragraph()
            target_kind = "itemize" if bullet else "enumerate"
            if list_kind != target_kind:
                close_list()
                output.append(rf"\begin{{{target_kind}}}")
                list_kind = target_kind
            output.append(rf"\item {inline_latex((bullet or numbered).group(1))}")
            index += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            close_list()
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip("> "))
                index += 1
            output.append(
                r"\begin{quote}\itshape\color{reportblue} "
                + inline_latex(" ".join(quote_lines))
                + r" \end{quote}"
            )
            continue

        if not stripped or stripped == "---":
            flush_paragraph()
            close_list()
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    close_list()
    return "\n".join(output)


def latex_document(body: str) -> str:
    return (
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.15cm,headheight=15pt]{geometry}
\usepackage{fontspec}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{pdflscape}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{hyperref}
\definecolor{reportblue}{HTML}{244A73}
\definecolor{linkblue}{HTML}{1769AA}
\hypersetup{colorlinks=true,linkcolor=reportblue,urlcolor=linkblue,citecolor=reportblue,
  pdftitle={Global Comparison of Protein Intrinsic Disorder Predictors},
  pdfsubject={Agreement and systematic disagreement across the human proteome}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.62em}
\setlength{\emergencystretch}{3em}
\setcounter{tocdepth}{2}
\setlist{itemsep=0.2em,topsep=0.35em,leftmargin=1.8em}
\titleformat{\section}{\Large\bfseries\color{reportblue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\color{reportblue}}{\thesubsection}{0.7em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.7em}{}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Global Predictor Comparison}
\fancyhead[R]{\small Human Proteome}
\fancyfoot[C]{\thepage}
\begin{document}
"""
        + body
        + "\n"
        + r"\end{document}"
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("docs/predictor_comparison_report.md")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/predictor_comparison_report.pdf")
    )
    parser.add_argument("--keep-tex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = shutil.which("xelatex")
    if engine is None:
        raise RuntimeError("xelatex is required to render the report PDF")
    markdown_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = latex_document(report_body(markdown_path))

    with tempfile.TemporaryDirectory(prefix="predictor-report-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        tex_path = temp_dir / "predictor_comparison_report.tex"
        tex_path.write_text(document, encoding="utf-8")
        command = [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        for _ in range(2):
            result = subprocess.run(
                command, cwd=temp_dir, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                log_path = temp_dir / "predictor_comparison_report.log"
                log = log_path.read_text(errors="replace") if log_path.exists() else result.stdout
                raise RuntimeError("xelatex failed:\n" + log[-5000:])
        shutil.copy2(temp_dir / "predictor_comparison_report.pdf", output_path)
        if args.keep_tex:
            output_path.with_suffix(".tex").write_text(document, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
