from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fitz


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parents[2]
REPORT = ROOT / "Aiert_Task8_Report.md"
OUTPUT = ROOT / "Task8_Aiert_Individual_Report.pdf"
LOGO_DIR = PROJECT_ROOT / "tasks" / "Task_06_Cooling_Tower" / "img"
ESTIA_LOGO = LOGO_DIR / "logo-estia.png"
CRANFIELD_LOGO = LOGO_DIR / "Cranfield_logo.png"
REPORT_DATE = "May 14, 2026"

PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT = 0.105, 0.895
TOP, BOTTOM = 0.895, 0.09
BODY_FONT = "DejaVu Serif"
TITLE_COLOR = "#111111"
FOOTER_COLOR = "#555555"


@dataclass
class Element:
    kind: str
    text: str = ""
    rows: list[list[str]] | None = None


def strip_inline(markdown: str) -> str:
    markdown = re.sub(r"`([^`]+)`", r"\1", markdown)
    markdown = re.sub(r"\*\*([^*]+)\*\*", r"\1", markdown)
    markdown = markdown.replace("->", "->")
    return markdown


def parse_markdown(text: str) -> tuple[str, list[Element]]:
    lines = text.splitlines()
    title = "Task 8 Individual Reflective Report - Aiert Ceccon"
    elements: list[Element] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = strip_inline(raw)

        if raw.startswith("# "):
            title = strip_inline(raw[2:])
        elif raw.startswith("## "):
            elements.append(Element("h2", strip_inline(raw[3:])))
        elif raw.startswith("### "):
            elements.append(Element("h3", strip_inline(raw[4:])))
        elif raw.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            rows = []
            for table_line in table_lines:
                cells = [strip_inline(c.strip()) for c in table_line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                    continue
                rows.append(cells)
            if rows:
                elements.append(Element("table", rows=rows))
            continue
        elif raw.startswith("- "):
            elements.append(Element("bullet", strip_inline(raw[2:])))
        elif raw.strip() == "---":
            elements.append(Element("space"))
        elif raw.strip():
            elements.append(Element("body", line))
        else:
            elements.append(Element("space"))
        i += 1
    return title, elements


def new_page(page_no: int | None = None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if page_no is not None:
        fig.text(0.08, 0.045, "Montrésor group", fontsize=8.5, family=BODY_FONT, color=FOOTER_COLOR)
        fig.text(0.50, 0.045, str(page_no), fontsize=8.5, family=BODY_FONT, color=FOOTER_COLOR, ha="center")
        fig.text(0.92, 0.045, REPORT_DATE, fontsize=8.5, family=BODY_FONT, color=FOOTER_COLOR, ha="right")
        fig.lines.append(plt.Line2D([0.08, 0.92], [0.065, 0.065], color="#777777", linewidth=0.6, transform=fig.transFigure))
    return fig


def add_logo(fig, path: Path, box: list[float]):
    if not path.exists():
        return
    ax = fig.add_axes(box)
    ax.imshow(plt.imread(path))
    ax.axis("off")


def make_title_page() -> plt.Figure:
    fig = new_page(None)
    add_logo(fig, ESTIA_LOGO, [0.08, 0.85, 0.16, 0.08])
    add_logo(fig, CRANFIELD_LOGO, [0.73, 0.80, 0.17, 0.14])
    fig.text(0.5, 0.68, "Task 8 Individual Reflective Report", ha="center", fontsize=19, weight="bold", family=BODY_FONT, color=TITLE_COLOR)
    fig.text(0.5, 0.61, "Group Project Spring 2026", ha="center", fontsize=11.5, family=BODY_FONT, color=TITLE_COLOR)
    fig.text(0.5, 0.50, "Aiert CECCON", ha="center", fontsize=10.5, family=BODY_FONT, color=TITLE_COLOR)
    fig.text(0.5, 0.465, "Montrésor Group", ha="center", fontsize=10.5, family=BODY_FONT, color=TITLE_COLOR)
    return fig


def estimate_height(el: Element) -> float:
    if el.kind == "h2":
        return 0.150
    if el.kind == "h3":
        return 0.042
    if el.kind == "space":
        return 0.014
    if el.kind == "table" and el.rows:
        return min(0.33, 0.040 + 0.035 * len(el.rows))
    width = 92 if el.kind == "body" else 86
    lines = max(1, len(textwrap.wrap(el.text, width=width)))
    return 0.022 * lines + 0.012


def draw_wrapped(fig, x: float, y: float, text: str, *, width: int, fontsize: float, weight: str = "normal", indent: str = "", color: str = "#222222") -> float:
    wrapped = textwrap.wrap(text, width=width) or [""]
    first = True
    for part in wrapped:
        prefix = indent if first else " " * len(indent)
        fig.text(x, y, prefix + part, fontsize=fontsize, weight=weight, family=BODY_FONT, color=color)
        y -= 0.022
        first = False
    return y


def draw_table(fig, y: float, rows: list[list[str]]) -> float:
    headers = rows[0]
    body = rows[1:]
    short_headers = {
        "Technical Quality": "Tech.\nQuality",
        "Delivery Reliability": "Delivery\nReliab.",
        "Integration and Handover": "Integration",
        "Problem-Solving Under Constraints": "Problem-\nSolving",
        "Collaboration Quality": "Collab.",
        "Adaptability and Learning": "Adapt.",
        "Weighted score (/5)": "Weighted\n/5",
    }
    headers = [short_headers.get(h, h) for h in headers]
    table_ax = fig.add_axes([0.07, max(BOTTOM, y - 0.24), 0.86, 0.22])
    table_ax.axis("off")
    table = table_ax.table(cellText=body, colLabels=headers, loc="upper center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    return y - 0.265


def build_body_pages(elements: list[Element], start_page_no: int) -> tuple[list[plt.Figure], dict[str, int]]:
    pages: list[plt.Figure] = []
    section_pages: dict[str, int] = {}
    page_no = start_page_no
    fig = new_page(page_no)
    y = TOP

    for el in elements:
        required = estimate_height(el)
        if y - required < BOTTOM or (el.kind == "h2" and y < 0.31):
            pages.append(fig)
            page_no += 1
            fig = new_page(page_no)
            y = TOP

        if el.kind == "h2":
            section_pages[el.text] = page_no
            y -= 0.010
            fig.text(LEFT, y, el.text, fontsize=15, weight="bold", family=BODY_FONT, color=TITLE_COLOR)
            y -= 0.046
        elif el.kind == "h3":
            fig.text(LEFT, y, el.text, fontsize=11.5, weight="bold", family=BODY_FONT, color=TITLE_COLOR)
            y -= 0.034
        elif el.kind == "body":
            y = draw_wrapped(fig, LEFT, y, el.text, width=95, fontsize=9.1)
            y -= 0.012
        elif el.kind == "bullet":
            y = draw_wrapped(fig, LEFT + 0.025, y, el.text, width=88, fontsize=9.1, indent="• ")
            y -= 0.006
        elif el.kind == "table" and el.rows:
            y = draw_table(fig, y, el.rows)
            y -= 0.018
        else:
            y -= 0.014

    pages.append(fig)
    return pages, section_pages


def make_toc_page(section_pages: dict[str, int]) -> plt.Figure:
    fig = new_page(1)
    fig.text(LEFT, TOP, "Contents", fontsize=18, weight="bold", family=BODY_FONT, color=TITLE_COLOR)
    y = TOP - 0.07
    for section, page in section_pages.items():
        label = re.sub(r"\s+", " ", section).strip()
        fig.text(LEFT, y, label, fontsize=10.2, family=BODY_FONT, color=TITLE_COLOR)
        fig.text(0.88, y, str(page), fontsize=10.2, family=BODY_FONT, color=TITLE_COLOR, ha="right")
        fig.lines.append(plt.Line2D([0.15, 0.85], [y - 0.005, y - 0.005], color="#dddddd", linewidth=0.4, transform=fig.transFigure))
        y -= 0.038
    return fig


def main():
    _title, elements = parse_markdown(REPORT.read_text(encoding="utf-8"))
    body_pages, section_pages = build_body_pages(elements, start_page_no=2)
    with PdfPages(OUTPUT) as pdf:
        pdf.savefig(make_title_page())
        plt.close()
        pdf.savefig(make_toc_page(section_pages))
        plt.close()
        for fig in body_pages:
            pdf.savefig(fig)
            plt.close(fig)
    sanitize_metadata()
    print(OUTPUT)


def sanitize_metadata():
    doc = fitz.open(OUTPUT)
    doc.set_metadata(
        {
            "title": "Task 8 Individual Reflective Report",
            "author": "Aiert Ceccon",
            "subject": "Group Project Spring 2026",
            "keywords": "Task 8; individual reflective report; Montresor group",
            "creator": "Task 8 report source",
            "producer": "PDF report build",
            "creationDate": "D:20260514120000+02'00'",
            "modDate": "D:20260514120000+02'00'",
        }
    )
    tmp = OUTPUT.with_suffix(".sanitized.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(OUTPUT)


if __name__ == "__main__":
    main()
