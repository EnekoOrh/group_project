import argparse
import os
import re
from typing import List

# Match inline markdown like **text** or `text`
BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
CODE_RE = re.compile(r"`(.*?)`")
# Match mathematical inline like $E = 33 GPa$
MATH_RE = re.compile(r"\$(.*?)\$")
# Standard markdown image ![caption](path)
IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def _clean_heading(text: str) -> str:
    text = text.replace("[Annex]", "").strip()
    return text


def _convert_inline(text: str) -> str:
    text = text.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_").replace("^", r"\textasciicircum{}")
    text = BOLD_RE.sub(r"\\textbf{\1}", text)
    text = CODE_RE.sub(r"\\texttt{\1}", text)
    text = MATH_RE.sub(r"\\(\1\\)", text)
    return text


def _normalize_image_path(path: str) -> str:
    # LaTeX project structure handles paths slightly differently, normalize path so latexmk works
    return path.replace("results/figures/", "../results/figures/")


def _render_table(block: List[str]) -> List[str]:
    out: List[str] = []
    if len(block) < 3:
        return out

    parts = [p.strip() for p in block[0].split("|")[1:-1]]
    num_cols = len(parts)

    out.append(r"\begin{table}[H]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.1}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{@" + "{}" + ("l" * num_cols) + "@{}}")
    out.append(r"\hline")

    for i, line in enumerate(block):
        if i == 1:
            # Skip divider
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        safe_cells = [_convert_inline(c) for c in cells]
        tex_line = " & ".join(safe_cells) + r" \\"
        out.append(tex_line)
        if i == 0:
            out.append(r"\hline")

    out.append(r"\hline")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return out


def convert_markdown_to_tex(md_text: str) -> str:
    lines = md_text.splitlines()
    out: List[str] = []
    in_list = False
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("Generated:"):
            i += 1
            continue

        # Display math blocks.
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\[")
            out.append(stripped[2:-2].strip())
            out.append(r"\]")
            out.append("")
            i += 1
            continue
        if stripped == "$$":
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            display_lines: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                display_lines.append(lines[i].rstrip())
                i += 1
            out.append(r"\[")
            out.extend(display_lines)
            out.append(r"\]")
            out.append("")
            if i < len(lines) and lines[i].strip() == "$$":
                i += 1
            continue

        # Table block.
        if stripped.startswith("|"):
            table_block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i])
                i += 1
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.extend(_render_table(table_block))
            out.append("")
            continue

        # Headings.
        if stripped.startswith("# "):
            # Skip top-level markdown title; LaTeX title page already handles this.
            i += 1
            continue
        if stripped.startswith("## "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\chapter{" + _convert_inline(_clean_heading(stripped[3:])) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\section{" + _convert_inline(_clean_heading(stripped[4:])) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("#### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\subsection{" + _convert_inline(_clean_heading(stripped[5:])) + "}")
            out.append("")
            i += 1
            continue

        # Images.
        if "![" in stripped and "](" in stripped:
            images = re.findall(r"!\[(.*?)\]\((.*?)\)", stripped)
            if images:
                if in_list:
                    out.append(r"\end{itemize}")
                    in_list = False
                out.append(r"\begin{figure}[H]")
                out.append(r"\centering")
                
                if len(images) == 3:
                    for j, (img_caption, img_path) in enumerate(images):
                        img_caption = _convert_inline(img_caption.strip() or "Figure")
                        path = _normalize_image_path(img_path)
                        out.append(r"\begin{subfigure}{0.32\textwidth}")
                        out.append(r"\centering")
                        out.append(r"\includegraphics[width=\textwidth]{" + path + "}")
                        out.append(r"\caption{" + img_caption + "}")
                        out.append(r"\end{subfigure}" + (r"\hfill" if j < 2 else ""))
                    out.append(r"\caption{Combined visual fields}")
                else:
                    for img_caption, img_path in images:
                        img_caption = _convert_inline(img_caption.strip() or "Figure")
                        path = _normalize_image_path(img_path)
                        out.append(r"\includegraphics[width=0.95\textwidth]{" + path + "}")
                        out.append(r"\caption{" + img_caption + "}")
                
                out.append(r"\end{figure}")
                out.append("")
                i += 1
                continue

        # Bullets.
        if stripped.startswith("- "):
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + _convert_inline(stripped[2:]))
            i += 1
            continue

        # Blank line.
        if stripped == "":
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append("")
            i += 1
            continue

        # Normal paragraph text.
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False
        out.append(_convert_inline(stripped))
        out.append("")
        i += 1

    if in_list:
        out.append(r"\end{itemize}")

    return "\n".join(out).rstrip() + "\n"

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Task 7 markdown report into a LaTeX section file.")
    parser.add_argument(
        "--input-md",
        type=str,
        default=os.path.join("tasks", "Task_07_Abaqus", "Task7_Report.md"),
        help="Path to source markdown report.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=os.path.join(
            "tasks", "Task_07_Abaqus", "report_latex", "sections", "generated", "00_report_from_md.tex"
        ),
        help="Path to generated TeX section file.",
    )
    args = parser.parse_args()

    input_md = os.path.abspath(args.input_md)
    output_tex = os.path.abspath(args.output_tex)
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)

    with open(input_md, "r", encoding="utf-8") as f:
        md = f.read()

    tex = convert_markdown_to_tex(md)

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Generated TeX from markdown: {output_tex}")

if __name__ == "__main__":
    main()
