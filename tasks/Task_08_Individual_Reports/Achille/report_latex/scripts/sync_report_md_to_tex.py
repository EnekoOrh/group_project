import argparse
import os
import re
from typing import List

BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
CODE_RE = re.compile(r"`(.*?)`")
MATH_RE = re.compile(r"\$(.*?)\$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")


def _clean_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
    return text


def _convert_inline(text: str) -> str:
    math_chunks: List[str] = []

    def _stash_math(match: re.Match[str]) -> str:
        math_chunks.append(match.group(1))
        return f"@@MATH{len(math_chunks) - 1}@@"

    text = MATH_RE.sub(_stash_math, text)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    text = text.replace("^", r"\textasciicircum{}")
    text = LINK_RE.sub(lambda m: r"\href{" + m.group(2) + "}{" + m.group(1) + "}", text)
    text = BOLD_RE.sub(r"\\textbf{\1}", text)
    text = CODE_RE.sub(r"\\texttt{\1}", text)

    for idx, chunk in enumerate(math_chunks):
        text = text.replace(f"@@MATH{idx}@@", r"\(" + chunk + r"\)")
    return text


def _render_table(block: List[str]) -> List[str]:
    out: List[str] = []
    if len(block) < 3:
        return out

    headers = [p.strip() for p in block[0].split("|")[1:-1]]
    num_cols = len(headers)
    col_spec = "@{}" + "l" * num_cols + "@{}"

    out.append(r"\begin{table}[H]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.1}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")

    for i, line in enumerate(block):
        if i == 1:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        safe_cells = [_convert_inline(c) for c in cells]
        out.append(" & ".join(safe_cells) + r" \\")
        if i == 0:
            out.append(r"\midrule")

    out.append(r"\bottomrule")
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

        if stripped == "---":
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append("")
            i += 1
            continue

        if stripped.startswith("|"):
            table_block: List[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i])
                i += 1
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.extend(_render_table(table_block))
            out.append("")
            continue

        if stripped.startswith("# "):
            i += 1
            continue
        if stripped.startswith("## "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\Needspace{10\baselineskip}")
            out.append(r"\chapter{" + _convert_inline(_clean_heading(stripped[3:])) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\Needspace{8\baselineskip}")
            out.append(r"\section{" + _convert_inline(_clean_heading(stripped[4:])) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("#### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\Needspace{6\baselineskip}")
            out.append(r"\subsection{" + _convert_inline(_clean_heading(stripped[5:])) + "}")
            out.append("")
            i += 1
            continue

        if stripped.startswith("- "):
            if not in_list:
                out.append(r"\begin{itemize}[leftmargin=1.5em]")
                in_list = True
            out.append(r"\item " + _convert_inline(stripped[2:]))
            i += 1
            continue

        if stripped == "":
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append("")
            i += 1
            continue

        if in_list:
            out.append(r"\end{itemize}")
            in_list = False
        out.append(r"\noindent " + _convert_inline(stripped))
        out.append("")
        i += 1

    if in_list:
        out.append(r"\end{itemize}")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Task 8 markdown report into a LaTeX section file.")
    parser.add_argument(
        "--input-md",
        type=str,
        default=os.path.join("tasks", "Task_08_Individual_Reports", "Achille", "Achille_Task8_Report.md"),
        help="Path to source markdown report.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=os.path.join(
            "tasks",
            "Task_08_Individual_Reports",
            "Achille",
            "report_latex",
            "sections",
            "generated",
            "00_report_from_md.tex",
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
