import argparse
import os
import re
from typing import List


IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def _escape_latex(text: str) -> str:
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
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _convert_inline(text: str) -> str:
    # Preserve inline code first.
    code_spans: List[str] = []

    def repl_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"@@CODE{len(code_spans)-1}@@"

    text = re.sub(r"`([^`]+)`", repl_code, text)

    # Bold markdown.
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: r"\textbf{" + _escape_latex(m.group(1)) + "}", text)

    # Escape remaining text.
    text = _escape_latex(text)

    # Restore code spans.
    for i, code in enumerate(code_spans):
        text = text.replace(f"@@CODE{i}@@", r"\texttt{" + _escape_latex(code) + "}")

    return text


def _normalize_image_path(path: str) -> str:
    path = path.strip()
    path = path.replace("\\", "/")
    if path.startswith("results/figures/"):
        return "../" + path
    return path


def _render_table(table_lines: List[str]) -> List[str]:
    rows = []
    for line in table_lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return [_convert_inline(line) + r"\\" for line in table_lines]

    header = rows[0]
    body = rows[2:]  # Skip markdown separator row.
    cols = len(header)
    colspec = " | ".join(["p{0.13\\textwidth}"] * cols)

    out = []
    out.append(r"\begin{longtable}{" + colspec + "}")
    out.append(r"\hline")
    out.append(" & ".join(_convert_inline(c) for c in header) + r" \\")
    out.append(r"\hline")
    out.append(r"\endfirsthead")
    out.append(r"\hline")
    out.append(" & ".join(_convert_inline(c) for c in header) + r" \\")
    out.append(r"\hline")
    out.append(r"\endhead")
    for row in body:
        padded = row + [""] * (cols - len(row))
        out.append(" & ".join(_convert_inline(c) for c in padded[:cols]) + r" \\")
    out.append(r"\hline")
    out.append(r"\end{longtable}")
    return out


def convert_markdown_to_tex(md_text: str) -> str:
    lines = md_text.splitlines()
    out: List[str] = []
    in_list = False
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

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
            out.append(r"\chapter{" + _convert_inline(stripped[3:]) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\section{" + _convert_inline(stripped[4:]) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("#### "):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(r"\subsection{" + _convert_inline(stripped[5:]) + "}")
            out.append("")
            i += 1
            continue

        # Images.
        image_match = IMAGE_RE.match(stripped)
        if image_match:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            caption = _convert_inline(image_match.group(1).strip() or "Figure")
            path = _normalize_image_path(image_match.group(2))
            out.append(r"\begin{figure}[H]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=0.95\textwidth]{" + path + "}")
            out.append(r"\caption{" + caption + "}")
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
    parser = argparse.ArgumentParser(description="Sync Task 6 markdown report into a LaTeX section file.")
    parser.add_argument(
        "--input-md",
        type=str,
        default=os.path.join("tasks", "Task_06_Cooling_Tower", "Report.md"),
        help="Path to source markdown report.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=os.path.join("tasks", "Task_06_Cooling_Tower", "report_latex", "sections", "00_report_from_md.tex"),
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
