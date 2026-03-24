import argparse
import os
import re
from typing import List


IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
HEADING_PREFIX_RE = re.compile(r"^\d+(\.\d+)*\.?\s+")


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


def _clean_heading(text: str) -> str:
    return HEADING_PREFIX_RE.sub("", text.strip())


def _is_math_code(code: str) -> bool:
    lowered = code.lower()
    if "/" in code or "\\" in code:
        return False
    if lowered.endswith(".csv") or lowered.endswith(".json") or lowered.endswith(".png") or lowered.endswith(".md"):
        return False
    if lowered.startswith("results/") or lowered.startswith("tasks/"):
        return False

    math_tokens = ["=", "pi", "sqrt", "^", "_", "<=", ">=", "sum(", "A_i", "V_i", "J("]
    return any(tok in code for tok in math_tokens)


def _code_to_math(code: str) -> str:
    expr = code.strip()
    expr = expr.replace("<=", r"\leq")
    expr = expr.replace(">=", r"\geq")
    expr = re.sub(r"\bpi\b", r"\\pi", expr)
    expr = re.sub(r"\bsum\((.*?)\)", r"\\sum(\1)", expr)
    expr = expr.replace("*", r" \cdot ")
    expr = expr.replace("..", r"\ldots ")
    expr = re.sub(r"\b(\d+)e-([0-9]+)\b", r"\1 \\times 10^{-\2}", expr)
    expr = re.sub(r"\b([A-Za-z])_([A-Za-z][A-Za-z0-9]*)\b", r"\1_{\2}", expr)

    # Convert sqrt(...) to \sqrt{...} with shallow balanced parsing.
    while "sqrt(" in expr:
        start = expr.find("sqrt(")
        idx = start + 5
        depth = 1
        end = idx
        while end < len(expr) and depth > 0:
            if expr[end] == "(":
                depth += 1
            elif expr[end] == ")":
                depth -= 1
            end += 1
        if depth != 0:
            break
        inside = expr[idx : end - 1]
        expr = expr[:start] + r"\sqrt{" + inside + "}" + expr[end:]

    return r"\(" + expr + r"\)"


def _convert_inline(text: str) -> str:
    # Preserve inline code first.
    code_spans: List[str] = []
    bold_spans: List[str] = []

    def repl_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"@@CODE{len(code_spans)-1}@@"

    text = re.sub(r"`([^`]+)`", repl_code, text)

    # Preserve bold markdown before escaping to avoid turning \textbf into literal text.
    def repl_bold(match: re.Match[str]) -> str:
        bold_spans.append(match.group(1))
        return f"@@BOLD{len(bold_spans)-1}@@"

    text = re.sub(r"\*\*(.+?)\*\*", repl_bold, text)

    # Escape remaining text.
    text = _escape_latex(text)

    # Restore code spans.
    for i, code in enumerate(code_spans):
        replacement = _code_to_math(code) if _is_math_code(code) else r"\texttt{" + _escape_latex(code) + "}"
        text = text.replace(f"@@CODE{i}@@", replacement)

    # Restore bold spans.
    for i, content in enumerate(bold_spans):
        text = text.replace(f"@@BOLD{i}@@", r"\textbf{" + _escape_latex(content) + "}")

    text = text.replace("m²", r"m$^2$")
    text = text.replace("m³", r"m$^3$")
    text = text.replace("m\u00c2\u00b2", r"m$^2$")
    text = text.replace("m\u00c2\u00b3", r"m$^3$")

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
    colspec = "@{}" + "l" * cols + "@{}"

    out = []
    out.append(r"\begin{table}[H]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{1.1}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{" + colspec + "}")
    out.append(r"\hline")
    out.append(" & ".join(r"\textbf{" + _convert_inline(c) + "}" for c in header) + r" \\")
    out.append(r"\hline")
    for row in body:
        padded = row + [""] * (cols - len(row))
        out.append(" & ".join(_convert_inline(c) for c in padded[:cols]) + r" \\")
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
