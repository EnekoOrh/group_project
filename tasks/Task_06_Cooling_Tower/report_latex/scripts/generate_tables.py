import argparse
import csv
import os
from typing import List


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt_float(value: str, digits: int = 2) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def _fmt_sci(value: str, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}e}"
    except ValueError:
        return value


def _fmt_pct01(value: str, digits: int = 1) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{100.0 * float(value):.{digits}f}\\%"
    except ValueError:
        return value


def _write_table(path: str, caption: str, label: str, columns: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{6pt}\n")
        f.write(f"\\begin{{tabular}}{{{'l' * len(columns)}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from Task 6 CSV exports.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join("tasks", "Task_06_Cooling_Tower", "results", "latex"),
        help="Directory containing CSV files exported by run_experiments.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("tasks", "Task_06_Cooling_Tower", "report_latex", "tables"),
        help="Directory where .tex tables are written",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    _ensure_dir(output_dir)

    scenario_csv = os.path.join(input_dir, "scenario_summary_table.csv")
    algo_csv = os.path.join(input_dir, "algorithm_summary_table.csv")
    finding_csv = os.path.join(input_dir, "key_findings_table.csv")

    if not os.path.exists(scenario_csv):
        raise FileNotFoundError(f"Missing input CSV: {scenario_csv}")
    if not os.path.exists(algo_csv):
        raise FileNotFoundError(f"Missing input CSV: {algo_csv}")
    if not os.path.exists(finding_csv):
        raise FileNotFoundError(f"Missing input CSV: {finding_csv}")

    scenario_rows = _read_csv(scenario_csv)
    algo_rows = _read_csv(algo_csv)
    finding_rows = _read_csv(finding_csv)

    scenario_table_rows = []
    for row in scenario_rows:
        scenario_table_rows.append(
            [
                row["scenario_id"],
                row["algorithm"],
                _fmt_float(row["mean_area"], 2),
                _fmt_float(row["std_area"], 2),
                _fmt_sci(row["mean_rel_volume_error"], 3),
                _fmt_float(row["mean_evals"], 1),
                _fmt_float(row["mean_time_s"], 4),
                _fmt_pct01(row["feasibility_rate"], 1),
                _fmt_pct01(row.get("engineering_feasibility_rate", ""), 1),
            ]
        )

    _write_table(
        os.path.join(output_dir, "scenario_summary_table.tex"),
        caption="Scenario-level performance summary across algorithms.",
        label="tab:scenario-summary",
        columns=[
            "Scenario",
            "Algorithm",
            "Mean Area",
            "Std Area",
            "Mean Rel Vol Err",
            "Mean Evals",
            "Mean Time (s)",
            "Math Feasibility",
            "Eng. Feasibility",
        ],
        rows=scenario_table_rows,
    )

    algo_table_rows = []
    for row in algo_rows:
        algo_table_rows.append(
            [
                row["algorithm"],
                row["total_runs"],
                _fmt_float(row["mean_area"], 2),
                _fmt_sci(row["mean_rel_volume_error"], 3),
                _fmt_float(row["mean_evals"], 1),
                _fmt_float(row["mean_time_s"], 4),
                _fmt_pct01(row["overall_feasibility_rate"], 1),
                _fmt_pct01(row.get("overall_engineering_feasibility_rate", ""), 1),
            ]
        )

    _write_table(
        os.path.join(output_dir, "algorithm_summary_table.tex"),
        caption="Cross-scenario algorithm summary.",
        label="tab:algorithm-summary",
        columns=[
            "Algorithm",
            "Total Runs",
            "Mean Area",
            "Mean Rel Vol Err",
            "Mean Evals",
            "Mean Time (s)",
            "Math Feasibility",
            "Eng. Feasibility",
        ],
        rows=algo_table_rows,
    )

    finding_table_rows = []
    for row in finding_rows:
        finding_table_rows.append(
            [
                row["scenario_id"],
                row["best_tradeoff_algorithm"],
                _fmt_pct01(row["best_tradeoff_feasibility_rate"], 1),
                _fmt_float(row["best_tradeoff_mean_area"], 2),
                row["fastest_algorithm"],
                _fmt_float(row["fastest_mean_evals"], 1),
            ]
        )

    _write_table(
        os.path.join(output_dir, "key_findings_table.tex"),
        caption="Per-scenario best feasibility-area tradeoff and fastest algorithm.",
        label="tab:key-findings",
        columns=["Scenario", "Best Tradeoff Algo", "Feasibility", "Mean Area", "Fastest Algo", "Fastest Mean Evals"],
        rows=finding_table_rows,
    )

    print(f"Wrote LaTeX tables to: {output_dir}")


if __name__ == "__main__":
    main()
