import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _readable_case_name(case_id):
    mapping = {
        "radius_baseline": "Radius baseline (S2 / BFGS)",
        "height_variant": "Height variant (S6 / BFGS)",
        "joint_reference": "Joint reference (S7 / BFGS)",
    }
    return mapping.get(case_id, case_id)


def _recommended_case(rows):
    comparison_rows = [row for row in rows if row["case_variant"] == "comparison"]
    comparison_rows.sort(
        key=lambda row: (
            -float(row["buckling_factor_1"]),
            float(row["max_displacement_m"]),
            float(row["max_mises_pa"]),
            float(row["task6_area_m2"]),
        )
    )
    return comparison_rows[0]


def _build_comparison_plot(rows, output_path: Path):
    comparison_rows = [row for row in rows if row["case_variant"] == "comparison"]
    labels = [_readable_case_name(row["case_id"]) for row in comparison_rows]
    displacements_mm = [1000.0 * float(row["max_displacement_m"]) for row in comparison_rows]
    stresses_mpa = [float(row["max_mises_pa"]) / 1e6 for row in comparison_rows]
    buckling = [float(row["buckling_factor_1"]) for row in comparison_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#C0392B", "#1F618D", "#117A65"]

    axes[0].bar(labels, displacements_mm, color=colors)
    axes[0].set_title("Max displacement")
    axes[0].set_ylabel("mm")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, stresses_mpa, color=colors)
    axes[1].set_title("Max von Mises stress")
    axes[1].set_ylabel("MPa")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, buckling, color=colors)
    axes[2].set_title("First buckling factor")
    axes[2].set_ylabel("-")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Task 7 Cross-Tower Structural Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _format_table(rows):
    lines = [
        "| Tower | Task 6 area (m^2) | Max disp. (mm) | Max von Mises (MPa) | Base reaction resultant (kN) | First buckling factor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["case_variant"] != "comparison":
            continue
        lines.append(
            "| {tower} | {area:.2f} | {disp:.3f} | {stress:.3f} | {reaction:.2f} | {buckling:.3f} |".format(
                tower=_readable_case_name(row["case_id"]),
                area=float(row["task6_area_m2"]),
                disp=1000.0 * float(row["max_displacement_m"]),
                stress=float(row["max_mises_pa"]) / 1e6,
                reaction=float(row["base_reaction_resultant_n"]) / 1000.0,
                buckling=float(row["buckling_factor_1"]),
            )
        )
    return "\n".join(lines)


def _mesh_section(mesh_rows):
    if not mesh_rows:
        return "Mesh sensitivity was not available."

    lines = [
        "| Metric | Baseline | Refined | Relative change (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in mesh_rows:
        baseline = float(row["baseline"])
        refined = float(row["refined"])
        rel_change = float(row["relative_change_percent"])
        lines.append(f"| {row['metric']} | {baseline:.6g} | {refined:.6g} | {rel_change:.2f} |")
    return "\n".join(lines)


def _feedback_text(rows, recommended):
    comparison_rows = [row for row in rows if row["case_variant"] == "comparison"]
    worst_buckling = min(comparison_rows, key=lambda row: float(row["buckling_factor_1"]))
    worst_displacement = max(comparison_rows, key=lambda row: float(row["max_displacement_m"]))

    lines = []
    if recommended["case_id"] == "joint_reference":
        lines.append(
            "The joint radii-height tower remains the preferred design under structural loading, which supports the Task 6 decision to treat coupled geometry optimization as the leading engineering configuration."
        )
    elif recommended["case_id"] == "radius_baseline":
        lines.append(
            "The simpler radius-only tower outperformed the richer Task 6 variants structurally, which suggests Task 6 should keep a strong bias toward smooth radius control before adding extra height freedom."
        )
    else:
        lines.append(
            "The height-variant tower performed best structurally, which suggests axial segmentation deserves stronger weight in later optimization passes than Task 6 currently gives it."
        )

    if worst_buckling["case_id"] == "height_variant":
        lines.append(
            "The weakest buckling response came from the height-only wide-bound case, so future feasibility checks should continue to penalize abrupt height redistribution even when the geometry stays mathematically compliant."
        )
    if worst_displacement["case_id"] == "radius_baseline":
        lines.append(
            "The largest displacement demand appeared in the radius-only case, which indicates that low shell area alone is not sufficient and that wind response should be included in the final tower selection logic."
        )

    return "\n\n".join(lines)


def _mesh_caution(mesh_rows):
    if not mesh_rows:
        return ""
    for row in mesh_rows:
        if row["metric"] == "buckling_factor_1":
            rel_change = abs(float(row["relative_change_percent"]))
            if rel_change > 10.0:
                return (
                    "The joint-reference mesh check showed a large change in first buckling factor when the mesh was refined. "
                    "That means the current buckling ranking is informative for screening, but it is not yet converged strongly enough to be treated as a final design decision without at least one more refinement pass."
                )
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate Task 7 markdown report from Abaqus outputs.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--cases", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json"))
    parser.add_argument("--summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv"))
    parser.add_argument("--mesh-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_sensitivity.csv"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus"))
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    cases = _load_json(Path(args.cases))
    rows = _load_csv(Path(args.summary_csv))
    mesh_rows = _load_csv(Path(args.mesh_csv)) if Path(args.mesh_csv).exists() else []
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "results" / "report"
    figures_dir = output_dir / "results" / "figures"
    data_dir = output_dir / "results" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _build_comparison_plot(rows, figures_dir / "comparison_metrics.png")
    recommended = _recommended_case(rows)

    q_ref = 0.5 * float(config["wind"]["air_density_kg_m3"]) * float(config["wind"]["reference_speed_m_s"]) ** 2
    table_md = _format_table(rows)
    mesh_md = _mesh_section(mesh_rows)
    feedback_md = _feedback_text(rows, recommended)
    mesh_caution = _mesh_caution(mesh_rows)

    lines = []
    lines.append("# Task 7 Report: Abaqus Wind Loading on Cooling Towers")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This Task 7 study transfers three selected Task 6 cooling-tower geometries into Abaqus shell models in order to compare their structural response under self-weight and one-direction wind loading."
    )
    lines.append(
        "The engineering question is not which geometry had the lowest optimized shell area alone, but which geometry remains the most convincing once displacement, stress, and shell-buckling sensitivity are checked under a common structural load case."
    )
    lines.append("")
    lines.append("## 2. Towers Carried Forward from Task 6")
    lines.append("")
    lines.append("| Case ID | Task 6 source | Why it was kept |")
    lines.append("|---|---|---|")
    for case in cases:
        lines.append(f"| {_readable_case_name(case['case_id'])} | {case['scenario_id']} / {case['algorithm']} | {case['description']} |")
    lines.append("")
    lines.append("![Task 7 candidate profiles](../figures/candidate_profiles.png)")
    lines.append("")
    lines.append("## 3. Material and Loading Assumptions")
    lines.append("")
    lines.append("- Equivalent reinforced concrete: `E = 33 GPa`, `nu = 0.20`, `rho = 2500 kg/m^3`.")
    lines.append("- Constant shell thickness: `0.20 m`.")
    lines.append(
        f"- Reference wind speed: `{config['wind']['reference_speed_m_s']} m/s`, giving `q = {q_ref:.1f} Pa` from `q = 0.5 rho V^2`."
    )
    lines.append(
        f"- Circumferential wind coefficient law: `{config['wind']['pressure_coefficient_model']}` with wind acting from the `{config['wind']['wind_direction_axis']}` direction."
    )
    lines.append("- Base boundary condition: fully fixed support ring at the tower base.")
    lines.append("- Analysis sequence: static response under self-weight plus wind, followed by linear buckling on the preloaded state.")
    lines.append("")
    lines.append("## 4. Abaqus Model Setup")
    lines.append("")
    lines.append(
        "Each tower is meshed as a full 3D shell surface generated from the Task 6 meridian profile. The study uses the same material, shell thickness, and wind basis for all three towers so the structural comparison remains controlled."
    )
    lines.append(
        f"The comparison mesh uses `{config['mesh']['circumferential_divisions']}` circumferential sectors and `{config['mesh']['axial_subdivisions_per_segment']}` axial subdivision per Task 6 frustum segment."
    )
    lines.append("")
    lines.append("## 5. Results Summary")
    lines.append("")
    lines.append("All figures below are rendered from actual Abaqus ODB field data exported with `abaqus python`; they are not screenshots from Abaqus/CAE.")
    lines.append("")
    lines.append(table_md)
    lines.append("")
    lines.append("![Task 7 comparison metrics](../figures/comparison_metrics.png)")
    lines.append("")
    lines.append("## 6. Field Visualizations")
    lines.append("")
    for case in cases:
        case_name = _readable_case_name(case["case_id"])
        job_name = f"task7_{case['case_id']}"
        lines.append(f"### {case_name}")
        lines.append("")
        lines.append(f"![{case_name} stress](../figures/{job_name}_stress.png)")
        lines.append("")
        lines.append(f"![{case_name} displacement](../figures/{job_name}_displacement.png)")
        lines.append("")
        lines.append(f"![{case_name} buckling mode 1](../figures/{job_name}_buckling_mode1.png)")
        lines.append("")
    lines.append("## 7. Mesh Sensitivity")
    lines.append("")
    lines.append(
        "The reference mesh-sensitivity check was run on the joint-reference tower because it is the most realistic candidate carried over from Task 6."
    )
    lines.append("")
    lines.append(mesh_md)
    if mesh_caution:
        lines.append("")
        lines.append(mesh_caution)
    lines.append("")
    lines.append("## 8. Recommendation")
    lines.append("")
    lines.append(
        "The recommended tower is **{name}**. The selection rule is deliberately transparent: prioritize the highest first buckling factor, then the lower maximum displacement, then the lower maximum von Mises stress, and finally the lower Task 6 shell area.".format(
            name=_readable_case_name(recommended["case_id"])
        )
    )
    lines.append("")
    lines.append(
        "Under that rule, `{name}` ranked first with a first buckling factor of `{buckling:.3f}`, a maximum displacement of `{disp:.3f} mm`, and a maximum von Mises stress of `{stress:.3f} MPa`.".format(
            name=_readable_case_name(recommended["case_id"]),
            buckling=float(recommended["buckling_factor_1"]),
            disp=1000.0 * float(recommended["max_displacement_m"]),
            stress=float(recommended["max_mises_pa"]) / 1e6,
        )
    )
    if mesh_caution:
        lines.append("")
        lines.append(
            "This recommendation should therefore be read as a **baseline screening result**, not as a mesh-converged final structural verdict."
        )
    lines.append("")
    lines.append("## 9. Feedback into Task 6")
    lines.append("")
    lines.append(feedback_md)
    lines.append("")
    lines.append("## 10. Limitations and Next Steps")
    lines.append("")
    lines.append(
        "This is a comparative baseline study. The shell is linear elastic, the thickness is constant, and the wind field is represented by an explicit sector-based circumferential pressure law rather than a full code-based site model."
    )
    lines.append(
        "If the project is extended, the next upgrades should be: refined wind action based on a chosen design standard, shell-thickness sensitivity, and possibly geometric or material nonlinearity for the recommended tower only."
    )

    report_text = "\n".join(lines) + "\n"
    report_path = report_dir / "Task7_Report.md"
    root_report_path = output_dir / "Task7_Report.md"
    report_path.write_text(report_text, encoding="utf-8")
    root_report_path.write_text(report_text, encoding="utf-8")
    print("Generated Task 7 markdown report.")


if __name__ == "__main__":
    main()
