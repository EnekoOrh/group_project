import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task7_common import STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING, status_rank


STATUS_TEXT = {
    STATUS_ENGINEERING: "Engineering-feasible",
    STATUS_MATH_FALLBACK: "Math fallback",
    STATUS_WARNING: "Warning case",
}
STATUS_COLOR = {
    STATUS_ENGINEERING: "#117A65",
    STATUS_MATH_FALLBACK: "#B9770E",
    STATUS_WARNING: "#922B21",
}
METRIC_CONFIG = [
    ("buckling_factor_1", "Buckling factor", True, 0.45),
    ("max_displacement_m", "Max displacement", False, 0.25),
    ("max_mises_pa", "Max stress", False, 0.20),
    ("task6_area_m2", "Task 6 area", False, 0.10),
]
DEFAULT_STATUS_PENALTY = {
    STATUS_ENGINEERING: 0.0,
    STATUS_MATH_FALLBACK: 10.0,
    STATUS_WARNING: 20.0,
}
ABAQUS_LE_NODE_CAP = 1000


def _load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_case(rows):
    grouped = {}
    for row in rows:
        if row["case_variant"] != "comparison":
            continue
        grouped.setdefault(row["case_id"], {})[row["mesh_level"]] = row
    return grouped


def _preferred_row(case_id, row_map):
    case_rows = row_map[case_id]
    return case_rows.get("refined", case_rows.get("coarse"))


def _load_ranking_config(config):
    ranking_cfg = config.get("ranking", {})
    default_weights = {key: weight for key, _, _, weight in METRIC_CONFIG}
    weight_cfg = ranking_cfg.get("weights", {})
    weights = {key: float(weight_cfg.get(key, default_weights[key])) for key, _, _, _ in METRIC_CONFIG}
    status_penalty_cfg = ranking_cfg.get("status_penalty_rank_points", {})
    penalties = {
        STATUS_ENGINEERING: float(status_penalty_cfg.get(STATUS_ENGINEERING, DEFAULT_STATUS_PENALTY[STATUS_ENGINEERING])),
        STATUS_MATH_FALLBACK: float(status_penalty_cfg.get(STATUS_MATH_FALLBACK, DEFAULT_STATUS_PENALTY[STATUS_MATH_FALLBACK])),
        STATUS_WARNING: float(status_penalty_cfg.get(STATUS_WARNING, DEFAULT_STATUS_PENALTY[STATUS_WARNING])),
    }
    return {
        "weights": weights,
        "penalties": penalties,
        "criterion_top_count": int(ranking_cfg.get("criterion_top_count", 3)),
    }


def _metric_rank_map(rows, metric_key, descending):
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[metric_key]) if descending else float(row[metric_key]),
            row["scenario_id"],
            row["algorithm"],
            row["case_id"],
        ),
    )
    return {row["case_id"]: index + 1 for index, row in enumerate(ordered)}


def _weighted_sort_key(entry):
    row = entry["row"]
    return (
        entry["weighted_score"],
        status_rank(row),
        -float(row["buckling_factor_1"]),
        float(row["max_displacement_m"]),
        float(row["max_mises_pa"]),
        float(row["task6_area_m2"]),
        row["case_id"],
    )


def _compute_weighted_ranking(refined_rows, config):
    ranking_cfg = _load_ranking_config(config)
    metric_rank_maps = {}
    for metric_key, _, descending, _ in METRIC_CONFIG:
        metric_rank_maps[metric_key] = _metric_rank_map(refined_rows, metric_key, descending)

    ranked_entries = []
    for row in refined_rows:
        case_id = row["case_id"]
        metric_ranks = {metric_key: metric_rank_maps[metric_key][case_id] for metric_key, _, _, _ in METRIC_CONFIG}
        weighted_metric_score = sum(ranking_cfg["weights"][metric_key] * metric_ranks[metric_key] for metric_key, _, _, _ in METRIC_CONFIG)
        penalty_points = ranking_cfg["penalties"][row["selection_status"]]
        weighted_score = weighted_metric_score + penalty_points
        ranked_entries.append(
            {
                "row": row,
                "metric_ranks": metric_ranks,
                "weighted_metric_score": weighted_metric_score,
                "penalty_points": penalty_points,
                "weighted_score": weighted_score,
            }
        )

    ranked_entries.sort(key=_weighted_sort_key)
    for index, entry in enumerate(ranked_entries, start=1):
        entry["overall_rank"] = index
    return ranked_entries


def _criterion_top_entries(refined_rows, top_count, allowed_statuses: Optional[Sequence[str]] = None):
    filtered_rows = list(refined_rows)
    if allowed_statuses is not None:
        allowed = set(allowed_statuses)
        filtered_rows = [row for row in refined_rows if row["selection_status"] in allowed]

    criterion_entries = []
    for metric_key, label, descending, _ in METRIC_CONFIG:
        ordered = sorted(
            filtered_rows,
            key=lambda row: (
                -float(row[metric_key]) if descending else float(row[metric_key]),
                status_rank(row),
                row["case_id"],
            ),
        )
        for rank_index, row in enumerate(ordered[:top_count], start=1):
            criterion_entries.append(
                {
                    "criterion_key": metric_key,
                    "criterion_label": label,
                    "rank": rank_index,
                    "row": row,
                }
            )
    return criterion_entries


def _format_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _selection_matrix(cases):
    rows = []
    for case in cases:
        rows.append(
            [
                case["label"],
                STATUS_TEXT[case["selection_status"]],
                case["selection_basis"],
                str(case["source_run_index"]),
                case["decision_mode"],
                f"{case['task6_area_m2']:.2f}",
            ]
        )
    return _format_table(["Case", "Status", "Selection basis", "Task 6 run", "Decision mode", "Task 6 area (m^2)"], rows)


def _refined_table(refined_rows):
    rows = []
    for row in refined_rows:
        rows.append(
            [
                row["case_label"],
                STATUS_TEXT[row["selection_status"]],
                f"{float(row['task6_area_m2']):.2f}",
                f"{1000.0 * float(row['max_displacement_m']):.3f}",
                f"{float(row['max_mises_pa']) / 1e6:.3f}",
                f"{float(row['buckling_factor_1']):.3f}",
            ]
        )
    return _format_table(["Case", "Status", "Task 6 area (m^2)", "Max disp. (mm)", "Max stress (MPa)", "Buckling factor"], rows)


def _scenario_table(rows, ranking_by_case):
    body = []
    for row in rows:
        rank_entry = ranking_by_case[row["case_id"]]
        body.append(
            [
                row["algorithm"],
                STATUS_TEXT[row["selection_status"]],
                f"{rank_entry['weighted_score']:.3f}",
                f"{float(row['task6_area_m2']):.2f}",
                f"{1000.0 * float(row['max_displacement_m']):.3f}",
                f"{float(row['max_mises_pa']) / 1e6:.3f}",
                f"{float(row['buckling_factor_1']):.3f}",
            ]
        )
    return _format_table(
        ["Algorithm", "Status", "Weighted score", "Task 6 area (m^2)", "Max disp. (mm)", "Max stress (MPa)", "Buckling factor"],
        body,
    )


def _convergence_table(rows):
    body = []
    for row in rows:
        body.append(
            [
                row["case_label"],
                STATUS_TEXT[row["selection_status"]],
                f"{float(row['delta_max_displacement_percent']):.2f}",
                row["pass_max_displacement"],
                f"{float(row['delta_max_mises_percent']):.2f}",
                row["pass_max_mises"],
                f"{float(row['delta_buckling_factor_1_percent']):.2f}",
                row["pass_buckling_factor_1"],
                row["all_criteria_pass"],
            ]
        )
    return _format_table(
        ["Case", "Status", "Delta disp. (%)", "Pass", "Delta stress (%)", "Pass", "Delta buckling (%)", "Pass", "All pass"],
        body,
    )


def _geometry_summary(rows):
    by_mesh = defaultdict(list)
    for row in rows:
        by_mesh[row["mesh_level"]].append(row)
    body = []
    for mesh_level in ["coarse", "refined"]:
        subset = by_mesh.get(mesh_level, [])
        if not subset:
            continue
        body.append(
            [
                mesh_level,
                f"{max(float(row['abs_height_diff_m']) for row in subset):.6e}",
                f"{max(float(row['max_source_ring_radius_abs_diff_m']) for row in subset):.6e}",
                f"{max(float(row['max_source_ring_z_abs_diff_m']) for row in subset):.6e}",
            ]
        )
    return _format_table(["Mesh", "Max total-height diff (m)", "Max ring-radius diff (m)", "Max ring-z diff (m)"], body)


def _mesh_policy_summary(rows, config):
    if not rows:
        return "A uniform refined mesh policy is applied to all refined jobs."

    refined_rows = [row for row in rows if row["mesh_level"] == "refined"]
    if not refined_rows:
        return "A uniform refined mesh policy is applied to all refined jobs."

    refined_theta = int(config["mesh"]["refined_circumferential_divisions"])
    refined_axial = int(config["mesh"]["refined_axial_subdivisions_per_segment"])
    refined_nodes = [int(row["nodes"]) for row in refined_rows]
    min_nodes = min(refined_nodes)
    max_nodes = max(refined_nodes)

    return (
        f"The refined study uses one uniform high mesh policy for all 24 towers (`{refined_theta}` circumferential divisions, `{refined_axial}` axial subdivisions per segment). "
        f"Refined node counts range from `{min_nodes}` to `{max_nodes}` nodes, which keeps every case within the Abaqus Learning Edition limit while using the highest common density."
    )


def _scenario_commentary_lines(rows, ranking_by_case):
    ranked = sorted(rows, key=lambda row: (ranking_by_case[row["case_id"]]["weighted_score"], row["case_id"]))
    winner = ranked[0]
    winner_entry = ranking_by_case[winner["case_id"]]
    comments = [f"**Winner:** {winner['case_label']} with weighted score `{winner_entry['weighted_score']:.3f}`."]
    if len(ranked) > 1:
        runner_up = ranked[1]
        runner_entry = ranking_by_case[runner_up["case_id"]]
        score_gap = runner_entry["weighted_score"] - winner_entry["weighted_score"]
        comments.append(f"The score gap to second place ({runner_up['case_label']}) is `{score_gap:.3f}` points.")
    else:
        score_gap = 0.0
    disp_values = [1000.0 * float(row["max_displacement_m"]) for row in rows]
    stress_values = [float(row["max_mises_pa"]) / 1e6 for row in rows]
    buck_values = [float(row["buckling_factor_1"]) for row in rows]
    comments.append(
        "**Spread/Risk:** displacement spans `{:.3f}` to `{:.3f}` mm, stress spans `{:.3f}` to `{:.3f}` MPa, and buckling spans `{:.3f}` to `{:.3f}`.".format(
            min(disp_values),
            max(disp_values),
            min(stress_values),
            max(stress_values),
            min(buck_values),
            max(buck_values),
        )
    )
    fallbacks = [row["algorithm"] for row in rows if row["selection_status"] == STATUS_MATH_FALLBACK]
    warnings = [row["algorithm"] for row in rows if row["selection_status"] == STATUS_WARNING]
    if fallbacks:
        comments.append(f"**Status caveat:** mathematical fallback entries are present for {', '.join(fallbacks)}.")
    elif warnings:
        comments.append(f"**Status caveat:** warning-only entries are present for {', '.join(warnings)} and are not decision-grade designs.")
    else:
        comments.append("**Status caveat:** all three entries are engineering-feasible.")

    mixed_status = len({row["selection_status"] for row in rows}) > 1
    if mixed_status:
        comments.append(
            "**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs."
        )
    elif score_gap < 1.0:
        comments.append(
            "**Critical note:** this scenario is tightly clustered, so small modeling shifts can reorder first and second place."
        )
    else:
        comments.append(
            "**Critical note:** the winner remains robust inside this scenario under the current weighted scoring contract."
        )
    return comments


def _weighted_top_table(top_entries):
    rows = []
    for entry in top_entries:
        row = entry["row"]
        rows.append(
            [
                str(entry["overall_rank"]),
                row["case_label"],
                STATUS_TEXT[row["selection_status"]],
                f"{entry['weighted_score']:.3f}",
                f"{entry['penalty_points']:.1f}",
                f"{float(row['buckling_factor_1']):.3f}",
                f"{1000.0 * float(row['max_displacement_m']):.3f}",
                f"{float(row['max_mises_pa']) / 1e6:.3f}",
                f"{float(row['task6_area_m2']):.2f}",
            ]
        )
    return _format_table(
        ["Rank", "Case", "Status", "Weighted score", "Penalty", "Buckling factor", "Max disp. (mm)", "Max stress (MPa)", "Area (m^2)"],
        rows,
    )


def _criterion_top_table(criterion_entries):
    metric_order = {metric_key: index for index, (metric_key, _, _, _) in enumerate(METRIC_CONFIG)}
    rows = []
    for entry in sorted(
        criterion_entries, key=lambda item: (metric_order[item["criterion_key"]], item["rank"], item["row"]["case_id"])
    ):
        metric_key = entry["criterion_key"]
        row = entry["row"]
        if metric_key == "max_displacement_m":
            value_text = f"{1000.0 * float(row[metric_key]):.3f} mm"
        elif metric_key == "max_mises_pa":
            value_text = f"{float(row[metric_key]) / 1e6:.3f} MPa"
        elif metric_key == "task6_area_m2":
            value_text = f"{float(row[metric_key]):.2f} m^2"
        else:
            value_text = f"{float(row[metric_key]):.3f}"
        rows.append(
            [
                entry["criterion_label"],
                str(entry["rank"]),
                row["case_label"],
                STATUS_TEXT[row["selection_status"]],
                value_text,
            ]
        )
    return _format_table(["Criterion", "Rank", "Case", "Status", "Value"], rows)


def _write_weighted_ranking_csv(path, ranked_entries):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "overall_rank",
                "case_id",
                "case_label",
                "scenario_id",
                "algorithm",
                "selection_status",
                "weighted_score",
                "weighted_metric_score",
                "penalty_points",
                "rank_buckling_factor_1",
                "rank_max_displacement_m",
                "rank_max_mises_pa",
                "rank_task6_area_m2",
                "buckling_factor_1",
                "max_displacement_m",
                "max_mises_pa",
                "task6_area_m2",
            ]
        )
        for entry in ranked_entries:
            row = entry["row"]
            writer.writerow(
                [
                    entry["overall_rank"],
                    row["case_id"],
                    row["case_label"],
                    row["scenario_id"],
                    row["algorithm"],
                    row["selection_status"],
                    f"{entry['weighted_score']:.6f}",
                    f"{entry['weighted_metric_score']:.6f}",
                    f"{entry['penalty_points']:.6f}",
                    entry["metric_ranks"]["buckling_factor_1"],
                    entry["metric_ranks"]["max_displacement_m"],
                    entry["metric_ranks"]["max_mises_pa"],
                    entry["metric_ranks"]["task6_area_m2"],
                    f"{float(row['buckling_factor_1']):.6f}",
                    f"{float(row['max_displacement_m']):.12f}",
                    f"{float(row['max_mises_pa']):.12f}",
                    f"{float(row['task6_area_m2']):.6f}",
                ]
            )


def _write_criterion_top_csv(path, criterion_entries):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["criterion_key", "criterion_label", "rank", "case_id", "case_label", "selection_status", "value"])
        for entry in criterion_entries:
            metric_key = entry["criterion_key"]
            row = entry["row"]
            value = float(row[metric_key])
            writer.writerow(
                [
                    metric_key,
                    entry["criterion_label"],
                    entry["rank"],
                    row["case_id"],
                    row["case_label"],
                    row["selection_status"],
                    f"{value:.12f}",
                ]
            )


def _solver_mesh_context(mesh_summary_rows):
    refined_rows = [row for row in mesh_summary_rows if row.get("mesh_level") == "refined"]
    if not refined_rows:
        return None

    nodes = [int(row["nodes"]) for row in refined_rows]
    return {
        "case_count": len(refined_rows),
        "node_min": min(nodes),
        "node_max": max(nodes),
        "within_le_cap": max(nodes) <= ABAQUS_LE_NODE_CAP,
    }


def _load_cae_audit(cae_audit_json_path: Path):
    if not cae_audit_json_path.exists():
        return None
    return _load_json(cae_audit_json_path)


def _run_report_consistency_audit(
    *,
    cases,
    refined_rows,
    convergence_rows,
    ranked_entries,
    criterion_entries_raw,
    criterion_entries_engineering,
    weighted_csv_path: Path,
    criterion_csv_path: Path,
    criterion_engineering_csv_path: Path,
    mesh_summary_rows,
    cae_audit_json_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
):
    checks: List[Dict[str, str]] = []

    def add_check(name: str, passed: bool, message: str):
        checks.append({"check_name": name, "status": "pass" if passed else "fail", "message": message})

    add_check(
        "case_count_matches_selection",
        len(refined_rows) == len(cases),
        f"refined_rows={len(refined_rows)}, selected_cases={len(cases)}",
    )

    add_check(
        "convergence_rows_cover_all_cases",
        len(convergence_rows) == len(cases),
        f"convergence_rows={len(convergence_rows)}, selected_cases={len(cases)}",
    )

    weighted_csv_rows = _load_csv(weighted_csv_path)
    weighted_ids_csv = [row["case_id"] for row in weighted_csv_rows]
    weighted_ids_expected = [entry["row"]["case_id"] for entry in ranked_entries]
    add_check(
        "weighted_csv_matches_computed_order",
        weighted_ids_csv == weighted_ids_expected,
        f"csv_rows={len(weighted_ids_csv)}, expected_rows={len(weighted_ids_expected)}",
    )

    raw_csv_rows = _load_csv(criterion_csv_path)
    raw_rows_expected = [
        (entry["criterion_key"], str(entry["rank"]), entry["row"]["case_id"]) for entry in criterion_entries_raw
    ]
    raw_rows_csv = [(row["criterion_key"], row["rank"], row["case_id"]) for row in raw_csv_rows]
    add_check(
        "criterion_top3_raw_matches_expected",
        sorted(raw_rows_csv) == sorted(raw_rows_expected),
        f"csv_rows={len(raw_rows_csv)}, expected_rows={len(raw_rows_expected)}",
    )

    eng_csv_rows = _load_csv(criterion_engineering_csv_path)
    eng_rows_expected = [
        (entry["criterion_key"], str(entry["rank"]), entry["row"]["case_id"]) for entry in criterion_entries_engineering
    ]
    eng_rows_csv = [(row["criterion_key"], row["rank"], row["case_id"]) for row in eng_csv_rows]
    add_check(
        "criterion_top3_engineering_matches_expected",
        sorted(eng_rows_csv) == sorted(eng_rows_expected),
        f"csv_rows={len(eng_rows_csv)}, expected_rows={len(eng_rows_expected)}",
    )

    add_check(
        "criterion_top3_engineering_status_only_engineering",
        all(row["selection_status"] == STATUS_ENGINEERING for row in eng_csv_rows),
        "engineering-only criterion leaderboard contains only engineering-feasible cases",
    )

    solver_mesh = _solver_mesh_context(mesh_summary_rows)
    if solver_mesh is None:
        add_check("solver_mesh_context_present", False, "No refined rows found in mesh_summary.csv")
    else:
        add_check(
            "solver_mesh_within_le_cap",
            solver_mesh["within_le_cap"],
            f"solver refined node range = {solver_mesh['node_min']}..{solver_mesh['node_max']} (LE cap={ABAQUS_LE_NODE_CAP})",
        )

    cae_audit = _load_cae_audit(cae_audit_json_path)
    if cae_audit is None:
        add_check("cae_audit_present", False, f"Missing {cae_audit_json_path}")
    else:
        cae_min = int(cae_audit.get("node_count_min", -1))
        cae_max = int(cae_audit.get("node_count_max", -1))
        add_check(
            "cae_audit_status_pass",
            cae_audit.get("status") == "pass",
            f"cae status={cae_audit.get('status')}, node range={cae_min}..{cae_max}",
        )
        if solver_mesh is not None:
            envelopes_solver = cae_min <= solver_mesh["node_min"] and cae_max >= solver_mesh["node_max"]
            add_check(
                "mesh_narrative_non_contradictory",
                envelopes_solver,
                f"solver refined node range={solver_mesh['node_min']}..{solver_mesh['node_max']} ; CAE integrity range={cae_min}..{cae_max}",
            )

    all_pass = all(item["status"] == "pass" for item in checks)
    summary = {
        "status": "pass" if all_pass else "fail",
        "checks_total": len(checks),
        "checks_failed": sum(1 for item in checks if item["status"] != "pass"),
        "checks": checks,
        "solver_mesh_refined": solver_mesh,
        "cae_audit_path": str(cae_audit_json_path),
    }
    output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_name", "status", "message"])
        writer.writeheader()
        writer.writerows(checks)

    if not all_pass:
        failed = [item["check_name"] for item in checks if item["status"] != "pass"]
        raise ValueError("Task 7 report consistency audit failed: " + ", ".join(failed))

    return summary


def _plot_weighted_ranking(ranked_entries, output_path):
    labels = [entry["row"]["case_label"] for entry in ranked_entries]
    values = [entry["weighted_score"] for entry in ranked_entries]
    colors = [STATUS_COLOR[entry["row"]["selection_status"]] for entry in ranked_entries]

    fig, axis = plt.subplots(figsize=(12.5, 9.5))
    y_positions = list(range(len(ranked_entries)))
    axis.barh(y_positions, values, color=colors)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Weighted score (lower is better)")
    axis.set_title("Task 7 global weighted ranking (refined models)")
    axis.grid(axis="x", alpha=0.25)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLOR[key], label=STATUS_TEXT[key]) for key in [STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING]]
    axis.legend(handles=legend_handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _criterion_value_for_plot(entry):
    metric_key = entry["criterion_key"]
    value = float(entry["row"][metric_key])
    if metric_key == "max_displacement_m":
        return value * 1000.0, "mm"
    if metric_key == "max_mises_pa":
        return value / 1e6, "MPa"
    if metric_key == "task6_area_m2":
        return value, "m^2"
    return value, "-"


def _plot_criterion_top(criterion_entries, output_path):
    grouped = defaultdict(list)
    for entry in criterion_entries:
        grouped[entry["criterion_key"]].append(entry)

    metric_sequence = [key for key, _, _, _ in METRIC_CONFIG]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    axes_flat = list(axes.flatten())
    for axis, metric_key in zip(axes_flat, metric_sequence):
        entries = sorted(grouped[metric_key], key=lambda item: item["rank"])
        labels = [item["row"]["case_label"] for item in entries]
        values = []
        unit = "-"
        for item in entries:
            converted, unit = _criterion_value_for_plot(item)
            values.append(converted)
        colors = [STATUS_COLOR[item["row"]["selection_status"]] for item in entries]
        axis.bar(labels, values, color=colors)
        axis.set_title(next(label for key, label, _, _ in METRIC_CONFIG if key == metric_key))
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Top-3 by criterion (refined models)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Task 7 markdown report from Abaqus outputs.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--cases", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json"))
    parser.add_argument("--summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv"))
    parser.add_argument("--mesh-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_sensitivity.csv"))
    parser.add_argument("--mesh-summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_summary.csv"))
    parser.add_argument("--geometry-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "geometry_verification.csv"))
    parser.add_argument("--cae-audit-json", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "cae_integrity_audit.json"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus"))
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    cases = _load_json(Path(args.cases))
    rows = _load_csv(Path(args.summary_csv))
    convergence_rows = _load_csv(Path(args.mesh_csv)) if Path(args.mesh_csv).exists() else []
    mesh_summary_rows = _load_csv(Path(args.mesh_summary_csv)) if Path(args.mesh_summary_csv).exists() else []
    geometry_rows = _load_csv(Path(args.geometry_csv)) if Path(args.geometry_csv).exists() else []
    row_map = _rows_by_case(rows)
    output_dir = Path(args.output_dir)
    results_data_dir = output_dir / "results" / "data"
    figures_dir = output_dir / "results" / "figures"
    report_dir = output_dir / "results" / "report"
    results_data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ordered_case_ids = [case["case_id"] for case in cases]
    refined_rows = [_preferred_row(case_id, row_map) for case_id in ordered_case_ids]
    ranked_entries = _compute_weighted_ranking(refined_rows, config)
    ranking_by_case = {entry["row"]["case_id"]: entry for entry in ranked_entries}
    ranking_cfg = _load_ranking_config(config)
    criterion_entries = _criterion_top_entries(refined_rows, ranking_cfg["criterion_top_count"])
    criterion_entries_engineering = _criterion_top_entries(
        refined_rows, ranking_cfg["criterion_top_count"], allowed_statuses=[STATUS_ENGINEERING]
    )
    top3_overall = ranked_entries[:3]
    winner = top3_overall[0]["row"]

    weighted_csv_path = results_data_dir / "weighted_ranking.csv"
    criterion_csv_path = results_data_dir / "criterion_top3.csv"
    criterion_engineering_csv_path = results_data_dir / "criterion_top3_engineering.csv"
    weighted_figure_path = figures_dir / "task7_weighted_ranking.png"
    criterion_figure_path = figures_dir / "task7_criterion_top3.png"
    criterion_engineering_figure_path = figures_dir / "task7_criterion_top3_engineering.png"
    consistency_audit_json_path = results_data_dir / "report_consistency_audit.json"
    consistency_audit_csv_path = results_data_dir / "report_consistency_audit.csv"
    _write_weighted_ranking_csv(weighted_csv_path, ranked_entries)
    _write_criterion_top_csv(criterion_csv_path, criterion_entries)
    _write_criterion_top_csv(criterion_engineering_csv_path, criterion_entries_engineering)
    _plot_weighted_ranking(ranked_entries, weighted_figure_path)
    _plot_criterion_top(criterion_entries, criterion_figure_path)
    _plot_criterion_top(criterion_entries_engineering, criterion_engineering_figure_path)

    consistency_audit = _run_report_consistency_audit(
        cases=cases,
        refined_rows=refined_rows,
        convergence_rows=convergence_rows,
        ranked_entries=ranked_entries,
        criterion_entries_raw=criterion_entries,
        criterion_entries_engineering=criterion_entries_engineering,
        weighted_csv_path=weighted_csv_path,
        criterion_csv_path=criterion_csv_path,
        criterion_engineering_csv_path=criterion_engineering_csv_path,
        mesh_summary_rows=mesh_summary_rows,
        cae_audit_json_path=Path(args.cae_audit_json),
        output_json_path=consistency_audit_json_path,
        output_csv_path=consistency_audit_csv_path,
    )

    q_ref = 0.5 * float(config["wind"]["air_density_kg_m3"]) * float(config["wind"]["reference_speed_m_s"]) ** 2
    engineering_count = sum(1 for case in cases if case["selection_status"] == STATUS_ENGINEERING)
    fallback_count = sum(1 for case in cases if case["selection_status"] == STATUS_MATH_FALLBACK)
    warning_count = sum(1 for case in cases if case["selection_status"] == STATUS_WARNING)
    converged_count = sum(1 for row in convergence_rows if row["all_criteria_pass"] == "yes")
    provisional_winner = converged_count < len(convergence_rows)

    lines = []
    lines.append("# Task 7 Report: Abaqus Wind Loading on Cooling Towers")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This Task 7 study expands the previous three-tower screening into a full structural comparison across the eight Task 6 scenarios and the three optimization algorithms SA, PSO, and BFGS."
    )
    lines.append(
        "The engineering objective is to identify which Task 6 concepts remain convincing once the towers are checked under self-weight, one-direction wind loading, and linear buckling in Abaqus."
    )
    lines.append("")
    lines.append("## 2. Selection Rule and Scenario Matrix")
    lines.append("")
    lines.append(
        f"The Task 7 matrix contains **{len(cases)} refined presentation models**. For scenarios S1 to S7, the preferred choice is the lowest-area engineering-feasible Task 6 run for each scenario/algorithm pair. If no engineering-feasible run exists, the workflow falls back to the lowest-area mathematically compliant run. If neither exists, the best penalized run is kept only as a warning case. Scenario S8 is intentionally carried as warning-only because it has no compliant Task 6 solution in the current study."
    )
    lines.append("")
    lines.append(
        f"Across the 24 selected cases, there are **{engineering_count} engineering-feasible selections**, **{fallback_count} mathematical fallbacks**, and **{warning_count} warning-only cases**."
    )
    lines.append("")
    lines.append(_selection_matrix(cases))
    lines.append("")
    lines.append("![Task 7 candidate profiles](../figures/candidate_profiles.png)")
    lines.append("")
    lines.append("## 3. Material, Loading, and CAE Setup")
    lines.append("")
    lines.append(f"The structural baseline keeps the same assumptions for every model: equivalent reinforced concrete with `E = 33 GPa`, `nu = 0.20`, and `rho = 2500 kg/m^3`, together with a constant shell thickness of `0.20 m`.")
    lines.append(f"The reference wind speed is `{config['wind']['reference_speed_m_s']} m/s`, which gives a dynamic pressure of `{q_ref:.1f} Pa` through `q = 0.5 rho V^2`.")
    lines.append(
        "Wind loading is interpreted as one-direction external action: windward sectors receive positive external pressure and leeward sectors receive suction. This is not internal cabin-style pressurization."
    )
    lines.append("The directional pressure model is:")
    lines.append("$$")
    lines.append("C_p(\\theta) = \\mathrm{clip}(0.8\\cos\\theta, -0.5, 0.8), \\quad p(\\theta) = q\\,C_p(\\theta)")
    lines.append("$$")
    lines.append("Task 7 uses a strict `Y`-up coordinate convention: `Y` is vertical, the horizontal wind plane is `X/Z`, and self-weight acts along `-Y`.")
    lines.append("The clean user-facing Abaqus/CAE database contains the **24 refined models only**, with one gravity load and one wind load per model. The automated solver decks still use the same circumferential pressure law in sectorized form for coarse-versus-refined verification.")
    lines.append(_mesh_policy_summary(mesh_summary_rows, config))
    solver_mesh = consistency_audit.get("solver_mesh_refined") or {}
    lines.append(
        f"Solver-deck mesh compliance (`mesh_summary.csv`) is tracked separately from CAE integrity: refined solver decks span `{solver_mesh.get('node_min', 'n/a')}` to `{solver_mesh.get('node_max', 'n/a')}` nodes and remain within the Abaqus LE cap."
    )
    cae_audit = _load_cae_audit(Path(args.cae_audit_json))
    if cae_audit:
        lines.append(
            f"CAE-native integrity (`cae_integrity_audit.json`) reports a model-node range of `{cae_audit.get('node_count_min')}` to `{cae_audit.get('node_count_max')}`. This audit validates model-tree integrity and CAE consistency, not LE solve-cap compliance."
        )
    lines.append("A deterministic input-deck audit runs before solving and confirms that all 48 decks use identical material constants, gravity definition, and wind-pressure sector logic.")
    lines.append("The towers look squat in Abaqus because the geometry is genuinely squat: the Task 6 cooling towers are about `36.5 m` tall for a base diameter of about `78.6 m`, so the aspect ratio is below one. This is source geometry, not an Abaqus distortion.")
    lines.append("")
    lines.append("## 4. Geometry Integrity Check")
    lines.append("")
    lines.append("The Task 7 generator preserves the exact Task 6 meridian coordinates. A geometry verification pass is written before solving and checks the mesh rings against the selected Task 6 source profile for every coarse and refined job.")
    lines.append("")
    lines.append(_geometry_summary(geometry_rows))
    lines.append("")
    lines.append("## 5. Refined Structural Comparison Across All 24 Cases")
    lines.append("")
    lines.append(_refined_table(refined_rows))
    lines.append("")
    lines.append("![Task 7 refined comparison metrics](../figures/comparison_metrics.png)")
    lines.append("")
    lines.append("## 6. Methodological Clarity and Global Ranking")
    lines.append("")
    lines.append("### 6.1 Methodological Clarity")
    lines.append("")
    lines.append("The global ranking is a rank-based weighted score on the refined results, designed to stay robust when one or two cases have extreme values.")
    lines.append("For each metric, rank `1` is best and rank `24` is worst, with deterministic tie-breaks by scenario/algorithm identifiers.")
    lines.append("")
    lines.append("$$")
    lines.append("R_{k,i} = \\\\text{rank of case } i \\\\text{ on metric } k")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append("S_i = 0.45 R_{\\\\mathrm{buckling},i} + 0.25 R_{\\\\mathrm{disp},i} + 0.20 R_{\\\\mathrm{stress},i} + 0.10 R_{\\\\mathrm{area},i}")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append("J_i = S_i + P_i, \\\\quad P_i \\\\in \\\\{0,10,20\\\\}")
    lines.append("$$")
    lines.append("")
    lines.append("Penalty points are `0` for engineering-feasible, `10` for mathematical fallback, and `20` for warning cases. The best model is the one with the lowest `J_i`.")
    lines.append("")
    lines.append("### 6.2 Global Weighted Top-3")
    lines.append("")
    lines.append(_weighted_top_table(top3_overall))
    lines.append("")
    lines.append("![Task 7 weighted ranking](../figures/task7_weighted_ranking.png)")
    lines.append("")
    lines.append("### 6.3 Top-3 by Criterion")
    lines.append("")
    lines.append("#### 6.3.1 Raw Top-3 (all statuses)")
    lines.append("")
    lines.append("This raw criterion table keeps all statuses visible, so warning and fallback cases can appear if they are numerically extreme on a single indicator.")
    lines.append("")
    lines.append(_criterion_top_table(criterion_entries))
    lines.append("")
    lines.append("![Task 7 criterion leaders](../figures/task7_criterion_top3.png)")
    lines.append("")
    lines.append("#### 6.3.2 Engineering-eligible Top-3")
    lines.append("")
    lines.append("This decision-grade table filters to engineering-feasible entries only.")
    lines.append("")
    lines.append(_criterion_top_table(criterion_entries_engineering))
    lines.append("")
    lines.append("![Task 7 engineering criterion leaders](../figures/task7_criterion_top3_engineering.png)")
    lines.append("")
    lines.append("## 7. Scenario-by-Scenario Comparison")
    lines.append("")
    for scenario_id in config["selection"]["scenario_ids"]:
        scenario_rows = [row for row in refined_rows if row["scenario_id"] == scenario_id]
        scenario_rows.sort(key=lambda row: row["algorithm"])
        lines.append(f"### {scenario_id}")
        lines.append("")
        lines.append(_scenario_table(scenario_rows, ranking_by_case))
        lines.append("")
        for scenario_line in _scenario_commentary_lines(scenario_rows, ranking_by_case):
            lines.append(scenario_line)
        lines.append("")

    lines.append("## 8. Convergence Summary")
    lines.append("")
    lines.append(
        "The verification workflow solves both coarse and refined meshes for every case. The acceptance criteria are: displacement change below `{0:.1f}%`, stress change below `{1:.1f}%`, and first buckling-factor change below `{2:.1f}%`.".format(
            float(config["convergence_criteria_percent"]["max_displacement_m"]),
            float(config["convergence_criteria_percent"]["max_mises_pa"]),
            float(config["convergence_criteria_percent"]["buckling_factor_1"]),
        )
    )
    lines.append("")
    lines.append(f"At the current stage, **{converged_count} of {len(convergence_rows)} comparison pairs** satisfy all three convergence criteria.")
    lines.append(
        f"Because the study is currently at `{converged_count}/{len(convergence_rows)}` all-pass convergence, the ranking should be interpreted as comparative screening quality, not final structural qualification."
    )
    lines.append("")
    lines.append(_convergence_table(convergence_rows))
    lines.append("")
    lines.append(f"![Winner mesh comparison](../figures/task7_{winner['case_id']}_mesh_comparison.png)")
    lines.append("")
    if provisional_winner:
        lines.append("The global winner is therefore reported as **provisional**: the ranking is valid for comparative screening, but convergence evidence is not yet strong enough for a final structural sign-off.")
    else:
        lines.append("The global winner is reported as **converged** under the current criteria.")
    lines.append("")
    lines.append("## 9. Warning Cases from Scenario S8")
    lines.append("")
    lines.append("Scenario S8 is kept intentionally as a warning family. None of the Task 6 S8 runs are mathematically compliant or engineering-feasible, so these Abaqus models are useful only as structural cautionary examples and not as candidate towers for recommendation.")
    lines.append("")
    lines.append("![Scenario S8 warning metrics](../figures/s8_warning_metrics.png)")
    lines.append("")
    lines.append("## 10. Field Visualizations of the Global Top-3")
    lines.append("")
    lines.append("All detailed field figures are rendered from actual Abaqus ODB data with Python, not from Abaqus screenshots.")
    lines.append("Task 7 figures use a true-scale equal-axis policy (no axis stretching). This differs from some legacy Task 6 renderings that used a different plotting style.")
    lines.append("")
    for entry in top3_overall:
        row = entry["row"]
        lines.append(f"### {row['case_label']}")
        lines.append("")
        lines.append(f"![{row['case_label']} stress](../figures/task7_{row['case_id']}_stress.png)")
        lines.append("")
        lines.append(f"![{row['case_label']} displacement](../figures/task7_{row['case_id']}_displacement.png)")
        lines.append("")
        lines.append(f"![{row['case_label']} buckling](../figures/task7_{row['case_id']}_buckling_mode1.png)")
        lines.append("")

    lines.append("## 11. Recommendation for Later Tasks")
    lines.append("")
    winner_entry = ranking_by_case[winner["case_id"]]
    recommendation_status = "provisional" if provisional_winner else "converged"
    lines.append(
        f"The current Task 7 recommendation is **{winner['case_label']}** with a `{recommendation_status}` status. Its weighted score is `{winner_entry['weighted_score']:.3f}` and its refined response gives a first buckling factor of `{float(winner['buckling_factor_1']):.3f}`, a maximum displacement of `{1000.0 * float(winner['max_displacement_m']):.3f} mm`, and a maximum stress of `{float(winner['max_mises_pa']) / 1e6:.3f} MPa`."
    )
    lines.append("")
    lines.append(
        "The main practical outcome for the project is that Task 6 geometric alternatives can now be compared with one consistent structural score, explicit warning handling, and scenario-level interpretation suitable for Task 9 communication."
    )

    report_text = "\n".join(lines) + "\n"
    root_report_text = report_text.replace("(../figures/", "(results/figures/")
    report_path = report_dir / "Task7_Report.md"
    root_report_path = output_dir / "Task7_Report.md"
    report_path.write_text(report_text, encoding="utf-8")
    root_report_path.write_text(root_report_text, encoding="utf-8")
    print("Generated Task 7 markdown report.")


if __name__ == "__main__":
    main()
