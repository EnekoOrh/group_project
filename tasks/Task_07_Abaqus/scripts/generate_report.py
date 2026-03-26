import argparse
import csv
import json
import math
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
        if row["case_variant"] != "comparison" or row.get("load_case", "combined") != "combined":
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
    return _format_table(["Case", "Status", "Selection basis", "Task 6 run", "Decision mode", "Task 6 area (m²)"], rows)


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
    return _format_table(["Case", "Status", "Task 6 area (m²)", "Max disp. (mm)", "Max stress (MPa)", "Buckling factor"], rows)


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
        ["Algorithm", "Status", "Weighted score", "Task 6 area (m²)", "Max disp. (mm)", "Max stress (MPa)", "Buckling factor"],
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
        if row.get("case_variant") and row.get("case_variant") != "comparison":
            continue
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


def _scale_audit_summary(rows):
    if not rows:
        return "Scale audit file is missing."

    comparison_rows = [row for row in rows if row.get("case_variant", "comparison") == "comparison"]
    if not comparison_rows:
        return "Scale audit has no comparison rows."

    max_radius = max(float(row["max_radius_abs_diff_m"]) for row in comparison_rows)
    max_segment_h = max(float(row["max_segment_height_abs_diff_m"]) for row in comparison_rows)
    max_total_h = max(float(row["total_height_abs_diff_m"]) for row in comparison_rows)
    max_base_d = max(float(row["base_diameter_abs_diff_m"]) for row in comparison_rows)
    max_top_d = max(float(row["top_diameter_abs_diff_m"]) for row in comparison_rows)
    tol = float(comparison_rows[0]["tolerance_m"])
    all_pass = all(row["status"] == "pass" for row in comparison_rows)

    status_text = "all checks pass" if all_pass else "at least one check failed"
    return (
        f"Scale audit (`scale_geometry_audit.csv`) confirms {status_text} for `{len(comparison_rows)}` comparison jobs "
        f"with tolerance `{tol:.1e} m`: max radius diff `{max_radius:.3e} m`, max segment-height diff `{max_segment_h:.3e} m`, "
        f"max total-height diff `{max_total_h:.3e} m`, max base-diameter diff `{max_base_d:.3e} m`, max top-diameter diff `{max_top_d:.3e} m`."
    )


def _decomposition_case_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row.get("case_variant") != "load_decomposition":
            continue
        if row.get("mesh_level") != "refined":
            continue
        grouped[row["case_id"]][row.get("load_case", "")] = row
    return grouped


def _safe_ratio(num, den):
    if abs(float(den)) < 1.0e-12:
        return float("nan")
    return float(num) / float(den)


def _median(values):
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return float("nan")
    count = len(clean)
    mid = count // 2
    if count % 2 == 1:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _percentile(values, quantile):
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    q = min(max(float(quantile), 0.0), 1.0)
    idx = q * (len(clean) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return clean[lo]
    alpha = idx - lo
    return (1.0 - alpha) * clean[lo] + alpha * clean[hi]


def _build_decomposition_entries(refined_rows, all_rows):
    decomp_map = _decomposition_case_rows(all_rows)
    entries = []
    for row in refined_rows:
        case_id = row["case_id"]
        decomp = decomp_map.get(case_id, {})
        gravity = decomp.get("gravity_only")
        wind = decomp.get("wind_only")
        if gravity is None or wind is None:
            continue

        combined_disp = 1000.0 * float(row["max_displacement_m"])
        gravity_disp = 1000.0 * float(gravity["max_displacement_m"])
        wind_disp = 1000.0 * float(wind["max_displacement_m"])
        combined_stress = float(row["max_mises_pa"]) / 1.0e6
        gravity_stress = float(gravity["max_mises_pa"]) / 1.0e6
        wind_stress = float(wind["max_mises_pa"]) / 1.0e6
        reaction_combined = float(row["base_reaction_resultant_n"])
        reaction_gravity = float(gravity["base_reaction_resultant_n"])
        reaction_wind = float(wind["base_reaction_resultant_n"])

        disp_ratio = _safe_ratio(wind_disp, gravity_disp)
        stress_ratio = _safe_ratio(wind_stress, gravity_stress)
        reaction_ratio = _safe_ratio(reaction_wind, reaction_gravity)
        disp_driver = "wind" if wind_disp >= gravity_disp else "gravity"
        stress_driver = "wind" if wind_stress >= gravity_stress else "gravity"

        entries.append(
            {
                "case_id": case_id,
                "case_label": row["case_label"],
                "scenario_id": row["scenario_id"],
                "algorithm": row["algorithm"],
                "selection_status": row["selection_status"],
                "combined_max_displacement_mm": combined_disp,
                "gravity_only_max_displacement_mm": gravity_disp,
                "wind_only_max_displacement_mm": wind_disp,
                "wind_to_gravity_disp_ratio": disp_ratio,
                "displacement_driver": disp_driver,
                "combined_max_mises_mpa": combined_stress,
                "gravity_only_max_mises_mpa": gravity_stress,
                "wind_only_max_mises_mpa": wind_stress,
                "wind_to_gravity_stress_ratio": stress_ratio,
                "stress_driver": stress_driver,
                "combined_reaction_resultant_n": reaction_combined,
                "gravity_only_reaction_resultant_n": reaction_gravity,
                "wind_only_reaction_resultant_n": reaction_wind,
                "wind_to_gravity_reaction_ratio": reaction_ratio,
            }
        )
    return entries


def _write_decomposition_csv(path, entries):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "case_label",
                "scenario_id",
                "algorithm",
                "selection_status",
                "combined_max_displacement_mm",
                "gravity_only_max_displacement_mm",
                "wind_only_max_displacement_mm",
                "wind_to_gravity_disp_ratio",
                "displacement_driver",
                "combined_max_mises_mpa",
                "gravity_only_max_mises_mpa",
                "wind_only_max_mises_mpa",
                "wind_to_gravity_stress_ratio",
                "stress_driver",
                "combined_reaction_resultant_n",
                "gravity_only_reaction_resultant_n",
                "wind_only_reaction_resultant_n",
                "wind_to_gravity_reaction_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)


def _write_decomposition_summary_csv(path, entries):
    disp_ratios = [entry["wind_to_gravity_disp_ratio"] for entry in entries]
    stress_ratios = [entry["wind_to_gravity_stress_ratio"] for entry in entries]
    reaction_ratios = [entry["wind_to_gravity_reaction_ratio"] for entry in entries]
    disp_wind = sum(1 for entry in entries if entry["displacement_driver"] == "wind")
    stress_wind = sum(1 for entry in entries if entry["stress_driver"] == "wind")
    total = len(entries)

    rows = [
        {
            "metric": "displacement",
            "cases_total": total,
            "wind_dominant_cases": disp_wind,
            "gravity_dominant_cases": total - disp_wind,
            "median_wind_to_gravity_ratio": _median(disp_ratios),
            "p90_wind_to_gravity_ratio": _percentile(disp_ratios, 0.90),
        },
        {
            "metric": "stress",
            "cases_total": total,
            "wind_dominant_cases": stress_wind,
            "gravity_dominant_cases": total - stress_wind,
            "median_wind_to_gravity_ratio": _median(stress_ratios),
            "p90_wind_to_gravity_ratio": _percentile(stress_ratios, 0.90),
        },
        {
            "metric": "reaction_resultant",
            "cases_total": total,
            "wind_dominant_cases": sum(1 for ratio in reaction_ratios if math.isfinite(ratio) and ratio >= 1.0),
            "gravity_dominant_cases": sum(1 for ratio in reaction_ratios if math.isfinite(ratio) and ratio < 1.0),
            "median_wind_to_gravity_ratio": _median(reaction_ratios),
            "p90_wind_to_gravity_ratio": _percentile(reaction_ratios, 0.90),
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "cases_total",
                "wind_dominant_cases",
                "gravity_dominant_cases",
                "median_wind_to_gravity_ratio",
                "p90_wind_to_gravity_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _decomposition_table(entries):
    body = []
    for entry in entries:
        body.append(
            [
                entry["case_label"],
                STATUS_TEXT[entry["selection_status"]],
                f"{entry['gravity_only_max_displacement_mm']:.3f}",
                f"{entry['wind_only_max_displacement_mm']:.3f}",
                f"{entry['wind_to_gravity_disp_ratio']:.3f}" if math.isfinite(entry["wind_to_gravity_disp_ratio"]) else "n/a",
                entry["displacement_driver"],
                f"{entry['gravity_only_max_mises_mpa']:.3f}",
                f"{entry['wind_only_max_mises_mpa']:.3f}",
                f"{entry['wind_to_gravity_stress_ratio']:.3f}" if math.isfinite(entry["wind_to_gravity_stress_ratio"]) else "n/a",
                entry["stress_driver"],
            ]
        )
    return _format_table(
        [
            "Case",
            "Status",
            "Disp gravity (mm)",
            "Disp wind (mm)",
            "Wind/Gravity disp ratio",
            "Disp driver",
            "Stress gravity (MPa)",
            "Stress wind (MPa)",
            "Wind/Gravity stress ratio",
            "Stress driver",
        ],
        body,
    )


def _mesh_policy_summary(rows, config):
    if not rows:
        return "A uniform refined mesh policy is applied to all refined jobs."

    refined_rows = [
        row
        for row in rows
        if row["mesh_level"] == "refined" and row.get("case_variant", "comparison") == "comparison"
    ]
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


def _decomposition_map(entries):
    return {entry["case_id"]: entry for entry in entries}


def _convergence_map(rows):
    return {row["case_id"]: row for row in rows}


def _scenario_commentary_lines(rows, ranking_by_case, decomposition_by_case, convergence_by_case):
    ranked = sorted(rows, key=lambda row: (ranking_by_case[row["case_id"]]["weighted_score"], row["case_id"]))
    winner = ranked[0]
    winner_entry = ranking_by_case[winner["case_id"]]
    comments = []
    if len(ranked) > 1:
        runner_up = ranked[1]
        runner_entry = ranking_by_case[runner_up["case_id"]]
        score_gap = runner_entry["weighted_score"] - winner_entry["weighted_score"]
        winner_text = (
            f"**Winner:** {winner['case_label']} with weighted score `{winner_entry['weighted_score']:.3f}`; "
            f"gap to second place ({runner_up['case_label']}) is `{score_gap:.3f}`."
        )
    else:
        score_gap = 0.0
        winner_text = f"**Winner:** {winner['case_label']} with weighted score `{winner_entry['weighted_score']:.3f}`."
    comments.append(winner_text)
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

    winner_decomp = decomposition_by_case.get(winner["case_id"])
    if winner_decomp is not None:
        disp_ratio = float(winner_decomp["wind_to_gravity_disp_ratio"])
        stress_ratio = float(winner_decomp["wind_to_gravity_stress_ratio"])
        disp_ratio_text = f"{disp_ratio:.3f}" if math.isfinite(disp_ratio) else "n/a"
        stress_ratio_text = f"{stress_ratio:.3f}" if math.isfinite(stress_ratio) else "n/a"
        comments.append(
            f"**Load dominance:** winner ratios are disp wind/gravity `{disp_ratio_text}` and stress wind/gravity `{stress_ratio_text}`; dominant drivers are displacement `{winner_decomp['displacement_driver']}` and stress `{winner_decomp['stress_driver']}`."
        )

    winner_convergence = convergence_by_case.get(winner["case_id"], {})
    mixed_status = len({row["selection_status"] for row in rows}) > 1
    if mixed_status:
        comments.append(
            "**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs."
        )
    elif winner_convergence.get("all_criteria_pass") != "yes":
        comments.append(
            "**Critical note:** winner ranking is not yet convergence-locked on this scenario; keep this as screening guidance until mesh convergence is improved."
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
        ["Rank", "Case", "Status", "Weighted score", "Penalty", "Buckling factor", "Max disp. (mm)", "Max stress (MPa)", "Area (m²)"],
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
            value_text = f"{float(row[metric_key]):.2f} m²"
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
    refined_rows = [
        row
        for row in mesh_summary_rows
        if row.get("mesh_level") == "refined" and row.get("case_variant", "comparison") == "comparison"
    ]
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
    scale_audit_rows,
    unit_audit_rows,
    decomposition_entries,
    cae_audit_json_path: Path,
    plot_audit_csv_path: Path,
    plot_audit_json_path: Path,
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

    add_check(
        "decomposition_rows_cover_all_cases",
        len(decomposition_entries) == len(cases),
        f"decomposition_rows={len(decomposition_entries)}, selected_cases={len(cases)}",
    )

    plot_audit_rows = _load_csv(plot_audit_csv_path) if plot_audit_csv_path.exists() else []
    add_check(
        "plot_audit_rows_cover_all_cases",
        len(plot_audit_rows) == len(refined_rows),
        f"plot_audit_rows={len(plot_audit_rows)}, refined_rows={len(refined_rows)}",
    )
    add_check(
        "plot_audit_all_pass",
        bool(plot_audit_rows) and all(row.get("overall_pass") == "yes" for row in plot_audit_rows),
        "all plot-view audit rows have overall_pass=yes",
    )
    add_check(
        "plot_audit_perpendicularity_pass",
        bool(plot_audit_rows) and all(row.get("perpendicular_pass") == "yes" for row in plot_audit_rows),
        "all plot-view audit rows pass perpendicularity",
    )
    add_check(
        "plot_audit_left_to_right_pass",
        bool(plot_audit_rows) and all(row.get("left_to_right_pass") == "yes" for row in plot_audit_rows),
        "all plot-view audit rows pass left-to-right projection",
    )
    add_check(
        "plot_audit_arrow_clearance_pass",
        bool(plot_audit_rows) and all(row.get("arrow_clearance_pass") == "yes" for row in plot_audit_rows),
        "all plot-view audit rows pass arrow-clearance checks",
    )

    plot_audit_json = _load_json(plot_audit_json_path) if plot_audit_json_path.exists() else {}
    add_check(
        "plot_audit_json_status_pass",
        bool(plot_audit_json) and plot_audit_json.get("status") == "pass",
        f"plot_audit_json_status={plot_audit_json.get('status', 'missing')}",
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

    scale_rows = [row for row in scale_audit_rows if row.get("case_variant", "comparison") == "comparison"]
    add_check(
        "scale_audit_present",
        bool(scale_rows),
        f"scale_audit_rows={len(scale_rows)}",
    )
    add_check(
        "scale_audit_all_pass",
        bool(scale_rows) and all(row.get("status") == "pass" for row in scale_rows),
        "all comparison rows in scale_geometry_audit.csv are pass",
    )

    add_check(
        "unit_contract_rows_cover_manifest",
        len(unit_audit_rows) >= len(refined_rows),
        f"unit_audit_rows={len(unit_audit_rows)}, refined_rows={len(refined_rows)}",
    )
    add_check(
        "unit_contract_all_pass",
        bool(unit_audit_rows) and all(row.get("status") == "pass" for row in unit_audit_rows),
        "all rows in unit_load_contract_audit.csv are pass",
    )

    all_pass = all(item["status"] == "pass" for item in checks)
    summary = {
        "status": "pass" if all_pass else "fail",
        "checks_total": len(checks),
        "checks_failed": sum(1 for item in checks if item["status"] != "pass"),
        "checks": checks,
        "solver_mesh_refined": solver_mesh,
        "cae_audit_path": str(cae_audit_json_path),
        "plot_audit_csv_path": str(plot_audit_csv_path),
        "plot_audit_json_path": str(plot_audit_json_path),
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
        return value, "m²"
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
    parser.add_argument("--scale-audit-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "scale_geometry_audit.csv"))
    parser.add_argument("--input-audit-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "input_contract_audit.csv"))
    parser.add_argument("--unit-audit-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "unit_load_contract_audit.csv"))
    parser.add_argument("--cae-audit-json", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "cae_integrity_audit.json"))
    parser.add_argument("--plot-audit-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "plot_view_audit.csv"))
    parser.add_argument("--plot-audit-json", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "plot_view_audit.json"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus"))
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    cases = _load_json(Path(args.cases))
    rows = _load_csv(Path(args.summary_csv))
    convergence_rows = _load_csv(Path(args.mesh_csv)) if Path(args.mesh_csv).exists() else []
    mesh_summary_rows = _load_csv(Path(args.mesh_summary_csv)) if Path(args.mesh_summary_csv).exists() else []
    geometry_rows = _load_csv(Path(args.geometry_csv)) if Path(args.geometry_csv).exists() else []
    scale_audit_rows = _load_csv(Path(args.scale_audit_csv)) if Path(args.scale_audit_csv).exists() else []
    input_audit_rows = _load_csv(Path(args.input_audit_csv)) if Path(args.input_audit_csv).exists() else []
    unit_audit_rows = _load_csv(Path(args.unit_audit_csv)) if Path(args.unit_audit_csv).exists() else []
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
    top5_overall = ranked_entries[:5]
    winner = top5_overall[0]["row"]

    weighted_csv_path = results_data_dir / "weighted_ranking.csv"
    criterion_csv_path = results_data_dir / "criterion_top3.csv"
    criterion_engineering_csv_path = results_data_dir / "criterion_top3_engineering.csv"
    weighted_figure_path = figures_dir / "task7_weighted_ranking.png"
    criterion_figure_path = figures_dir / "task7_criterion_top3.png"
    criterion_engineering_figure_path = figures_dir / "task7_criterion_top3_engineering.png"
    consistency_audit_json_path = results_data_dir / "report_consistency_audit.json"
    consistency_audit_csv_path = results_data_dir / "report_consistency_audit.csv"
    decomposition_csv_path = results_data_dir / "load_decomposition_refined.csv"
    decomposition_summary_csv_path = results_data_dir / "load_dominance_summary.csv"
    _write_weighted_ranking_csv(weighted_csv_path, ranked_entries)
    _write_criterion_top_csv(criterion_csv_path, criterion_entries)
    _write_criterion_top_csv(criterion_engineering_csv_path, criterion_entries_engineering)
    _plot_weighted_ranking(ranked_entries, weighted_figure_path)
    _plot_criterion_top(criterion_entries, criterion_figure_path)
    _plot_criterion_top(criterion_entries_engineering, criterion_engineering_figure_path)
    decomposition_entries = _build_decomposition_entries(refined_rows, rows)
    _write_decomposition_csv(decomposition_csv_path, decomposition_entries)
    decomposition_summary_rows = _write_decomposition_summary_csv(decomposition_summary_csv_path, decomposition_entries)

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
        scale_audit_rows=scale_audit_rows,
        unit_audit_rows=unit_audit_rows,
        decomposition_entries=decomposition_entries,
        cae_audit_json_path=Path(args.cae_audit_json),
        plot_audit_csv_path=Path(args.plot_audit_csv),
        plot_audit_json_path=Path(args.plot_audit_json),
        output_json_path=consistency_audit_json_path,
        output_csv_path=consistency_audit_csv_path,
    )

    q_ref = 0.5 * float(config["wind"]["air_density_kg_m3"]) * float(config["wind"]["reference_speed_m_s"]) ** 2
    engineering_count = sum(1 for case in cases if case["selection_status"] == STATUS_ENGINEERING)
    fallback_count = sum(1 for case in cases if case["selection_status"] == STATUS_MATH_FALLBACK)
    warning_count = sum(1 for case in cases if case["selection_status"] == STATUS_WARNING)
    converged_count = sum(1 for row in convergence_rows if row["all_criteria_pass"] == "yes")
    provisional_winner = converged_count < len(convergence_rows)
    decomp_summary_by_metric = {row["metric"]: row for row in decomposition_summary_rows}
    disp_summary = decomp_summary_by_metric.get("displacement", {})
    stress_summary = decomp_summary_by_metric.get("stress", {})
    decomposition_by_case = _decomposition_map(decomposition_entries)
    convergence_by_case = _convergence_map(convergence_rows)
    top_engineering_entry = next((entry for entry in ranked_entries if entry["row"]["selection_status"] == STATUS_ENGINEERING), None)

    disp_ratios = [float(entry["wind_to_gravity_disp_ratio"]) for entry in decomposition_entries if math.isfinite(float(entry["wind_to_gravity_disp_ratio"]))]
    stress_ratios = [float(entry["wind_to_gravity_stress_ratio"]) for entry in decomposition_entries if math.isfinite(float(entry["wind_to_gravity_stress_ratio"]))]
    disp_ratio_p25 = _percentile(disp_ratios, 0.25) if disp_ratios else float("nan")
    disp_ratio_p75 = _percentile(disp_ratios, 0.75) if disp_ratios else float("nan")
    stress_ratio_p25 = _percentile(stress_ratios, 0.25) if stress_ratios else float("nan")
    stress_ratio_p75 = _percentile(stress_ratios, 0.75) if stress_ratios else float("nan")

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
    lines.append("### 2.1 Selection and Compliance Semantics")
    lines.append("")
    lines.append("To keep Task 6 -> Task 7 interpretation unambiguous, each selected case carries its original status tier.")
    lines.append("Task 6 mathematical compliance is tied to normalized target constraints:")
    lines.append("$$")
    lines.append("g_V = \\frac{|V - V^*|}{V^*}, \\qquad g_H = \\frac{|H - H^*|}{H^*}")
    lines.append("$$")
    lines.append("A mathematically compliant case satisfies $g_V \\le \\varepsilon_V$ and $g_H \\le \\varepsilon_H$.")
    lines.append("Engineering-feasible means mathematically compliant **and** passing the additional engineering plausibility checks from Task 6.")
    lines.append("The Task 7 selection rule per scenario/optimizer pair is:")
    lines.append("$$")
    lines.append("\\begin{aligned}")
    lines.append("r^* &= \\arg\\min A(r) && \\text{over engineering-feasible runs} \\\\")
    lines.append("&\\text{else }= \\arg\\min A(r) && \\text{over mathematically compliant runs} \\\\")
    lines.append("&\\text{else }= \\arg\\min F_{pen}(r) && \\text{(warning fallback only)}")
    lines.append("\\end{aligned}")
    lines.append("$$")
    lines.append("where `A` is shell area and `F_{pen}` is the penalized Task 6 objective.")
    lines.append("This defines exactly what \"best\" means in this report and why fallback/warning statuses appear in some scenario rows.")
    lines.append("")
    lines.append("## 3. Material, Loading, and CAE Setup")
    lines.append("")
    lines.append("Each model is built from the selected Task 6 meridian profile and revolved into a thin-shell cooling-tower surface; Task 7 keeps the exact Task 6 geometry scale.")
    lines.append("The global coordinate convention is `Y`-up, with `X/Z` as the horizontal plane.")
    lines.append(
        "Material assumptions are uniform across all cases: equivalent reinforced concrete with `E = 33 GPa`, `nu = 0.20`, density `rho = 2500 kg/m³`, and shell thickness `t = 0.20 m`."
    )
    lines.append(
        f"Loading combines self-weight and one-direction wind. The reference wind speed is `{config['wind']['reference_speed_m_s']} m/s`, giving `q = {q_ref:.1f} Pa` from $q = 0.5\\,\\rho_{{air}}V^2$."
    )
    lines.append(
        "Wind is modeled as external pressure on the shell with windward pressure and leeward suction (not internal cabin pressure). The circumferential pressure law is:"
    )
    lines.append("$$")
    lines.append("C_p(\\theta) = \\mathrm{clip}(0.8\\cos\\theta, -0.5, 0.8), \\quad p(\\theta) = q\\,C_p(\\theta)")
    lines.append("$$")
    lines.append(
        "Boundary conditions represent the tower support at the foundation/piler ring: base nodes are fixed (`U_x = U_y = U_z = 0`), while the shell remains free elsewhere."
    )
    lines.append(
        "Each model uses one static step (`STATIC_WIND`) for combined loading and one linear buckling step (`BUCKLING`) to extract the first instability modes about the preloaded state."
    )
    lines.append("")
    lines.append("## 4. Refined Structural Comparison Across All 24 Cases")
    lines.append("")
    lines.append(_refined_table(refined_rows))
    lines.append("")
    lines.append("![Task 7 refined comparison metrics](../figures/comparison_metrics.png)")
    lines.append("")
    lines.append("Figure 4.1 stacks displacement, stress, and buckling vertically to improve visibility and avoid compressed labels.")
    lines.append("The first reading pass should prioritize buckling spread, then check whether displacement and stress trends confirm the same structural leader.")
    lines.append("")
    lines.append("## 5. Methodological Clarity and Global Ranking")
    lines.append("")
    lines.append("### 5.1 Methodological Clarity")
    lines.append("")
    lines.append("The global ranking is a rank-based weighted score on the refined results, designed to stay robust when one or two cases have extreme values.")
    lines.append("For each metric, rank `1` is best and rank `24` is worst, with deterministic tie-breaks by scenario/algorithm identifiers.")
    lines.append("")
    lines.append("$$")
    lines.append("r_i^{(k)} = \\operatorname{rank}_k(i)")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append("S_i = 0.45\\,r_i^{(b)} + 0.25\\,r_i^{(d)} + 0.20\\,r_i^{(s)} + 0.10\\,r_i^{(a)}")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append("J_i = S_i + P_i, \\quad P_i \\in \\{0,10,20\\}")
    lines.append("$$")
    lines.append("")
    lines.append("Here $r_i^{(b)}$ is the buckling rank (higher buckling is better), $r_i^{(d)}$ is the displacement rank, $r_i^{(s)}$ is the stress rank, and $r_i^{(a)}$ is the area rank (lower is better for the last three).")
    lines.append("Penalty points are `0` for engineering-feasible, `10` for mathematical fallback, and `20` for warning cases. The best model is the one with the lowest `J_i`.")
    lines.append("")
    lines.append("### 5.2 Global Weighted Top-5")
    lines.append("")
    lines.append(_weighted_top_table(top5_overall))
    lines.append("")
    lines.append("![Task 7 weighted ranking](../figures/task7_weighted_ranking.png)")
    lines.append("")
    lines.append("### 5.3 Top-3 by Criterion")
    lines.append("")
    lines.append("#### 5.3.1 Raw Top-3 (all statuses)")
    lines.append("")
    lines.append("This raw criterion table keeps all statuses visible, so warning and fallback cases can appear if they are numerically extreme on a single indicator.")
    lines.append("")
    lines.append(_criterion_top_table(criterion_entries))
    lines.append("")
    lines.append("![Task 7 criterion leaders](../figures/task7_criterion_top3.png)")
    lines.append("")
    lines.append("#### 5.3.2 Engineering-eligible Top-3")
    lines.append("")
    lines.append("This decision-grade table filters to engineering-feasible entries only.")
    lines.append("")
    lines.append(_criterion_top_table(criterion_entries_engineering))
    lines.append("")
    lines.append("![Task 7 engineering criterion leaders](../figures/task7_criterion_top3_engineering.png)")
    lines.append("")
    lines.append("### 5.4 Reproducibility and Fairness Controls")
    lines.append("")
    lines.append("Task 7 ranking is deterministic: every score is computed from the latest refined Abaqus results with fixed tie-breaks, so repeated report generation gives the same ordering.")
    lines.append("Fairness is handled in two layers: warning/fallback cases remain visible in the global ranking through explicit penalties, while the engineering-eligible Top-3 table isolates decision-grade candidates.")
    lines.append("This avoids the presentation ambiguity highlighted in Task 6 feedback by defining \"best\" as the minimum penalized weighted rank `J_i` and by keeping compliance status explicit in every comparison table.")
    lines.append("")
    lines.append("## 6. Scenario-by-Scenario Comparison")
    lines.append("")
    for scenario_id in config["selection"]["scenario_ids"]:
        scenario_rows = [row for row in refined_rows if row["scenario_id"] == scenario_id]
        scenario_rows.sort(key=lambda row: row["algorithm"])
        lines.append(f"### {scenario_id}")
        lines.append("")
        lines.append(_scenario_table(scenario_rows, ranking_by_case))
        lines.append("")
        for scenario_line in _scenario_commentary_lines(
            scenario_rows, ranking_by_case, decomposition_by_case, convergence_by_case
        ):
            lines.append(scenario_line)
        lines.append("")

    lines.append("## 7. Convergence Summary")
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
    lines.append("## 8. Wind-vs-Self-Weight Decomposition (Refined Cases)")
    lines.append("")
    lines.append(
        "To separate load influence explicitly, each refined comparison tower is re-solved with gravity-only and wind-only static steps."
    )
    lines.append("The combined-load ranking remains unchanged; decomposition is used as an interpretation layer.")
    lines.append(
        f"Across all `{len(decomposition_entries)}` cases, displacement is wind-dominant in `{disp_summary.get('wind_dominant_cases', 'n/a')}` cases and stress is wind-dominant in `{stress_summary.get('wind_dominant_cases', 'n/a')}` cases."
    )
    lines.append(
        f"Median wind-to-gravity ratios are `{float(disp_summary.get('median_wind_to_gravity_ratio', float('nan'))):.3f}` for displacement and `{float(stress_summary.get('median_wind_to_gravity_ratio', float('nan'))):.3f}` for stress, with IQR ranges `{disp_ratio_p25:.3f}`-`{disp_ratio_p75:.3f}` (displacement) and `{stress_ratio_p25:.3f}`-`{stress_ratio_p75:.3f}` (stress)."
    )
    lines.append(
        f"Tail sensitivity remains important: P90 wind-to-gravity ratios are `{float(disp_summary.get('p90_wind_to_gravity_ratio', float('nan'))):.3f}` for displacement and `{float(stress_summary.get('p90_wind_to_gravity_ratio', float('nan'))):.3f}` for stress."
    )
    lines.append(
        "Critical interpretation: most towers remain gravity-led in global reaction, while wind can still control local displacement/stress behavior for specific geometries. This is why the report keeps both decomposition evidence and weighted ranking instead of collapsing to one indicator."
    )
    lines.append(
        f"Because convergence is currently `{converged_count}/{len(convergence_rows)}` all-pass, decomposition trends are used as high-value screening evidence but not as final qualification proof."
    )
    lines.append("")
    lines.append(_decomposition_table(decomposition_entries))
    lines.append("")
    lines.append("## 9. Warning Cases from Scenario S8")
    lines.append("")
    lines.append("Scenario S8 is kept intentionally as a warning family. None of the Task 6 S8 runs are mathematically compliant or engineering-feasible, so these Abaqus models are useful only as structural cautionary examples and not as candidate towers for recommendation.")
    lines.append("")
    lines.append("![Scenario S8 warning metrics](../figures/s8_warning_metrics.png)")
    lines.append("")
    lines.append("## 10. Final Critical Conclusions")
    lines.append("")
    winner_entry = ranking_by_case[winner["case_id"]]
    winner_status = STATUS_TEXT[winner["selection_status"]]
    lines.append(
        f"The current global leader is `{winner['case_label']}` ({winner_status}) with weighted score `{winner_entry['weighted_score']:.3f}`."
    )
    if top_engineering_entry is not None and top_engineering_entry["row"]["case_id"] != winner["case_id"]:
        top_eng_row = top_engineering_entry["row"]
        lines.append(
            f"The strongest engineering-feasible alternative is `{top_eng_row['case_label']}` at overall rank `{top_engineering_entry['overall_rank']}`, which remains the primary backup if compliance-only filtering is enforced."
        )
    lines.append(
        f"Convergence confidence is still limited (`{converged_count}/{len(convergence_rows)}` all-pass), so the current winner must be treated as provisional screening output rather than final structural qualification."
    )
    lines.append(
        f"Load decomposition shows the action split clearly: displacement is wind-dominant in `{disp_summary.get('wind_dominant_cases', 'n/a')}/{len(decomposition_entries)}` cases and stress is wind-dominant in `{stress_summary.get('wind_dominant_cases', 'n/a')}/{len(decomposition_entries)}` cases."
    )
    lines.append(
        f"Median wind/gravity ratios are `{float(disp_summary.get('median_wind_to_gravity_ratio', float('nan'))):.3f}` for displacement and `{float(stress_summary.get('median_wind_to_gravity_ratio', float('nan'))):.3f}` for stress, which indicates a predominantly self-weight-driven response in this dataset with wind becoming locally decisive only in a small displacement subset."
    )
    lines.append(
        "The ranking remains fair for screening because buckling, displacement, stress, and area are combined with explicit compliance penalties; final design lock-in should wait for stronger convergence closure and, if schedule allows, a higher-fidelity wind-standard calibration."
    )
    lines.append("")
    lines.append("## 11. Field Visualizations of the Global Top-5")
    lines.append("")
    lines.append("All detailed field figures are rendered from actual Abaqus ODB data with Python, not from Abaqus screenshots.")
    lines.append("Simulation geometry remains true-scale from Task 6. Rendered views use consistent camera framing and a left-to-right wind-view policy for direct cross-case comparison.")
    lines.append(f"Wind arrows are bound to the configured physical wind axis (`{config['wind'].get('wind_direction_axis', '+X')}`), so they indicate actual load direction rather than decorative annotation.")
    lines.append("Per-case camera/arrow verification is enforced numerically through `plot_view_audit.csv`: the camera is perpendicular to wind direction, wind projects left-to-right on screen, and arrows stay outside the tower silhouette.")
    lines.append("")
    for entry in top5_overall:
        row = entry["row"]
        lines.append(f"### {row['case_label']}")
        lines.append("")
        lines.append(f"![{row['case_label']} stress](../figures/task7_{row['case_id']}_stress.png) ![{row['case_label']} displacement](../figures/task7_{row['case_id']}_displacement.png) ![{row['case_label']} buckling](../figures/task7_{row['case_id']}_buckling_mode1.png)")
        lines.append("")
        
        rank = entry["overall_rank"]
        score = entry["weighted_score"]
        buckling = float(row["buckling_factor_1"])
        disp = 1000.0 * float(row["max_displacement_m"])
        stress = float(row["max_mises_pa"]) / 1e6
        status_text = STATUS_TEXT[row["selection_status"]]
        
        discussion = f"**Rank/Status:** Rank `{rank}`, `{status_text}`, weighted score `{score:.3f}`."
        discussion += f" **Metrics:** buckling `{buckling:.3f}`, max displacement `{disp:.3f} mm`, max stress `{stress:.3f} MPa`."

        if rank == 1:
            discussion += " **Interpretation:** best current compromise across stability, stiffness, and stress."
        elif rank in [2, 3]:
            discussion += " **Interpretation:** high-value runner-up with a narrow gap to the leader."
        else:
            discussion += " **Interpretation:** structurally credible but less balanced than the first three ranks."
        
        lines.append(discussion)
        lines.append("")

    lines.append("## 12. Annex: Complete Field Visualizations")
    lines.append("")
    lines.append("This section contains the field visualizations for the remaining 19 candidates out of the total 24 refined presentation models, organized by Scenario.")
    lines.append("")
    annex_entries = [entry for entry in ranked_entries if entry not in top5_overall]
    
    annex_by_scenario = defaultdict(list)
    for entry in annex_entries:
        annex_by_scenario[entry["row"]["scenario_id"]].append(entry)
        
    for scenario_id in config["selection"]["scenario_ids"]:
        if scenario_id in annex_by_scenario:
            lines.append(f"### Scenario {scenario_id}")
            lines.append("")
            scenario_entries = sorted(annex_by_scenario[scenario_id], key=lambda e: e["row"]["algorithm"])
            for entry in scenario_entries:
                row = entry["row"]
                lines.append(f"#### {row['case_label']}")
                lines.append("")
                lines.append(f"![{row['case_label']} stress](../figures/task7_{row['case_id']}_stress.png) ![{row['case_label']} displacement](../figures/task7_{row['case_id']}_displacement.png) ![{row['case_label']} buckling](../figures/task7_{row['case_id']}_buckling_mode1.png)")
                lines.append("")

    report_text = "\n".join(lines) + "\n"
    root_report_text = report_text.replace("(../figures/", "(results/figures/")
    report_path = report_dir / "Task7_Report.md"
    root_report_path = output_dir / "Task7_Report.md"
    report_path.write_text(report_text, encoding="utf-8")
    root_report_path.write_text(root_report_text, encoding="utf-8")
    print("Generated Task 7 markdown report.")


if __name__ == "__main__":
    main()

