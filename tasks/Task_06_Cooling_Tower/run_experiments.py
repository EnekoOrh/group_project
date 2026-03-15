import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.deterministic import BFGS
from src.algorithms.stochastics import ParticleSwarm, SimulatedAnnealing
from src.benchmarks.cooling_tower import (
    build_scenario_problem,
    evaluate_decision,
    get_all_scenarios,
    get_decision_bounds,
    get_scenario,
    get_solver_settings,
    scenario_to_dict,
)
from src.visualization.cooling_tower_plotting import (
    plot_convergence,
    plot_cross_scenario_area_bar,
    plot_profile_overlay,
    plot_tower_triptych,
)


ALGORITHMS = ["SA", "PSO", "BFGS"]
ALGO_SEED_OFFSETS = {"SA": 0, "PSO": 10000, "BFGS": 20000}
VOLUME_FEASIBILITY_THRESHOLD = 1e-3
HEIGHT_FEASIBILITY_THRESHOLD = 1e-3
SHAPE_FEASIBILITY_THRESHOLD = 1e-6
ENGINEERING_MIN_NECK_RADIUS = 18.0
ENGINEERING_MAX_REL_RADIUS_STEP = 0.35
ENGINEERING_MAX_SECOND_DIFF = 6.0
ENGINEERING_MAX_HEIGHT_RATIO_DEFAULT = 2.5


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_scenarios(arg_value: str, all_ids: List[str]) -> List[str]:
    if not arg_value:
        return all_ids

    requested = [sid.strip() for sid in arg_value.split(",") if sid.strip()]
    unknown = [sid for sid in requested if sid not in all_ids]
    if unknown:
        raise ValueError(f"Unknown scenario ids: {', '.join(unknown)}")

    ordered = [sid for sid in all_ids if sid in requested]
    return ordered


def _create_algorithm(algo_name: str, objective, bounds, dim: int, seed: int, settings: Dict[str, float]):
    if algo_name == "SA":
        return SimulatedAnnealing(
            func=objective,
            bounds=bounds,
            dim=dim,
            seed=seed,
            temp_init=settings["temp_init"],
            cooling_rate=settings["cooling_rate"],
            step_size=settings["step_size"],
            max_evals=int(settings["max_evals"]),
        )

    if algo_name == "PSO":
        return ParticleSwarm(
            func=objective,
            bounds=bounds,
            dim=dim,
            seed=seed,
            w=settings["w"],
            c1=settings["c1"],
            c2=settings["c2"],
            num_particles=int(settings["num_particles"]),
            max_evals=int(settings["max_evals"]),
        )

    if algo_name == "BFGS":
        return BFGS(
            func=objective,
            bounds=bounds,
            dim=dim,
            seed=seed,
            tol=settings["tol"],
            max_iter=int(settings["max_iter"]),
            max_evals=int(settings["max_evals"]),
        )

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def _pick_best_run(run_rows: List[Dict[str, object]]) -> Dict[str, object]:
    feasible_rows = [row for row in run_rows if row["feasible"]]
    if feasible_rows:
        return min(feasible_rows, key=lambda row: row["area"])
    return min(run_rows, key=lambda row: row["penalized_objective"])


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def _mean_ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(arr.size))


def _median_q1_q3(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def _scenario_intent(scenario_id: str) -> Dict[str, str]:
    intent = {
        "S1": {
            "change": "Bounded radii with required ring heights and smoothness.",
            "hypothesis": "Reference constrained case should be feasible and stable across methods.",
            "relevance": "Baseline required assignment case for fair method comparison.",
        },
        "S2": {
            "change": "Wide radii range with reduced practical constraints.",
            "hypothesis": "Constraint relaxation improves search freedom but can produce impractical geometries.",
            "relevance": "Separates mathematical feasibility from engineering practicality.",
        },
        "S3": {
            "change": "Bounded radii with uniform height discretization.",
            "hypothesis": "Uniform axial spacing isolates radial optimization effects.",
            "relevance": "Controls for geometry discretization bias in radii-only optimization.",
        },
        "S4": {
            "change": "Higher radial discretization (m=12) with smoothness.",
            "hypothesis": "Higher dimensionality increases search difficulty but allows finer profiles.",
            "relevance": "Tests resolution sensitivity and smoothness regularization impact.",
        },
        "S5": {
            "change": "Heights-only optimization with bounded heights and fixed radii template.",
            "hypothesis": "Vertical segmentation can be optimized reliably under practical bounds.",
            "relevance": "Isolates height design effects from radius-shape effects.",
        },
        "S6": {
            "change": "Heights-only optimization with wide height bounds.",
            "hypothesis": "Relaxed height limits expose robustness and potential non-practical spacing patterns.",
            "relevance": "Unconstrained-like counterpart to S5 for sensitivity analysis.",
        },
        "S7": {
            "change": "Joint optimization of radii and heights under bounded settings.",
            "hypothesis": "Coupled variables increase nonlinearity and challenge feasibility preservation.",
            "relevance": "Most representative realistic design-optimization setup before stress case.",
        },
        "S8": {
            "change": "Joint optimization with larger target volume and constructability constraints.",
            "hypothesis": "Combined demand increase and constructability controls create a feasibility bottleneck.",
            "relevance": "Industrial stress test of methods under tight and conflicting requirements.",
        },
    }
    return intent.get(
        scenario_id,
        {
            "change": "Scenario variant.",
            "hypothesis": "Assesses method sensitivity to changed constraints.",
            "relevance": "Contributes to robustness evaluation.",
        },
    )


def _feasibility_checks_for_scenario(scenario, metrics: Dict[str, float]) -> List[Dict[str, object]]:
    checks = [
        {
            "name": "volume",
            "value": float(metrics["rel_volume_error"]),
            "threshold": VOLUME_FEASIBILITY_THRESHOLD,
        }
    ]

    heights_are_variables = scenario.decision_mode in ("heights", "joint")
    radii_are_variables = scenario.decision_mode in ("radii", "joint")

    if heights_are_variables:
        checks.append(
            {
                "name": "height",
                "value": float(metrics["rel_height_error"]),
                "threshold": HEIGHT_FEASIBILITY_THRESHOLD,
            }
        )

    if radii_are_variables:
        checks.append(
            {
                "name": "shape",
                "value": float(metrics["monotonic_violation"]),
                "threshold": SHAPE_FEASIBILITY_THRESHOLD,
            }
        )

    for check in checks:
        check["passed"] = bool(float(check["value"]) <= float(check["threshold"]))

    return checks


def _engineering_checks_for_scenario(scenario, metrics: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    radii = np.asarray(metrics["radii"], dtype=float)
    heights = np.asarray(metrics["heights"], dtype=float)
    radii_are_variables = scenario.decision_mode in ("radii", "joint")
    heights_are_variables = scenario.decision_mode in ("heights", "joint")

    checks.append(
        {
            "name": "neck_radius",
            "value": float(np.min(radii)),
            "threshold": ENGINEERING_MIN_NECK_RADIUS,
            "comparator": ">=",
        }
    )

    if radii_are_variables and radii.size >= 2:
        rel_step = np.abs(np.diff(radii)) / np.maximum(np.maximum(radii[:-1], radii[1:]), 1e-9)
        checks.append(
            {
                "name": "rel_radius_step",
                "value": float(np.max(rel_step)),
                "threshold": ENGINEERING_MAX_REL_RADIUS_STEP,
                "comparator": "<=",
            }
        )

    if radii_are_variables and radii.size >= 3:
        second_diff = np.abs(radii[:-2] - 2.0 * radii[1:-1] + radii[2:])
        checks.append(
            {
                "name": "radius_second_diff",
                "value": float(np.max(second_diff)),
                "threshold": ENGINEERING_MAX_SECOND_DIFF,
                "comparator": "<=",
            }
        )

    if heights_are_variables and heights.size >= 2:
        ratio_limit = (
            float(scenario.adjacent_height_ratio_limit)
            if scenario.adjacent_height_ratio_limit is not None
            else ENGINEERING_MAX_HEIGHT_RATIO_DEFAULT
        )
        ratio = np.maximum(
            heights[:-1] / np.maximum(heights[1:], 1e-9),
            heights[1:] / np.maximum(heights[:-1], 1e-9),
        )
        checks.append(
            {
                "name": "adj_height_ratio",
                "value": float(np.max(ratio)),
                "threshold": ratio_limit,
                "comparator": "<=",
            }
        )

    for check in checks:
        if check["comparator"] == "<=":
            check["passed"] = bool(float(check["value"]) <= float(check["threshold"]))
        else:
            check["passed"] = bool(float(check["value"]) >= float(check["threshold"]))

    return checks


def _format_visual_feasibility_line(
    algo: str,
    scenario,
    metrics: Dict[str, float],
    selection_basis: str,
) -> str:
    checks = _feasibility_checks_for_scenario(scenario, metrics)
    failed = [check for check in checks if not check["passed"]]

    def _format_check(check: Dict[str, object], comparator: str) -> str:
        return (
            f"{check['name']} "
            f"({float(check['value']):.3e} {comparator} {float(check['threshold']):.3e})"
        )

    eng_checks = _engineering_checks_for_scenario(scenario, metrics)
    eng_failed = [check for check in eng_checks if not check["passed"]]
    eng_gate_failed = not bool(metrics["feasible"])
    eng_status = " engineering status: FEASIBLE."
    if eng_failed or eng_gate_failed:
        reasons = []
        if eng_gate_failed:
            reasons.append("math_compliance prerequisite not met")
        if eng_failed:
            reasons.extend(
                f"{check['name']} ({float(check['value']):.3e} {check['comparator']} {float(check['threshold']):.3e})"
                for check in eng_failed
            )
        reason_text = ", ".join(reasons)
        eng_status = f" engineering status: INFEASIBLE ({reason_text})."

    if bool(metrics["feasible"]):
        reason = ", ".join(_format_check(check, "<=") for check in checks)
        return f"- {algo}: **FEASIBLE** - passes {reason}.{eng_status}"

    if not failed:
        failed = checks
    reason = ", ".join(_format_check(check, ">") for check in failed)
    line = f"- {algo}: **INFEASIBLE** - fails {reason}."
    if selection_basis == "best_penalized_objective":
        line += " visualized run is lowest penalized objective among infeasible runs."
    return line + eng_status


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _export_latex_csv_inputs(
    output_dir: str,
    scenario_summary_rows: List[Dict[str, object]],
    algorithm_summary_rows: List[Dict[str, object]],
) -> None:
    _ensure_dir(output_dir)

    scenario_table_rows = []
    for row in scenario_summary_rows:
        scenario_table_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "algorithm": row["algorithm"],
                "mean_area": row["mean_area"],
                "std_area": row["std_area"],
                "mean_rel_volume_error": row["mean_rel_volume_error"],
                "mean_evals": row["mean_evals"],
                "mean_time_s": row["mean_time_s"],
                "feasibility_rate": row["feasibility_rate"],
                "engineering_feasibility_rate": row["engineering_feasibility_rate"],
            }
        )

    _write_csv(
        os.path.join(output_dir, "scenario_summary_table.csv"),
        scenario_table_rows,
        fieldnames=[
            "scenario_id",
            "algorithm",
            "mean_area",
            "std_area",
            "mean_rel_volume_error",
            "mean_evals",
            "mean_time_s",
            "feasibility_rate",
            "engineering_feasibility_rate",
        ],
    )

    algorithm_table_rows = []
    for row in algorithm_summary_rows:
        algorithm_table_rows.append(
            {
                "algorithm": row["algorithm"],
                "total_runs": row["total_runs"],
                "mean_area": row["mean_area"],
                "mean_rel_volume_error": row["mean_rel_volume_error"],
                "mean_evals": row["mean_evals"],
                "mean_time_s": row["mean_time_s"],
                "overall_feasibility_rate": row["overall_feasibility_rate"],
                "overall_engineering_feasibility_rate": row["overall_engineering_feasibility_rate"],
            }
        )

    _write_csv(
        os.path.join(output_dir, "algorithm_summary_table.csv"),
        algorithm_table_rows,
        fieldnames=[
            "algorithm",
            "total_runs",
            "mean_area",
            "mean_rel_volume_error",
            "mean_evals",
            "mean_time_s",
            "overall_feasibility_rate",
            "overall_engineering_feasibility_rate",
        ],
    )

    key_findings_rows = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in scenario_summary_rows})
    for sid in scenario_ids:
        rows = [row for row in scenario_summary_rows if row["scenario_id"] == sid]
        if not rows:
            continue

        best_tradeoff_row = max(rows, key=lambda item: (float(item["feasibility_rate"]), -float(item["mean_area"])))
        fastest_row = min(rows, key=lambda item: float(item["mean_evals"]))

        key_findings_rows.append(
            {
                "scenario_id": sid,
                "best_tradeoff_algorithm": best_tradeoff_row["algorithm"],
                "best_tradeoff_feasibility_rate": best_tradeoff_row["feasibility_rate"],
                "best_tradeoff_mean_area": best_tradeoff_row["mean_area"],
                "fastest_algorithm": fastest_row["algorithm"],
                "fastest_mean_evals": fastest_row["mean_evals"],
            }
        )

    _write_csv(
        os.path.join(output_dir, "key_findings_table.csv"),
        key_findings_rows,
        fieldnames=[
            "scenario_id",
            "best_tradeoff_algorithm",
            "best_tradeoff_feasibility_rate",
            "best_tradeoff_mean_area",
            "fastest_algorithm",
            "fastest_mean_evals",
        ],
    )


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_failed_check_names(checks: List[Dict[str, object]]) -> str:
    failed = [str(check["name"]) for check in checks if not bool(check.get("passed", False))]
    return ", ".join(failed) if failed else "None"


def _generate_report(
    report_path: str,
    scenario_ids: List[str],
    scenario_definitions: List[Dict[str, object]],
    scenario_summary_rows: List[Dict[str, object]],
    algorithm_summary_rows: List[Dict[str, object]],
    visual_selection_by_scenario: Dict[str, Dict[str, Dict[str, object]]],
    include_3d: bool,
    runs: int,
    seed_offset: int,
    figures_dir: str,
) -> None:
    scenario_summary_by_id = defaultdict(list)
    for row in scenario_summary_rows:
        scenario_summary_by_id[row["scenario_id"]].append(row)

    lines = []
    report_dir = os.path.dirname(os.path.abspath(report_path))
    figure_link_prefix = os.path.relpath(os.path.abspath(figures_dir), report_dir).replace(os.sep, "/")

    def _figure_link(filename: str) -> str:
        return f"{figure_link_prefix}/{filename}" if figure_link_prefix != "." else filename

    lines.append("# Task 6 Report: Hyperboloid Cooling Tower Optimization")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This study addresses constrained shape optimization of a hyperboloid cooling tower, where the engineering target is to reduce lateral shell area while preserving required cooling capacity."
    )
    lines.append(
        "The optimization is performed with Simulated Annealing (SA), Particle Swarm Optimization (PSO), and BFGS under explicit feasibility constraints on volume, total height (when variable), and hyperboloid monotonicity (when radii are variable)."
    )
    lines.append(
        "Eight scenarios are used to increase complexity from constrained baseline cases to a high-demand constructability stress case, allowing method behavior to be compared under a controlled progression of design difficulty."
    )
    lines.append(
        "Success is evaluated first by feasibility robustness, then by feasible shell-area quality and computational efficiency."
    )
    lines.append("")
    lines.append("## 2. Geometry Model")
    lines.append("The cooling-tower shell is modeled as a stack of conical frustums. For segment `i`, the slant height is:")
    lines.append("")
    lines.append(r"$$s_i = \sqrt{(r_i - r_{i-1})^2 + h_i^2}$$")
    lines.append("")
    lines.append("The lateral shell area and enclosed volume of each frustum follow the same notation used in the benchmark implementation:")
    lines.append("")
    lines.append(r"$$A_i = \pi (r_{i-1} + r_i) s_i$$")
    lines.append("")
    lines.append(r"$$V_i = \frac{\pi h_i}{3} \left(r_{i-1}^2 + r_{i-1} r_i + r_i^2\right)$$")
    lines.append("")
    lines.append("The total tower area and total tower volume are then obtained by summing the segment contributions:")
    lines.append("")
    lines.append(r"$$A = \sum_{i=1}^{m} A_i$$")
    lines.append("")
    lines.append(r"$$V = \sum_{i=1}^{m} V_i$$")
    lines.append("")
    lines.append(
        "Lateral shell area is optimized (top and bottom caps excluded), which directly corresponds to shell-construction material for fixed end radii."
    )
    lines.append(
        "Analytic gradients were used for both `A` and `V`, and for all penalty terms, so BFGS receives exact first-order information."
    )
    lines.append("")
    lines.append("### 2.1 Design Variables by Scenario Type")
    lines.append(
        "In `radii` mode, interior radii `r1..r_{m-1}` are optimized while ring heights are fixed from predefined `z` levels."
    )
    lines.append("In `heights` mode, segment heights `h1..hm` are optimized while ring radii are fixed.")
    lines.append("In `joint` mode, both interior radii and segment heights are optimized simultaneously.")
    lines.append("")
    lines.append("## 3. Optimization Methods")
    lines.append("")
    lines.append("### 3.1 Simulated Annealing (SA)")
    lines.append(
        "SA performs single-solution stochastic search. At each evaluation, a local perturbation is proposed and accepted if it improves objective, or probabilistically accepted otherwise using the temperature schedule."
    )
    lines.append(
        "This supports escape from local minima early in search and gradual exploitation as temperature cools."
    )
    lines.append("")
    lines.append("### 3.2 Particle Swarm Optimization (PSO)")
    lines.append(
        "PSO evolves a population of particles with velocity updates using inertia, cognitive pull to personal best, and social pull to global best."
    )
    lines.append(
        "It is typically strong on global exploration and multimodal landscapes, but can require many evaluations to satisfy strict constraints."
    )
    lines.append("")
    lines.append("### 3.3 BFGS (Deterministic Quasi-Newton)")
    lines.append(
        "BFGS updates an inverse-Hessian approximation and computes descent directions with line search, yielding rapid local convergence when gradients are informative."
    )
    lines.append(
        "It is efficient in evaluation count but can be sensitive to initialization and non-convex penalties."
    )
    lines.append("")
    lines.append("## 4. Penalized Objective and Geometrical Constraints")
    lines.append(
        "The optimized scalar objective is `J(x) = A(x) + P_volume + P_height + P_shape + P_smooth + P_bounds + P_ratio` (terms enabled per scenario)."
    )
    lines.append("")
    lines.append("### 4.1 Constraint Terms and Their Function")
    lines.append(
        "- `P_volume`: enforces target cooling capacity by penalizing relative deviation from target volume."
    )
    lines.append(
        "- `P_height`: enforces total tower height in scenarios where heights vary, preserving comparable overall structure."
    )
    lines.append(
        "- `P_shape`: enforces hyperboloid-like monotonic contraction to neck and expansion above neck."
    )
    lines.append(
        "- `P_smooth`: penalizes second differences in radii to avoid oscillatory/sawtooth shell profiles."
    )
    lines.append(
        "- `P_bounds`: hinge penalty for design-variable bounds to keep variables in practical and constructable ranges."
    )
    lines.append(
        "- `P_ratio` (S8): limits adjacent-height ratios to promote constructability and avoid abrupt segment transitions."
    )
    lines.append("")
    lines.append("### 4.2 Feasibility Rule")
    lines.append("- Relative volume error must be `<= 1e-3`.")
    lines.append("- If heights vary, relative total-height error must be `<= 1e-3`.")
    lines.append("- If radii vary, monotonic hyperboloid violation must be `<= 1e-6`.")
    lines.append("")
    lines.append("### 4.3 Engineering Feasibility Criteria")
    lines.append(
        "Engineering feasibility is evaluated in addition to mathematical compliance in order to flag designs that are formally valid but impractical. "
        "The checks are: minimum neck radius `>= 18.0 m`, maximum relative adjacent radius step `<= 0.35`, maximum radius second-difference `<= 6.0`, "
        "and, when heights vary, maximum adjacent-height ratio `<= 2.5` (or scenario-specific limit, e.g. `1.8` in S8)."
    )
    lines.append("In the scenario result sections, both statuses are reported side-by-side for each shown 3D tower.")
    lines.append("")
    lines.append("## 5. Optimization Protocol")
    lines.append(
        f"Each algorithm is run {runs} times per scenario with deterministic seed control (seed offset {seed_offset}). "
        "SA uses `temp_init=120`, `cooling_rate=0.997`, and scenario-dependent step size; PSO uses `w=0.65`, `c1=1.6`, `c2=1.7` with particle count tied to dimension; "
        "BFGS uses `tol=1e-7` and `max_iter=1200`."
    )
    lines.append(
        "All runs share the same penalized objective structure (volume, height sum where applicable, shape monotonicity, smoothness, bounds, and S8 ratio constraint)."
    )
    lines.append("")
    lines.append(
        "Stochastic budgets are 10k evaluations for lower-dimensional setups and 16k for joint/high-dimensional setups. BFGS uses smaller budgets due to faster local convergence."
    )
    lines.append("")
    lines.append("### 5.1 Methodology Quality Checks")
    lines.append(
        "Feasibility is evaluated independently from objective value using explicit thresholds, and scenario comparisons report both all-run and feasible-only statistics."
    )
    lines.append(
        "Because evaluation budgets differ between SA/PSO and BFGS, quality and efficiency are interpreted together rather than from a single metric."
    )
    lines.append("For statistically tighter uncertainty on difficult cases, hard scenarios should be repeated with at least 20 runs.")
    lines.append("")
    lines.append("## 6. Scenario Definitions")

    scenario_table_rows = []
    for scenario in scenario_definitions:
        scenario_table_rows.append(
            [
                str(scenario["scenario_id"]),
                str(scenario["decision_mode"]),
                str(scenario["m"]),
                f"{float(scenario['target_volume']):.0f}",
                f"{float(scenario['r0']):.1f}",
                f"{float(scenario['rm']):.1f}",
                str(scenario["description"]),
            ]
        )

    lines.append(
        _markdown_table(
            ["ID", "Mode", "m", "Target Volume", "r0", "rm", "Description"],
            scenario_table_rows,
        )
    )
    lines.append("")
    lines.append("### 6.1 Why These 8 Scenarios Are Relevant")
    for sid in scenario_ids:
        intent = _scenario_intent(sid)
        hypothesis = str(intent["hypothesis"]).rstrip(".")
        relevance = str(intent["relevance"]).rstrip(".")
        relevance_l = relevance[0].lower() + relevance[1:] if relevance else relevance
        direct_verb_starts = ("separates", "controls", "tests", "isolates", "improves", "reduces")
        if relevance_l.startswith(direct_verb_starts):
            relevance_sentence = f"This is relevant because it {relevance_l}."
        else:
            relevance_sentence = f"This is relevant because it serves as {relevance_l}."
        lines.append(
            f"{sid}. Design change: {intent['change']} Hypothesis: {hypothesis}. {relevance_sentence}"
        )
        lines.append("")

    lines.append("## 7. Cross-Algorithm Summary")
    best_math_algo = max(algorithm_summary_rows, key=lambda row: float(row["overall_feasibility_rate"]))
    best_eng_algo = max(
        algorithm_summary_rows, key=lambda row: float(row["overall_engineering_feasibility_rate"])
    )
    fastest_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_evals"]))
    lowest_area_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_area"]))

    lines.append(
        "This summary aggregates all selected scenarios for each method. Mean and median area describe central tendency, IQR and 95% CI indicate dispersion, and the two feasibility columns separate strict mathematical compliance from the stricter engineering constructability screen."
    )
    lines.append("")
    algo_rows = []
    for row in algorithm_summary_rows:
        algo_rows.append(
            [
                str(row["algorithm"]),
                str(int(row["total_runs"])),
                f"{float(row['mean_area']):.2f}",
                f"{float(row['median_area']):.2f}",
                f"{float(row['area_iqr_q1']):.2f}-{float(row['area_iqr_q3']):.2f}",
                f"{float(row['area_ci95']):.2f}",
                f"{float(row['mean_rel_volume_error']):.3e}",
                f"{float(row['mean_evals']):.1f}",
                f"{float(row['mean_time_s']):.4f}",
                f"{100.0 * float(row['overall_feasibility_rate']):.1f}%",
                f"{100.0 * float(row['overall_engineering_feasibility_rate']):.1f}%",
                f"{float(row['feasible_area_mean']):.2f}" if row["feasible_runs"] > 0 else "N/A",
                f"{float(row['feasible_area_median']):.2f}" if row["feasible_runs"] > 0 else "N/A",
            ]
        )

    lines.append(
        _markdown_table(
            [
                "Algorithm",
                "Total Runs",
                "Mean Area",
                "Median Area",
                "IQR (Q1-Q3)",
                "95% CI",
                "Mean Rel Vol Err",
                "Mean Evals",
                "Mean Time (s)",
                "Math Feasibility",
                "Engineering Feasibility",
                "Feasible Mean Area",
                "Feasible Median Area",
            ],
            algo_rows,
        )
    )
    lines.append("")
    lines.append(f"![Cross-scenario area comparison]({_figure_link('cross_scenario_area_bar.png')})")
    lines.append("")
    if best_math_algo["algorithm"] == best_eng_algo["algorithm"]:
        lines.append(
            f"Across the current aggregated run set, {best_math_algo['algorithm']} achieved the highest mathematical compliance "
            f"({100.0 * float(best_math_algo['overall_feasibility_rate']):.1f}%) and the highest engineering feasibility "
            f"({100.0 * float(best_eng_algo['overall_engineering_feasibility_rate']):.1f}%). "
            f"{fastest_algo['algorithm']} remained the most evaluation-efficient, while {lowest_area_algo['algorithm']} delivered the lowest raw mean area."
        )
    else:
        lines.append(
            f"Across the current aggregated run set, {best_math_algo['algorithm']} achieved the highest mathematical compliance "
            f"({100.0 * float(best_math_algo['overall_feasibility_rate']):.1f}%), whereas {best_eng_algo['algorithm']} achieved the highest engineering feasibility "
            f"({100.0 * float(best_eng_algo['overall_engineering_feasibility_rate']):.1f}%). "
            f"{fastest_algo['algorithm']} remained the most evaluation-efficient, while {lowest_area_algo['algorithm']} delivered the lowest raw mean area."
        )
    lines.append("")

    lines.append("## 8. Scenario Results")
    for sid in scenario_ids:
        lines.append("")
        lines.append(f"### {sid}")

        rows = []
        for row in sorted(scenario_summary_by_id[sid], key=lambda item: item["algorithm"]):
            rows.append(
                [
                    str(row["algorithm"]),
                    f"{float(row['mean_area']):.2f}",
                    f"{float(row['median_area']):.2f}",
                    f"{float(row['area_iqr_q1']):.2f}-{float(row['area_iqr_q3']):.2f}",
                    f"{float(row['area_ci95']):.2f}",
                    f"{float(row['std_area']):.2f}",
                    f"{float(row['mean_rel_volume_error']):.3e}",
                    f"{float(row['mean_evals']):.1f}",
                    f"{float(row['mean_time_s']):.4f}",
                    f"{100.0 * float(row['feasibility_rate']):.1f}%",
                    f"{100.0 * float(row['engineering_feasibility_rate']):.1f}%",
                    f"{int(row['feasible_runs'])}/{int(row['runs'])}",
                    f"{float(row['feasible_area_median']):.2f}" if int(row["feasible_runs"]) > 0 else "N/A",
                ]
            )

        lines.append(
            _markdown_table(
                [
                    "Algorithm",
                    "Mean Area",
                    "Median Area",
                    "IQR (Q1-Q3)",
                    "95% CI",
                    "Std Area",
                    "Mean Rel Vol Err",
                    "Mean Evals",
                    "Mean Time (s)",
                    "Math Feasibility",
                    "Engineering Feasibility",
                    "Feasible Runs",
                    "Feasible Median Area",
                ],
                rows,
            )
        )
        lines.append("")
        lines.append(f"![{sid} convergence]({_figure_link(f'{sid}_convergence.png')})")
        lines.append("")
        lines.append(f"![{sid} profile]({_figure_link(f'{sid}_profile_overlay.png')})")
        lines.append("")
        scenario = get_scenario(sid)
        selected_visual_runs = visual_selection_by_scenario.get(sid, {})
        if include_3d:
            lines.append(f"![{sid} 3D towers]({_figure_link(f'{sid}_tower_3d.png')})")
            lines.append("")
            lines.append("Compliance and feasibility status of the shown 3D towers (same selected runs as profile overlay):")
            status_rows: List[List[str]] = []
            for algo in ALGORITHMS:
                selected = selected_visual_runs.get(algo)
                if not selected:
                    continue
                metrics = selected["metrics"]
                math_checks = _feasibility_checks_for_scenario(scenario, metrics)
                math_ok = all(bool(check["passed"]) for check in math_checks)
                math_status = "COMPLIANT" if math_ok else "NON-COMPLIANT"
                math_failed = _format_failed_check_names(math_checks)

                eng_checks = _engineering_checks_for_scenario(scenario, metrics)
                eng_ok = math_ok and all(bool(check["passed"]) for check in eng_checks)
                eng_status = "FEASIBLE" if eng_ok else "INFEASIBLE"
                eng_failed = _format_failed_check_names(eng_checks)
                if not math_ok:
                    eng_failed = "math_compliance" if eng_failed == "None" else f"math_compliance, {eng_failed}"

                status_rows.append([algo, math_status, math_failed, eng_status, eng_failed])

            lines.append(
                _markdown_table(
                    ["Algorithm", "Math Status", "Math Failed Checks", "Engineering Status", "Engineering Failed Checks"],
                    status_rows,
                )
            )
            lines.append("")

        sid_rows = scenario_summary_by_id[sid]
        sid_max_feas = max(float(item["feasibility_rate"]) for item in sid_rows)
        fastest = min(sid_rows, key=lambda item: float(item["mean_evals"]))
        intent = _scenario_intent(sid)
        best_math_rate_row = max(
            sid_rows,
            key=lambda item: (
                float(item["feasibility_rate"]),
                -(float(item["feasible_area_median"]) if int(item["feasible_runs"]) > 0 else float("inf")),
            ),
        )
        best_eng_rate_row = max(
            sid_rows,
            key=lambda item: (
                float(item["engineering_feasibility_rate"]),
                -(
                    float(item["engineering_feasible_area_median"])
                    if int(item["engineering_feasible_runs"]) > 0
                    else float("inf")
                ),
            ),
        )
        math_cost_candidates = [row for row in sid_rows if int(row["feasible_runs"]) > 0]
        eng_cost_candidates = [row for row in sid_rows if int(row["engineering_feasible_runs"]) > 0]

        paragraph = (
            f"In {sid}, the design change is: {intent['change']} The working hypothesis is: {intent['hypothesis']} "
            f"The highest mathematical compliance rate was achieved by {best_math_rate_row['algorithm']} "
            f"({100.0 * float(best_math_rate_row['feasibility_rate']):.1f}%), while the highest engineering-feasibility rate was achieved by "
            f"{best_eng_rate_row['algorithm']} ({100.0 * float(best_eng_rate_row['engineering_feasibility_rate']):.1f}%). "
        )
        if sid_max_feas <= 0.0:
            best_infeasible = min(sid_rows, key=lambda item: float(item["best_penalized_objective"]))
            paragraph += (
                f"No mathematically compliant run was found in this configuration, and the lowest penalized infeasible result came from {best_infeasible['algorithm']}. "
            )
        if math_cost_candidates:
            best_math_cost = min(math_cost_candidates, key=lambda item: float(item["feasible_area_median"]))
            paragraph += (
                f"Among mathematically compliant runs, the lowest median area (cost proxy) was obtained by {best_math_cost['algorithm']} "
                f"at {float(best_math_cost['feasible_area_median']):.2f} m². "
            )
        else:
            paragraph += "No mathematically compliant runs were available for cost comparison. "
        if eng_cost_candidates:
            best_eng_cost = min(eng_cost_candidates, key=lambda item: float(item["engineering_feasible_area_median"]))
            paragraph += (
                f"Among engineering-feasible runs, the lowest median area was obtained by {best_eng_cost['algorithm']} "
                f"at {float(best_eng_cost['engineering_feasible_area_median']):.2f} m². "
            )
        else:
            paragraph += "No engineering-feasible runs were available for cost comparison. "
        paragraph += (
            f"In terms of evaluation efficiency, {fastest['algorithm']} required the fewest mean evaluations "
            f"({float(fastest['mean_evals']):.1f}). Practical selection should still combine strict mathematical compliance with constructability review from the profile shape."
        )
        lines.append(paragraph)
        lines.append("")

    lines.append("## 9. Key Findings")
    lines.append("")
    for sid in scenario_ids:
        rows = scenario_summary_by_id[sid]
        if not rows:
            continue

        max_feas = max(float(row["feasibility_rate"]) for row in rows)
        if max_feas <= 0.0:
            best_penalized_row = min(rows, key=lambda row: float(row["best_penalized_objective"]))
            lines.append(
                f"{sid}: no mathematically compliant solutions were found (all algorithms at 0.0% compliance), and the lowest penalized infeasible objective was obtained by {best_penalized_row['algorithm']}."
            )
        else:
            best_row = max(
                rows,
                key=lambda row: (
                    float(row["feasibility_rate"]),
                    -(float(row["feasible_area_median"]) if int(row["feasible_runs"]) > 0 else float("inf")),
                ),
            )
            lines.append(
                f"{sid}: the best mathematical compliance-area tradeoff was achieved by {best_row['algorithm']} "
                f"(compliance {100.0 * float(best_row['feasibility_rate']):.1f}%, feasible median area {float(best_row['feasible_area_median']):.2f} m²)."
            )

        eng_rows = [row for row in rows if int(row["engineering_feasible_runs"]) > 0]
        if eng_rows:
            best_eng_row = max(
                eng_rows,
                key=lambda row: (
                    float(row["engineering_feasibility_rate"]),
                    -float(row["engineering_feasible_area_median"]),
                ),
            )
            lines.append(
                f"{sid}: the best engineering-feasibility tradeoff was achieved by {best_eng_row['algorithm']} "
                f"(engineering feasibility {100.0 * float(best_eng_row['engineering_feasibility_rate']):.1f}%, engineering-feasible median area {float(best_eng_row['engineering_feasible_area_median']):.2f} m²)."
            )
        else:
            lines.append(f"{sid}: current constructability checks yielded no engineering-feasible solutions.")

        fastest_row = min(rows, key=lambda row: float(row["mean_evals"]))
        lines.append(
            f"{sid}: the lowest mean evaluation count was achieved by {fastest_row['algorithm']} "
            f"({float(fastest_row['mean_evals']):.1f} evals)."
        )

    hard_scenarios = []
    for sid in scenario_ids:
        rows = scenario_summary_by_id[sid]
        if rows and max(float(row["feasibility_rate"]) for row in rows) <= 0.0:
            hard_scenarios.append(sid)

    if hard_scenarios:
        lines.append("")
        lines.append(
            f"Mathematical-compliance bottleneck scenarios: {', '.join(hard_scenarios)} "
            "(all tested methods yielded 0% compliant runs under current constraints)."
        )

    engineering_hard_scenarios = []
    for sid in scenario_ids:
        rows = scenario_summary_by_id[sid]
        if rows and max(float(row["engineering_feasibility_rate"]) for row in rows) <= 0.0:
            engineering_hard_scenarios.append(sid)

    if engineering_hard_scenarios:
        lines.append(
            f"Engineering-feasibility bottleneck scenarios: {', '.join(engineering_hard_scenarios)} "
            "(no algorithm produced engineering-feasible solutions under current checks)."
        )

    lines.append("")
    lines.append("## 10. Discussion and Conclusions")
    lines.append("")

    best_feas_algo = max(algorithm_summary_rows, key=lambda row: float(row["overall_feasibility_rate"]))
    best_eng_feas_algo = max(
        algorithm_summary_rows, key=lambda row: float(row["overall_engineering_feasibility_rate"])
    )
    fastest_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_evals"]))
    lowest_area_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_area"]))

    lines.append(
        f"Across all scenarios, **{best_feas_algo['algorithm']}** achieved the highest overall mathematical compliance "
        f"({100.0 * float(best_feas_algo['overall_feasibility_rate']):.1f}%)."
    )
    lines.append(
        f"Across all scenarios, **{best_eng_feas_algo['algorithm']}** achieved the highest overall engineering feasibility "
        f"({100.0 * float(best_eng_feas_algo['overall_engineering_feasibility_rate']):.1f}%)."
    )
    lines.append(
        f"**{fastest_algo['algorithm']}** required the fewest evaluations on average "
        f"({float(fastest_algo['mean_evals']):.1f}), confirming its efficiency for local convergence."
    )
    lines.append(
        f"By raw mean area across all runs, **{lowest_area_algo['algorithm']}** produced the lowest average shell area "
        f"({float(lowest_area_algo['mean_area']):.2f} m²), but this must be interpreted jointly with compliance and feasibility rates."
    )

    feasible_median_candidates = [row for row in algorithm_summary_rows if int(row["feasible_runs"]) > 0]
    if feasible_median_candidates:
        best_feasible_median_algo = min(
            feasible_median_candidates,
            key=lambda row: float(row["feasible_area_median"]),
        )
        lines.append(
            f"Considering compliant runs only, **{best_feasible_median_algo['algorithm']}** achieved the lowest compliant median area "
            f"({float(best_feasible_median_algo['feasible_area_median']):.2f} m²)."
        )

    engineering_feasible_median_candidates = [
        row for row in algorithm_summary_rows if int(row["engineering_feasible_runs"]) > 0
    ]
    if engineering_feasible_median_candidates:
        best_eng_feasible_median_algo = min(
            engineering_feasible_median_candidates,
            key=lambda row: float(row["engineering_feasible_area_median"]),
        )
        lines.append(
            f"On engineering-feasible central tendency, **{best_eng_feasible_median_algo['algorithm']}** achieved the lowest engineering-feasible median area "
            f"({float(best_eng_feasible_median_algo['engineering_feasible_area_median']):.2f} m²)."
        )

    lines.append("")
    lines.append(
        "These comparisons should be interpreted with budget asymmetry in mind: SA and PSO use larger evaluation budgets than BFGS in most scenarios, so quality and efficiency must be read together."
    )
    lines.append("")
    lines.append(
        "Result patterns are consistent with expected method behavior: BFGS is computationally efficient and strong when gradients and local curvature align with feasible basins; "
        "SA is often robust under hard nonlinear penalties because probabilistic acceptance helps transition between constrained basins; "
        "PSO explores broadly but can underperform compliance when penalty landscapes are steep and high-dimensional."
    )
    lines.append("")
    lines.append(
        "The split between mathematical compliance and engineering feasibility is necessary for this task. Some shapes can satisfy the formal optimization constraints yet remain impractical because of neck narrowing, abrupt radius transitions, or uneven segment heights. Those cases should be reported as compliant but not engineering-feasible rather than accepted as final designs."
    )
    lines.append("")
    lines.append(
        "For practical engineering usage in this problem family, the recommended workflow is a hybrid strategy: use SA or PSO for broader search and feasible-region discovery, then warm-start BFGS for rapid refinement of the best compliant candidate after constructability screening."
    )
    lines.append("")
    if hard_scenarios or engineering_hard_scenarios:
        bottlenecks = sorted(set(hard_scenarios + engineering_hard_scenarios))
        lines.append(
            f"The principal bottleneck scenarios are {', '.join(bottlenecks)}. These cases combine tighter capacity, geometric, and constructability demands, so future refinement should focus on adaptive penalties, better initialization, and stronger smoothing or ratio constraints."
        )
        lines.append("")
    lines.append("### 10.1 Planned Structural Validation in Abaqus")
    lines.append("")
    lines.append(
        "The next step is to transfer the best candidate geometries into Abaqus shell models with realistic material properties and shell-thickness assumptions for reinforced-concrete cooling towers. These models will be checked under self-weight and representative environmental loading to evaluate displacement, stress distribution, and buckling sensitivity."
    )
    lines.append("")
    lines.append(
        "Multiple candidate towers will be compared, not only the minimum-area design, so selection can be based on engineering feasibility as well as geometric efficiency. The Abaqus results will then be used to refine the optimization model by tightening feasibility rules or adding constraints linked to smoothness, curvature, and stability."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6 cooling tower experiments")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset applied to all runs")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per algorithm per scenario")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario IDs (default: all S1..S8)",
    )
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D tower plots")
    parser.add_argument(
        "--report-outdir",
        type=str,
        default="",
        help="Optional directory for report artifacts (Report.md). Default: results/reports",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional root directory under which Task 6 generated outputs are written as results/{data,figures,latex,reports}.",
    )
    parser.set_defaults(export_latex_tables=True)
    parser.add_argument(
        "--export-latex-tables",
        dest="export_latex_tables",
        action="store_true",
        help="Export CSV inputs under results/latex for LaTeX table generation (default: enabled).",
    )
    parser.add_argument(
        "--no-export-latex-tables",
        dest="export_latex_tables",
        action="store_false",
        help="Disable CSV export used by the LaTeX report pipeline.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.abspath(args.output_root) if args.output_root else script_dir
    results_dir = os.path.join(output_root, "results")
    figures_dir = os.path.join(results_dir, "figures")
    data_dir = os.path.join(results_dir, "data")
    report_outdir = os.path.abspath(args.report_outdir) if args.report_outdir else os.path.join(results_dir, "reports")
    _ensure_dir(results_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(data_dir)
    _ensure_dir(report_outdir)

    all_scenarios = get_all_scenarios()
    all_ids = sorted(all_scenarios.keys())
    scenario_ids = _parse_scenarios(args.scenarios, all_ids)

    raw_rows: List[Dict[str, object]] = []
    scenario_summary_rows: List[Dict[str, object]] = []
    algorithm_summary_rows: List[Dict[str, object]] = []
    visual_selection_by_scenario: Dict[str, Dict[str, Dict[str, object]]] = {}

    print(f"Running Task 6 scenarios: {', '.join(scenario_ids)}")
    print(f"Runs per algorithm: {args.runs} | Seed offset: {args.seed_offset}")

    for sid in scenario_ids:
        scenario = get_scenario(sid)
        problem = build_scenario_problem(scenario)
        bounds = problem["bounds"]
        dim = int(problem["dim"])
        objective = problem["objective"]

        solver_settings = get_solver_settings(scenario)

        print(f"\n=== Scenario {sid}: {scenario.description}")

        scenario_run_rows = defaultdict(list)
        histories_by_algorithm: Dict[str, List[List[tuple]]] = defaultdict(list)
        profiles_by_algorithm: Dict[str, Dict[str, np.ndarray]] = {}
        visual_selection_by_scenario[sid] = {}

        max_eval_for_plot = 0

        for algo_name in ALGORITHMS:
            settings = solver_settings[algo_name]
            max_eval_for_plot = max(max_eval_for_plot, int(settings["max_evals"]))
            print(f"  -> {algo_name} ({args.runs} runs)")

            for run_idx in range(args.runs):
                seed = int(args.seed_offset + ALGO_SEED_OFFSETS[algo_name] + run_idx)
                algo = _create_algorithm(algo_name, objective, bounds, dim, seed, settings)
                result = algo.solve()

                metrics = evaluate_decision(scenario, result["best_x"])
                eng_checks = _engineering_checks_for_scenario(scenario, metrics)
                engineering_feasible = bool(metrics["feasible"]) and all(bool(check["passed"]) for check in eng_checks)
                eng_failed = [str(check["name"]) for check in eng_checks if not check["passed"]]

                row = {
                    "scenario_id": sid,
                    "algorithm": algo_name,
                    "run_index": run_idx,
                    "seed": seed,
                    "penalized_objective": float(result["best_val"]),
                    "area": float(metrics["area"]),
                    "volume": float(metrics["volume"]),
                    "total_height": float(metrics["total_height"]),
                    "rel_volume_error": float(metrics["rel_volume_error"]),
                    "rel_height_error": float(metrics["rel_height_error"]),
                    "monotonic_violation": float(metrics["monotonic_violation"]),
                    "feasible": bool(metrics["feasible"]),
                    "engineering_feasible": bool(engineering_feasible),
                    "engineering_failed_checks": ",".join(eng_failed),
                    "n_evals": int(result["n_evals"]),
                    "time_s": float(result["time"]),
                    "budget": int(settings["max_evals"]),
                    "decision_vector": json.dumps(np.asarray(result["best_x"], dtype=float).tolist()),
                }

                raw_rows.append(row)
                scenario_run_rows[algo_name].append(row)
                histories_by_algorithm[algo_name].append(result["history"])

            best_row = _pick_best_run(scenario_run_rows[algo_name])
            best_x = np.array(json.loads(best_row["decision_vector"]), dtype=float)
            best_metrics = evaluate_decision(scenario, best_x)
            best_eng_checks = _engineering_checks_for_scenario(scenario, best_metrics)
            best_engineering_feasible = bool(best_metrics["feasible"]) and all(
                bool(check["passed"]) for check in best_eng_checks
            )
            selection_basis = "best_feasible_area" if bool(best_row["feasible"]) else "best_penalized_objective"
            profiles_by_algorithm[algo_name] = {
                "radii": best_metrics["radii"],
                "z": best_metrics["z"],
                "area": best_metrics["area"],
                "rel_volume_error": best_metrics["rel_volume_error"],
            }
            visual_selection_by_scenario[sid][algo_name] = {
                "run_index": int(best_row["run_index"]),
                "seed": int(best_row["seed"]),
                "selection_basis": selection_basis,
                "metrics": {
                    "feasible": bool(best_metrics["feasible"]),
                    "area": float(best_metrics["area"]),
                    "rel_volume_error": float(best_metrics["rel_volume_error"]),
                    "rel_height_error": float(best_metrics["rel_height_error"]),
                    "monotonic_violation": float(best_metrics["monotonic_violation"]),
                    "engineering_feasible": bool(best_engineering_feasible),
                    "radii": np.asarray(best_metrics["radii"], dtype=float).tolist(),
                    "heights": np.asarray(best_metrics["heights"], dtype=float).tolist(),
                },
            }

            rows = scenario_run_rows[algo_name]
            areas_all = [float(row["area"]) for row in rows]
            feasible_areas = [float(row["area"]) for row in rows if row["feasible"]]
            engineering_feasible_areas = [float(row["area"]) for row in rows if row["engineering_feasible"]]
            all_stats = _median_q1_q3(areas_all)
            feas_stats = _median_q1_q3(feasible_areas)
            eng_feas_stats = _median_q1_q3(engineering_feasible_areas)

            summary_row = {
                "scenario_id": sid,
                "algorithm": algo_name,
                "runs": len(rows),
                "mean_area": _safe_mean(areas_all),
                "median_area": all_stats["median"],
                "area_iqr_q1": all_stats["q1"],
                "area_iqr_q3": all_stats["q3"],
                "area_ci95": _mean_ci95(areas_all),
                "std_area": _safe_std(areas_all),
                "mean_rel_volume_error": _safe_mean([float(row["rel_volume_error"]) for row in rows]),
                "mean_rel_height_error": _safe_mean([float(row["rel_height_error"]) for row in rows]),
                "mean_monotonic_violation": _safe_mean([float(row["monotonic_violation"]) for row in rows]),
                "mean_evals": _safe_mean([float(row["n_evals"]) for row in rows]),
                "mean_time_s": _safe_mean([float(row["time_s"]) for row in rows]),
                "feasibility_rate": _safe_mean([1.0 if row["feasible"] else 0.0 for row in rows]),
                "engineering_feasibility_rate": _safe_mean(
                    [1.0 if row["engineering_feasible"] else 0.0 for row in rows]
                ),
                "feasible_runs": len(feasible_areas),
                "feasible_area_mean": _safe_mean(feasible_areas) if feasible_areas else float("nan"),
                "feasible_area_median": feas_stats["median"] if feasible_areas else float("nan"),
                "engineering_feasible_runs": len(engineering_feasible_areas),
                "engineering_feasible_area_mean": (
                    _safe_mean(engineering_feasible_areas) if engineering_feasible_areas else float("nan")
                ),
                "engineering_feasible_area_median": (
                    eng_feas_stats["median"] if engineering_feasible_areas else float("nan")
                ),
                "best_feasible_area": min(feasible_areas) if feasible_areas else "",
                "best_penalized_objective": min(float(row["penalized_objective"]) for row in rows),
            }
            scenario_summary_rows.append(summary_row)

        plot_convergence(
            histories_by_algorithm,
            output_path=os.path.join(figures_dir, f"{sid}_convergence.png"),
            title=f"{sid} Convergence (Penalized Objective)",
            max_eval=max_eval_for_plot,
        )

        plot_profile_overlay(
            profiles_by_algorithm,
            output_path=os.path.join(figures_dir, f"{sid}_profile_overlay.png"),
            title=f"{sid} Tower Profile Overlay",
        )

        if not args.no_3d:
            plot_tower_triptych(
                profiles_by_algorithm,
                output_path=os.path.join(figures_dir, f"{sid}_tower_3d.png"),
                title=f"{sid} 3D Tower Designs",
            )

    # Algorithm summary across all selected scenarios and runs.
    for algo_name in ALGORITHMS:
        rows = [row for row in raw_rows if row["algorithm"] == algo_name]
        if not rows:
            continue
        areas_all = [float(row["area"]) for row in rows]
        feasible_areas = [float(row["area"]) for row in rows if row["feasible"]]
        engineering_feasible_areas = [float(row["area"]) for row in rows if row["engineering_feasible"]]
        all_stats = _median_q1_q3(areas_all)
        feas_stats = _median_q1_q3(feasible_areas)
        eng_feas_stats = _median_q1_q3(engineering_feasible_areas)

        algorithm_summary_rows.append(
            {
                "algorithm": algo_name,
                "total_runs": len(rows),
                "mean_area": _safe_mean(areas_all),
                "median_area": all_stats["median"],
                "area_iqr_q1": all_stats["q1"],
                "area_iqr_q3": all_stats["q3"],
                "area_ci95": _mean_ci95(areas_all),
                "mean_rel_volume_error": _safe_mean([float(row["rel_volume_error"]) for row in rows]),
                "mean_rel_height_error": _safe_mean([float(row["rel_height_error"]) for row in rows]),
                "mean_evals": _safe_mean([float(row["n_evals"]) for row in rows]),
                "mean_time_s": _safe_mean([float(row["time_s"]) for row in rows]),
                "overall_feasibility_rate": _safe_mean([1.0 if row["feasible"] else 0.0 for row in rows]),
                "overall_engineering_feasibility_rate": _safe_mean(
                    [1.0 if row["engineering_feasible"] else 0.0 for row in rows]
                ),
                "feasible_runs": len(feasible_areas),
                "feasible_area_mean": _safe_mean(feasible_areas) if feasible_areas else float("nan"),
                "feasible_area_median": feas_stats["median"] if feasible_areas else float("nan"),
                "engineering_feasible_runs": len(engineering_feasible_areas),
                "engineering_feasible_area_mean": (
                    _safe_mean(engineering_feasible_areas) if engineering_feasible_areas else float("nan")
                ),
                "engineering_feasible_area_median": (
                    eng_feas_stats["median"] if engineering_feasible_areas else float("nan")
                ),
            }
        )

    plot_cross_scenario_area_bar(
        scenario_summary_rows,
        output_path=os.path.join(figures_dir, "cross_scenario_area_bar.png"),
        title="Mean Area per Scenario and Algorithm",
    )

    scenario_definitions = [scenario_to_dict(get_scenario(sid)) for sid in scenario_ids]

    _write_csv(
        os.path.join(data_dir, "raw_runs.csv"),
        raw_rows,
        fieldnames=[
            "scenario_id",
            "algorithm",
            "run_index",
            "seed",
            "penalized_objective",
            "area",
            "volume",
            "total_height",
            "rel_volume_error",
            "rel_height_error",
            "monotonic_violation",
            "feasible",
            "engineering_feasible",
            "engineering_failed_checks",
            "n_evals",
            "time_s",
            "budget",
            "decision_vector",
        ],
    )

    _write_csv(
        os.path.join(data_dir, "scenario_summary.csv"),
        scenario_summary_rows,
        fieldnames=[
            "scenario_id",
            "algorithm",
            "runs",
            "mean_area",
            "median_area",
            "area_iqr_q1",
            "area_iqr_q3",
            "area_ci95",
            "std_area",
            "mean_rel_volume_error",
            "mean_rel_height_error",
            "mean_monotonic_violation",
            "mean_evals",
            "mean_time_s",
            "feasibility_rate",
            "engineering_feasibility_rate",
            "feasible_runs",
            "feasible_area_mean",
            "feasible_area_median",
            "engineering_feasible_runs",
            "engineering_feasible_area_mean",
            "engineering_feasible_area_median",
            "best_feasible_area",
            "best_penalized_objective",
        ],
    )

    _write_csv(
        os.path.join(data_dir, "algorithm_summary.csv"),
        algorithm_summary_rows,
        fieldnames=[
            "algorithm",
            "total_runs",
            "mean_area",
            "median_area",
            "area_iqr_q1",
            "area_iqr_q3",
            "area_ci95",
            "mean_rel_volume_error",
            "mean_rel_height_error",
            "mean_evals",
            "mean_time_s",
            "overall_feasibility_rate",
            "overall_engineering_feasibility_rate",
            "feasible_runs",
            "feasible_area_mean",
            "feasible_area_median",
            "engineering_feasible_runs",
            "engineering_feasible_area_mean",
            "engineering_feasible_area_median",
        ],
    )

    with open(os.path.join(data_dir, "scenario_definitions.json"), "w", encoding="utf-8") as f:
        json.dump(scenario_definitions, f, indent=2)

    if args.export_latex_tables:
        latex_csv_dir = os.path.join(results_dir, "latex")
        _export_latex_csv_inputs(
            output_dir=latex_csv_dir,
            scenario_summary_rows=scenario_summary_rows,
            algorithm_summary_rows=algorithm_summary_rows,
        )

    report_path = os.path.join(report_outdir, "Report.md")
    _generate_report(
        report_path=report_path,
        scenario_ids=scenario_ids,
        scenario_definitions=scenario_definitions,
        scenario_summary_rows=scenario_summary_rows,
        algorithm_summary_rows=algorithm_summary_rows,
        visual_selection_by_scenario=visual_selection_by_scenario,
        include_3d=(not args.no_3d),
        runs=args.runs,
        seed_offset=args.seed_offset,
        figures_dir=figures_dir,
    )

    live_report_path = os.path.join(script_dir, "Report.md")
    if os.path.abspath(output_root) == script_dir and os.path.abspath(report_path) != os.path.abspath(live_report_path):
        shutil.copyfile(report_path, live_report_path)

    print("\nTask 6 experiments complete.")
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    if args.export_latex_tables:
        print(f"LaTeX CSV inputs: {os.path.join(results_dir, 'latex')}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
