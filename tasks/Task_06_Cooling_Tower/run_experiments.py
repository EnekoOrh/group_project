import argparse
import csv
import json
import os
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

    if bool(metrics["feasible"]):
        reason = ", ".join(_format_check(check, "<=") for check in checks)
        return f"- {algo}: **FEASIBLE** - passes {reason}."

    if not failed:
        failed = checks
    reason = ", ".join(_format_check(check, ">") for check in failed)
    line = f"- {algo}: **INFEASIBLE** - fails {reason}."
    if selection_basis == "best_penalized_objective":
        line += " visualized run is lowest penalized objective among infeasible runs."
    return line


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


def _format_pct(rate: float) -> str:
    return f"{100.0 * float(rate):.1f}%"


def _scenario_commentary(
    scenario_id: str,
    rows: List[Dict[str, object]],
) -> List[str]:
    by_algo = {str(row["algorithm"]): row for row in rows}
    best_tradeoff = max(rows, key=lambda row: (float(row["feasibility_rate"]), -float(row["mean_area"])))
    fastest = min(rows, key=lambda row: float(row["mean_evals"]))

    sa = by_algo["SA"]
    pso = by_algo["PSO"]
    bfgs = by_algo["BFGS"]

    if scenario_id == "S1":
        return [
            (
                f"S1 remains a difficult required case even though the plotted profiles are visually very similar. "
                f"**{best_tradeoff['algorithm']}** gives the best feasibility-area tradeoff with "
                f"{_format_pct(best_tradeoff['feasibility_rate'])} feasibility and mean area "
                f"{float(best_tradeoff['mean_area']):.2f} m^2, while **PSO** improves feasibility only modestly and still pays a clear area penalty. "
                f"**{fastest['algorithm']}** is the fastest method, but here speed does not translate into reliability because the same low-area local outcome remains volume-infeasible."
            ),
            (
                "The main point in S1 is that the search space contains a narrow feasible basin around the required geometry. "
                "The algorithms do not differ much in the shapes they ultimately prefer; they differ in how often they can satisfy the strict volume tolerance while keeping the surface area low."
            ),
        ]

    if scenario_id == "S2":
        return [
            (
                f"S2 again shows why mathematical feasibility is not the same as practical plausibility. "
                f"**{best_tradeoff['algorithm']}** is the strongest method here with {_format_pct(best_tradeoff['feasibility_rate'])} feasibility and mean area "
                f"{float(best_tradeoff['mean_area']):.2f} m^2, while **BFGS** remains mostly feasible but much less robust in mean area and **PSO** remains weak on both area and consistency."
            ),
            (
                "Because S2 deliberately removes smoothness control and allows a wide radii range, feasible profiles can still look structurally awkward. "
                "That is visible in the PSO profile, which has a sharper throat and more abrupt expansion than the smoother SA and BFGS solutions. "
                "This is the clearest scenario where the report must distinguish formal feasibility from practical shape quality."
            ),
        ]

    if scenario_id == "S3":
        return [
            (
                f"S3 is more stable than S2 because the uniform-height setting simplifies the search. "
                f"**SA** and **BFGS** both remain fully feasible, with SA giving the lower mean area at {float(sa['mean_area']):.2f} m^2 and "
                f"BFGS remaining the fastest method at {float(bfgs['mean_evals']):.1f} evaluations on average. "
                f"**PSO** can still find a good design, but its lower feasibility and higher variance show that it reaches that quality less consistently."
            ),
            (
                "The plotted profiles support this interpretation: SA and BFGS are almost indistinguishable visually, suggesting that the bounded uniform-height radii problem has a clear preferred geometry. "
                "The main remaining uncertainty is therefore algorithmic consistency rather than disagreement about the tower shape itself."
            ),
        ]

    if scenario_id == "S4":
        return [
            (
                f"S4 combines finer discretization with smoothness control, so it is a stronger test of repeated convergence than the simpler radii-only cases. "
                f"**BFGS** remains dominant with {_format_pct(bfgs['feasibility_rate'])} feasibility and mean area {float(bfgs['mean_area']):.2f} m^2, "
                f"whereas **SA** and **PSO** find feasible solutions only occasionally even though their selected plotted towers are competitive in shape and area."
            ),
            (
                "The correct interpretation is not that the stochastic methods produce bad geometries when they succeed. "
                "Rather, the extra discretization and smoothness penalty make this scenario harder to solve consistently, and BFGS handles that additional structure much more reliably than SA or PSO."
            ),
        ]

    if scenario_id == "S5":
        return [
            (
                f"S5 is a comparatively well-behaved bounded heights-only problem. "
                f"**BFGS** is strongest on both accuracy and efficiency, reaching {_format_pct(bfgs['feasibility_rate'])} feasibility with mean area {float(bfgs['mean_area']):.2f} m^2 and "
                f"the lowest evaluation count. **SA** is also fully feasible but slightly less efficient, while **PSO** is the least reliable method in this setting."
            ),
            (
                "The three plotted profiles are close, which means the scenario is not about discovering radically different shapes. "
                "Instead, it is about how precisely each method can fine-tune the segment heights while keeping both volume and total height within tolerance."
            ),
        ]

    if scenario_id == "S6":
        return [
            (
                f"S6 becomes easier once the height bounds are widened. "
                f"All three methods are fully feasible in this run, but **BFGS** still gives the lowest mean area at {float(bfgs['mean_area']):.2f} m^2 and is by far the fastest method at "
                f"{float(bfgs['mean_evals']):.1f} evaluations. **PSO** becomes competitive on feasibility, but not on area, and **SA** remains the largest of the three mean designs."
            ),
            (
                "This scenario shows that relaxing the height bounds can remove much of the feasibility difficulty without changing the overall algorithm ranking. "
                "Greater design freedom helps every method, but it does not overturn the advantage of BFGS on efficiency or final area."
            ),
        ]

    if scenario_id == "S7":
        return [
            (
                f"S7 is the clearest joint optimization case in the study. "
                f"**{best_tradeoff['algorithm']}** is the strongest method here with {_format_pct(best_tradeoff['feasibility_rate'])} feasibility, mean area "
                f"{float(best_tradeoff['mean_area']):.2f} m^2, and the lowest evaluation count in the scenario. "
                f"**SA** and **PSO** can still produce feasible towers, but they do so less reliably and with materially larger mean areas."
            ),
            (
                "The profile overlay helps explain the ranking. "
                "BFGS maintains the cleanest transition through the neck region while staying close to the lowest-area profile, whereas PSO shows a boxier middle section and SA settles on a smoother but noticeably larger tower. "
                "This is a good example of a coupled radii-height problem where strong local refinement is especially valuable."
            ),
        ]

    if scenario_id == "S8":
        return [
            (
                f"S8 remains the strongest bottleneck in the whole study. "
                f"All three methods have {_format_pct(sa['feasibility_rate'])} feasibility under the current constraints, even though **SA** still reaches the lowest mean area "
                f"at {float(sa['mean_area']):.2f} m^2 and the lowest penalized objective among the infeasible methods."
            ),
            (
                "The convergence and profile plots show that this is not simply a random failure case. "
                "The enlarged target volume, smoothness requirement, and adjacent-height ratio control together create a genuinely hard joint problem. "
                "BFGS remains more controlled than PSO, but neither method can satisfy the full feasibility rule, so S8 still marks the limit of the current formulation."
            ),
        ]

    return [
        (
            f"In {scenario_id}, **{best_tradeoff['algorithm']}** gives the best feasibility-area tradeoff with "
            f"{_format_pct(best_tradeoff['feasibility_rate'])} feasibility and mean area {float(best_tradeoff['mean_area']):.2f} m^2, while "
            f"**{fastest['algorithm']}** is the fastest method at {float(fastest['mean_evals']):.1f} evaluations."
        ),
        "The profile and convergence plots should be interpreted alongside the summary statistics so that visual plausibility is not confused with repeatable numerical performance.",
    ]


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
        "Minimize cooling-tower lateral shell area using SA, PSO, and BFGS while enforcing fixed target volume and scenario-specific constructability constraints."
    )
    lines.append(
        "The study compares stochastic and deterministic optimization behavior across eight structurally different problem setups while keeping common engineering requirements (capacity and realistic shape) explicit in the objective."
    )
    lines.append("")
    lines.append("## 2. Geometry Model")
    lines.append("The tower is modeled as a stack of frustums.")
    lines.append("")
    lines.append("- Frustum area: `A_i = pi * (r_{i-1} + r_i) * sqrt((r_i - r_{i-1})^2 + h_i^2)`")
    lines.append("- Frustum volume: `V_i = (pi * h_i / 3) * (r_{i-1}^2 + r_{i-1}*r_i + r_i^2)`")
    lines.append("- Tower totals: `A = sum(A_i)`, `V = sum(V_i)`")
    lines.append("")
    lines.append(
        "Lateral shell area is optimized (top and bottom caps excluded), which directly corresponds to shell-construction material for fixed end radii."
    )
    lines.append(
        "Analytic gradients were used for both `A` and `V`, and for all penalty terms, so BFGS receives exact first-order information."
    )
    lines.append("")
    lines.append("### 2.1 Design Variables by Scenario Type")
    lines.append("- `radii` mode: optimize interior radii `r1..r_{m-1}` while ring heights are fixed from `z` levels.")
    lines.append("- `heights` mode: optimize segment heights `h1..hm` while all ring radii are fixed.")
    lines.append("- `joint` mode: optimize both interior radii and all heights simultaneously.")
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
        "- `P_bounds`: hinge penalty for design-variable bounds to keep variables in practical/constructable ranges."
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
    lines.append("## 5. Optimization Protocol")
    lines.append(f"- Runs per algorithm per scenario: {runs}")
    lines.append(f"- Seed offset: {seed_offset}")
    lines.append("- SA: `temp_init=120`, `cooling_rate=0.997`, scenario-dependent `step_size`")
    lines.append("- PSO: `w=0.65`, `c1=1.6`, `c2=1.7`, particles = 40 or 50 by dimension")
    lines.append("- BFGS: `tol=1e-7`, `max_iter=1200`")
    lines.append("- Penalties: volume, height sum (when applicable), shape monotonicity, smoothness, bounds, ratio (S8)")
    lines.append("")
    lines.append(
        "Stochastic budgets are 10k evaluations for lower-dimensional setups and 16k for joint/high-dimensional setups. BFGS uses smaller budgets due to faster local convergence."
    )
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

    lines.append("## 7. Scenario Results")
    for sid in scenario_ids:
        lines.append("")
        lines.append(f"### {sid}")

        rows = []
        for row in sorted(scenario_summary_by_id[sid], key=lambda item: item["algorithm"]):
            rows.append(
                [
                    str(row["algorithm"]),
                    f"{float(row['mean_area']):.2f}",
                    f"{float(row['std_area']):.2f}",
                    f"{float(row['mean_rel_volume_error']):.3e}",
                    f"{float(row['mean_evals']):.1f}",
                    f"{float(row['mean_time_s']):.4f}",
                    f"{100.0 * float(row['feasibility_rate']):.1f}%",
                ]
            )

        lines.append(
            _markdown_table(
                [
                    "Algorithm",
                    "Mean Area",
                    "Std Area",
                    "Mean Rel Vol Err",
                    "Mean Evals",
                    "Mean Time (s)",
                    "Feasibility Rate",
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
            lines.append("Feasibility of the shown 3D towers (same selected runs as profile overlay):")
            for algo in ALGORITHMS:
                selected = selected_visual_runs.get(algo)
                if not selected:
                    continue
                lines.append(
                    _format_visual_feasibility_line(
                        algo=algo,
                        scenario=scenario,
                        metrics=selected["metrics"],
                        selection_basis=str(selected["selection_basis"]),
                    )
                )
            lines.append("")

        for paragraph in _scenario_commentary(sid, scenario_summary_by_id[sid]):
            lines.append(paragraph)
            lines.append("")

    lines.append("## 8. Cross-Algorithm Summary")
    algo_rows = []
    for row in algorithm_summary_rows:
        algo_rows.append(
            [
                str(row["algorithm"]),
                str(int(row["total_runs"])),
                f"{float(row['mean_area']):.2f}",
                f"{float(row['mean_rel_volume_error']):.3e}",
                f"{float(row['mean_evals']):.1f}",
                f"{float(row['mean_time_s']):.4f}",
                f"{100.0 * float(row['overall_feasibility_rate']):.1f}%",
            ]
        )

    lines.append(
        _markdown_table(
            [
                "Algorithm",
                "Total Runs",
                "Mean Area",
                "Mean Rel Vol Err",
                "Mean Evals",
                "Mean Time (s)",
                "Feasibility Rate",
            ],
            algo_rows,
        )
    )
    lines.append("")
    lines.append(f"![Cross-scenario area comparison]({_figure_link('cross_scenario_area_bar.png')})")
    lines.append("")

    # High-level aggregate conclusions computed from summary table.
    best_feas_algo = max(algorithm_summary_rows, key=lambda row: float(row["overall_feasibility_rate"]))
    fastest_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_evals"]))
    lowest_area_algo = min(algorithm_summary_rows, key=lambda row: float(row["mean_area"]))
    hard_scenarios = [
        sid
        for sid in scenario_ids
        if scenario_summary_by_id[sid]
        and max(float(row["feasibility_rate"]) for row in scenario_summary_by_id[sid]) <= 0.0
    ]

    lines.append("## 9. Discussion and Conclusions")
    lines.append("")

    lines.append(
        f"Across all scenarios, **{best_feas_algo['algorithm']}** achieved the highest overall feasibility "
        f"({100.0 * float(best_feas_algo['overall_feasibility_rate']):.1f}%)."
    )
    lines.append(
        f"**{fastest_algo['algorithm']}** required the fewest evaluations on average "
        f"({float(fastest_algo['mean_evals']):.1f}), confirming its efficiency for local convergence."
    )
    lines.append(
        f"By raw mean area across all runs, **{lowest_area_algo['algorithm']}** produced the lowest average shell area "
        f"({float(lowest_area_algo['mean_area']):.2f} m^2), but this must be interpreted jointly with feasibility rates."
    )
    lines.append("")
    lines.append(
        "Result patterns are consistent with expected method behavior: BFGS is computationally efficient and strong when gradients and local curvature align with feasible basins; "
        "SA is often robust under hard nonlinear penalties because probabilistic acceptance helps transition between constrained basins; "
        "PSO explores broadly but can underperform feasibility when penalty landscapes are steep and high-dimensional."
    )
    lines.append("")
    lines.append(
        "For practical engineering usage in this problem family, the recommended workflow is a hybrid strategy: "
        "use SA/PSO for global search and feasibility discovery, then warm-start BFGS for rapid refinement of the best feasible candidate."
    )
    lines.append("")
    lines.append(
        "The remaining limitation is that optimization does not by itself confirm structural suitability. "
        "S2 shows that mathematically feasible shapes can still look impractical, while S8 shows that higher-capacity, constructability-aware designs remain difficult under the current formulation."
    )
    lines.append("")
    if hard_scenarios:
        lines.append(
            f"Under the current penalty and budget settings, {', '.join(hard_scenarios)} remains the clearest feasibility bottleneck, "
            "indicating that the present formulation still struggles when constructability and higher-capacity requirements are combined."
        )
        lines.append("")
    lines.append("### 9.1 Planned Structural Validation in Abaqus")
    lines.append("")
    lines.append(
        "The next step is to transfer the best candidate geometries into Abaqus shell models with realistic material properties and shell-thickness assumptions for reinforced-concrete cooling towers. "
        "These models will be checked under self-weight and representative environmental loading to evaluate displacement, stress distribution, and buckling sensitivity."
    )
    lines.append("")
    lines.append(
        "Multiple candidate towers will be compared, not only the minimum-area design, so selection can be based on structural feasibility as well as geometric efficiency. "
        "The Abaqus results will then be used to refine the optimization model by tightening feasibility rules or adding constraints linked to smoothness, curvature, and stability."
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
                },
            }

            rows = scenario_run_rows[algo_name]
            feasible_areas = [float(row["area"]) for row in rows if row["feasible"]]

            summary_row = {
                "scenario_id": sid,
                "algorithm": algo_name,
                "runs": len(rows),
                "mean_area": _safe_mean([float(row["area"]) for row in rows]),
                "std_area": _safe_std([float(row["area"]) for row in rows]),
                "mean_rel_volume_error": _safe_mean([float(row["rel_volume_error"]) for row in rows]),
                "mean_rel_height_error": _safe_mean([float(row["rel_height_error"]) for row in rows]),
                "mean_monotonic_violation": _safe_mean([float(row["monotonic_violation"]) for row in rows]),
                "mean_evals": _safe_mean([float(row["n_evals"]) for row in rows]),
                "mean_time_s": _safe_mean([float(row["time_s"]) for row in rows]),
                "feasibility_rate": _safe_mean([1.0 if row["feasible"] else 0.0 for row in rows]),
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

        algorithm_summary_rows.append(
            {
                "algorithm": algo_name,
                "total_runs": len(rows),
                "mean_area": _safe_mean([float(row["area"]) for row in rows]),
                "mean_rel_volume_error": _safe_mean([float(row["rel_volume_error"]) for row in rows]),
                "mean_rel_height_error": _safe_mean([float(row["rel_height_error"]) for row in rows]),
                "mean_evals": _safe_mean([float(row["n_evals"]) for row in rows]),
                "mean_time_s": _safe_mean([float(row["time_s"]) for row in rows]),
                "overall_feasibility_rate": _safe_mean([1.0 if row["feasible"] else 0.0 for row in rows]),
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
            "std_area",
            "mean_rel_volume_error",
            "mean_rel_height_error",
            "mean_monotonic_violation",
            "mean_evals",
            "mean_time_s",
            "feasibility_rate",
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
            "mean_rel_volume_error",
            "mean_rel_height_error",
            "mean_evals",
            "mean_time_s",
            "overall_feasibility_rate",
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

    print("\nTask 6 experiments complete.")
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    if args.export_latex_tables:
        print(f"LaTeX CSV inputs: {os.path.join(results_dir, 'latex')}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
