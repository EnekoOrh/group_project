import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.algorithms.deterministic import BFGS
from src.algorithms.stochastics import ParticleSwarm, SimulatedAnnealing
from src.benchmarks.cooling_tower import build_scenario_problem, evaluate_decision, get_scenario, get_solver_settings


ALGORITHMS = ["SA", "PSO", "BFGS"]
ALGO_SEED_OFFSETS = {"SA": 0, "PSO": 10000, "BFGS": 20000}
ALGO_COLORS = {"SA": "magenta", "PSO": "orange", "BFGS": "red"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _surface_from_profile(radii: np.ndarray, z: np.ndarray, n_theta: int = 80):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    rr = np.repeat(radii[:, None], n_theta, axis=1)
    zz = np.repeat(z[:, None], n_theta, axis=1)
    xx = rr * np.cos(theta)[None, :]
    yy = rr * np.sin(theta)[None, :]
    return xx, yy, zz


def _plot_profile_contrast(feasible: Dict[str, object], relaxed: Dict[str, object], output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    for label, metrics, color in [
        ("Feasible constrained", feasible, "tab:blue"),
        ("Relaxed over-optimized", relaxed, "tab:red"),
    ]:
        z = np.asarray(metrics["z"], dtype=float)
        radii = np.asarray(metrics["radii"], dtype=float)
        plt.plot(radii, z, color=color, linewidth=2.2, label=f"{label} (A={metrics['area']:.1f})")
        plt.plot(-radii, z, color=color, linewidth=1.2, linestyle="--", alpha=0.7)

    plt.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    plt.title("Cooling Tower Contrast: Feasible vs Relaxed Optimization")
    plt.xlabel("Radius r (m)")
    plt.ylabel("Height z (m)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_tower_contrast_3d(feasible: Dict[str, object], relaxed: Dict[str, object], output_path: str) -> None:
    fig = plt.figure(figsize=(12, 5))
    for idx, (label, metrics, color) in enumerate(
        [
            ("Feasible constrained", feasible, "tab:blue"),
            ("Relaxed over-optimized", relaxed, "tab:red"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        x, y, z = _surface_from_profile(np.asarray(metrics["radii"]), np.asarray(metrics["z"]))
        ax.plot_surface(x, y, z, color=color, alpha=0.85, linewidth=0, antialiased=False)
        ax.set_title(label)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=45)

    fig.suptitle("Task 9 Poster Concept: Optimization Without Constraints Can Be Non-Buildable")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_notes(path: str, feasible: Dict[str, object], relaxed: Dict[str, object]) -> None:
    lines = []
    lines.append("# Poster Notes: Feasible vs Over-Optimized Cooling Tower")
    lines.append("")
    lines.append("## Intended Message")
    lines.append("- These two towers are both optimization outputs for similar objectives.")
    lines.append("- The constrained case respects practical/shape conditions and is closer to buildable geometry.")
    lines.append("- The relaxed case can reduce objective value while violating practical design realism.")
    lines.append("- Conclusion: optimization quality depends on objective **and** boundary conditions.")
    lines.append("")
    lines.append("## Numerical Snapshot")
    lines.append(f"- Feasible constrained area: `{feasible['area']:.2f}` m^2")
    lines.append(f"- Feasible constrained rel. volume error: `{feasible['rel_volume_error']:.3e}`")
    lines.append(f"- Relaxed over-optimized area: `{relaxed['area']:.2f}` m^2")
    lines.append(f"- Relaxed over-optimized rel. volume error: `{relaxed['rel_volume_error']:.3e}`")
    lines.append(f"- Relaxed monotonic violation: `{relaxed['monotonic_violation']:.3e}`")
    lines.append("")
    lines.append("## Usage Warning")
    lines.append("- Use this comparison in Task 9 as communication material, not as final engineering recommendation.")
    lines.append("- For compliance conclusions, rely on Task 6 constrained scenario analyses.")

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 9 poster contrast for Task 6 cooling tower.")
    parser.add_argument("--runs", type=int, default=2, help="Runs per algorithm and regime.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset.")
    parser.add_argument("--eval-scale", type=float, default=0.6, help="Scale solver evaluation budgets.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    data_dir = os.path.join(results_dir, "data")
    figures_dir = os.path.join(results_dir, "figures")
    _ensure_dir(data_dir)
    _ensure_dir(figures_dir)

    base = get_scenario("S1")
    feasible_scenario = base
    relaxed_scenario = replace(
        base,
        description="Relaxed radii design for poster awareness (non-buildable risk).",
        radii_bounds=(3.0, 95.0),
        apply_shape_penalty=False,
        apply_smoothness_penalty=False,
        apply_bounds_penalty=False,
    )

    regime_specs = [
        ("feasible", feasible_scenario, None),
        (
            "relaxed",
            relaxed_scenario,
            {
                "lambda_shape": 0.0,
                "lambda_smooth": 0.0,
                "lambda_bounds": 0.0,
                "lambda_vol": 2e8,
            },
        ),
    ]

    rows: List[Dict[str, object]] = []
    best_metrics_by_regime: Dict[str, Dict[str, object]] = {}

    for regime_name, scenario, penalty_overrides in regime_specs:
        problem = build_scenario_problem(scenario, penalty_overrides=penalty_overrides)
        settings = get_solver_settings(scenario)
        best_row = None

        for algo_name in ALGORITHMS:
            algo_settings = dict(settings[algo_name])
            algo_settings["max_evals"] = max(200, int(algo_settings["max_evals"] * args.eval_scale))
            if algo_name == "BFGS":
                algo_settings["max_iter"] = int(min(algo_settings["max_iter"], algo_settings["max_evals"]))

            for run_idx in range(args.runs):
                seed = int(args.seed_offset + ALGO_SEED_OFFSETS[algo_name] + run_idx)
                algo = _create_algorithm(
                    algo_name,
                    problem["objective"],
                    problem["bounds"],
                    int(problem["dim"]),
                    seed,
                    algo_settings,
                )
                t0 = time.time()
                result = algo.solve()
                elapsed = time.time() - t0
                metrics = evaluate_decision(scenario, result["best_x"])

                row = {
                    "regime": regime_name,
                    "algorithm": algo_name,
                    "run_index": run_idx,
                    "seed": seed,
                    "area": float(metrics["area"]),
                    "volume": float(metrics["volume"]),
                    "rel_volume_error": float(metrics["rel_volume_error"]),
                    "monotonic_violation": float(metrics["monotonic_violation"]),
                    "feasible_by_task6_rule": bool(metrics["feasible"]),
                    "objective": float(result["best_val"]),
                    "n_evals": int(result["n_evals"]),
                    "time_s": float(elapsed),
                    "decision_vector": json.dumps(np.asarray(result["best_x"], dtype=float).tolist()),
                }
                rows.append(row)

                if best_row is None or row["area"] < best_row["area"]:
                    best_row = row

        if best_row is None:
            raise RuntimeError(f"No optimization result for regime: {regime_name}")

        best_x = np.array(json.loads(best_row["decision_vector"]), dtype=float)
        best_metrics = evaluate_decision(scenario, best_x)
        best_metrics_by_regime[regime_name] = best_metrics

    _write_csv(
        os.path.join(data_dir, "summary.csv"),
        rows,
        fieldnames=[
            "regime",
            "algorithm",
            "run_index",
            "seed",
            "area",
            "volume",
            "rel_volume_error",
            "monotonic_violation",
            "feasible_by_task6_rule",
            "objective",
            "n_evals",
            "time_s",
            "decision_vector",
        ],
    )

    _plot_profile_contrast(
        feasible=best_metrics_by_regime["feasible"],
        relaxed=best_metrics_by_regime["relaxed"],
        output_path=os.path.join(figures_dir, "profile_contrast.png"),
    )
    _plot_tower_contrast_3d(
        feasible=best_metrics_by_regime["feasible"],
        relaxed=best_metrics_by_regime["relaxed"],
        output_path=os.path.join(figures_dir, "tower_contrast_3d.png"),
    )
    _write_notes(
        os.path.join(results_dir, "PosterNotes.md"),
        feasible=best_metrics_by_regime["feasible"],
        relaxed=best_metrics_by_regime["relaxed"],
    )

    print("Task 9 poster exploration complete.")
    print(f"Results: {results_dir}")
    print(f"Data: {os.path.join(data_dir, 'summary.csv')}")
    print(f"Figures: {os.path.join(figures_dir, 'profile_contrast.png')}, {os.path.join(figures_dir, 'tower_contrast_3d.png')}")


if __name__ == "__main__":
    main()
