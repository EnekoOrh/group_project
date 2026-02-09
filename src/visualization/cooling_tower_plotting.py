import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALGO_ORDER = ["SA", "PSO", "BFGS"]
ALGO_COLORS = {
    "SA": "magenta",
    "PSO": "orange",
    "BFGS": "red",
}


def _interpolate_histories(histories: List[List[tuple]], common_evals: np.ndarray) -> np.ndarray:
    rows = []
    for hist in histories:
        if not hist:
            continue

        evals = np.array([float(x[0]) for x in hist], dtype=float)
        vals = np.array([max(float(x[1]), 1e-18) for x in hist], dtype=float)

        order = np.argsort(evals)
        evals = evals[order]
        vals = vals[order]

        unique_evals, unique_idx = np.unique(evals, return_index=True)
        unique_vals = vals[unique_idx]

        rows.append(np.interp(common_evals, unique_evals, unique_vals))

    if not rows:
        return np.zeros((0, len(common_evals)), dtype=float)

    return np.vstack(rows)


def plot_convergence(
    histories_by_algorithm: Dict[str, List[List[tuple]]],
    output_path: str,
    title: str,
    max_eval: int,
) -> None:
    common_evals = np.linspace(1.0, float(max_eval), 250)

    plt.figure(figsize=(10, 6))

    for algo in ALGO_ORDER:
        histories = histories_by_algorithm.get(algo, [])
        matrix = _interpolate_histories(histories, common_evals)
        if matrix.shape[0] == 0:
            continue

        mean_vals = np.mean(matrix, axis=0)
        std_vals = np.std(matrix, axis=0)
        color = ALGO_COLORS.get(algo, "black")

        plt.plot(common_evals, mean_vals, color=color, linewidth=2.0, label=f"{algo} (mean final: {mean_vals[-1]:.2e})")
        plt.fill_between(common_evals, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

    plt.title(title)
    plt.xlabel("Function Evaluations")
    plt.ylabel("Penalized Objective Value (log scale)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_profile_overlay(
    profiles_by_algorithm: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(9, 8))

    for algo in ALGO_ORDER:
        profile = profiles_by_algorithm.get(algo)
        if not profile:
            continue

        z = np.asarray(profile["z"], dtype=float)
        radii = np.asarray(profile["radii"], dtype=float)
        color = ALGO_COLORS.get(algo, "black")

        area = profile.get("area")
        rel_v = profile.get("rel_volume_error")

        label = f"{algo}"
        if area is not None and rel_v is not None:
            label = f"{algo} (A={area:.1f}, relV={rel_v:.2e})"

        plt.plot(radii, z, color=color, linewidth=2.0, label=label)
        plt.plot(-radii, z, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    plt.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    plt.title(title)
    plt.xlabel("Radius r (m)")
    plt.ylabel("Height z (m)")
    plt.grid(True, alpha=0.25)
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _surface_from_profile(radii: np.ndarray, z: np.ndarray, n_theta: int = 80):
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    rr = np.repeat(radii[:, None], n_theta, axis=1)
    zz = np.repeat(z[:, None], n_theta, axis=1)

    xx = rr * np.cos(theta)[None, :]
    yy = rr * np.sin(theta)[None, :]
    return xx, yy, zz


def plot_tower_triptych(
    profiles_by_algorithm: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    title: str,
) -> None:
    fig = plt.figure(figsize=(15, 5))

    for idx, algo in enumerate(ALGO_ORDER, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        profile = profiles_by_algorithm.get(algo)

        if profile:
            radii = np.asarray(profile["radii"], dtype=float)
            z = np.asarray(profile["z"], dtype=float)
            x, y, zz = _surface_from_profile(radii, z)
            color = ALGO_COLORS.get(algo, "gray")

            ax.plot_surface(x, y, zz, color=color, alpha=0.85, linewidth=0, antialiased=False)
            ax.set_title(f"{algo}")
        else:
            ax.set_title(f"{algo} (no data)")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=45)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cross_scenario_area_bar(
    scenario_summary_rows: List[Dict[str, object]],
    output_path: str,
    title: str,
) -> None:
    scenarios = sorted({row["scenario_id"] for row in scenario_summary_rows})
    algos = [algo for algo in ALGO_ORDER if any(row["algorithm"] == algo for row in scenario_summary_rows)]

    x = np.arange(len(scenarios), dtype=float)
    width = 0.22 if len(algos) >= 3 else 0.35

    plt.figure(figsize=(12, 6))

    for idx, algo in enumerate(algos):
        means = []
        for sid in scenarios:
            matches = [row for row in scenario_summary_rows if row["scenario_id"] == sid and row["algorithm"] == algo]
            if matches:
                means.append(float(matches[0]["mean_area"]))
            else:
                means.append(np.nan)

        offset = (idx - (len(algos) - 1) / 2.0) * width
        plt.bar(x + offset, means, width=width, color=ALGO_COLORS.get(algo, "gray"), label=algo)

    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel("Mean Lateral Surface Area (m^2)")
    plt.xticks(x, scenarios)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
