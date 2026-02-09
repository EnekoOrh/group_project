import copy
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_PENALTIES = {
    "lambda_vol": 2e8,
    "lambda_hsum": 3e7,
    "lambda_shape": 5e6,
    "lambda_bounds": 2e7,
    "lambda_smooth": 2e4,
    "lambda_ratio": 5e6,
}


@dataclass(frozen=True)
class CoolingTowerScenario:
    scenario_id: str
    description: str
    m: int
    decision_mode: str  # "radii", "heights", "joint"
    r0: float = 39.3
    rm: float = 27.4
    target_volume: float = 70320.0
    target_total_height: float = 36.5
    fixed_z: Optional[List[float]] = None
    fixed_radii: Optional[List[float]] = None
    neck_index: Optional[int] = None
    radii_bounds: Tuple[float, float] = (12.0, 45.0)
    heights_bounds: Tuple[float, float] = (1.5, 6.0)
    apply_shape_penalty: bool = True
    apply_smoothness_penalty: bool = False
    apply_bounds_penalty: bool = True
    adjacent_height_ratio_limit: Optional[float] = None


def _build_scenarios() -> Dict[str, CoolingTowerScenario]:
    z_required = [0.0, 3.6, 7.3, 10.9, 14.6, 18.2, 21.9, 25.5, 29.1, 32.8, 36.5]
    radii_template = [39.3, 34.94, 30.73, 27.01, 23.77, 21.42, 20.15, 20.22, 21.58, 24.08, 27.4]

    scenarios = {
        "S1": CoolingTowerScenario(
            scenario_id="S1",
            description="Required case with bounded radii and required ring heights.",
            m=10,
            decision_mode="radii",
            fixed_z=z_required,
            neck_index=6,
            radii_bounds=(12.0, 45.0),
            apply_shape_penalty=True,
            apply_smoothness_penalty=True,
            apply_bounds_penalty=True,
        ),
        "S2": CoolingTowerScenario(
            scenario_id="S2",
            description="Required case with wide radii range (unconstrained-like).",
            m=10,
            decision_mode="radii",
            fixed_z=z_required,
            neck_index=6,
            radii_bounds=(5.0, 80.0),
            apply_shape_penalty=True,
            apply_smoothness_penalty=False,
            apply_bounds_penalty=False,
        ),
        "S3": CoolingTowerScenario(
            scenario_id="S3",
            description="Uniform-height radii design with bounded radii.",
            m=10,
            decision_mode="radii",
            fixed_z=np.linspace(0.0, 36.5, 11).tolist(),
            neck_index=6,
            radii_bounds=(12.0, 45.0),
            apply_shape_penalty=True,
            apply_smoothness_penalty=False,
            apply_bounds_penalty=True,
        ),
        "S4": CoolingTowerScenario(
            scenario_id="S4",
            description="Finer discretization (m=12) with bounded radii and smoothness.",
            m=12,
            decision_mode="radii",
            fixed_z=np.linspace(0.0, 36.5, 13).tolist(),
            neck_index=7,
            radii_bounds=(12.0, 45.0),
            apply_shape_penalty=True,
            apply_smoothness_penalty=True,
            apply_bounds_penalty=True,
        ),
        "S5": CoolingTowerScenario(
            scenario_id="S5",
            description="Fixed radii, optimize bounded heights.",
            m=10,
            decision_mode="heights",
            fixed_radii=radii_template,
            heights_bounds=(2.0, 5.5),
            apply_shape_penalty=False,
            apply_smoothness_penalty=False,
            apply_bounds_penalty=True,
        ),
        "S6": CoolingTowerScenario(
            scenario_id="S6",
            description="Fixed radii, optimize wide-range heights (unconstrained-like).",
            m=10,
            decision_mode="heights",
            fixed_radii=radii_template,
            heights_bounds=(0.8, 8.0),
            apply_shape_penalty=False,
            apply_smoothness_penalty=False,
            apply_bounds_penalty=False,
        ),
        "S7": CoolingTowerScenario(
            scenario_id="S7",
            description="Joint bounded optimization of radii and heights.",
            m=10,
            decision_mode="joint",
            neck_index=6,
            radii_bounds=(12.0, 45.0),
            heights_bounds=(1.5, 6.0),
            apply_shape_penalty=True,
            apply_smoothness_penalty=False,
            apply_bounds_penalty=True,
        ),
        "S8": CoolingTowerScenario(
            scenario_id="S8",
            description="Joint constructability-aware design with larger target volume.",
            m=10,
            decision_mode="joint",
            neck_index=6,
            radii_bounds=(14.0, 50.0),
            heights_bounds=(1.5, 7.0),
            target_volume=90000.0,
            apply_shape_penalty=True,
            apply_smoothness_penalty=True,
            apply_bounds_penalty=True,
            adjacent_height_ratio_limit=1.8,
        ),
    }

    return scenarios


_SCENARIOS = _build_scenarios()


def get_all_scenarios() -> Dict[str, CoolingTowerScenario]:
    return dict(_SCENARIOS)


def get_scenario(scenario_id: str) -> CoolingTowerScenario:
    if scenario_id not in _SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_id}'")
    return _SCENARIOS[scenario_id]


def scenario_to_dict(scenario: CoolingTowerScenario) -> Dict[str, object]:
    data = asdict(scenario)
    data["radii_bounds"] = list(scenario.radii_bounds)
    data["heights_bounds"] = list(scenario.heights_bounds)
    return data


def get_decision_bounds(scenario: CoolingTowerScenario) -> List[Tuple[float, float]]:
    if scenario.decision_mode == "radii":
        return [tuple(scenario.radii_bounds) for _ in range(scenario.m - 1)]
    if scenario.decision_mode == "heights":
        return [tuple(scenario.heights_bounds) for _ in range(scenario.m)]
    if scenario.decision_mode == "joint":
        return [tuple(scenario.radii_bounds) for _ in range(scenario.m - 1)] + [
            tuple(scenario.heights_bounds) for _ in range(scenario.m)
        ]
    raise ValueError(f"Unsupported decision mode: {scenario.decision_mode}")


def decode_decision_vector(
    scenario: CoolingTowerScenario, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)

    if scenario.decision_mode == "radii":
        expected_dim = scenario.m - 1
        if x.size != expected_dim:
            raise ValueError(f"Expected {expected_dim} radii variables, got {x.size}")
        if scenario.fixed_z is None:
            raise ValueError("Radii mode requires fixed_z")

        radii = np.concatenate(([scenario.r0], x, [scenario.rm]))
        z = np.asarray(scenario.fixed_z, dtype=float)
        heights = np.diff(z)

    elif scenario.decision_mode == "heights":
        expected_dim = scenario.m
        if x.size != expected_dim:
            raise ValueError(f"Expected {expected_dim} height variables, got {x.size}")
        if scenario.fixed_radii is None:
            raise ValueError("Heights mode requires fixed_radii")

        heights = x.copy()
        z = np.concatenate(([0.0], np.cumsum(heights)))
        radii = np.asarray(scenario.fixed_radii, dtype=float)

    elif scenario.decision_mode == "joint":
        expected_dim = (scenario.m - 1) + scenario.m
        if x.size != expected_dim:
            raise ValueError(f"Expected {expected_dim} joint variables, got {x.size}")

        radii = np.concatenate(([scenario.r0], x[: scenario.m - 1], [scenario.rm]))
        heights = x[scenario.m - 1 :].copy()
        z = np.concatenate(([0.0], np.cumsum(heights)))

    else:
        raise ValueError(f"Unsupported decision mode: {scenario.decision_mode}")

    if radii.size != scenario.m + 1:
        raise ValueError("Radii length must be m+1")
    if heights.size != scenario.m:
        raise ValueError("Heights length must be m")

    return radii, heights, z


def tower_area_volume(radii: np.ndarray, heights: np.ndarray) -> Tuple[float, float]:
    radii = np.asarray(radii, dtype=float)
    heights = np.asarray(heights, dtype=float)

    r_lower = radii[:-1]
    r_upper = radii[1:]
    dr = r_upper - r_lower
    slant = np.sqrt(dr * dr + heights * heights)

    area = np.pi * np.sum((r_lower + r_upper) * slant)
    volume = (np.pi / 3.0) * np.sum(heights * (r_lower * r_lower + r_lower * r_upper + r_upper * r_upper))
    return float(area), float(volume)


def tower_area_volume_with_gradients(
    radii: np.ndarray, heights: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = np.asarray(radii, dtype=float)
    heights = np.asarray(heights, dtype=float)

    r_lower = radii[:-1]
    r_upper = radii[1:]
    dr = r_upper - r_lower
    slant = np.sqrt(dr * dr + heights * heights)
    slant_safe = np.maximum(slant, 1e-12)

    area_terms = np.pi * (r_lower + r_upper) * slant
    volume_terms = (np.pi / 3.0) * heights * (r_lower * r_lower + r_lower * r_upper + r_upper * r_upper)

    area = np.sum(area_terms)
    volume = np.sum(volume_terms)

    dA_dr_lower = np.pi * (slant + (r_lower + r_upper) * (-dr / slant_safe))
    dA_dr_upper = np.pi * (slant + (r_lower + r_upper) * (dr / slant_safe))
    dA_dh = np.pi * (r_lower + r_upper) * (heights / slant_safe)

    dV_dr_lower = (np.pi / 3.0) * heights * (2.0 * r_lower + r_upper)
    dV_dr_upper = (np.pi / 3.0) * heights * (r_lower + 2.0 * r_upper)
    dV_dh = (np.pi / 3.0) * (r_lower * r_lower + r_lower * r_upper + r_upper * r_upper)

    grad_area_r = np.zeros_like(radii)
    grad_volume_r = np.zeros_like(radii)

    grad_area_r[:-1] += dA_dr_lower
    grad_area_r[1:] += dA_dr_upper

    grad_volume_r[:-1] += dV_dr_lower
    grad_volume_r[1:] += dV_dr_upper

    grad_area_h = dA_dh.copy()
    grad_volume_h = dV_dh.copy()

    return (
        float(area),
        float(volume),
        grad_area_r,
        grad_area_h,
        grad_volume_r,
        grad_volume_h,
    )


def compute_monotonic_violation(radii: np.ndarray, neck_index: int) -> float:
    if neck_index <= 0 or neck_index >= (len(radii) - 1):
        return 0.0

    left_violation = 0.0
    right_violation = 0.0

    for i in range(1, neck_index + 1):
        left_violation = max(left_violation, radii[i] - radii[i - 1])
    for i in range(neck_index + 1, len(radii)):
        right_violation = max(right_violation, radii[i - 1] - radii[i])

    return float(max(left_violation, right_violation, 0.0))


def shape_monotonic_penalty(
    radii: np.ndarray, neck_index: int, weight: float
) -> Tuple[float, np.ndarray, float]:
    grad = np.zeros_like(radii)
    if neck_index <= 0 or neck_index >= len(radii) - 1:
        return 0.0, grad, 0.0

    penalty = 0.0
    max_violation = 0.0

    for i in range(1, neck_index + 1):
        violation = radii[i] - radii[i - 1]
        if violation > 0.0:
            max_violation = max(max_violation, violation)
            penalty += weight * violation * violation
            grad[i] += 2.0 * weight * violation
            grad[i - 1] -= 2.0 * weight * violation

    for i in range(neck_index + 1, len(radii)):
        violation = radii[i - 1] - radii[i]
        if violation > 0.0:
            max_violation = max(max_violation, violation)
            penalty += weight * violation * violation
            grad[i - 1] += 2.0 * weight * violation
            grad[i] -= 2.0 * weight * violation

    return float(penalty), grad, float(max_violation)


def smoothness_penalty(radii: np.ndarray, weight: float) -> Tuple[float, np.ndarray]:
    grad = np.zeros_like(radii)
    if len(radii) < 3:
        return 0.0, grad

    penalty = 0.0
    for i in range(1, len(radii) - 1):
        second_diff = radii[i - 1] - 2.0 * radii[i] + radii[i + 1]
        penalty += weight * second_diff * second_diff
        grad[i - 1] += 2.0 * weight * second_diff
        grad[i] += -4.0 * weight * second_diff
        grad[i + 1] += 2.0 * weight * second_diff

    return float(penalty), grad


def bounds_penalty(
    x: np.ndarray, bounds: List[Tuple[float, float]], weight: float
) -> Tuple[float, np.ndarray]:
    penalty = 0.0
    grad = np.zeros_like(x)

    for idx, (low, high) in enumerate(bounds):
        val = x[idx]
        if val < low:
            diff = low - val
            penalty += weight * diff * diff
            grad[idx] -= 2.0 * weight * diff
        elif val > high:
            diff = val - high
            penalty += weight * diff * diff
            grad[idx] += 2.0 * weight * diff

    return float(penalty), grad


def adjacent_height_ratio_penalty(
    heights: np.ndarray, ratio_limit: float, weight: float
) -> Tuple[float, np.ndarray]:
    heights = np.asarray(heights, dtype=float)
    grad = np.zeros_like(heights)
    penalty = 0.0

    if len(heights) < 2:
        return 0.0, grad

    for i in range(len(heights) - 1):
        h0 = heights[i]
        h1 = heights[i + 1]

        v01 = h0 - ratio_limit * h1
        if v01 > 0.0:
            penalty += weight * v01 * v01
            grad[i] += 2.0 * weight * v01
            grad[i + 1] -= 2.0 * weight * ratio_limit * v01

        v10 = h1 - ratio_limit * h0
        if v10 > 0.0:
            penalty += weight * v10 * v10
            grad[i + 1] += 2.0 * weight * v10
            grad[i] -= 2.0 * weight * ratio_limit * v10

    return float(penalty), grad


def _pack_gradient(
    scenario: CoolingTowerScenario, grad_radii: np.ndarray, grad_heights: np.ndarray
) -> np.ndarray:
    if scenario.decision_mode == "radii":
        return grad_radii[1:-1].copy()
    if scenario.decision_mode == "heights":
        return grad_heights.copy()
    if scenario.decision_mode == "joint":
        return np.concatenate([grad_radii[1:-1], grad_heights])
    raise ValueError(f"Unsupported decision mode: {scenario.decision_mode}")


def build_scenario_problem(
    scenario: CoolingTowerScenario,
    penalty_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    penalties = copy.deepcopy(DEFAULT_PENALTIES)
    if penalty_overrides:
        penalties.update(penalty_overrides)

    bounds = get_decision_bounds(scenario)
    decision_dim = len(bounds)

    radii_are_variables = scenario.decision_mode in ("radii", "joint")
    heights_are_variables = scenario.decision_mode in ("heights", "joint")

    def objective(x: np.ndarray, grad: bool = False):
        x = np.asarray(x, dtype=float)
        radii, heights, _ = decode_decision_vector(scenario, x)

        area, volume, gA_r, gA_h, gV_r, gV_h = tower_area_volume_with_gradients(radii, heights)

        total = area
        grad_total_r = gA_r.copy()
        grad_total_h = gA_h.copy()

        rel_volume_error = (volume - scenario.target_volume) / scenario.target_volume
        vol_penalty = penalties["lambda_vol"] * rel_volume_error * rel_volume_error
        total += vol_penalty

        vol_factor = 2.0 * penalties["lambda_vol"] * rel_volume_error / scenario.target_volume
        grad_total_r += vol_factor * gV_r
        grad_total_h += vol_factor * gV_h

        if heights_are_variables:
            total_height = float(np.sum(heights))
            rel_height_error = (total_height - scenario.target_total_height) / scenario.target_total_height
            h_penalty = penalties["lambda_hsum"] * rel_height_error * rel_height_error
            total += h_penalty

            h_factor = 2.0 * penalties["lambda_hsum"] * rel_height_error / scenario.target_total_height
            grad_total_h += h_factor

        if radii_are_variables and scenario.apply_shape_penalty and scenario.neck_index is not None:
            shape_penalty, grad_shape_r, _ = shape_monotonic_penalty(
                radii,
                scenario.neck_index,
                penalties["lambda_shape"],
            )
            total += shape_penalty
            grad_total_r += grad_shape_r

        if radii_are_variables and scenario.apply_smoothness_penalty:
            smooth_penalty, grad_smooth_r = smoothness_penalty(radii, penalties["lambda_smooth"])
            total += smooth_penalty
            grad_total_r += grad_smooth_r

        if scenario.adjacent_height_ratio_limit is not None:
            ratio_penalty, grad_ratio_h = adjacent_height_ratio_penalty(
                heights,
                scenario.adjacent_height_ratio_limit,
                penalties["lambda_ratio"],
            )
            total += ratio_penalty
            grad_total_h += grad_ratio_h

        grad_decision = _pack_gradient(scenario, grad_total_r, grad_total_h)

        bound_weight = penalties["lambda_bounds"] if scenario.apply_bounds_penalty else penalties["lambda_bounds"] * 1e-2
        if bound_weight > 0.0:
            bounds_pen, bounds_grad = bounds_penalty(x, bounds, bound_weight)
            total += bounds_pen
            grad_decision += bounds_grad

        if grad:
            return float(total), grad_decision

        return float(total)

    return {
        "scenario": scenario,
        "objective": objective,
        "bounds": bounds,
        "dim": decision_dim,
        "penalties": penalties,
    }


def evaluate_decision(scenario: CoolingTowerScenario, x: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x, dtype=float)
    radii, heights, z = decode_decision_vector(scenario, x)
    area, volume = tower_area_volume(radii, heights)

    total_height = float(np.sum(heights))
    rel_volume_error = abs(volume - scenario.target_volume) / scenario.target_volume
    rel_height_error = abs(total_height - scenario.target_total_height) / scenario.target_total_height

    monotonic_violation = 0.0
    radii_are_variables = scenario.decision_mode in ("radii", "joint")
    heights_are_variables = scenario.decision_mode in ("heights", "joint")

    if radii_are_variables and scenario.neck_index is not None:
        monotonic_violation = compute_monotonic_violation(radii, scenario.neck_index)

    feasible = rel_volume_error <= 1e-3
    if heights_are_variables:
        feasible = feasible and (rel_height_error <= 1e-3)
    if radii_are_variables:
        feasible = feasible and (monotonic_violation <= 1e-6)

    return {
        "area": float(area),
        "volume": float(volume),
        "total_height": float(total_height),
        "rel_volume_error": float(rel_volume_error),
        "rel_height_error": float(rel_height_error),
        "monotonic_violation": float(monotonic_violation),
        "feasible": bool(feasible),
        "radii": radii,
        "heights": heights,
        "z": z,
    }


def get_solver_settings(scenario: CoolingTowerScenario) -> Dict[str, Dict[str, float]]:
    dim = len(get_decision_bounds(scenario))

    stochastic_budget = 10000 if dim <= 11 else 16000
    bfgs_budget = 5000 if dim <= 11 else 7000
    pso_particles = 40 if dim <= 11 else 50

    if scenario.decision_mode == "radii":
        sa_step = 0.25
    elif scenario.decision_mode == "heights":
        sa_step = 0.12
    else:
        sa_step = 0.20

    return {
        "SA": {
            "temp_init": 120.0,
            "cooling_rate": 0.997,
            "step_size": sa_step,
            "max_evals": stochastic_budget,
        },
        "PSO": {
            "w": 0.65,
            "c1": 1.6,
            "c2": 1.7,
            "num_particles": pso_particles,
            "max_evals": stochastic_budget,
        },
        "BFGS": {
            "tol": 1e-7,
            "max_iter": 1200,
            "max_evals": bfgs_budget,
        },
    }
