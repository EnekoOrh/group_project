import math
import unittest

import numpy as np

from src.benchmarks.cooling_tower import (
    build_scenario_problem,
    get_scenario,
    tower_area_volume,
    tower_area_volume_with_gradients,
)


def _finite_difference(fun, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += eps
        x_m[i] -= eps
        grad[i] = (fun(x_p) - fun(x_m)) / (2.0 * eps)
    return grad


class CoolingTowerMathTests(unittest.TestCase):
    def test_cylinder_special_case(self):
        radii = np.array([10.0, 10.0, 10.0, 10.0])
        heights = np.array([3.0, 4.0, 5.0])

        area, volume = tower_area_volume(radii, heights)

        expected_area = 2.0 * math.pi * 10.0 * np.sum(heights)
        expected_volume = math.pi * (10.0 ** 2) * np.sum(heights)

        self.assertAlmostEqual(area, expected_area, places=8)
        self.assertAlmostEqual(volume, expected_volume, places=8)

    def test_single_frustum_spot_check(self):
        radii = np.array([4.0, 2.0])
        heights = np.array([5.0])

        area, volume = tower_area_volume(radii, heights)

        slant = math.sqrt((2.0 - 4.0) ** 2 + 5.0 ** 2)
        expected_area = math.pi * (4.0 + 2.0) * slant
        expected_volume = (math.pi * 5.0 / 3.0) * (4.0 ** 2 + 4.0 * 2.0 + 2.0 ** 2)

        self.assertAlmostEqual(area, expected_area, places=10)
        self.assertAlmostEqual(volume, expected_volume, places=10)

    def test_area_volume_gradients_against_finite_difference(self):
        radii = np.array([39.3, 33.0, 29.0, 25.0, 22.0, 20.0, 19.0, 20.0, 22.0, 24.0, 27.4])
        heights = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 3.8, 3.5, 4.0])

        area, volume, gA_r, gA_h, gV_r, gV_h = tower_area_volume_with_gradients(radii, heights)

        def area_wrt_r(rr):
            a, _ = tower_area_volume(rr, heights)
            return a

        def area_wrt_h(hh):
            a, _ = tower_area_volume(radii, hh)
            return a

        def vol_wrt_r(rr):
            _, v = tower_area_volume(rr, heights)
            return v

        def vol_wrt_h(hh):
            _, v = tower_area_volume(radii, hh)
            return v

        num_gA_r = _finite_difference(area_wrt_r, radii)
        num_gA_h = _finite_difference(area_wrt_h, heights)
        num_gV_r = _finite_difference(vol_wrt_r, radii)
        num_gV_h = _finite_difference(vol_wrt_h, heights)

        self.assertLess(np.linalg.norm(gA_r - num_gA_r) / max(1.0, np.linalg.norm(num_gA_r)), 1e-5)
        self.assertLess(np.linalg.norm(gA_h - num_gA_h) / max(1.0, np.linalg.norm(num_gA_h)), 1e-5)
        self.assertLess(np.linalg.norm(gV_r - num_gV_r) / max(1.0, np.linalg.norm(num_gV_r)), 1e-5)
        self.assertLess(np.linalg.norm(gV_h - num_gV_h) / max(1.0, np.linalg.norm(num_gV_h)), 1e-5)

        self.assertGreater(area, 0.0)
        self.assertGreater(volume, 0.0)

    def test_objective_gradient_against_finite_difference_s7(self):
        scenario = get_scenario("S7")
        problem = build_scenario_problem(scenario)
        objective = problem["objective"]

        x = np.array([
            33.0,
            29.0,
            25.0,
            22.0,
            20.0,
            19.0,
            20.0,
            22.0,
            24.0,
            3.0,
            3.2,
            3.4,
            3.6,
            3.8,
            4.0,
            4.2,
            3.8,
            3.5,
            4.0,
        ])

        val, grad = objective(x, grad=True)
        num_grad = _finite_difference(lambda xx: objective(xx, grad=False), x)

        rel_err = np.linalg.norm(grad - num_grad) / max(1.0, np.linalg.norm(num_grad))
        self.assertLess(rel_err, 1e-5)
        self.assertGreater(val, 0.0)

    def test_s1_integrity(self):
        scenario = get_scenario("S1")

        self.assertEqual(scenario.m, 10)
        self.assertAlmostEqual(scenario.r0, 39.3)
        self.assertAlmostEqual(scenario.rm, 27.4)
        self.assertEqual(
            scenario.fixed_z,
            [0.0, 3.6, 7.3, 10.9, 14.6, 18.2, 21.9, 25.5, 29.1, 32.8, 36.5],
        )


if __name__ == "__main__":
    unittest.main()
