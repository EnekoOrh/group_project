import tempfile
import unittest
from pathlib import Path

from src.benchmarks.cooling_tower import get_scenario, scenario_to_dict
from tasks.Task_06_Cooling_Tower.run_experiments import (
    _engineering_checks_for_scenario,
    _feasibility_checks_for_scenario,
    _format_visual_feasibility_line,
    _generate_report,
)


class ReportFeasibilityCaptionTests(unittest.TestCase):
    def test_feasibility_checks_radii_mode(self):
        scenario = get_scenario("S1")
        checks = _feasibility_checks_for_scenario(
            scenario,
            {
                "rel_volume_error": 5e-4,
                "rel_height_error": 0.0,
                "monotonic_violation": 0.0,
            },
        )
        names = [str(check["name"]) for check in checks]
        self.assertEqual(names, ["volume", "shape"])
        self.assertTrue(all(bool(check["passed"]) for check in checks))

    def test_feasibility_checks_heights_mode_infeasible_height(self):
        scenario = get_scenario("S5")
        checks = _feasibility_checks_for_scenario(
            scenario,
            {
                "rel_volume_error": 2e-4,
                "rel_height_error": 2.5e-3,
                "monotonic_violation": 0.0,
            },
        )
        names = [str(check["name"]) for check in checks]
        self.assertEqual(names, ["volume", "height"])
        self.assertTrue(bool(checks[0]["passed"]))
        self.assertFalse(bool(checks[1]["passed"]))

    def test_engineering_checks_flag_impractical_shape(self):
        scenario = get_scenario("S1")
        checks = _engineering_checks_for_scenario(
            scenario,
            {
                "radii": [40.0, 14.0, 8.0, 14.0, 40.0],
                "heights": [9.0, 9.0, 9.0, 9.0],
            },
        )
        failed = [str(check["name"]) for check in checks if not bool(check["passed"])]
        self.assertIn("neck_radius", failed)
        self.assertIn("rel_radius_step", failed)
        self.assertIn("radius_second_diff", failed)

    def test_format_line_joint_infeasible_with_fallback_note(self):
        scenario = get_scenario("S7")
        line = _format_visual_feasibility_line(
            algo="PSO",
            scenario=scenario,
            metrics={
                "feasible": False,
                "rel_volume_error": 2.0e-3,
                "rel_height_error": 3.0e-3,
                "monotonic_violation": 2.5e-6,
                "radii": [40.0, 20.0, 8.0, 20.0, 40.0],
                "heights": [6.0, 15.0, 4.0, 20.0],
            },
            selection_basis="best_penalized_objective",
        )
        self.assertIn("**INFEASIBLE**", line)
        self.assertIn("volume", line)
        self.assertIn("height", line)
        self.assertIn("shape", line)
        self.assertIn("engineering status: INFEASIBLE", line)
        self.assertIn("lowest penalized objective among infeasible runs", line)

    def test_format_line_joint_feasible_reports_all_checks(self):
        scenario = get_scenario("S7")
        line = _format_visual_feasibility_line(
            algo="BFGS",
            scenario=scenario,
            metrics={
                "feasible": True,
                "rel_volume_error": 3.2e-4,
                "rel_height_error": 8.0e-4,
                "monotonic_violation": 0.0,
                "radii": [40.0, 36.0, 34.0, 36.0, 40.0],
                "heights": [8.0, 8.0, 8.0, 8.0],
            },
            selection_basis="best_feasible_area",
        )
        self.assertIn("**FEASIBLE**", line)
        self.assertIn("volume", line)
        self.assertIn("height", line)
        self.assertIn("shape", line)
        self.assertIn("engineering status: FEASIBLE", line)
        self.assertNotIn("lowest penalized objective among infeasible runs", line)

    def test_generate_report_includes_dual_status_sections(self):
        scenario_ids = ["S1"]
        scenario_definitions = [scenario_to_dict(get_scenario("S1"))]
        scenario_summary_rows = [
            {
                "scenario_id": "S1",
                "algorithm": "SA",
                "runs": 1,
                "mean_area": 7780.0,
                "median_area": 7780.0,
                "area_iqr_q1": 7780.0,
                "area_iqr_q3": 7780.0,
                "area_ci95": 0.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 4.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "feasibility_rate": 1.0,
                "engineering_feasibility_rate": 1.0,
                "feasible_runs": 1,
                "feasible_area_mean": 7780.0,
                "feasible_area_median": 7780.0,
                "engineering_feasible_runs": 1,
                "engineering_feasible_area_mean": 7780.0,
                "engineering_feasible_area_median": 7780.0,
                "best_feasible_area": 7780.0,
                "best_penalized_objective": 7781.0,
            },
            {
                "scenario_id": "S1",
                "algorithm": "PSO",
                "runs": 1,
                "mean_area": 7900.0,
                "median_area": 7900.0,
                "area_iqr_q1": 7900.0,
                "area_iqr_q3": 7900.0,
                "area_ci95": 0.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 2.0e-3,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 1.0e-2,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "feasibility_rate": 0.0,
                "engineering_feasibility_rate": 0.0,
                "feasible_runs": 0,
                "feasible_area_mean": float("nan"),
                "feasible_area_median": float("nan"),
                "engineering_feasible_runs": 0,
                "engineering_feasible_area_mean": float("nan"),
                "engineering_feasible_area_median": float("nan"),
                "best_feasible_area": "",
                "best_penalized_objective": 9000.0,
            },
            {
                "scenario_id": "S1",
                "algorithm": "BFGS",
                "runs": 1,
                "mean_area": 7790.0,
                "median_area": 7790.0,
                "area_iqr_q1": 7790.0,
                "area_iqr_q3": 7790.0,
                "area_ci95": 0.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 5.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 0.0,
                "mean_evals": 3000.0,
                "mean_time_s": 0.1,
                "feasibility_rate": 1.0,
                "engineering_feasibility_rate": 0.0,
                "feasible_runs": 1,
                "feasible_area_mean": 7790.0,
                "feasible_area_median": 7790.0,
                "engineering_feasible_runs": 0,
                "engineering_feasible_area_mean": float("nan"),
                "engineering_feasible_area_median": float("nan"),
                "best_feasible_area": 7790.0,
                "best_penalized_objective": 7790.0,
            },
        ]
        algorithm_summary_rows = [
            {
                "algorithm": "SA",
                "total_runs": 1,
                "mean_area": 7780.0,
                "median_area": 7780.0,
                "area_iqr_q1": 7780.0,
                "area_iqr_q3": 7780.0,
                "area_ci95": 0.0,
                "mean_rel_volume_error": 4.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "overall_feasibility_rate": 1.0,
                "overall_engineering_feasibility_rate": 1.0,
                "feasible_runs": 1,
                "feasible_area_mean": 7780.0,
                "feasible_area_median": 7780.0,
                "engineering_feasible_runs": 1,
                "engineering_feasible_area_mean": 7780.0,
                "engineering_feasible_area_median": 7780.0,
            },
            {
                "algorithm": "PSO",
                "total_runs": 1,
                "mean_area": 7900.0,
                "median_area": 7900.0,
                "area_iqr_q1": 7900.0,
                "area_iqr_q3": 7900.0,
                "area_ci95": 0.0,
                "mean_rel_volume_error": 2.0e-3,
                "mean_rel_height_error": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "overall_feasibility_rate": 0.0,
                "overall_engineering_feasibility_rate": 0.0,
                "feasible_runs": 0,
                "feasible_area_mean": float("nan"),
                "feasible_area_median": float("nan"),
                "engineering_feasible_runs": 0,
                "engineering_feasible_area_mean": float("nan"),
                "engineering_feasible_area_median": float("nan"),
            },
            {
                "algorithm": "BFGS",
                "total_runs": 1,
                "mean_area": 7790.0,
                "median_area": 7790.0,
                "area_iqr_q1": 7790.0,
                "area_iqr_q3": 7790.0,
                "area_ci95": 0.0,
                "mean_rel_volume_error": 5.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_evals": 3000.0,
                "mean_time_s": 0.1,
                "overall_feasibility_rate": 1.0,
                "overall_engineering_feasibility_rate": 0.0,
                "feasible_runs": 1,
                "feasible_area_mean": 7790.0,
                "feasible_area_median": 7790.0,
                "engineering_feasible_runs": 0,
                "engineering_feasible_area_mean": float("nan"),
                "engineering_feasible_area_median": float("nan"),
            },
        ]
        visual_selection_by_scenario = {
            "S1": {
                "SA": {
                    "run_index": 0,
                    "seed": 0,
                    "selection_basis": "best_feasible_area",
                    "metrics": {
                        "feasible": True,
                        "engineering_feasible": True,
                        "area": 7780.0,
                        "rel_volume_error": 4.0e-4,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 0.0,
                        "radii": [40.0, 36.0, 34.0, 36.0, 40.0],
                        "heights": [9.0, 9.0, 9.0, 9.0],
                    },
                },
                "PSO": {
                    "run_index": 0,
                    "seed": 10000,
                    "selection_basis": "best_penalized_objective",
                    "metrics": {
                        "feasible": False,
                        "engineering_feasible": False,
                        "area": 7900.0,
                        "rel_volume_error": 2.0e-3,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 1.0e-2,
                        "radii": [40.0, 20.0, 8.0, 20.0, 40.0],
                        "heights": [9.0, 9.0, 9.0, 9.0],
                    },
                },
                "BFGS": {
                    "run_index": 0,
                    "seed": 20000,
                    "selection_basis": "best_feasible_area",
                    "metrics": {
                        "feasible": True,
                        "engineering_feasible": False,
                        "area": 7790.0,
                        "rel_volume_error": 5.0e-4,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 0.0,
                        "radii": [40.0, 32.0, 24.0, 32.0, 40.0],
                        "heights": [9.0, 9.0, 9.0, 9.0],
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "Report.md"
            _generate_report(
                report_path=str(report_path),
                scenario_ids=scenario_ids,
                scenario_definitions=scenario_definitions,
                scenario_summary_rows=scenario_summary_rows,
                algorithm_summary_rows=algorithm_summary_rows,
                visual_selection_by_scenario=visual_selection_by_scenario,
                include_3d=True,
                runs=1,
                seed_offset=0,
                figures_dir=str(Path(tmpdir) / "results" / "figures"),
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("### 4.3 Engineering Feasibility Criteria", report)
            self.assertIn("### 5.1 Methodology Quality Checks", report)
            self.assertIn("### 6.1 Why These 8 Scenarios Are Relevant", report)
            self.assertIn("## 9. Key Findings", report)
            self.assertIn("### 10.1 Planned Structural Validation in Abaqus", report)
            self.assertIn("Compliance and feasibility status of the shown 3D towers", report)
            self.assertIn("Math Status", report)
            self.assertIn("Engineering Status", report)
            self.assertIn("COMPLIANT", report)
            self.assertIn("NON-COMPLIANT", report)
            self.assertIn("INFEASIBLE", report)
            self.assertIn("engineering-feasible median area", report)
            self.assertIn("m²", report)
            self.assertNotIn("## 11. Deliverables", report)
            self.assertNotIn("Generated:", report)
            self.assertIn("![S1 convergence](results/figures/S1_convergence.png)", report)
            self.assertIn("![Cross-scenario area comparison](results/figures/cross_scenario_area_bar.png)", report)
            self.assertLess(report.index("## 7. Cross-Algorithm Summary"), report.index("## 8. Scenario Results"))
            self.assertLess(
                report.index("![S1 profile](results/figures/S1_profile_overlay.png)"),
                report.index("![S1 3D towers](results/figures/S1_tower_3d.png)"),
            )
            self.assertLess(
                report.index("![S1 3D towers](results/figures/S1_tower_3d.png)"),
                report.index("Compliance and feasibility status of the shown 3D towers"),
            )


if __name__ == "__main__":
    unittest.main()
