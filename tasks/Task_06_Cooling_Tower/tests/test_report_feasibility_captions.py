import tempfile
import unittest
from pathlib import Path

from src.benchmarks.cooling_tower import get_scenario, scenario_to_dict
from tasks.Task_06_Cooling_Tower.run_experiments import (
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
            },
            selection_basis="best_penalized_objective",
        )
        self.assertIn("**INFEASIBLE**", line)
        self.assertIn("volume", line)
        self.assertIn("height", line)
        self.assertIn("shape", line)
        self.assertIn(">", line)
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
            },
            selection_basis="best_feasible_area",
        )
        self.assertIn("**FEASIBLE**", line)
        self.assertIn("volume", line)
        self.assertIn("height", line)
        self.assertIn("shape", line)
        self.assertIn("<=", line)
        self.assertNotIn("lowest penalized objective among infeasible runs", line)

    def test_generate_report_includes_3d_feasibility_captions_only(self):
        scenario_ids = ["S1"]
        scenario_definitions = [scenario_to_dict(get_scenario("S1"))]
        scenario_summary_rows = [
            {
                "scenario_id": "S1",
                "algorithm": "SA",
                "runs": 1,
                "mean_area": 7780.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 4.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "feasibility_rate": 1.0,
                "best_feasible_area": 7780.0,
                "best_penalized_objective": 7781.0,
            },
            {
                "scenario_id": "S1",
                "algorithm": "PSO",
                "runs": 1,
                "mean_area": 7900.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 2.0e-3,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 1.0e-2,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "feasibility_rate": 0.0,
                "best_feasible_area": "",
                "best_penalized_objective": 9000.0,
            },
            {
                "scenario_id": "S1",
                "algorithm": "BFGS",
                "runs": 1,
                "mean_area": 7790.0,
                "std_area": 0.0,
                "mean_rel_volume_error": 5.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_monotonic_violation": 0.0,
                "mean_evals": 3000.0,
                "mean_time_s": 0.1,
                "feasibility_rate": 1.0,
                "best_feasible_area": 7790.0,
                "best_penalized_objective": 7790.0,
            },
        ]
        algorithm_summary_rows = [
            {
                "algorithm": "SA",
                "total_runs": 1,
                "mean_area": 7780.0,
                "mean_rel_volume_error": 4.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "overall_feasibility_rate": 1.0,
            },
            {
                "algorithm": "PSO",
                "total_runs": 1,
                "mean_area": 7900.0,
                "mean_rel_volume_error": 2.0e-3,
                "mean_rel_height_error": 0.0,
                "mean_evals": 10000.0,
                "mean_time_s": 0.5,
                "overall_feasibility_rate": 0.0,
            },
            {
                "algorithm": "BFGS",
                "total_runs": 1,
                "mean_area": 7790.0,
                "mean_rel_volume_error": 5.0e-4,
                "mean_rel_height_error": 0.0,
                "mean_evals": 3000.0,
                "mean_time_s": 0.1,
                "overall_feasibility_rate": 1.0,
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
                        "area": 7780.0,
                        "rel_volume_error": 4.0e-4,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 0.0,
                    },
                },
                "PSO": {
                    "run_index": 0,
                    "seed": 10000,
                    "selection_basis": "best_penalized_objective",
                    "metrics": {
                        "feasible": False,
                        "area": 7900.0,
                        "rel_volume_error": 2.0e-3,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 1.0e-2,
                    },
                },
                "BFGS": {
                    "run_index": 0,
                    "seed": 20000,
                    "selection_basis": "best_feasible_area",
                    "metrics": {
                        "feasible": True,
                        "area": 7790.0,
                        "rel_volume_error": 5.0e-4,
                        "rel_height_error": 0.0,
                        "monotonic_violation": 0.0,
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
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertNotIn("Feasibility of the plotted profiles:", report)
            self.assertIn("Feasibility of the shown 3D towers", report)
            self.assertIn("**FEASIBLE**", report)
            self.assertIn("**INFEASIBLE**", report)
            self.assertIn("lowest penalized objective among infeasible runs", report)
            self.assertIn("volume", report)
            self.assertIn("shape", report)


if __name__ == "__main__":
    unittest.main()
