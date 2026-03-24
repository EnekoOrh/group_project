import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tasks" / "Task_07_Abaqus" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task7_common import STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING, build_case_payload, load_csv, load_json, select_task7_case


class Task7SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_json(ROOT / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json")
        cls.rows = load_csv(ROOT / "tasks" / "Task_06_Cooling_Tower" / "results" / "data" / "raw_runs.csv")

    def test_selection_matrix_has_24_slots(self):
        selection = self.config["selection"]
        cases = []
        for scenario_id in selection["scenario_ids"]:
            for algorithm in selection["algorithms"]:
                chosen = select_task7_case(self.rows, scenario_id, algorithm, selection["warning_scenarios"])
                cases.append(chosen)
        self.assertEqual(len(cases), 24)

    def test_statuses_cover_engineering_fallback_and_warning(self):
        warning_scenarios = self.config["selection"]["warning_scenarios"]
        _, status_s2_bfgs, _, _ = select_task7_case(self.rows, "S2", "BFGS", warning_scenarios)
        _, status_s2_pso, _, _ = select_task7_case(self.rows, "S2", "PSO", warning_scenarios)
        _, status_s1_bfgs, _, _ = select_task7_case(self.rows, "S1", "BFGS", warning_scenarios)
        _, status_s8_sa, _, _ = select_task7_case(self.rows, "S8", "SA", warning_scenarios)

        self.assertEqual(status_s2_bfgs, STATUS_ENGINEERING)
        self.assertEqual(status_s2_pso, STATUS_MATH_FALLBACK)
        self.assertEqual(status_s1_bfgs, STATUS_WARNING)
        self.assertEqual(status_s8_sa, STATUS_WARNING)

    def test_build_case_payload_preserves_task6_identity_fields(self):
        warning_scenarios = self.config["selection"]["warning_scenarios"]
        row, status, basis, note = select_task7_case(self.rows, "S6", "BFGS", warning_scenarios)
        payload = build_case_payload(row, "S6", "BFGS", status, basis, note)
        self.assertEqual(payload["case_id"], "s6_bfgs")
        self.assertEqual(payload["model_name"], "S6_BFGS")
        self.assertEqual(payload["selection_status"], STATUS_ENGINEERING)
        self.assertGreater(payload["task6_total_height_m"], 30.0)


if __name__ == "__main__":
    unittest.main()
