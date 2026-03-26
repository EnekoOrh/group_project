import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tasks" / "Task_07_Abaqus" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_report  # noqa: E402


class Task7DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_report._load_json(ROOT / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json")
        cls.summary_rows = generate_report._load_csv(ROOT / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv")
        row_map = generate_report._rows_by_case(cls.summary_rows)
        ordered_case_ids = [case["case_id"] for case in cls.cases]
        cls.refined_rows = [generate_report._preferred_row(case_id, row_map) for case_id in ordered_case_ids]
        cls.decomp_entries = generate_report._build_decomposition_entries(cls.refined_rows, cls.summary_rows)

    def test_decomposition_covers_all_refined_cases(self):
        self.assertEqual(len(self.decomp_entries), len(self.cases))
        self.assertEqual({entry["case_id"] for entry in self.decomp_entries}, {case["case_id"] for case in self.cases})

    def test_decomposition_has_expected_driver_labels(self):
        for entry in self.decomp_entries:
            self.assertIn(entry["displacement_driver"], {"gravity", "wind"})
            self.assertIn(entry["stress_driver"], {"gravity", "wind"})
            self.assertGreaterEqual(float(entry["combined_max_displacement_mm"]), 0.0)
            self.assertGreaterEqual(float(entry["combined_max_mises_mpa"]), 0.0)


if __name__ == "__main__":
    unittest.main()
