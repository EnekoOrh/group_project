import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tasks" / "Task_07_Abaqus" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task7_common import build_case_payload, build_mesh, geometry_verification, load_csv, load_json, select_task7_case


class Task7GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_json(ROOT / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json")
        cls.rows = load_csv(ROOT / "tasks" / "Task_06_Cooling_Tower" / "results" / "data" / "raw_runs.csv")
        cls.warning_scenarios = cls.config["selection"]["warning_scenarios"]

    def _selected_payload(self, scenario_id, algorithm):
        row, status, basis, note = select_task7_case(self.rows, scenario_id, algorithm, self.warning_scenarios)
        return build_case_payload(row, scenario_id, algorithm, status, basis, note)

    def test_geometry_verification_matches_s2_source(self):
        payload = self._selected_payload("S2", "BFGS")
        mesh_data = build_mesh(payload, self.config["mesh"]["refined_circumferential_divisions"], self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        check = geometry_verification(payload, mesh_data, self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        self.assertAlmostEqual(check["abs_height_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_radius_abs_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_z_abs_diff_m"], 0.0, places=12)

    def test_geometry_verification_matches_s6_source(self):
        payload = self._selected_payload("S6", "BFGS")
        mesh_data = build_mesh(payload, self.config["mesh"]["refined_circumferential_divisions"], self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        check = geometry_verification(payload, mesh_data, self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        self.assertAlmostEqual(check["abs_height_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_radius_abs_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_z_abs_diff_m"], 0.0, places=12)

    def test_geometry_verification_matches_s7_source(self):
        payload = self._selected_payload("S7", "BFGS")
        mesh_data = build_mesh(payload, self.config["mesh"]["refined_circumferential_divisions"], self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        check = geometry_verification(payload, mesh_data, self.config["mesh"]["refined_axial_subdivisions_per_segment"])
        self.assertAlmostEqual(check["abs_height_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_radius_abs_diff_m"], 0.0, places=12)
        self.assertAlmostEqual(check["max_source_ring_z_abs_diff_m"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
