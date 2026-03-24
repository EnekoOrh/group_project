import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tasks" / "Task_07_Abaqus" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_report  # noqa: E402


class Task7RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = generate_report._load_json(ROOT / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json")
        cls.cases = generate_report._load_json(ROOT / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json")
        cls.summary_rows = generate_report._load_csv(ROOT / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv")
        row_map = generate_report._rows_by_case(cls.summary_rows)
        ordered_case_ids = [case["case_id"] for case in cls.cases]
        cls.refined_rows = [generate_report._preferred_row(case_id, row_map) for case_id in ordered_case_ids]

    def test_weighted_ranking_is_deterministic(self):
        ranked_a = generate_report._compute_weighted_ranking(self.refined_rows, self.config)
        ranked_b = generate_report._compute_weighted_ranking(self.refined_rows, self.config)
        self.assertEqual(len(ranked_a), 24)
        self.assertEqual([entry["row"]["case_id"] for entry in ranked_a], [entry["row"]["case_id"] for entry in ranked_b])

    def test_penalties_match_status_config(self):
        ranking_cfg = generate_report._load_ranking_config(self.config)
        ranked = generate_report._compute_weighted_ranking(self.refined_rows, self.config)
        for entry in ranked:
            status = entry["row"]["selection_status"]
            self.assertAlmostEqual(entry["penalty_points"], ranking_cfg["penalties"][status], places=9)

    def test_top3_by_criterion_matches_raw_metric_order(self):
        ranking_cfg = generate_report._load_ranking_config(self.config)
        top_entries = generate_report._criterion_top_entries(self.refined_rows, ranking_cfg["criterion_top_count"])
        for metric_key, _, descending, _ in generate_report.METRIC_CONFIG:
            selected = [entry["row"]["case_id"] for entry in top_entries if entry["criterion_key"] == metric_key]
            ordered = sorted(
                self.refined_rows,
                key=lambda row: (
                    -float(row[metric_key]) if descending else float(row[metric_key]),
                    generate_report.status_rank(row),
                    row["case_id"],
                ),
            )
            expected = [row["case_id"] for row in ordered[: ranking_cfg["criterion_top_count"]]]
            self.assertEqual(selected, expected)

    def test_top3_engineering_only_filters_non_engineering(self):
        ranking_cfg = generate_report._load_ranking_config(self.config)
        top_entries = generate_report._criterion_top_entries(
            self.refined_rows, ranking_cfg["criterion_top_count"], allowed_statuses=[generate_report.STATUS_ENGINEERING]
        )
        self.assertTrue(top_entries)
        self.assertTrue(all(entry["row"]["selection_status"] == generate_report.STATUS_ENGINEERING for entry in top_entries))

        engineering_rows = [row for row in self.refined_rows if row["selection_status"] == generate_report.STATUS_ENGINEERING]
        for metric_key, _, descending, _ in generate_report.METRIC_CONFIG:
            selected = [entry["row"]["case_id"] for entry in top_entries if entry["criterion_key"] == metric_key]
            ordered = sorted(
                engineering_rows,
                key=lambda row: (
                    -float(row[metric_key]) if descending else float(row[metric_key]),
                    generate_report.status_rank(row),
                    row["case_id"],
                ),
            )
            expected = [row["case_id"] for row in ordered[: ranking_cfg["criterion_top_count"]]]
            self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
