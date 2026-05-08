# Task 8 Individual Reports

This folder contains the Task 8 individual-report deliverables.

## Achille deliverables

- `Achille/Achille_Task8_Report.md`: final individual reflective report (target ~2000 words)
- `Achille/appendices/achille_reconstructed_timesheet.csv`: reconstructed weekly timesheet appendix
- `Achille/evidence/claim_to_evidence_matrix.csv`: claim-to-evidence matrix
- `Achille/evidence/contribution_timeline.md`: phase-based contribution timeline
- `Achille/evidence/task8_requirement_traceability_checklist.md`: requirement coverage + quality gate
- `Achille/Task8_Achille_Individual_Report.pdf`: polished PDF output

## Eneko deliverables

- `Eneko/Eneko_Task8_Report.md`: final individual reflective report (~2,000 words)
- `Eneko/appendices/eneko_reconstructed_timesheet.csv`: reconstructed weekly timesheet appendix
- `Eneko/evidence/claim_to_evidence_matrix.csv`: claim-to-evidence matrix
- `Eneko/evidence/contribution_timeline.md`: phase-based contribution timeline
- `Eneko/Task8_Eneko_Individual_Report.pdf`: polished PDF output

## Reusable template for teammates

- `templates/Task8_Individual_Report_Template.md`
- `templates/peer_scoring_rubric.csv`
- `templates/reconstructed_timesheet_template.csv`
- `templates/claim_to_evidence_matrix_template.csv`

## Shared baseline artifacts (all members)

- `common/task8_group_narrative_baseline.md`
- `common/evidence_index.csv`
- `common/peer_scoring_calibration.md`

## Build PDF (Achille)

From repo root:

```powershell
python tasks/Task_08_Individual_Reports/Achille/report_latex/scripts/sync_report_md_to_tex.py
latexmk -r tasks/Task_08_Individual_Reports/Achille/report_latex/latexmkrc -cd tasks/Task_08_Individual_Reports/Achille/report_latex/main.tex
```

The generated PDF is copied to:

- `tasks/Task_08_Individual_Reports/Achille/Task8_Achille_Individual_Report.pdf`

## Build PDF (Eneko)

From repo root:

```bash
python tasks/Task_08_Individual_Reports/Eneko/report_latex/scripts/sync_report_md_to_tex.py
cd tasks/Task_08_Individual_Reports/Eneko/report_latex && tectonic -X compile main.tex
cp main.pdf ../Task8_Eneko_Individual_Report.pdf
```

The generated PDF is copied to:

- `tasks/Task_08_Individual_Reports/Eneko/Task8_Eneko_Individual_Report.pdf`
