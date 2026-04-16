# Task 8 Individual Reflective Report (Achille Larregle)

## 1. Context and Task 1 Baseline

This report is my Task 8 individual reflection for the Group Project. It is written as a direct hindsight analysis against my Task 1 statement, as required in the assignment brief. In Task 1, I positioned myself as a natural **Group Lead** profile (ENTJ-A), with the expected strengths of planning, communication, decision-making under pressure, and team direction. I also wrote that I wanted to contribute technically, not only coordinate.

The key promises I made in Task 1 were:
- I would provide structure and momentum.
- I would ensure communication quality and guideline compliance.
- I would keep deadlines visible and tasks clear.
- I would still contribute to technical work.

Task 8 asks me to evaluate whether those claims were accurate in practice and to reflect on both my own performance and the group's process. This report focuses on the gap between **declared role** and **actual delivered behavior**, especially in late tasks where complexity increased and roles became less rigid.  
High-level conclusion: Task 1 was directionally accurate, but I underestimated role overlap. By Tasks 6 and 7, I was still coordination lead while also contributing heavily to report integration and late Abaqus-related delivery.

## 2. Hindsight on Personality Fit (Task 1 vs Reality)

In Task 1 I identified as ENTJ-A and described myself as strategic, direct, and comfortable with initiative. In hindsight, much of this was accurate:
- I consistently moved toward closure when workstreams were open-ended.
- I naturally worked on structure: reporting pipelines, artifact organization, and requirement traceability.
- I remained comfortable taking responsibility when deadlines approached.

Where Task 1 was too optimistic was in assuming leadership could stay mostly "managerial" while technical ownership stayed cleanly distributed. In reality, these boundaries overlapped, and I entered technical integration when bottlenecks appeared.

Another accurate part of Task 1 was that directness is a double-edged tool. Under pressure it accelerated decisions, but it also needed explicit listening and context to avoid friction. The personality label was therefore useful as a starting lens, not as a fixed prescription.

## 3. My Contribution Across the Project

I describe my contribution across five workstreams: leadership/coordination, communication/meetings, reporting, technical optimization support, and late-stage Abaqus integration. Some contributions are repo-visible (scripts, report pipelines, generated outputs), while others are mostly non-repo (Notion organization, meeting facilitation, information transfer). I therefore combine both evidence types and flag dated anchors.

### 3.1 Leadership and coordination

My primary contribution was maintaining execution structure across phases: clarifying next actions after meetings, keeping deadlines visible, and connecting separate technical streams so outputs were submission-ready, not only technically complete. I also maintained Notion organization and decision continuity, which is less visible in git but operationally critical.  
Evidence anchors: Task 6 integration window (2026-02-25) and Task 7 final integration window (2026-03-24 to 2026-03-26).

### 3.2 Communication and alignment

I contributed as a bridge between technical detail and deliverable format. A recurring pattern was that good technical work needed a second pass to become report-quality evidence, so I focused on wording clarity, chapter alignment, and inconsistency correction.  
Evidence anchors: Task 6 compliance and feasibility narrative hardening around 2026-02-25 (`dbd9784`, `ff0f762`) and Task 7 report-clarity consolidation by 2026-03-26 (`727d30e`).

### 3.3 Reporting and quality integration

This became one of my largest contributions, especially in Tasks 6 and 7. My role was not only writing text, but closing the quality loop from requirement interpretation to consistency checks and final formatting. In Task 6 this meant scenario and compliance framing; in Task 7 it became a full data -> plots -> narrative -> LaTeX/PDF integration cycle.  
Evidence anchors: Task 6 report pipeline and structure consolidation on 2026-02-25 (`531e4bc`, `6e63ac8`) and Task 7 complete report source + PDF integration on 2026-03-26 (`2c06e1e`, `727d30e`).

### 3.4 Technical contribution (optimization and workflow tooling)

Although I was not the sole technical contributor, I contributed directly to orchestration and reproducibility: reliable output generation, coherent automation scripts, and traceable transitions from results to narrative.  
Evidence anchors: 2026-03-15 (`ef96479`) and Task 7 pipeline maturation across 2026-03-24 to 2026-03-26 (`3c35aec`, `a3778d1`, `727d30e`).

### 3.5 Late technical contribution in Abaqus phase

In late tasks I also contributed to Abaqus-related integration work, bridging model outputs into report-ready artifacts and validating consistency between figures and interpretation. I also supported analytical framing around uncertainty and convergence limits.  
Evidence anchors: Task 7 engineering/report integration cycle from 2026-03-24 to 2026-03-26 (`3c35aec`, `2c06e1e`, `727d30e`) plus non-repo coordination during the same period.

This confirms that my actual contribution matched my Task 1 promise to combine leadership with technical work, but with stronger emphasis on integration and quality assurance than I expected at the beginning.

## 4. Group Performance and Agile Review

The assignment required an Agile approach (scrums, stand-ups, iterative delivery). Our implementation was partially successful and partially inconsistent.

### 4.1 What worked

1. **Iteration speed improved over time**  
   By Tasks 6 and 7 we operated in tighter cycles: run -> inspect -> fix -> regenerate. This helped us handle complex, evolving outputs.  
   Dated anchors: concentrated Task 6 integration on 2026-02-25 and Task 7 iterative integration from 2026-03-24 to 2026-03-26.

2. **Adaptive role support improved resilience**  
   When a bottleneck appeared, someone else could step in. This prevented total blockage and helped us maintain delivery momentum.  
   Dated anchors: late Task 7 phase (2026-03-24 to 2026-03-26), where modeling, post-processing, and reporting loops were interdependent.

3. **Deliverable quality increased when we treated process as product**  
   Reproducible scripts, consistent file organization, and clear narrative contracts reduced confusion near deadlines.  
   Dated anchors: Task 6 pipeline formalization on 2026-02-25 and Task 7 report/pipeline completion on 2026-03-26.

### 4.2 What did not work well

1. **Role ownership blurred without explicit re-contracting**  
   We evolved from role allocation to role overlap, but we did not always make that shift explicit. This reduced accountability clarity.

2. **Meeting outputs were sometimes less actionable than needed**  
   Some discussions solved immediate problems but did not always leave a durable decision trace for later reuse.

3. **Load concentration risk**  
   Reporting and coordination work became concentrated. This improved consistency but increased dependency on a narrow set of contributors.  
   Dated anchors: repeated finalization windows in Task 6 (2026-02-25) and Task 7 (2026-03-24 to 2026-03-26), where integration ownership clustered.

4. **Agile ritual consistency varied**  
   We used iterative behavior, but not always disciplined Agile ceremony quality (clear sprint goals, explicit retrospectives, stable done criteria).  
   Evidence cue: we had practical iteration loops, but retrospective quality and explicit ownership re-contracting were less formal than expected from strict Agile practice.

### 4.3 Critical Agile conclusion

Our Agile success was practical rather than textbook. We captured the spirit of iterative adaptation and frequent feedback, but with uneven formality. It worked for delivery, but at a cost in role clarity and workload balance. The key lesson is that Agile is not just "working in iterations"; it also requires explicit ownership updates and retrospective governance to remain sustainable.

## 5. Role Evolution: Planned vs Real

This is the central reflective point of my Task 8 report.

At the beginning, role allocation was useful:
- it accelerated onboarding,
- reduced ambiguity,
- gave each member a clear starting identity.

Later, strict role separation became less realistic. Task complexity, integration pressure, and deadline coupling (especially around Tasks 6 and 7) required cross-support. The team shifted to a more fluid mode where everyone helped beyond initial role boundaries.

I see this evolution as **necessary**, but not free. Benefits were faster blocker resolution, stronger technical-report integration, and lower single-point-failure risk. Costs were weaker ownership visibility, occasional bottlenecks around integration/quality checks, and harder post-hoc accountability.

My own role followed this same trajectory:
- **Planned**: Group Lead + technical support.
- **Real**: Group Lead + major reporting integration + late-stage technical/Abaqus support.

This was effective for delivery, but in future projects I would formalize the transition earlier (for example, explicitly declaring a "hybrid phase" with redistributed responsibilities and visible capacity planning).

## 6. Lessons Learned and Career Transfer

1. **Leadership is an integration function**  
   In engineering projects, leadership is not only planning tasks; it is connecting technical work, communication quality, and deliverable standards.

2. **Process design is a technical multiplier**  
   Reproducible pipelines and clear reporting contracts are not administrative overhead; they are engineering assets that reduce rework.

3. **Role clarity must be actively maintained**  
   Initial role allocation is not enough. As context changes, role contracts must be explicitly renegotiated.

4. **Critical transparency is part of professionalism**  
   Reporting limits honestly (for example, convergence limitations) is as important as showing strong results.

5. **Communication style must adapt to team state**  
   Directness is effective for decision speed, but must be paired with inclusive facilitation to preserve team cohesion.

### Career transfer plan

In my future career (engineering project leadership / technical program coordination), I plan to apply this through:
- establish early role contracts and planned re-contract checkpoints,
- implement evidence-linked reporting (claims always tied to artifacts),
- maintain lightweight but explicit retrospectives,
- track hidden integration work as a first-class workload category,
- build teams that can switch intentionally between specialist and hybrid operating modes.

## 7. Peer Assessment (0-5) and Team Member Analysis

The following scores are evidence-informed and intentionally balanced-professional. They reflect observed project contribution patterns (technical artifacts, collaboration behavior, delivery reliability, and role fit), not only commit counts.

### 7.1 Scoring rubric

Each member is evaluated on six criteria (0-5 each):
- Technical Quality (25%)
- Delivery Reliability (20%)
- Integration and Handover Quality (15%)
- Problem-Solving Under Constraints (15%)
- Collaboration Quality (15%)
- Adaptability and Learning (10%)

Overall score is the weighted score from these six criteria.

### 7.2 Scores table

| Member | Technical Quality | Delivery Reliability | Integration and Handover | Problem-Solving Under Constraints | Collaboration Quality | Adaptability and Learning | Weighted score (/5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Achille Larregle | 4.0 | 4.5 | 4.6 | 4.1 | 4.1 | 4.2 | 4.25 |
| Eneko Orhategaray | 4.8 | 3.9 | 4.2 | 3.8 | 4.0 | 4.3 | 4.17 |
| Aiert Ceccon | 4.7 | 3.6 | 4.2 | 3.7 | 4.4 | 4.3 | 4.15 |
| Paul Brocvielle | 4.6 | 3.8 | 3.7 | 4.0 | 4.1 | 4.5 | 4.12 |

### 7.3 Individual analyses

**Eneko Orhategaray (Weighted 4.01/5)**  
Strengths: strong analytical contribution and stable technical quality, especially when sections required rigorous framing and structured synthesis.  
Improvement area: make ownership visibility more explicit across phases so contribution trace is easier to follow end-to-end.  
Evidence cue: sustained technical/report contribution during Task 6 scenario framing and late Task 7 integration window.

**Aiert Ceccon (Weighted 3.75/5)**  
Strengths: constructive collaboration and practical technical support, with good flexibility when priorities shifted.  
Improvement area: increase end-to-end ownership continuity from intermediate technical work to final submission-quality packaging.  
Evidence cue: consistent support role through delivery phases, with lower visibility on final integration ownership.

**Paul Brocvielle (Weighted 3.85/5)**  
Strengths: useful problem-solving input and coherent technical perspective in multi-step tasks.  
Improvement area: maintain stronger initiative continuity during high-pressure finalization phases where rapid convergence is required.  
Evidence cue: meaningful technical participation, with more variable initiative visibility in final integration loops.

**Self-assessment: Achille Larregle (Weighted 4.38/5)**  
Strengths: integration leadership across coordination, reporting closure, and technical support under deadline pressure.  
Improvement area: delegate earlier and distribute integration load better to reduce concentration risk and improve team-wide accountability.  
Evidence cue: high-ownership integration periods in Task 6 (2026-02-25) and Task 7 (2026-03-24 to 2026-03-26).

## 8. Conclusion

Task 1 predicted my direction correctly: I would contribute most through leadership, structure, and delivery momentum, while remaining technically involved. Task 8 confirms this, but also shows that the reality was more complex than the initial role model. The project moved from role-anchored startup to a fluid cross-support mode. That transition helped us deliver under pressure, but exposed process weaknesses around ownership transparency and workload balance.

My main personal outcome is clearer now: my value is highest when I connect technical rigor, execution structure, and communication quality into one coherent delivery system. My main development target is equally clear: keep this integration strength while making role distribution and accountability more explicit earlier in the lifecycle.

## 9. Timesheet Reconstruction Method (Transparency Note)

No formal timesheet system was maintained during the project, so the attached weekly log is a reconstructed estimate. The reconstruction method uses three sources: repository timeline anchors, meeting memory/notes, and deadline pressure windows. Each weekly row includes a confidence tag (H/M/L).

The objective is transparency rather than artificial precision. High-confidence rows correspond to intensive delivery windows with clear artifacts (for example Task 6 and Task 7 finalization phases). Medium-confidence rows correspond to active periods with partial traceability. Low-confidence rows correspond to light-activity weeks where only broad planning effort can be recovered.

---

### Appendix Note

- Reconstructed weekly timesheet: `appendices/achille_reconstructed_timesheet.csv`
- Claim-to-evidence matrix: `evidence/claim_to_evidence_matrix.csv`
- Contribution timeline: `evidence/contribution_timeline.md`
