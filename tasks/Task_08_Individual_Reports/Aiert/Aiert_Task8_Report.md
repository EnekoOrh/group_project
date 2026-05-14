# Task 8 Individual Reflective Report - Aiert Ceccon

## 1. Context and Task 1 Baseline

This report is my individual Task 8 reflection for the group project. I use it to compare what I expected from myself in Task 1 with what I actually contributed across the project, especially when the work moved from algorithm comparison into cooling-tower optimisation and Abaqus validation.

In Task 1 I described myself as an ENTP-A "Debater". My profile was 78% Extraverted, 78% Intuitive, 56% Thinking, 65% Prospecting, and 68% Assertive. I interpreted this as a profile suited to technical discussion, idea generation, fast problem solving, and calmness under pressure. My strongest team-role indicators were Chair and Shaper, with secondary strengths in innovation, investigation, evaluation, teamwork, and finishing when I commit to a task.

The role I proposed was **Technical Lead & Meeting Chair**. In Task 2, the group confirmed this direction by assigning me the Technical Lead role. The practical promises behind that role were that I would coordinate implementation work, support code quality, guide technical sprint planning, facilitate discussions when needed, and contribute directly to programming, testing, and technical problem solving. I also recognised a personal risk: I tend to prioritise technical work and discussion over formal documentation, so I expected another member to carry more of the report-writing load.

Looking back, Task 1 was accurate about my technical and discussion strengths, but too optimistic about how naturally I would keep my work visible through final submission packaging. I was strongest in Tasks 3 and 4, where the project needed experimentation, plots, evaluation budgets, statistics, and penalty-function refinement. In Tasks 6 and 7 my contribution became more supportive and less traceable, which is the main development point I take from the project.

## 2. Hindsight on Personality Fit

The ENTP-A description fits my real behaviour well. I was comfortable challenging assumptions, asking whether comparisons were fair, and pushing the group to test ideas rather than accept the first result. That was useful in the early optimisation work, because comparing SA, PSO, and BFGS required more than running scripts. We had to decide what counted as a fair evaluation budget, how to present stochastic variability, and how to interpret penalty functions without hiding constraint violations.

The Chair and Shaper parts of my profile also appeared in meetings. Task 2 records that I was expected to help move Kanban tasks through "To Do", "In Progress", and "Done", and to support technical sprint planning. In practice I did contribute to discussion structure, especially when the group needed to convert broad technical ideas into next actions. That part of Task 1 was not just a personality label; it showed up in the way I naturally tried to keep technical conversations active and concrete.

Where Task 1 was incomplete was documentation and end-to-end ownership. I identified reporting as a weaker area at the start, and the project confirmed it. My commits show strong early technical activity, but the final Task 6 and Task 7 packaging was more visibly owned by Achille and Eneko. This does not mean I was absent from later work; it means my contribution shifted toward support, review, and discussion. For assessment and professional engineering, that is not enough. Useful work needs an artifact, a handover, or a decision record attached to it.

## 3. My Contribution Across the Project

### 3.1 Technical leadership and implementation

My clearest contribution was in the early technical phase. In December 2025 I worked on Task 3 by adding surface plotting, experimental data, 3D surface plots, summary statistics, and evaluation-report material. The commits `f090a30`, `0dd7592`, `d16b6fa`, and `4d4900e` show a progression from visual evidence generation to evaluation-budget refinement, statistical analysis, and expanded report explanation.

This matched my Technical Lead role well. The group needed practical code outputs, not only conceptual discussion. My work helped turn SA and PSO comparison into something measurable: convergence plots, 3D surfaces, data collection, and summary statistics. What I learned here is that technical leadership is not just deciding the direction; it also means creating evidence that other people can inspect.

### 3.2 Experiment design, statistics, and reporting support

In Task 3 I contributed to the experiment pipeline around benchmark comparison. I added or modified `run_experiments.py`, `calc_stats.py`, results datasets, summary statistics, and report content. This made the comparison easier to defend because we could refer to repeated runs, evaluation budgets, and cleaned statistical outputs rather than isolated results.

The important lesson here is that fairness in optimisation comparisons has to be engineered. If algorithms are given different budgets or if results are shown without context, the conclusion becomes weak. My Task 3 work helped the team move toward a more evidence-based comparison.

### 3.3 Task 4 penalty-function and BFGS refinement

My strongest single evidence anchor is commit `c76d5de` on 2026-01-21. It added a Task 4 fix plan and changed the Task 4 experiment script to address missing requirements: adding the constrained Rosenbrock problem and testing multiple penalty factors. The Task 4 summary then showed BFGS behaviour across Rastrigin, Rosenbrock, and constrained Rosenbrock, including penalty-factor sensitivity.

This work reflected my ENTP strengths in a useful way. I noticed that the assignment requirement was not only "run BFGS" but also "explain how constraints and penalties affect optimisation". The fix plan converted that gap into a concrete implementation path. The contribution was technical, but it was also interpretive: it helped the report explain why deterministic local methods can be very efficient on smooth problems while still needing careful treatment under constraints.

### 3.4 Communication and meeting contribution

My communication contribution was strongest as a technical facilitator. In Task 2 I was explicitly connected with meeting guidance and sprint planning. I helped structure technical discussion, challenge unclear assumptions, and keep decisions moving when we were choosing how to compare methods. This is partly non-repo evidence, so I do not want to overstate it. Still, it matters because the project depended on turning discussion into implementation.

The limitation is that I did not always leave a durable trace of these contributions. Meeting facilitation is valuable, but if it does not produce a note, an issue, a commit, or a clear ownership update, it becomes hard to prove later. That is one reason Task 8 is useful: it shows the difference between real contribution and traceable contribution.

### 3.5 Later Task 6 and Task 7 support

In Task 6 the project became an engineering cooling-tower problem with eight scenarios, feasibility criteria, volume constraints, and method comparisons. In Task 7 it expanded into Abaqus wind-loading validation. I supported the group through technical discussion, method continuity, and review, but I did not carry the same visible artifact ownership as in Tasks 3 and 4.

This is the main gap between my planned and actual role. Task 2 expected me to oversee coding in Task 6, but the visible implementation and final report integration were more strongly carried by other teammates. My role became more fluid and supportive. That helped the team, but it also shows I need to claim clearer deliverables when the project enters a finalisation phase.

## 4. Group Performance and Agile Review

Our Agile process worked best when it was practical and iterative. By the middle and late phases, the team often followed a useful loop: generate results, inspect plots, identify problems, change the code or interpretation, and regenerate. That rhythm was especially important in Task 6, where mathematical compliance and engineering feasibility had to be separated, and in Task 7, where Abaqus outputs needed careful interpretation.

The strongest group habit was adaptability. Initial roles gave us a stable starting point, but the work naturally became cross-functional. Achille coordinated and integrated, Eneko developed visual and quality-control work, Paul supported architecture and reproducibility, and I contributed technical implementation and discussion leadership. When problems appeared, people helped beyond their original labels.

The weakness was ownership traceability. We were flexible, but we did not always explicitly re-contract roles when the project changed. That meant some workstreams had clear owners while others depended on informal support. The consequence was visible in Task 8: some contributions are easy to prove through commits and final artifacts, while others have to be reconstructed from memory and context.

Our Agile approach was therefore effective but incomplete. We had iteration, adaptation, and communication, but we lacked consistent sprint closure, written retrospectives, and durable ownership logs. Next time I would keep the flexibility but add a weekly "artifact owner" checkpoint: who owns the code, who owns the figure, who owns the report section, and what evidence proves it is done.

## 5. Role Evolution: Planned vs Real

At the start, my role was Technical Lead & Meeting Chair. In reality, it became **early technical implementer, experiment refiner, discussion facilitator, and later-stage support contributor**.

This role evolution had benefits. It let me apply my strengths where they were most useful: technical experimentation, code changes, statistics, penalty logic, and challenging assumptions. It also allowed the team to adapt when different people became stronger owners of specific phases.

The cost was that my role became less visible after the early technical tasks. I was comfortable discussing and supporting, but I did not always convert that support into named deliverables. In professional work, that is a risk because the team needs traceability, not only goodwill. A technical lead should not only help solve problems; they should also make sure the solution is documented, handed over, and connected to the final submission.

## 6. Lessons Learned and Career Transfer

The first lesson is that technical leadership requires evidence. Code, plots, statistics, and fix plans were my strongest contributions because they left clear artifacts. In future projects I will make sure that every technical decision I lead has an attached output: a commit, a design note, a test, or a report paragraph.

The second lesson is that documentation cannot be treated as someone else's problem. I knew from Task 1 that reporting was a weak area for me. The project confirmed that weakness, but also showed why it matters. If the final report does not clearly explain the technical work, the technical value is reduced.

The third lesson is that role fluidity needs explicit communication. I like flexible work, but flexibility without ownership records can make collaboration messy. In future engineering teams I will use short written handovers and weekly ownership checkpoints to keep flexibility without losing accountability.

The fourth lesson is personal: I need to stay engaged through finalisation, not only through the interesting technical phase. The most professional version of my ENTP profile is not just the person who generates ideas quickly. It is the person who turns those ideas into reliable, explained, and reusable deliverables.

## 7. Peer Assessment (0-5)

The scores use the common Task 8 rubric: Technical Quality (25%), Delivery Reliability (20%), Integration and Handover Quality (15%), Problem-Solving Under Constraints (15%), Collaboration Quality (15%), and Adaptability and Learning (10%).

| Member | Technical Quality | Delivery Reliability | Integration and Handover | Problem-Solving Under Constraints | Collaboration Quality | Adaptability and Learning | Weighted score (/5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Achille Larregle | 4.5 | 4.6 | 4.7 | 4.4 | 4.5 | 4.4 | 4.53 |
| Eneko Orhategaray | 4.7 | 4.1 | 4.3 | 4.2 | 4.2 | 4.4 | 4.34 |
| Aiert Ceccon | 4.5 | 4.0 | 4.1 | 4.3 | 4.5 | 4.4 | 4.30 |
| Paul Brocvielle | 4.4 | 3.9 | 4.0 | 4.3 | 4.1 | 4.5 | 4.19 |

**Achille Larregle (4.53/5)**  
Achille provided the strongest delivery and integration leadership. He kept deadlines visible, coordinated finalisation, and carried a large part of the Task 6 and Task 7 report packaging. His improvement area is delegation: some integration load could have been distributed earlier to reduce concentration risk.

**Eneko Orhategaray (4.34/5)**  
Eneko made a strong technical and visual contribution, especially in the cooling-tower phase. His geometry, plotting, and quality checks made the results easier to understand and trust. His improvement area is making reasoning and ownership more visible while work is happening.

**Paul Brocvielle (4.19/5)**  
Paul contributed strong conceptual reasoning, architecture thinking, and reproducibility support. His setup and workflow contributions helped the project become easier to run. His improvement area is maintaining visible initiative during late finalisation phases, where owned deliverables matter most.

**Self-assessment: Aiert Ceccon (4.30/5)**  
My strengths were early technical implementation, experiment refinement, communication, and practical problem solving. My main improvement area is end-to-end ownership: I need to connect technical support more consistently to final report-quality artifacts and handovers.

## 8. Conclusion

Task 1 predicted my direction well. I did contribute through technical leadership, discussion, coding, testing, and optimisation reasoning. The strongest evidence is in Task 3 and Task 4, where my commits show experiment generation, statistical analysis, surface plotting, evaluation-budget refinement, and penalty-function work.

Project reality also corrected my self-image. I cannot rely only on being energetic, technical, and useful in discussion. In an engineering team, contribution must remain visible through final packaging. My main development target is therefore clear: keep the technical curiosity and momentum of my ENTP profile, but pair it with stronger documentation, clearer ownership, and better handover discipline.

## 9. Timesheet Reconstruction Method

No formal timesheet was maintained consistently during the project, so the attached weekly timesheet is reconstructed. I used Task 1 and Task 2 documents, git commits, report artifacts, project milestones, and memory of meeting/review activity. High-confidence rows correspond to weeks with direct commits or dated artifacts. Medium-confidence rows correspond to active project windows with partial evidence. Low-confidence rows correspond to broad planning or lighter review periods.

The aim is transparency rather than false precision. The reconstructed pattern shows that my highest traceable effort was concentrated in Task 3 and Task 4, with lighter but still useful support during Task 6 and Task 7.

---

### Appendix Note

- Reconstructed timesheet: `appendices/aiert_reconstructed_timesheet.csv`
- Claim-to-evidence matrix: `evidence/claim_to_evidence_matrix.csv`
- Contribution timeline: `evidence/contribution_timeline.md`
- Requirement traceability checklist: `evidence/task8_requirement_traceability_checklist.md`
