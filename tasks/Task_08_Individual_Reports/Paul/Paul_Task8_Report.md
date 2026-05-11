# Task 8 Individual Reflective Report - Paul Brocvielle

## 1. Context and Task 1 Baseline

This report is my individual Task 8 reflection for the group project. It looks back at the role I proposed in Task 1, how I actually contributed across the project, and what the experience showed about my working style in a technical engineering team.

In Task 1 I described myself as an ENTP-A "Debater". The main traits I identified were curiosity, creativity, adaptability, confidence under pressure, and a preference for understanding how systems work before accepting a fixed solution. My strongest team roles were **Innovator**, **Evaluator**, and **Teamworker**, with **Investigator** as a secondary strength. Based on that profile, I proposed the role of **Technical Strategist & System Designer**. In Task 2, the group formalised this as **Technical Strategist & System Architect**.

The practical promises attached to that role were clear. I said I would help propose and compare technical approaches, design the overall architecture of the work, support debugging and optimisation, challenge weak design choices constructively, and contribute research or external references when needed. The project gives a useful test of those claims because it moved from abstract algorithm comparison in Tasks 3 and 4 to applied cooling-tower optimisation in Task 6 and structural validation in Task 7.

My high-level conclusion is that Task 1 was directionally accurate. I did bring value through technical reasoning, questioning assumptions, workflow setup, and reproducibility. However, the project also showed that idea generation and architecture thinking are not enough on their own. In the late phases, where fast finalisation and visible ownership mattered most, my contribution was less visible than it had been in the earlier technical stages.

## 2. Hindsight on Personality Fit

Several parts of the ENTP-A profile were accurate. I naturally engaged with the "why" behind technical decisions. In Tasks 3 and 4, this matched the need to compare Simulated Annealing, Particle Swarm Optimisation, and BFGS not only as code outputs, but as different search behaviours. I was comfortable questioning whether a result was meaningful, whether an algorithm comparison was fair, and whether the workflow was reproducible enough for someone else to rerun.

The Innovator and Evaluator parts of my Task 1 profile also appeared in the way I contributed to project structure. Around the Task 3 and Task 4 period, I helped turn the project from separate scripts and generated outputs into a cleaner workflow with setup scripts, dependency files, a clearer README, and ignored generated artifacts. That is not the most visible academic content, but it mattered because reproducibility is part of technical quality.

Where Task 1 was incomplete was on follow-through and final-stage visibility. I had already identified that I can move quickly, become impatient with repetitive tasks, and find detailed finishing less natural than conceptual work. That became relevant later. By Tasks 6 and 7, the work required a lot of sustained reporting, validation, figure consistency, and final integration. Achille and Eneko had more visible ownership in those final loops, while my contribution was more supportive and discussion-based. This does not mean the original profile was wrong; it means I need to convert strategic input into clearer, owned deliverables more consistently.

## 3. My Contribution Across the Project

### 3.1 Early role setup and technical direction

In the first project phase, my contribution was mainly conceptual. Task 2 records that Paul and Aiert were expected to lead the methodological research for Tasks 3 and 4, while Achille coordinated the group and Eneko supported quality analysis. This matched my proposed role well. I was useful in discussions where the team needed to decide how to compare methods fairly, how to structure benchmarks, and how to explain algorithm behaviour.

The technical direction in Tasks 3 and 4 needed more than running algorithms. The group had to explain why PSO behaved well on multimodal landscapes, why BFGS was efficient on smooth local problems, and why constraints changed the interpretation of "best" results. This kind of interpretation suited my Task 1 profile because it required connecting code behaviour, mathematical assumptions, and engineering judgement.

### 3.2 Tasks 3 and 4: optimisation strategy and experiment interpretation

Task 3 compared SA and PSO on benchmark optimisation problems. The final report emphasised global search capability, precision, constraint handling, repeated runs, and equal evaluation budgets. My role was aligned with technical strategy: discussing what a fair comparison should mean and helping keep the work focused on interpretable evidence rather than only raw outputs.

Task 4 added BFGS and made the comparison more interesting. The project journey records the key interpretation: BFGS was extremely efficient on Rosenbrock but fragile on Rastrigin because local gradient information can trap it in local minima. This became an important technical lesson for the group: no single method was universally best. The final recommendation of a hybrid strategy, using a global searcher before BFGS local refinement, matched the type of architectural reasoning I expected to provide in Task 1.

### 3.3 Reproducibility, workflow, and repository structure

My clearest repo-visible contribution is the 2026-02-04 work around project setup and workflow. Commit `75bef48` added setup scripts for Windows and Mac/Linux, a `requirements.txt`, a `.gitignore`, and a clearer README with project structure and run instructions. This supported the whole team because the project had multiple scripts, generated results, and environment assumptions.

I also contributed to output-path and plotting robustness. Commit `073e601` updated plotting helpers so output directories could be passed explicitly instead of being hard-coded. This affected 3D plots, trajectory plots, interactive plots, and Task 3/4 experiment scripts. In practical terms, this made generated outputs more predictable and reduced confusion when running scripts from different contexts.

I also cleaned the repository by removing generated or ignored artifacts from version control (`8ab3331`) and adding `__pycache__` handling (`cb0ec5f`). This is a modest contribution compared with building a full technical model, but it is aligned with my stated DevOps and system-structuring strengths. A technical project is easier to trust when the repo separates source, generated results, and environment setup.

### 3.4 Task 6 and Task 7 support

Task 6 was the cooling-tower optimisation stage. The group moved from mathematical benchmarks to a real engineering formulation: frustum area and volume, volume constraints, hyperboloid shape constraints, engineering feasibility, and comparison across eight scenarios. Eneko had strong visible ownership in the geometry, plotting, and quality checks, while Achille contributed heavily to report integration and pipeline closure. My contribution was lighter than in Tasks 3 and 4, but still connected to the architectural side: discussing scenario logic, method comparison, and how BFGS/SA/PSO results should be interpreted under constraints.

Task 7 added Abaqus wind-loading analysis. This stage required detailed simulation workflow, report generation, structural ranking, and careful distinction between engineering-feasible, mathematical fallback, and warning cases. Again, my direct repo-visible ownership was limited. My support was mainly through technical review and cross-task understanding: making sure the link from Task 6 optimisation to Task 7 structural interpretation stayed conceptually coherent. In hindsight, this was useful but not visible enough. If I were repeating the project, I would claim a defined subtask in the Task 7 pipeline rather than staying mostly in advisory mode.

## 4. Group Performance and Agile Review

The group used Agile in a practical way rather than a textbook way. Early on, Task 2 established Notion, Kanban-style tracking, a Gantt view, weekly stand-ups, and role allocation. This helped the team start with clarity. It also gave us a shared language for ownership, deadlines, and progress.

What worked best was iteration. By Tasks 6 and 7, the group had a repeated cycle of generating results, checking plots, correcting interpretation, and rebuilding reports. This was especially important because the technical outputs were not simple. A cooling-tower shape could be mathematically compliant but not engineering-feasible; an Abaqus ranking could identify a provisional winner but still have limited convergence confidence. Iteration helped the group make those distinctions more clearly.

The main weakness was ownership traceability. Roles became fluid, which was necessary because deadlines and task complexity increased, but the group did not always explicitly reassign responsibilities. That meant some people took strong ownership of final integration while others, including me, contributed in ways that were less visible. This was efficient in the short term but weaker for accountability and retrospective evidence.

Documentation also lagged behind code and results at times. The project improved a lot as reporting pipelines matured, but some decisions were easier to reconstruct from commits and final reports than from contemporaneous meeting records. For a future Agile project, I would keep the flexible cross-support model but add a lightweight weekly ownership log: who owns which artifact, what "done" means, and what evidence proves completion.

## 5. Role Evolution: Planned vs Real

My planned role was Technical Strategist & System Designer. My real role was closer to technical strategist plus reproducibility and workflow support, with less late-stage integration ownership than expected.

This evolution had benefits. I contributed where my strengths were strongest: conceptual comparison, method interpretation, architecture, and repository usability. The setup scripts, dependency file, README work, output-path cleanup, and generated-artifact cleanup all made the project easier to run and maintain.

The cost was that my role became less visible when the project shifted toward Task 6/7 finalisation. The late phases rewarded sustained artifact ownership: reports, figures, Abaqus outputs, LaTeX pipelines, and final PDFs. I supported the technical direction, but I did not always translate that support into a clearly owned deliverable. This is the main difference between my Task 1 expectation and Task 8 reality.

## 6. Lessons Learned and Career Transfer

The first lesson is that technical strategy must end in a deliverable. Good ideas, comparisons, and architecture discussions are valuable, but in a team assessment they need to be attached to visible artifacts. In my future engineering work, I will make sure that when I propose a direction, I also claim a concrete output or decision record.

The second lesson is that reproducibility is engineering work. Setup scripts, dependency management, clean repository structure, and deterministic outputs are not administrative details. They make technical results more credible and easier to validate. This lesson transfers directly to optimisation, design engineering, and CAE workflows.

The third lesson is that role fluidity needs explicit tracking. I like flexible problem-solving, but flexibility can reduce accountability if ownership is not updated. In future projects I would keep the adaptability, but pair it with short written handovers and weekly ownership checkpoints.

The fourth lesson is personal: I need to improve finalisation discipline. My natural strength is early-stage technical thinking and system design. My development target is to stay equally engaged when the work becomes repetitive, detailed, and close to submission. That is especially important in engineering, where the final traceability of a result can matter as much as the initial idea.

## 7. Peer Assessment (0-5)

### 7.1 Scoring rubric

Each member is scored using the common Task 8 criteria:
- Technical Quality (25%)
- Delivery Reliability (20%)
- Integration and Handover Quality (15%)
- Problem-Solving Under Constraints (15%)
- Collaboration Quality (15%)
- Adaptability and Learning (10%)

### 7.2 Scores table

| Member | Technical Quality | Delivery Reliability | Integration and Handover | Problem-Solving Under Constraints | Collaboration Quality | Adaptability and Learning | Weighted score (/5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Achille Larregle | 4.4 | 4.7 | 4.7 | 4.4 | 4.5 | 4.4 | 4.52 |
| Eneko Orhategaray | 4.7 | 4.1 | 4.2 | 4.3 | 4.1 | 4.4 | 4.33 |
| Aiert Ceccon | 4.4 | 4.0 | 4.0 | 4.2 | 4.3 | 4.2 | 4.20 |
| Paul Brocvielle | 4.3 | 3.8 | 4.0 | 4.3 | 4.1 | 4.4 | 4.14 |

### 7.3 Individual analyses

**Achille Larregle (4.52/5)**  
Achille provided the strongest coordination and final integration contribution. He kept deadlines visible, consolidated reports, and carried major Task 6/7 packaging work. His improvement area is delegation: some final integration work could have been shared earlier to reduce load concentration.

**Eneko Orhategaray (4.33/5)**  
Eneko made a strong technical and visual contribution, especially in Task 6. His geometry, plotting, and quality-checking work made the cooling-tower optimisation more understandable and defensible. His improvement area is making reasoning and task ownership more visible while work is happening, not only in final outputs.

**Aiert Ceccon (4.20/5)**  
Aiert contributed strongly in the early optimisation tasks and helped maintain technical progress. His strengths were practical coding support, collaboration, and discussion leadership. His improvement area is clearer end-to-end ownership from intermediate technical work to final submission artifacts.

**Self-assessment: Paul Brocvielle (4.14/5)**  
My strengths were technical reasoning, problem framing, workflow setup, and adaptability. My setup and reproducibility work helped the project become easier to run and maintain. My improvement area is sustained visible ownership during high-pressure finalisation phases, especially in Tasks 6 and 7.

## 8. Conclusion

Task 1 predicted my direction reasonably well. I did contribute most naturally through strategy, system thinking, questioning assumptions, and technical structure. The strongest evidence is in the Tasks 3-4 phase and the project reproducibility work around setup, README, dependency management, output paths, and repository cleanup.

The project also showed a clear development need. I need to turn strategic input into owned, traceable deliverables more consistently, especially near deadlines. My main takeaway is that my value is highest when I combine conceptual engineering judgement with practical reproducibility and visible follow-through. For my future technical engineering career, the next step is to keep the creativity and adaptability of the ENTP profile, but pair it with stronger final-stage ownership and clearer handover discipline.

## 9. Timesheet Reconstruction Method

No formal weekly timesheet was maintained consistently during the project, so the attached timesheet is a reconstructed estimate. I used three evidence sources: Task release/submission periods, repository commits and artifacts, and memory of meeting and review activity. Each row has a confidence tag: high confidence for weeks with direct artifact or commit evidence, medium confidence for active project windows with partial traceability, and low confidence for broad planning or light review periods.

The objective is not artificial precision. It is to give a transparent account of when my effort was highest and where the evidence is strongest. The reconstruction shows higher effort during Tasks 3-4 and the reproducibility/setup period, with lighter support during Tasks 6-7.

---

### Appendix Note

- Reconstructed timesheet: `appendices/paul_reconstructed_timesheet.csv`
- Claim-to-evidence matrix: `evidence/claim_to_evidence_matrix.csv`
- Contribution timeline: `evidence/contribution_timeline.md`
- Requirement traceability checklist: `evidence/task8_requirement_traceability_checklist.md`
