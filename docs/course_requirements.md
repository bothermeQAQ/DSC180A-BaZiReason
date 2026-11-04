# Course requirements audit

Verified 2026-11-03 against the live 2026-27 site. Times are interpreted in America/Los_Angeles; the public project page gives dates without a submission hour.

## Official requirements

- Checkpoint: **Friday, November 6, 2026**. Final: **Friday, December 4, 2026**. Submit code and report through Gradescope.
- Checkpoint PDF: title, abstract, introduction. Introduction covers motivation, prior work and relevant data; mentor additions may apply.
- Code checkpoint: a public GitHub repository URL with completed code. README must explain data access/storage, dependencies and reproduction commands.
- Final report uses the supplied LaTeX template. The methodology assignment says the template is required for Q1 reports generally. Use it for the checkpoint conservatively; see [template audit](template_audit.md).
- Mentor grades reports; TA grades code. The syllabus assigns Q1 project 50% overall (reports 35%, code 15%). It does not specify the checkpoint/final split.
- General rubric: A/B/C/F based on accuracy, completeness, clarity and effective code/writing. No project-specific numerical rubric was found.

Sources: [project](https://dsc-capstone.org/2026-27/assignments/projects/q1/), [syllabus](https://dsc-capstone.org/2026-27/syllabus/#general-rubric), [home](https://dsc-capstone.org/2026-27/), [methodology/template](https://dsc-capstone.org/2026-27/assignments/methodology/05/), [scientific writing](https://dsc-capstone.org/2026-27/lessons/04/), [TA check-in](https://dsc-capstone.org/2026-27/assignments/projects/q1-week5-checkin/).

## Generative AI and academic integrity

Re-read on 2026-11-03:

- [Syllabus](https://dsc-capstone.org/2026-27/syllabus/), Collaboration Policy and Academic Integrity:
  - Generative AI tools "should not be used on any methodology assignments".
  - For domain work, including the Quarter 1 Project, "whether and how you can use Generative AI is up to your mentor. Ask them!"
  - The same section requires citing "any concepts, writing, or code that originates from other sources", and lists not citing outside ideas, writing or code as cheating.
- [UC San Diego Academic Integrity Policy](https://senate.ucsd.edu/operating-procedures/senate-manual/appendices/2) (Senate Manual Appendix 2, amended 6/2/26):
  - Students may not "employ aids (including artificial intelligence)" in course work unless the instructor authorizes them.
  - Students may not submit the work of others "(human or machine)" as their own.
  - Students must submit work that is "an honest and fair representation of their knowledge and abilities".
  - It does not prescribe a specific disclosure statement; instructors state course rules in writing.

The D25 domain description lists coding agents (Claude Code, Codex) among its summer tasks. That is not an authorization for the Q1 Project. Permitted use, and any acknowledgment the mentor wants in the report, must be confirmed by the student with Professor Shang. The README states that AI assistance was used extensively.

## Uncertainties and conflicts

The writing lesson contains a stale Monday November 3 checkpoint reference. The project page and current home page agree on Friday November 6; use these and verify Gradescope/Piazza before submission. The syllabus contains other stale year/date text. The TA check-in page describes October 26-30 meetings, not a replacement PDF rubric. Gradescope/Piazza and mentor-specific instructions were not accessed. The user's account of mentor openness is not independent evidence of final topic approval. Confirm scope with Professor Shang. Nothing was submitted to Gradescope.

## Self-imposed quality rubric (not official)

| Dimension | Conservative standard | Evidence |
|---|---|---|
| Correctness | Explicit conventions, date validation, external spot checks; disclose disagreements | tests, diagnostics, source audit |
| Contribution | State overlap; focus on controlled boundaries and convention handling | literature review |
| Methodology | Same-model paired conditions, fixed schema, no direct-answer leakage | prompts, runner, design |
| Reproducibility | Seed, pinned versions, portable JSONL, full commands | README and lock files |
| Analysis | Correct denominators; separate mocks and real model observations | metrics, results notes |
| Writing | Verified citations; no unperformed experiments stated as results | PDF and editable LaTeX |

## Compliance checklist

- [x] PDF title, abstract and introduction, including literature/data.
- [x] Public repository prepared/published as recorded in CHECKPOINT_STATUS.md.
- [x] README with setup, data, experiments, checks and limitations.
- [x] Source and calendar assumptions documented.
- [ ] Student verifies exact Gradescope deadline/time and assignment.
- [ ] Student reviews content and mentor-specific expectations.
- [ ] Student submits PDF and repository URL to the correct assignments.
