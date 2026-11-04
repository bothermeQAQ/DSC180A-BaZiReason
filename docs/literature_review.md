# Literature review and contribution boundary

Sources verified 2026-11-03. Repository commits are recorded in [repositories.json](sources/repositories.json). This review distinguishes reported findings from work reproduced here.

## Existing research

**Gao et al. (2023), PAL.** [ICML/PMLR paper](https://proceedings.mlr.press/v202/gao23f.html). Read the paper and inspected the interpreter-based method. Models generate programs whose execution provides solutions. This motivates separating language generation from computation. Our treatment instead supplies a precomputed answer; it neither generates code nor tests PAL itself. We do not transfer PAL's reported gains to BaZiReason.

**Schick et al. (2023), Toolformer.** [NeurIPS paper](https://proceedings.neurips.cc/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html). The paper learns API selection and integration, including calendar-related tools. Our fixed application-side augmentation is a simpler intervention; calling it learned tool use or a tool-calling agent would be inaccurate.

**Chu et al. (2024), TimeBench.** [ACL paper](https://aclanthology.org/2024.acl-long.66/). The benchmark organizes temporal reasoning hierarchically and compares performance across phenomena. This supports category-wise evaluation as a design choice. It is not evidence that the particular boundary cases here cause model failures. The official BibTeX was fetched to check authors, venue and pages.

**Chen and Liu (2026), BaziQA-Benchmark.** [arXiv:2602.12889](https://arxiv.org/abs/2602.12889), [repository](https://github.com/ChenJiangxi/BaziQA). The paper's 200 contest questions (2021-2025) use fixed chart context to isolate downstream judgments from chart construction. The current repository reports expanded and revised data (555 contest questions); its instructions distinguish v1 paper data from current data. The answer keys score contest questions; they do not establish the scientific validity of astrological prediction. Our task is raw datetime-to-pillar computation. Current repository chart generation already uses lunar-python and exposes time conventions, so convention awareness alone is not novel. No contest questions or real birth records are reused.

**imly-sudo, Deterministic Bazi Calendar Rules Benchmark.** [repository](https://github.com/imly-sudo/bazi-llm-benchmark). This is the closest overlap: deterministic pillars, solar boundaries, Zi-hour and true-solar correction. Its README labels model numbers estimates. At the audited commit, the solar solver and model evaluator are unimplemented, the README imports a package absent from the tree, and the day-pillar self-check contains an inconsistent 2024 anchor. The project is relevant prior art, not a validated source of comparative model statistics. We neither reuse its test cases nor adopt its claimed precision.

## Gap and feasible contribution

BaZiReason can contribute an executable, small diagnostic suite with explicit per-record provenance; civil-time gap/fold validation; deliberately scoped clock profiles; paired direct/tool-context prompts; and separate uncertainty/unsupported-input scoring. This is a reproducibility and measurement contribution, not a first benchmark claim. Its strongest research value would be a reproducible same-model study reporting which computational facts are corrected by context and which output errors remain.

## Reference computation

[6tail/lunar-python](https://github.com/6tail/lunar-python) provides candidate labels. Its EightChar implementation uses exact Lichun/Jie times and exposes two sects; the displayed day and hour stem at 23:00 are not derived from the same day in sect 2. [sxtwl](https://github.com/skydancep/sxtwl) provides a second codebase, but both have ShouXing astronomical lineage. Current PyPI sxtwl documentation/API differs from the supplied repository's older examples. Date-level year/month comparison is deliberately excluded on solar-term dates.

[HKO February 2025 calendar](https://www.hko.gov.hk/en/gts/astron2025/files/2025cal02.pdf) supplies two independently published civil-day anchors and a minute-level Lichun anchor. Its page was downloaded, rendered and visually read. This only partially checks labels and timestamp plausibility.

## Threats to validity

- Reference agreement can conceal shared astronomical errors; observed 13.73-17.13 second timestamp differences remain unresolved.
- Synthetic diagnostic sampling is boundary enriched and correlated, with very small subgroups.
- The fixed UTC+8 reference clock is not a universal birth-location convention.
- Tool context contains reference answers, so gains may reflect copying/formatting, not reasoning improvement.
- Public source code, calendar rules and dates may appear in model training; secret new dates do not prove absence of contamination.
- Missing conventions are operationally required even if some dates yield the same chart under both profiles; clarification scores depend on this stated policy.
- Structured-output failures, provider errors, model drift and token truncation need separate logging.
- H3 needs its own convention-policy ablation. The implemented two-condition runner does not identify that causal effect.

## Optional training literature

[Hu et al., LoRA](https://arxiv.org/abs/2106.09685) and [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314) motivate parameter-efficient adaptation; neither implies success for this task. See [finetuning_plan.md](finetuning_plan.md). No training has been performed.
