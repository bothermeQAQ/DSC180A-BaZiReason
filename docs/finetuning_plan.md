# Optional final-project fine-tuning plan

Status: **NOT RUN**. Do not start training until reference uncertainty, held-out data and paired inference baselines are stable. Fine-tuning is optional for the December 4 final; a sound evaluation study is more valuable than rushed training.

## Candidate and licensing

Use [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) as an initial small multilingual candidate. Its official model card and LICENSE were checked on 2026-11-03 and show Apache-2.0. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is a smaller pilot candidate with the same declared license, subject to exact revision/license verification at execution. These are candidates, not claims of current optimality. Freeze model/tokenizer revision and chat template before comparing outcomes. No weights were downloaded.

[LoRA (Hu et al.)](https://arxiv.org/abs/2106.09685) freezes base weights and learns low-rank updates. [QLoRA (Dettmers et al.)](https://arxiv.org/abs/2305.14314) combines quantized bases with adapter training. Their general results motivate feasibility, not a predicted BaZi gain.

## Data and splitting

Generate a separate appropriately licensed synthetic training set (initially 2,000-10,000 cases, size subject to learning curves), never train on either the original96-case checkpoint suite or the revised152-case audit suite. Reserve held-out year blocks and transition groups, rather than random near-duplicate timestamp rows. Separate all variants of the same instant, transition, date/profile pair and wording template. Deduplicate exact inputs and normalized instants across splits. Keep externally checked anchors as validation evidence, not training demonstrations.

Use only original prompts and synthetic inputs/labels whose licensing has been audited; do not train on BaziQA contest questions or scraped personal records. Engine outputs are implementation labels with uncertainty; exclude unvalidated near-second transitions and disclose reference lineage. Publish split manifests/hashes before model tuning. Public pretraining contamination cannot be ruled out; memorization is tested through held-out years, unseen combinations, and controlled offsets, not asserted absent.

## Training and comparison

First reproduce zero-shot direct and tool-context baselines with the same base model. Then compare direct SFT/LoRA, context-conditioned SFT/LoRA, and their untuned counterparts under identical prompting, decoding, token budget, held-out cases and attempts. Start with one small pilot; predefine adapter rank (e.g.8), learning-rate grid and early-stopping on a disjoint development set. Record actual trainable parameters, optimization steps, context lengths, seeds, software versions and peak memory. These values are initial planning choices, not executed settings.

Evaluate rule-generalization: held-out years, 60-day cycle transformations, unseen solar boundaries, timezone-equivalent instants, missing/unsupported profiles, and sect-2 hour behavior. Report validity/coverage and refusal rates to prevent apparent uncertainty gains from universal abstention. Use repeated inference and paired clustered intervals. A memorized chart lookup without generalization does not establish improved reasoning.

## Hardware and feasibility

Plan a small GPU run only after checking available course resources and permissions. Approximate base weight storage at BF16 is 1.2GB for 0.6B and 3.4GB for 1.7B; optimizer states, activations, tokenizer, adapters, caches and framework overhead add memory. These are arithmetic storage estimates, **not measured VRAM requirements**. A short-sequence 1.7B LoRA pilot on a 16-24GB GPU is a reasonable engineering starting point, subject to profiling. QLoRA may reduce base storage but brings backend constraints. Apple Silicon may require a different supported adapter stack; do not assume CUDA/bitsandbytes works on this Mac. No GPU, training runtime or budget has been provisioned.

## Research decision

Proceed only if a clean base-model pilot exposes errors that training could plausibly address (schema, convention selection or rule transfer). Tools may already solve the factual computation more simply; training could instead improve faithful tool use and abstention. Stop if held-out reference quality is inadequate, the data split leaks adjacent boundaries, or training crowds out reproducible evaluation. No claim of improvement is permitted without real comparative held-out observations.
