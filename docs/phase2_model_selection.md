# Phase 2 model selection (measured on the training machine)

Date: 2026-11-10. Machine: MacBook Pro, Apple M5 (10 cores), 32 GB unified memory, macOS 26.6.2.
Software: Python 3.12.16, MLX 0.32.4, MLX-LM 0.32.2 (the current PyPI releases on this date).

**Selected base model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0, quantized here to 4 bits and fine-tuned with LoRA (QLoRA).**

The choice was made from measurements on this machine, not from the model family's reputation. The two newest candidate families (Qwen3.5 and Gemma 4) cannot be LoRA-trained here with MLX-LM at our sequence lengths, so they were ruled out on evidence.

## Candidates

All files were downloaded at pinned commits by `scripts/phase2/download_models.py` (weights, tokenizer and configuration only; no repository code is executed).

| Candidate | Repository and revision | License | Parameters | Architecture in MLX-LM |
|---|---|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` @ `851bf6e8` (converted here) | Apache-2.0 | 4.66 B (with vision tower) | `qwen3_5`: 24 of 32 layers are Gated DeltaNet "linear attention" |
| Qwen3.5-2B | `mlx-community/Qwen3.5-2B-MLX-4bit` @ `93760be4` | Apache-2.0 | 2.27 B | `qwen3_5`, same hybrid design |
| Gemma 4 E4B | `mlx-community/gemma-4-e4b-it-4bit` @ `475b9088` | Apache-2.0 upstream (`google/gemma-4-E4B-it`) | 8.0 B stored, "4 B effective" | `gemma4` |
| MiniCPM5-2B | `openbmb/MiniCPM5-2B` @ `42433e90` (converted here) | Apache-2.0 | 2.52 B | `llama` |
| **Qwen3-4B-Instruct-2507** | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (converted here) | Apache-2.0 | 4.02 B | `qwen3`, 36 standard transformer layers |

Gemma 3 was not considered: it uses the custom Gemma terms and is access-gated, so it cannot be fetched without an account.

## Measurements

`scripts/phase2/bakeoff.py` runs one candidate per process on a fixed synthetic pilot (seed 20261110): 30 Chinese rule questions with four options, 12 strict-JSON requests with the engine answer in context, 12 reasoned requests with intermediate tool facts, and 12 real LoRA optimizer steps (rank 8, last 16 layers, batch 4, about 1,180 tokens per sequence). No contest question is used. Raw files: `results/phase2/model_selection/`.

| Candidate (4-bit) | Rule questions (chance 0.25) | Strict JSON valid | Final JSON found in reasoned reply | LoRA training | Training peak memory |
|---|---:|---:|---:|---|---:|
| Qwen3.5-4B | 0.53 | 12/12 | 0/12 | **fails: Metal out of memory** | n/a |
| Qwen3.5-2B | 0.23 | 0/12 (12/12 after stripping code fences) | 11/12 | **fails: Metal out of memory** | n/a |
| Gemma 4 E4B | 0.40 | 0/12 (12/12 after stripping code fences) | 0/12 | **fails: Metal out of memory** | n/a |
| MiniCPM5-2B | 0.30 | 9/12 | 0/12 (never concludes within 700 tokens) | 0.256 it/s, about 1,200 tokens/s | 12.4 GB |
| **Qwen3-4B-Instruct-2507** | 0.33 | **12/12** | **12/12** | 0.122 it/s, about 580 tokens/s | 20.7 GB |

Uncontended generation speed of the two trainable models (`clean_speed.json`): Qwen3-4B-Instruct 46 tokens/s single stream and 170 tokens/s in a batch of 16 (5.2 GB peak); MiniCPM5-2B 80 and 250 tokens/s (2.4 GB peak). The speeds inside the individual bake-off files were taken while other work was running and are lower.

With 30 questions, a rule-question accuracy has a standard error of about 0.09, so only Qwen3.5-4B is clearly above chance; the other four are not distinguishable from each other on that column.

## Why the newest families were ruled out

Qwen3.5 small models replace most attention layers with a recurrent Gated DeltaNet layer. In MLX-LM 0.32.2 back-propagation through that recurrence keeps one state per token and layer, which exhausted GPU memory at batch 4 with 1,180-token sequences (and again in a fresh process). Our prompts carry a long rule card or tool block, so shortening sequences enough to fit would change the task. Inference with Qwen3.5 works and is fast; training is the obstacle. Gemma 4 E4B stores 8 B parameters and also ran out of memory in the same probe. Both findings are specific to this machine, this MLX-LM version and these sequence lengths.

## Why Qwen3-4B-Instruct-2507 over MiniCPM5-2B

- **Instruction following.** It was the only trainable candidate with perfect strict-JSON output and the only candidate of all five that always ended a reasoned reply with the requested final JSON. An untuned baseline that already follows the output format makes "fine-tuned versus base" a comparison of knowledge and reliability rather than of formatting.
- **Size.** 4.02 B parameters is at the top of the intended 1.5-4 B range and trains within 32 GB.
- **Cost.** It trains about half as fast as MiniCPM5-2B and needs more memory. The run uses batch 2 with gradient accumulation 2 to keep the peak near 22 GB.
- **License.** Apache-2.0; no field-of-use restriction on the weights.
- **Thinking mode.** This revision has no hidden reasoning mode, so base and tuned models receive byte-identical prompts.

MiniCPM5-2B remains the documented fallback if memory becomes a problem.

## Frozen artifacts

| Item | Value |
|---|---|
| Upstream weights | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` (8.06 GB, bfloat16) |
| Conversion | `python -m mlx_lm convert --hf-path <snapshot> --mlx-path models/qwen3-4b-instruct-2507-mlx-4bit -q --q-bits 4 --q-group-size 64` (4.501 bits per weight) |
| `model.safetensors` SHA-256 | `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f` (identical to `mlx-community/Qwen3-4B-Instruct-2507-4bit` @ `50d42775`) |
| `tokenizer.json` SHA-256 | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `chat_template.jinja` SHA-256 | `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326` |

Reproduce: `python scripts/phase2/download_models.py --only qwen3-4b-2507`, then the conversion command above.

## Existing BaZi-specialized models

A search of the Hugging Face Hub on 2026-11-10 found several community fine-tunes (for example `dclef/Bazi-Qwen3-1.7B`, Apache-2.0; `hadeseus/Qwen3-0.6B-bazi` and `jakeveo05/bazi-qwen2.5-3b-lora`, no license stated; many GGUF "fortunetelling" uploads without licenses or data descriptions). None documents its training data or whether contest questions were used, so none can serve as a clean comparison on BaziQA. They were not run. The comparison reported in this project is against the published BaziQA baselines and the archived human champion scores.

## Limits

The pilot is small (30 + 12 + 12 items) and synthetic; it ranks instruction following and feasibility, not general ability. The memory failures depend on software versions and may disappear in later MLX-LM releases. Vendor benchmark claims found on the web were not used as evidence.
