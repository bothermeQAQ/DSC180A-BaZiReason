"""Local MLX inference on Apple Silicon: one model, optionally with a LoRA adapter.

This module is imported only by Phase 2 scripts, so the calendar package and
its tests keep working without MLX installed. Inference never touches the
network: weights are read from a local directory or the local Hugging Face
cache (``HF_HUB_OFFLINE`` is set before MLX-LM is imported).
"""

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import batch_generate, stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from bazireason.llm_clients import Completion  # noqa: E402


def messages_sha256(messages) -> str:
    """Same prompt hash as bazireason.baselines.run_case."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MLXChatModel:
    """A chat model with deterministic prompt rendering and batched generation.

    ``enable_thinking`` is passed to the chat template. It is False for every
    Phase 2 experiment, so the base and the fine-tuned model see byte-identical
    prompts and neither spends tokens on a hidden reasoning block.
    """

    def __init__(self, model_path, adapter_path=None, label=None, enable_thinking=False):
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.enable_thinking = enable_thinking
        start = time.perf_counter()
        self.net, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)
        self.net.eval()
        self.load_seconds = time.perf_counter() - start
        self.model = label or (
            Path(self.model_path).name + ("+adapter" if self.adapter_path else "")
        )

    def prompt_ids(self, messages) -> list:
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=False,
            )
        )

    def generate(self, messages, max_tokens=512, temperature=0.0, seed=None):
        """One completion. Returns (text, stats)."""
        if seed is not None:
            mx.random.seed(seed)
        ids = self.prompt_ids(messages)
        pieces, last = [], None
        for last in stream_generate(
            self.net,
            self.tokenizer,
            ids,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            pieces.append(last.text)
        stats = {
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.generation_tokens,
            "prompt_tps": last.prompt_tps,
            "generation_tps": last.generation_tps,
            "peak_memory_gb": last.peak_memory,
            "finish_reason": last.finish_reason,
        }
        return "".join(pieces), stats

    def shared_prefix_cache(self, conversations):
        """(cache, prefix length, prompts) for conversations that start with the same tokens.

        The common prefix is run through the model once. ``batch_generate`` copies
        the cache for every prompt, so one cache serves all of them.
        """
        prompts = [self.prompt_ids(messages) for messages in conversations]
        shortest = min(len(p) for p in prompts)
        length = 0
        while length < shortest - 1 and all(p[length] == prompts[0][length] for p in prompts):
            length += 1
        cache = make_prompt_cache(self.net)
        if length:
            self.net(mx.array(prompts[0][:length])[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache, length, prompts

    def generate_with_prefix(
        self, prompts, cache, length, max_tokens, temperature=0.0, seed=None, top_p=0.0, top_k=0
    ):
        """Completions for prompts that share a cached prefix of ``length`` tokens."""
        if seed is not None:
            mx.random.seed(seed)
        response = batch_generate(
            self.net,
            self.tokenizer,
            [p[length:] for p in prompts],
            prompt_caches=[cache] * len(prompts) if length else None,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k),
            completion_batch_size=len(prompts),
            prefill_batch_size=min(len(prompts), 8),
        )
        return response.texts, response.stats

    def generate_batch(
        self,
        conversations,
        max_tokens=512,
        temperature=0.0,
        seed=None,
        batch=16,
        top_p=0.0,
        top_k=0,
    ):
        """Completions for many conversations, in order. Returns (texts, stats).

        Prompts are grouped by length so each batch pads little. With
        temperature 0 the result does not depend on the seed.
        """
        if seed is not None:
            mx.random.seed(seed)
        prompts = [self.prompt_ids(messages) for messages in conversations]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        texts = [None] * len(prompts)
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "peak_memory_gb": 0.0}
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            tic = time.perf_counter()
            response = batch_generate(
                self.net,
                self.tokenizer,
                [prompts[i] for i in chunk],
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k),
                completion_batch_size=batch,
                prefill_batch_size=min(batch, 8),
            )
            totals["seconds"] += time.perf_counter() - tic
            totals["prompt_tokens"] += response.stats.prompt_tokens
            totals["completion_tokens"] += response.stats.generation_tokens
            totals["peak_memory_gb"] = max(totals["peak_memory_gb"], response.stats.peak_memory)
            for index, text in zip(chunk, response.texts, strict=True):
                texts[index] = text
        return texts, totals

    def complete(self, messages, temperature=0, max_tokens=512):
        """The bazireason.llm_clients interface, for bazireason.baselines.run_case."""
        start = time.perf_counter()
        text, stats = self.generate(messages, max_tokens=max_tokens, temperature=temperature)
        return Completion(
            text=text,
            model=self.model,
            latency_seconds=time.perf_counter() - start,
            usage={
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
            },
            finish_reason=stats["finish_reason"],
        )


class ReplayClient:
    """Serves completions generated earlier in a batch, keyed by prompt hash.

    It lets the unchanged ``run_case`` build records for batched local
    inference: prompts are collected, generated together, then replayed.
    """

    def __init__(self, model_label):
        self.model = model_label
        self.pending = {}
        self.done = {}

    def complete(self, messages, temperature=0, max_tokens=512):
        key = messages_sha256(messages)
        if key in self.done:
            return self.done[key]
        self.pending[key] = messages
        raise LookupError("prompt collected for batched generation")

    def run_pending(self, chat_model, max_tokens, temperature=0.0, seed=None, batch=16):
        keys = [k for k in self.pending if k not in self.done]
        if not keys:
            return {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        start = time.perf_counter()
        texts, totals = chat_model.generate_batch(
            [self.pending[k] for k in keys], max_tokens, temperature, seed, batch
        )
        latency = (time.perf_counter() - start) / len(keys)
        limit = max_tokens
        for key, text in zip(keys, texts, strict=True):
            tokens = len(chat_model.tokenizer.encode(text, add_special_tokens=False))
            self.done[key] = Completion(
                text=text,
                model=chat_model.model,
                latency_seconds=latency,
                usage={"completion_tokens": tokens},
                finish_reason="length" if tokens >= limit else "stop",
            )
        self.pending = {}
        return totals
