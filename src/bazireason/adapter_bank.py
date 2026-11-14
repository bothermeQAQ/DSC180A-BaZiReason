"""One loaded base model serving every LoRA adapter by swapping adapter weights in place.

All BaZiReason adapters share one structure (rank 16, scale 2, the last 16 layers, the
seven projection matrices), so the LoRA layers are created once and only their weights
change. Swapping takes milliseconds and gives the same output as loading the adapter
afresh (``tests/test_phase5.py`` and ``scripts/phase5/check_bank.py`` check this).
"base" is the same model with every ``lora_b`` set to zero, which is exactly the base.

Interpolated adapters are supported because the adapters are nested: each one was
trained from the one before, so ``(1 - a) * R3 + a * A1`` stays on the path between two
related solutions. Adding or averaging unrelated adapters would double-count their
shared history and is not offered.
"""

import time
from pathlib import Path

import mlx.core as mx

from bazireason.mlx_backend import MLXChatModel

ADAPTER_ROOT = Path("artifacts/adapters")
ADAPTERS = {
    "p2": "bazireason-qwen3-4b-lora-v2",
    "p3r2": "bazireason-qwen3-4b-lora-p3-r2",
    "p3r3": "bazireason-qwen3-4b-lora-p3-r3",
    "p4a1": "bazireason-qwen3-4b-lora-p4-a1",
    "p5g1": "bazireason-qwen3-4b-lora-p5-g1",
}
DEFAULT_MODEL = "models/qwen3-4b-instruct-2507-mlx-4bit"


def adapter_file(name: str) -> Path:
    """The weights of a named adapter, an adapter folder under artifacts/adapters, or a path."""
    direct = Path(name) / "adapters.safetensors"
    if direct.exists():
        return direct
    return ADAPTER_ROOT / ADAPTERS.get(name, name) / "adapters.safetensors"


def parse_mix(name: str):
    """'mix:p3r3:p4a1:0.25' -> ('p3r3', 'p4a1', 0.25); anything else -> None."""
    if not name.startswith("mix:"):
        return None
    _, first, second, alpha = name.split(":")
    return first, second, float(alpha)


class AdapterBank(MLXChatModel):
    """An ``MLXChatModel`` whose adapter can be changed with ``use(name)``.

    Names: "base", a key of ``ADAPTERS``, a folder under ``artifacts/adapters``, or
    "mix:<first>:<second>:<alpha>" for the weights ``(1 - alpha) * first + alpha * second``.
    """

    def __init__(self, model_path=DEFAULT_MODEL, first="p3r3"):
        super().__init__(model_path, adapter_file(first).parent, label=first)
        self.weights = {}
        self.current = None
        self.swap_seconds = []
        self.use(first)

    def _load(self, name):
        if name not in self.weights:
            self.weights[name] = mx.load(str(adapter_file(name)))
        return self.weights[name]

    def weights_for(self, name: str) -> dict:
        mixed = parse_mix(name)
        if mixed:
            first, second, alpha = mixed
            a, b = self._load(first), self._load(second)
            return {k: (1 - alpha) * a[k] + alpha * b[k] for k in a}
        if name == "base":
            template = self._load(next(iter(ADAPTERS)))
            return {k: mx.zeros_like(v) if k.endswith("lora_b") else v for k, v in template.items()}
        return self._load(name)

    def use(self, name: str):
        if name == self.current:
            return self
        start = time.perf_counter()
        self.net.load_weights(list(self.weights_for(name).items()), strict=False)
        mx.eval(self.net.parameters())
        self.swap_seconds.append(time.perf_counter() - start)
        self.current = name
        self.model = name
        self.adapter_path = None if name == "base" else name
        return self
