"""Minimal bounded OpenAI-compatible chat interface; no implicit retries."""

import json
import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class Completion:
    text: str
    model: str
    latency_seconds: float
    usage: dict | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None


class OpenAICompatibleClient:
    def __init__(self, model, base_url, api_key=None, transport=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("BAZIREASON_API_KEY")
        self.transport = transport
        if not self.api_key:
            raise ValueError("BAZIREASON_API_KEY is required (use a dummy only for local servers)")

    def complete(self, messages, temperature=0, max_tokens=512):
        start = time.perf_counter()
        with httpx.Client(timeout=60, transport=self.transport) as client:
            response = client.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
        # Do not persist exception text/body, which might contain credentials or headers.
        if response.status_code != 200:
            raise RuntimeError(f"HTTP status {response.status_code}")
        data = response.json()
        choice = data["choices"][0]
        return Completion(
            text=choice["message"]["content"],
            model=data.get("model", self.model),
            latency_seconds=time.perf_counter() - start,
            usage=data.get("usage"),
            finish_reason=choice.get("finish_reason"),
            system_fingerprint=data.get("system_fingerprint"),
        )


class SyntheticMockClient:
    """Schema-only fixture. Never uses case labels and never represents a real LLM."""

    model = "SYNTHETIC-MOCK-NOT-AN-LLM"

    def complete(self, messages, temperature=0, max_tokens=512):
        result = {
            "status": "unsupported",
            "pillars": None,
            "reason_codes": ["synthetic_mock"],
            "conventions": None,
        }
        return Completion(json.dumps(result), self.model, 0.0)
