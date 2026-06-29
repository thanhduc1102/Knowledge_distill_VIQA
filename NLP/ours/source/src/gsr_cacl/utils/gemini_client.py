"""Thin, robust wrapper over the Gemini API (``google.genai``).

Features needed for reproducible, resumable research runs:
  * disk cache keyed by (model, messages, temperature, sample index) so re-runs and
    crashes never re-spend quota or change already-collected answers;
  * exponential backoff with jitter on 429/503/transient errors;
  * a simple requests-per-minute throttle (free-tier friendly);
  * optional structured-confidence parsing.

No secret is ever logged. The key is read via :mod:`gsr_cacl.utils.envload`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

from gsr_cacl.utils.envload import google_api_key

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CACHE = "outputs/cache/gemini"


class _RateLimiter:
    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            sleep = self._last + self.min_interval - now
            if sleep > 0:
                time.sleep(sleep)
            self._last = time.time()


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str = DEFAULT_CACHE,
        rpm: float = 0.0,              # 0 = no client-side throttle (rely on retries)
        max_retries: int = 6,
        timeout_s: float = 60.0,
        thinking_budget: int = 0,     # 0 disables "thinking" so small max_tokens still return text
    ):
        from google import genai  # lazy import

        self.model = model
        self.client = genai.Client(api_key=google_api_key())
        self.cache_dir = Path(cache_dir) / model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = _RateLimiter(rpm)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.thinking_budget = thinking_budget
        self.n_calls = 0
        self.n_cache_hits = 0

    # ---- cache -----------------------------------------------------------
    def _key(self, prompt: str, system: str, temperature: float, idx: int,
             max_tokens: int) -> str:
        h = hashlib.sha1()
        h.update(self.model.encode())
        h.update(b"\x00"); h.update(system.encode())
        h.update(b"\x00"); h.update(prompt.encode())
        h.update(f"\x00{temperature}\x00{idx}\x00{max_tokens}".encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ---- generation ------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        idx: int = 0,
        use_cache: bool = True,
    ) -> str:
        from google.genai import types

        key = self._key(prompt, system, temperature, idx, max_tokens)
        cp = self._cache_path(key)
        if use_cache and cp.exists():
            try:
                self.n_cache_hits += 1
                return json.loads(cp.read_text())["text"]
            except Exception:
                pass

        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
        )
        # Disable "thinking" on 2.5 models so a tight max_output_tokens still yields an answer.
        if "2.5" in self.model and self.thinking_budget is not None:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            except Exception:
                pass
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                text = (resp.text or "").strip()
                self.n_calls += 1
                if use_cache:
                    cp.write_text(json.dumps({"text": text}))
                return text
            except Exception as exc:  # noqa: BLE001 - need broad retry
                last_err = exc
                msg = str(exc).lower()
                # backoff on rate-limit / transient server errors
                transient = any(t in msg for t in ("429", "resource_exhausted", "rate",
                                                   "503", "500", "unavailable",
                                                   "deadline", "timeout"))
                wait = min(60.0, (2 ** attempt) + random.uniform(0, 1.5))
                if not transient and attempt >= 1:
                    # non-transient and already retried once: give up faster
                    break
                time.sleep(wait)
        raise RuntimeError(f"Gemini generate failed after {self.max_retries} retries: {last_err}")

    def stats(self) -> dict:
        return {"model": self.model, "api_calls": self.n_calls,
                "cache_hits": self.n_cache_hits}
