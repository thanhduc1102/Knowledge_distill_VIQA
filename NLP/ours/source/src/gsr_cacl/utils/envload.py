"""Robust .env loader for this project.

The repository's ``.env`` is hand-written and mixes shell-style assignments
(``KEY = value``) with Python-style lines (``os.environ['KEY'] = 'value'``),
sometimes with spaces around ``=`` and quotes. ``python-dotenv`` cannot parse it
reliably, so we extract ``KEY=VALUE`` pairs with a tolerant regex and inject them
into ``os.environ`` (without overwriting values already present).

Secrets are never logged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Candidate locations, nearest first.
_CANDIDATES = [
    Path(__file__).resolve().parents[4] / ".env",   # .../NLP/.env
    Path(__file__).resolve().parents[3] / ".env",   # .../ours/.env (unlikely)
    Path.cwd() / ".env",
    Path("/kaggle/Knowledge_distill_VIQA/NLP/.env"),
]

_KV = re.compile(r"([A-Z][A-Z0-9_]{2,})\s*['\]]*\s*=\s*['\"]?([^'\"#\n]+?)['\"]?\s*$")


def parse_env_file(path: str | os.PathLike) -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _KV.search(s)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip("'\"")
        # skip obvious non-values like a bare 'os'
        if val and val.lower() not in {"os", "true", "false"} or len(val) > 4:
            out[key] = val
    return out


def load_env(verbose: bool = False) -> dict[str, str]:
    """Load the first ``.env`` found and inject keys into ``os.environ``.

    Returns the parsed dict (values present, but callers should not print them).
    Does not overwrite variables already set in the environment.
    """
    parsed: dict[str, str] = {}
    for cand in _CANDIDATES:
        if Path(cand).exists():
            parsed = parse_env_file(cand)
            if verbose:
                print(f"[envload] loaded {cand}: keys={sorted(parsed)}")
            break
    for k, v in parsed.items():
        os.environ.setdefault(k, v)
    # Normalise the Gemini key alias.
    if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    return parsed


def google_api_key() -> str:
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        load_env()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not key:
        raise RuntimeError("No GOOGLE_API_KEY / GEMINI_API_KEY found in env or .env")
    return key
