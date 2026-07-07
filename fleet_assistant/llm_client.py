# llm_client.py
# Thin wrapper around a free, LOCAL Ollama server (https://ollama.com) - no API key, no
# per-token cost, no internet needed at runtime. If Ollama isn't installed or isn't
# running, generate() returns None instead of raising, so callers (agent.py) can fall
# back to a deterministic template response. The assistant must never hard-fail just
# because a generative model isn't available - see config.py's comment on this.

import time

import requests

from config import (OLLAMA_HOST, OLLAMA_MODEL, LLM_TIMEOUT_SECONDS, LLM_CONNECT_TIMEOUT_SECONDS,
                     LLM_MAX_OUTPUT_TOKENS, LLM_AVAILABILITY_CACHE_SECONDS)

_availability_cache = {"value": None, "checked_at": 0.0}


def is_available():
    # Cached for LLM_AVAILABILITY_CACHE_SECONDS - re-checking reachability on every single
    # message added a real, measurable delay for no benefit (Ollama's up/down status
    # doesn't change mid-conversation in practice).
    now = time.time()
    if now - _availability_cache["checked_at"] < LLM_AVAILABILITY_CACHE_SECONDS:
        return _availability_cache["value"]
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=LLM_CONNECT_TIMEOUT_SECONDS)
        result = resp.status_code == 200
    except requests.exceptions.RequestException:
        result = False
    _availability_cache["value"] = result
    _availability_cache["checked_at"] = now
    return result


def generate(prompt, model=OLLAMA_MODEL):
    # Returns the model's plain-text response, or None on any failure (connection refused,
    # timeout, model not pulled, etc.) - callers must handle None as "use the fallback".
    # num_predict bounds how many tokens the model may generate: unbounded generation was
    # the actual root cause of slow responses that silently fell back to templates anyway
    # (a long-winded reply took longer than LLM_TIMEOUT_SECONDS to finish, so the request
    # timed out AFTER burning most of that time, for nothing).
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"num_predict": LLM_MAX_OUTPUT_TOKENS}},
            timeout=(LLM_CONNECT_TIMEOUT_SECONDS, LLM_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except requests.exceptions.RequestException:
        return None
