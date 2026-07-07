# config.py
# Settings for the Fleet Assistant. Kept separate from src_v2/config.py because this
# assistant is a DECOUPLED consumer of the core pipeline's output files (fleet_status_v2.csv,
# work_orders_v2.json, model_metrics_v2.json, data_v2/fleet_telemetry_v2.csv) - it never
# imports src_v2 code directly, the same way two independent services in a real deployment
# would only share data files/APIs, not source code.

import os

# ---- where the core pipeline's output files live, relative to this project's root ----
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_v2")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_v2")

FLEET_STATUS_CSV = os.path.join(OUTPUTS_DIR, "fleet_status_v2.csv")
WORK_ORDERS_JSON = os.path.join(OUTPUTS_DIR, "work_orders_v2.json")
MODEL_METRICS_JSON = os.path.join(OUTPUTS_DIR, "model_metrics_v2.json")
TELEMETRY_HISTORY_CSV = os.path.join(DATA_DIR, "fleet_telemetry_v2.csv")

# ---- free, local LLM backend (Ollama) - entirely optional ----
# If Ollama isn't installed/running, every call gracefully falls back to a deterministic,
# template-based response instead of failing - the assistant ALWAYS answers, with or
# without an LLM available. See llm_client.py.
#
# NOTE: use 127.0.0.1, not "localhost" - on some Windows setups, resolving "localhost"
# tries IPv6 (::1) first and only falls back to IPv4 after a ~1-1.5s delay, adding that
# delay to every single request for no reason. 127.0.0.1 skips that resolution entirely.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
LLM_TIMEOUT_SECONDS = 15
LLM_CONNECT_TIMEOUT_SECONDS = 1.5   # fail fast if Ollama isn't running at all
# Measured on this machine: llama3.2 generates at roughly 10 tokens/second on CPU (no
# GPU). 90 tokens -> ~9s worst case generation time, leaving headroom under the read
# timeout above. Unbounded generation was the original cause of slow responses that
# silently fell back to templates anyway (a long reply took longer than the timeout).
# If responses still feel slow, either lower this further or use a smaller/faster local
# model (e.g. `ollama pull llama3.2:1b`) - CPU token-generation speed, not this code, is
# the limiting factor once the fixed overheads below are removed.
LLM_MAX_OUTPUT_TOKENS = 90
LLM_AVAILABILITY_CACHE_SECONDS = 30  # avoid re-checking reachability on every single message

# ---- assistant behavior ----
TOP_PRIORITY_DEFAULT_N = 5
ROLLING_TREND_CYCLES = 10   # how many recent cycles to summarize for a "trend" answer
