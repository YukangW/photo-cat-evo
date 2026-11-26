"""Settings for the paper data-extraction agent. Change these, or set them as environment variables."""
import os

# ---- Which LLM to use -------------------------------------------------------
# Local vLLM (default):   LLM_BASE_URL=http://localhost:8000/v1   LLM_MODEL=<name used in `vllm serve`>
# OpenRouter:             LLM_BASE_URL=https://openrouter.ai/api/v1  LLM_MODEL=<slug, e.g. anthropic/claude-opus-5.5>
#                         key: LLM_API_KEY, or OPENROUTER_API_KEY if LLM_API_KEY is not set
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "not-needed"  # vLLM ignores it
MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B")

# Set LLM_TEMPERATURE=none to use the model's default temperature
# (some reasoning models on OpenRouter don't accept a custom temperature).
_temp = os.getenv("LLM_TEMPERATURE", "0")
TEMPERATURE = None if _temp.lower() == "none" else float(_temp)

# Let the provider cache the paper text between requests (Anthropic models on OpenRouter only).
# The whole paper is resent on every step of every data item, so caching cuts the cost a lot.
USE_PROMPT_CACHE = "openrouter.ai" in BASE_URL and MODEL.startswith("anthropic/")

MAX_STEPS = 8            # max agent loop steps PER DATA ITEM (protects your request limit)
MAX_TARGETS = 10         # max number of separate data items taken from one prompt

# Used when no prompt is given on the command line (same behaviour as the original bandgap agent).
DEFAULT_PROMPT = "Find the bandgap of the main TiO2-based photocatalyst."

# ---- Parameter dictionary ---------------------------------------------------------
# The experimental quantities the agent knows about (band gap, BET surface area, ...): their aliases,
# usual unit, plausible range, search terms and notes. Add your own in parameters.json, or with
#   uv run python -m PhotoCatEvo.parameters add "name" --unit ... --range MIN MAX ...
# Found next to this file, so it works whichever folder you run the agent from.
PARAMETERS_FILE = os.getenv("PARAMETERS_FILE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.json"))

# ---- Folders ------------------------------------------------------------------
TEXT_DIR = "text"
RESULTS_DIR = "results"
