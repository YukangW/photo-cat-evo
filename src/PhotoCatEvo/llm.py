"""The only file that talks to the LLM (local vLLM or OpenRouter: same OpenAI-compatible API)."""
import asyncio
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types import CompletionUsage

from .config import API_KEY, BASE_URL, MODEL, TEMPERATURE

_REQUEST_TIMEOUT = 300

# Worth retrying: timeouts, rate limits and server-side errors. Anything else (bad key, no credits,
# unknown model, bad request) will fail the same way again, so stop straight away.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_HINTS = {
    401: "The API key is missing or invalid (set LLM_API_KEY or OPENROUTER_API_KEY).",
    402: "Out of OpenRouter credits: https://openrouter.ai/credits",
    404: "Unknown model or URL. Check LLM_MODEL and LLM_BASE_URL.",
}


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    model: str

    @property
    def use_prompt_cache(self) -> bool:
        return urlsplit(self.base_url).hostname == "openrouter.ai" and self.model.startswith("anthropic/")


model_settings = ContextVar("model_settings", default=ModelSettings(BASE_URL, MODEL))


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests_with_usage: int = 0
    requests_without_usage: int = 0

    def record(self, usage: CompletionUsage | None) -> None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if not all(isinstance(count, int) and count >= 0 for count in (prompt, completion, total)):
            self.requests_without_usage += 1
            return
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.requests_with_usage += 1

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.requests_with_usage += other.requests_with_usage
        self.requests_without_usage += other.requests_without_usage

    def as_dict(self) -> dict[str, int | None]:
        counts = asdict(self)
        if self.requests_without_usage and not self.requests_with_usage:
            counts.update(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        return counts


async def _create_completion(**kwargs):
    base_url = model_settings.get().base_url
    # Credentials belong to the server configured in the environment.
    api_key = API_KEY if base_url.rstrip("/") == BASE_URL.rstrip("/") else "not-needed"
    async with AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=_REQUEST_TIMEOUT, max_retries=0) as client:
        async with asyncio.timeout(_REQUEST_TIMEOUT):
            return await client.chat.completions.create(**kwargs)


def chat(messages: list, tools: list | None = None, retries: int = 4, *,
         token_usage: TokenUsage | None = None):
    """Send the conversation, return the assistant message. Retries on rate limits / server errors."""
    kwargs = {"model": model_settings.get().model, "messages": messages, "tools": tools}
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    for attempt in range(retries):
        resp = None
        try:
            resp = asyncio.run(_create_completion(**kwargs))
            if getattr(resp, "error", None):          # OpenRouter can put an error inside a 200 reply
                raise RuntimeError(f"LLM returned an error: {resp.error}")
            if not resp.choices:
                raise RuntimeError("LLM returned no choices.")
            return resp.choices[0].message
        except APIStatusError as e:
            if e.status_code not in _RETRY_STATUS:
                hint = _HINTS.get(e.status_code, "")
                raise RuntimeError(f"LLM request failed with HTTP {e.status_code}: {e.message} {hint}") from e
            err = f"HTTP {e.status_code}: {str(e.message)[:300]}"
        except APIConnectionError as e:
            err = e.__class__.__name__
        except TimeoutError:
            err = f"Total request timeout ({_REQUEST_TIMEOUT:g}s)"
        finally:
            if token_usage is not None:
                token_usage.record(getattr(resp, "usage", None))
        wait = 5 * 2 ** attempt          # 5, 10, 20, 40 s
        print(f"  [llm] {err}. Retrying in {wait}s...", flush=True)
        time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")
