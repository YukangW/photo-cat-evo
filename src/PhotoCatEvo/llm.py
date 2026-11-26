"""The only file that talks to the LLM (local vLLM or OpenRouter: same OpenAI-compatible API)."""
import time

from openai import APIConnectionError, APIStatusError, OpenAI

from .config import API_KEY, BASE_URL, MODEL, TEMPERATURE

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300, max_retries=0)

# Worth retrying: timeouts, rate limits and server-side errors. Anything else (bad key, no credits,
# unknown model, bad request) will fail the same way again, so stop straight away.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_HINTS = {
    401: "The API key is missing or invalid (set LLM_API_KEY or OPENROUTER_API_KEY).",
    402: "Out of OpenRouter credits: https://openrouter.ai/credits",
    404: "Unknown model or URL. Check LLM_MODEL and LLM_BASE_URL.",
}


def chat(messages: list, tools: list | None = None, retries: int = 4):
    """Send the conversation, return the assistant message. Retries on rate limits / server errors."""
    kwargs = {"model": MODEL, "messages": messages, "tools": tools}
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
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
        wait = 5 * 2 ** attempt          # 5, 10, 20, 40 s
        print(f"  [llm] {err}. Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")
