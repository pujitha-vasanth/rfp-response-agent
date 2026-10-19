"""
One small helper that every agent uses to talk to the AI model.

ask_json() sends a system prompt and a user prompt, and returns the model's
answer as a Python dictionary (the prompts always ask for JSON).

In demo mode no model is called. Each agent passes a "demo_answer" function
that produces a rule-based answer instead, so the whole system can run
offline without an API key.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from rfp_agent import config

_client = None
_stats_lock = threading.Lock()
stats = {"calls": 0, "seconds": 0.0}


def _get_client() -> OpenAI:
    """Create the OpenAI client once and reuse it."""
    global _client
    if _client is None:
        if config.LLM_PROVIDER == "azure":
            if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
                raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file.")
            # Azure OpenAI's v1 endpoint works with the standard OpenAI client.
            base_url = config.AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/"
            _client = OpenAI(base_url=base_url, api_key=config.AZURE_OPENAI_API_KEY)
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise RuntimeError("Set OPENAI_API_KEY in your .env file.")
            _client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            raise RuntimeError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use demo, azure or openai.")
    return _client


def _model_name() -> str:
    return config.AZURE_OPENAI_DEPLOYMENT if config.LLM_PROVIDER == "azure" else config.OPENAI_MODEL


def ask_json(system_prompt: str, user_prompt: str, demo_answer) -> dict:
    """
    Ask the model a question and get back a dictionary.

    demo_answer is a function with no arguments. It is only used in demo mode.
    """
    if config.LLM_PROVIDER == "demo":
        return demo_answer()

    started = time.time()
    last_error = None
    for attempt in range(3):  # retry a couple of times on network or rate-limit errors
        try:
            response = _get_client().chat.completions.create(
                model=_model_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content
            result = json.loads(text)
            with _stats_lock:
                stats["calls"] += 1
                stats["seconds"] += time.time() - started
            return result
        except Exception as error:  # keep it simple: wait and try again
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Model call failed after 3 attempts: {last_error}")


def run_in_parallel(function, items: list) -> list:
    """Run function(item) for every item at the same time and keep the order.

    Each requirement is independent, so this cuts the waiting time a lot.
    """
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        return list(pool.map(function, items))


def reset_stats() -> None:
    with _stats_lock:
        stats["calls"] = 0
        stats["seconds"] = 0.0
