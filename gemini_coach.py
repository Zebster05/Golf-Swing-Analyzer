"""Gemini coaching client with automatic model fallback."""

from __future__ import annotations

import json
import time

# Validated 2026-09-14 on this project's Gemini free-tier key.
# 3.8/3.7/flash-latest often 503 but kept; 2.5/2.0/1.5-flash 404 and omitted.
GEMINI_COACH_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
)

_RETRYABLE_CODES = {404, 429, 503}
_RETRYABLE_STATUSES = {"NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE"}
_RETRYABLE_MARKERS = (
    "429",
    "503",
    "404",
    "unavailable",
    "high demand",
    "resource exhausted",
    "not found",
    "no longer available",
)


def _normalize_error_text(exc):
    return str(exc).lower().replace("_", " ")


def _error_code(exc):
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _error_status(exc):
    return str(getattr(exc, "status", "") or "").upper()


def redact_secrets(message, api_key=None):
    text = str(message)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def is_rate_limit_error(exc):
    if _error_code(exc) == 429:
        return True
    if _error_status(exc) == "RESOURCE_EXHAUSTED":
        return True
    text = _normalize_error_text(exc)
    return "429" in str(exc) or "resource exhausted" in text


def is_retryable_gemini_error(exc):
    if _error_code(exc) in _RETRYABLE_CODES:
        return True
    if _error_status(exc) in _RETRYABLE_STATUSES:
        return True
    text = _normalize_error_text(exc)
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def generate_coaching_with_fallback(
    prompt,
    generate_content,
    models=GEMINI_COACH_MODELS,
    sleep=time.sleep,
    api_key=None,
    rate_limit_retries=2,
    base_delay=2,
    fallback_delay=1,
):
    """Try Gemini models in order; retry 429s on the same model, then fall back."""
    last_error = None
    attempts_per_model = 1 + rate_limit_retries
    last_index = len(models) - 1

    for index, model in enumerate(models):
        for attempt in range(attempts_per_model):
            try:
                response = generate_content(model, prompt)
                data = json.loads(response.text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if index < last_index:
                    sleep(fallback_delay)
                break
            except Exception as exc:
                last_error = exc
                if not is_retryable_gemini_error(exc):
                    return {"error": redact_secrets(exc, api_key)}
                if is_rate_limit_error(exc) and attempt < attempts_per_model - 1:
                    sleep(base_delay * (2**attempt))
                    continue
                if index < last_index:
                    sleep(fallback_delay)
                break
            else:
                if isinstance(data, dict):
                    data["_model_used"] = model
                return data

    last = redact_secrets(last_error, api_key) if last_error is not None else "unknown error"
    tried = ", ".join(models)
    return {
        "error": f"All Gemini model fallbacks failed ({tried}). Last error: {last}"
    }
