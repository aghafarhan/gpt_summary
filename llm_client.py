"""Shared OpenAI-compatible client configuration."""

import os
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_MODEL = "grok-4.6"
_PLACEHOLDER_API_KEY = "not-needed"
_CHAT_SUFFIXES = ("/chat/completions", "/completions")


def _looks_like_url(value: str | None) -> bool:
    print("FUNCTION: _looks_like_url")
    return bool(value) and value.startswith(("http://", "https://"))


def _normalize_base_url(raw_url: str) -> str:
    print("FUNCTION: _normalize_base_url")
    parsed = urlsplit(raw_url.strip().strip('"').strip("'"))
    path = parsed.path.rstrip("/")

    for suffix in _CHAT_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return normalized.rstrip("/")


def _configured_base_url() -> str | None:
    print("FUNCTION: _configured_base_url")
    explicit_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if explicit_base_url:
        return _normalize_base_url(explicit_base_url)

    api_key_or_url = (os.getenv("OPENAI_API_KEY") or "").strip()
    if _looks_like_url(api_key_or_url):
        return _normalize_base_url(api_key_or_url)

    return None


def _configured_api_key() -> str:
    print("FUNCTION: _configured_api_key")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if _looks_like_url(api_key):
        return (
            os.getenv("OPENAI_TOKEN")
            or os.getenv("OPENAI_API_TOKEN")
            or _PLACEHOLDER_API_KEY
        )
    return api_key


def has_llm_configuration() -> bool:
    print("FUNCTION: has_llm_configuration")
    return bool(_configured_api_key() or _configured_base_url())


def unwrap_llm_text(content: str | None) -> str:
    print("FUNCTION: unwrap_llm_text")
    return (content or "").strip()


def extract_json_text(content: str | None) -> str:
    print("FUNCTION: extract_json_text")
    cleaned = unwrap_llm_text(content)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return cleaned[start:end + 1]
    return cleaned


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    print("FUNCTION: get_llm_client")
    client_kwargs = {"api_key": _configured_api_key()}
    base_url = _configured_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


@lru_cache(maxsize=1)
def _discover_chat_model() -> str | None:
    print("FUNCTION: _discover_chat_model")
    if not _configured_base_url():
        return None

    try:
        models = get_llm_client().models.list()
    except Exception:
        return None

    for model in getattr(models, "data", []) or []:
        model_id = getattr(model, "id", "")
        if model_id:
            return model_id

    return None


@lru_cache(maxsize=8)
def get_chat_model(default_model: str = DEFAULT_MODEL) -> str:
    print("FUNCTION: get_chat_model")
    explicit_model = os.getenv("OPENAI_MODEL")
    if explicit_model:
        return explicit_model

    discovered_model = _discover_chat_model()
    if discovered_model:
        return discovered_model

    return default_model
