"""Loads the user-defined LangChain models from the root `model.py` and wraps them."""
from __future__ import annotations

import importlib
from typing import Any


def _user_module():
    return importlib.import_module("model")


def get_chat_model(role: str | None = None) -> Any:
    """role in {"reader", "judge", "extraction", "query"}: uses `<ROLE>_LLM` if model.py defines it, else `LLM`."""
    mod = _user_module()
    if role and hasattr(mod, f"{role.upper()}_LLM"):
        return getattr(mod, f"{role.upper()}_LLM")
    return mod.LLM


def get_embeddings() -> Any:
    return _user_module().EMBEDDINGS


def get_chat_pool() -> list[Any]:
    """model.py LLM_POOL if defined (same model, several providers), else [LLM]."""
    mod = _user_module()
    return list(getattr(mod, "LLM_POOL", None) or [mod.LLM])
