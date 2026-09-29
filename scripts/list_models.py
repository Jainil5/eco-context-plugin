"""List models available at each provider configured in .env (freellm, Ollama, NVIDIA). No generation calls."""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def show(name, url, key=None, field="data", idkey="id"):
    try:
        h = {"Authorization": f"Bearer {key}"} if key else {}
        r = requests.get(url, headers=h, timeout=15)
        ids = [m[idkey] for m in r.json().get(field, [])]
        sys.stdout.write(f"{name} ({r.status_code}): {ids}\n")
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(f"{name}: unreachable ({type(e).__name__}: {e})\n")


if os.getenv("FREELLM_BASE_URL"):
    show("freellm", os.environ["FREELLM_BASE_URL"].rstrip("/") + "/models", os.getenv("FREELLM_API_KEY"))
show("ollama", "http://127.0.0.1:11434/api/tags", field="models", idkey="name")
