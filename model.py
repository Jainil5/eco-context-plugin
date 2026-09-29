"""User-owned model definitions. Everything else imports from here and calls models only through eco.llm.client.
Keys/URLs come from .env, never from this file.

LLM_POOL: the SAME model (gpt-oss-20b) served by three providers. The runner assigns each question to one
provider (stable hash of the question id), so every method sees the same provider on the same question.
JUDGE_LLM: one fixed judge for every method and question.
"""
import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

FREELLM_URL = os.getenv("FREELLM_BASE_URL", "http://localhost:3001/v1")
FREELLM_KEY = os.getenv("FREELLM_API_KEY")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")  # the key in .env is an NVIDIA key

LLM_POOL = [
    # Pinned to gpt-oss-20b: freellm "auto" routes to its top-scored model (Nemotron/Gemma), which would change the reader.
    ChatOpenAI(model="gpt-oss-20b", base_url=FREELLM_URL, api_key=FREELLM_KEY, temperature=0, timeout=300, max_retries=2),
    ChatOllama(model="gpt-oss:20b-cloud", temperature=0),
    ChatNVIDIA(model="openai/gpt-oss-20b", api_key=NVIDIA_KEY, temperature=0, timeout=300),
]
LLM = LLM_POOL[1]

# Judge: one fixed model for every method and question. (2026-09-29: freellm's fixed models returned 503 "no usable
# provider key" and NVIDIA nemotron-70b returned 404 for this account.) gpt-oss reasons first, so the runner gives
# the judge 1024 output tokens instead of the official 10 (configs/poc.yaml judge_max_tokens).
JUDGE_LLM = ChatNVIDIA(model="openai/gpt-oss-20b", api_key=NVIDIA_KEY, temperature=0, timeout=300)

EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text:latest")


if __name__ == "__main__":  # quick check: python model.py
    for m in LLM_POOL + [JUDGE_LLM]:
        name = getattr(m, "model_name", None) or getattr(m, "model", None)
        try:
            print(f"{type(m).__name__:<12} {name:<24} ->", repr(m.invoke("Reply with just: OK").content[:40]))
        except Exception as e:  # noqa: BLE001
            print(f"{type(m).__name__:<12} {name:<24} -> FAILED {type(e).__name__}: {str(e)[:120]}")
    print("embedding dim:", len(EMBEDDINGS.embed_query("test")))
