#!/usr/bin/env bash
# One-time Google Colab setup for this repo. From a notebook cell:
#   %cd /content
#   !git clone -b <branch> https://github.com/<you>/<repo>.git eco && cd eco
#   !bash scripts/colab_setup.sh
#
# What this buys you on Colab: a GPU (T4) for the embedding phase, which is the only part of the experiment this
# sandbox is slow at (measured here: 1,008 sentences/min on 4 CPU cores for nomic-embed-text; the full-haystack dev
# run needs ~97K sentences). It buys nothing for Tier 2: reader and judge calls go to a chat API, and you still need
# a working key in .env (see step 4).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4 python + deps"
python - <<'PY'
import sys
assert (3, 11) <= sys.version_info[:2] < (3, 13), f"repo needs Python >=3.11,<3.13, got {sys.version}"
PY
pip install -q -r requirements.txt
pip install -q huggingface_hub

echo "== 2/4 Ollama (embeddings) + nomic-embed-text"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# The default 2048-token context thrashes on the 64-text batches the index builders send (measured 29-40 s per
# /api/embed request before, ~2 s after).
export OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=2
pgrep -x ollama >/dev/null || (nohup ollama serve </dev/null >/tmp/ollama.log 2>&1 &)
for _ in $(seq 30); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 2; done
ollama pull nomic-embed-text
nvidia-smi -L 2>/dev/null || echo "   no GPU visible: embeddings will run on CPU (slow, see the note above)"

echo "== 3/4 dataset (must match the sha256 the frozen dev/test split was made on)"
mkdir -p dataset
if [ ! -f dataset/longmemeval_s_cleaned.json ]; then
  hf download xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json --local-dir dataset
  mv -f dataset/dataset/longmemeval_s_cleaned.json dataset/ 2>/dev/null || true
fi
python - <<'PY'
import hashlib, json
from pathlib import Path
want = json.loads(Path("configs/splits/longmemeval_s_split.json").read_text())["dataset_sha256"]
p = Path("dataset/longmemeval_s_cleaned.json")
got = hashlib.sha256(p.read_bytes()).hexdigest()
assert got == want, f"dataset sha {got[:12]}… != frozen split sha {want[:12]}… : load_split will refuse to run.\n" \
                   f"Fetch the exact same file from xiaowu0162/longmemeval-cleaned, or re-freeze the split " \
                   f"(scripts/prepare_data.py) and accept that dev/test are then a different split."
print(f"   dataset ok ({got[:12]}…)")
PY

echo "== 4/4 chat model credentials"
if [ ! -f .env ]; then cp .env.example .env; fi
python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")            # explicit path: find_dotenv() cannot walk frames when python reads stdin
missing = [k for k in ("NVIDIA_API_KEY", "OPENAI_API_KEY") if not (os.getenv(k) or "").startswith(("nvapi-", "sk-"))]
if missing:
    print(f"   .env has no usable chat key for {missing}. Tier 1 (recall proxies) runs without any key;")
    print("   Tier 2 (QA accuracy vs ReFind / MemoryCPT) cannot. Put a real key in .env before compare.eval=qa.")
else:
    print("   chat key present")
PY

echo
echo "Ready (Colab has no uv; plain python works now that requirements.txt is installed):"
echo "  python scripts/run_compare.py                              # configs/compare.yaml: whole system vs controls, dev, oracle view"
echo "  python scripts/run_compare.py compare.view=full            # the 115K-token haystack (this is the GPU case)"
echo "  python scripts/run_compare.py config=ablations.yaml         # objective ablations"
echo "  python scripts/run_compare.py compare.eval=qa \"compare.systems=[eco_spans,hybrid,mmr,eco,refind,memorycpt,oracle]\"   # Tier 2, needs a key"
echo
echo "If the runtime disconnects: the SQLite embedding cache (.cache/) and results/runs/<id>/predictions.jsonl are"
echo "written incrementally, so keep the folder on Drive and rerun - cached calls are free and instant."
