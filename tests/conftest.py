import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.llm.cache import DiskCache
from core.llm.cost import CostLogger, Pricing


class CountingChat(FakeListChatModel):
    """Fake chat model that counts real invocations (to prove cache hits skip the model)."""
    calls: int = 0
    model_name: str = "fake-chat"
    temperature: float = 0.0

    def invoke(self, *a, **kw):
        object.__setattr__(self, "calls", self.calls + 1)
        return super().invoke(*a, **kw)


class FakeEmbeddings:
    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture
def cache(tmp_path):
    c = DiskCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


@pytest.fixture
def pricing():
    return Pricing({"fake-chat": {"input_per_1m": 1.0, "output_per_1m": 2.0},
                    "fake-embed": {"input_per_1m": 0.0}})


@pytest.fixture
def logger(tmp_path, pricing):
    return CostLogger(tmp_path / "costs.jsonl", pricing)


@pytest.fixture
def chat():
    return CountingChat(responses=["answer one", "answer two", "answer three"])


@pytest.fixture
def embeddings():
    return FakeEmbeddings()
