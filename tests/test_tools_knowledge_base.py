import pytest
from qdrant_client import QdrantClient

from app.services.vector_store import VectorStore
from app.tools import coaching_toolset, knowledge_base

_PROTEIN_CHUNK = {
    "text": "Protein intake for muscle gain should be 1.6 to 2.2 grams per kilogram of body weight per day.",
    "source": "protein-timing.md",
    "domain": "nutrition",
    "chunk_index": 0,
}
_KNEE_CHUNK = {
    "text": "For a knee injury, substitute back squats with leg press, step-ups, or Bulgarian split squats.",
    "source": "knee-safe-substitutions.md",
    "domain": "exercises",
    "chunk_index": 0,
}


@pytest.fixture
def store() -> VectorStore:
    vector_store = VectorStore(QdrantClient(":memory:"), "test_kb")
    vector_store.upsert_chunks([_PROTEIN_CHUNK, _KNEE_CHUNK])
    return vector_store


def test_search_returns_relevant_chunk(store):
    results = store.search("how much protein should I eat to build muscle", domain=None, top_k=5, score_threshold=0.3)
    assert results
    assert results[0]["source"] == "protein-timing.md"


def test_search_respects_domain_filter(store):
    results = store.search("body weight guidance", domain="nutrition", top_k=5, score_threshold=0.0)
    assert results
    assert all(r["domain"] == "nutrition" for r in results)


def test_search_returns_empty_for_unrelated_query(store):
    # Production threshold (knowledge_base._SCORE_THRESHOLD) must filter off-topic queries out.
    results = store.search(
        "what's the weather forecast for tomorrow",
        domain=None,
        top_k=5,
        score_threshold=knowledge_base._SCORE_THRESHOLD,
    )
    assert results == []


def test_reupsert_same_source_replaces_rather_than_duplicates(store):
    store.upsert_chunks(
        [
            {
                "text": "UPDATED: protein needs are 1.6 to 2.2 g/kg for muscle gain.",
                "source": "protein-timing.md",
                "domain": "nutrition",
                "chunk_index": 0,
            }
        ]
    )
    results = store.search("protein for muscle gain", domain=None, top_k=10, score_threshold=0.0)
    protein_matches = [r for r in results if r["source"] == "protein-timing.md"]
    assert len(protein_matches) == 1
    assert "UPDATED" in protein_matches[0]["text"]


async def test_search_knowledge_base_tool_returns_matches(monkeypatch, store):
    monkeypatch.setattr(knowledge_base, "get_vector_store", lambda: store)
    results = await knowledge_base.search_knowledge_base("protein for muscle gain")
    assert results
    assert "source" in results[0]


async def test_search_knowledge_base_tool_degrades_gracefully_on_error(monkeypatch):
    class BrokenStore:
        def search(self, *args, **kwargs):
            raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(knowledge_base, "get_vector_store", lambda: BrokenStore())
    assert await knowledge_base.search_knowledge_base("anything") == []


async def test_search_knowledge_base_tool_degrades_gracefully_when_store_cannot_be_built(monkeypatch):
    def _boom():
        raise RuntimeError("bad QDRANT_URL")

    monkeypatch.setattr(knowledge_base, "get_vector_store", _boom)
    assert await knowledge_base.search_knowledge_base("anything") == []


async def test_search_knowledge_base_tool_returns_empty_when_collection_missing(monkeypatch):
    # Search must not create the collection on read; a never-ingested Qdrant means "no results".
    empty_client = QdrantClient(":memory:")
    monkeypatch.setattr(knowledge_base, "get_vector_store", lambda: VectorStore(empty_client, "never_ingested"))

    assert await knowledge_base.search_knowledge_base("protein") == []
    assert not empty_client.collection_exists("never_ingested")


async def test_search_knowledge_base_tool_clamps_top_k(monkeypatch):
    seen: list[int] = []

    class RecordingStore:
        def search(self, query, domain, top_k, score_threshold):
            seen.append(top_k)
            return []

    monkeypatch.setattr(knowledge_base, "get_vector_store", lambda: RecordingStore())
    await knowledge_base.search_knowledge_base("anything", top_k=500)
    await knowledge_base.search_knowledge_base("anything", top_k=0)

    assert seen == [10, 1]


def test_settings_enable_cloud_inference_only_for_qdrant_cloud(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("QDRANT_CLOUD_INFERENCE", raising=False)
    monkeypatch.setenv("QDRANT_URL", "https://abc.eu-west-2-0.aws.cloud.qdrant.io")
    assert Settings().qdrant_cloud_inference is True

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    assert Settings().qdrant_cloud_inference is False

    monkeypatch.setenv("QDRANT_CLOUD_INFERENCE", "true")
    assert Settings().qdrant_cloud_inference is True


def test_embed_lock_only_serializes_local_inference():
    from app.services.vector_store import _EMBED_LOCK

    class FakeCloudClient:
        cloud_inference = True

    local = VectorStore(QdrantClient(":memory:"), "x")
    cloud = VectorStore(FakeCloudClient(), "x")  # type: ignore[arg-type]

    # Local fastembed isn't thread-safe -> serialized; cloud inference runs in parallel.
    assert local._embed_guard() is _EMBED_LOCK
    assert cloud._embed_guard() is not _EMBED_LOCK


async def test_search_failure_logs_status_detail_in_one_line(monkeypatch, caplog):
    class ForbiddenStore:
        def search(self, *args, **kwargs):
            raise RuntimeError(
                'Unexpected Response: 403 (Forbidden)\nRaw response content:\nb\'{"error":"forbidden"}\''
            )

    monkeypatch.setattr(knowledge_base, "get_vector_store", lambda: ForbiddenStore())
    with caplog.at_level("WARNING", logger="app.tools.knowledge_base"):
        assert await knowledge_base.search_knowledge_base("anything") == []

    (record,) = caplog.records
    assert "403" in record.getMessage()
    assert "\n" not in record.getMessage()
    assert record.exc_info is None  # no traceback for an expected, recoverable failure


def test_coaching_toolset_exposes_all_tools_with_parameter_docs():
    tools = coaching_toolset.tools
    assert set(tools) == {
        "calculate_protein_intake",
        "search_knowledge_base",
        "calculate_heart_rate_zones",
        "calculate_power_zones",
        "calculate_swim_pace_zones",
    }
    # The Args sections must reach the model as parameter descriptions, as they did over MCP.
    kb_schema = tools["search_knowledge_base"].tool_def.parameters_json_schema
    assert "physiotherapy" in kb_schema["properties"]["domain"]["description"]
    assert kb_schema["properties"]["domain"]["anyOf"][0]["enum"] == [
        "nutrition",
        "exercises",
        "training",
        "physiotherapy",
    ]
