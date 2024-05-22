import logging
from ulid import ULID
from itertools import cycle
from typing import List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import FakeListLLM, GenericFakeChatModel
from langchain_core.load.dump import dumps
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.outputs.llm_result import LLMResult
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache


def random_string() -> str:
    return str(ULID())


class DummyEmbeddings(Embeddings):
    def __init__(self, dims: int = 3):
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dims for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dims


@pytest.fixture
def redis_url():
    return "redis://localhost:6379"


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(size=768)


@pytest.fixture
def openai_embeddings():
    return OpenAIEmbeddings()


@pytest.fixture
def redis_cache(redis_url: str):
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        cache.clear()


@pytest.fixture
async def async_redis_cache(redis_url: str):
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        await cache.aclear()


@pytest.fixture(scope="function")
def redis_semantic_cache(openai_embeddings, redis_url: str):
    cache = RedisSemanticCache(
        name=f"semcache_{str(ULID())}",
        redis_url=redis_url,
        embedding=openai_embeddings,
    )
    try:
        yield cache
    finally:
        cache.clear()


class TestRedisCacheBasicIntegration:
    def test_redis_cache(self, redis_cache):
        redis_cache.update(
            "test_prompt", "test_llm", [Generation(text="test_response")]
        )
        result = redis_cache.lookup("test_prompt", "test_llm")
        assert len(result) == 1
        assert result[0].text == "test_response"
        redis_cache.clear()
        assert redis_cache.lookup("test_prompt", "test_llm") is None

    def test_redis_cache_ttl(self, redis_cache):
        llm = FakeListLLM(cache=redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        redis_cache.update(prompt, str(llm), [Generation(text="test response")])

        # Check that the TTL is set
        key = redis_cache._key(prompt, str(llm))
        ttl = redis_cache.redis.ttl(key)
        assert (
            ttl > 0 and ttl <= 3600
        )  # TTL should be positive and not exceed the set value

    @pytest.mark.asyncio
    async def test_async_redis_cache(self, async_redis_cache):
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result == [Generation(text="async test")]

    @pytest.mark.asyncio
    async def test_async_redis_cache_clear(self, async_redis_cache):
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        await async_redis_cache.aclear()

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result is None

    def test_redis_cache_chat(self, redis_cache):
        responses = cycle(
            [
                AIMessage(content="Hello from cache"),
                AIMessage(content="How are you from cache"),
            ]
        )
        chat_model = GenericFakeChatModel(messages=responses)

        human_message1 = HumanMessage(content="Hello")
        human_message2 = HumanMessage(content="How are you?")
        prompt1 = [human_message1]
        prompt2 = [human_message2]

        # First call should generate a response and cache it
        result1 = chat_model.generate([prompt1])

        # Instead of comparing the entire LLMResult, let's check specific parts
        assert len(result1.generations) == 1
        assert len(result1.generations[0]) == 1
        assert isinstance(result1.generations[0][0], ChatGeneration)
        assert result1.generations[0][0].message.content == "Hello from cache"

        # Cache the result manually (since GenericFakeChatModel doesn't use the cache internally)
        redis_cache.update(dumps(prompt1), str(chat_model), result1.generations[0])

        # Second call with the same prompt should hit the cache
        cached_result = redis_cache.lookup(dumps(prompt1), str(chat_model))
        assert cached_result is not None
        assert cached_result[0].message.content == "Hello from cache"

        # Call with a different prompt should generate a new response
        result3 = chat_model.generate([prompt2])

        # Check specific parts of the new result
        assert len(result3.generations) == 1
        assert len(result3.generations[0]) == 1
        assert isinstance(result3.generations[0][0], ChatGeneration)
        assert result3.generations[0][0].message.content == "How are you from cache"

        # Cache the new result
        redis_cache.update(dumps(prompt2), str(chat_model), result3.generations[0])

        # Verify that both prompts are in the cache
        cached_result1 = redis_cache.lookup(dumps(prompt1), str(chat_model))
        cached_result2 = redis_cache.lookup(dumps(prompt2), str(chat_model))
        assert cached_result1 is not None
        assert cached_result2 is not None
        assert cached_result1[0].message.content == "Hello from cache"
        assert cached_result2[0].message.content == "How are you from cache"


class TestRedisSemanticCacheBasicIntegration:
    def test_redis_semantic_cache_crud(self, redis_url):
        dummy_embeddings = DummyEmbeddings(dims=3)
        cache = RedisSemanticCache(redis_url, embedding=dummy_embeddings)
        cache.update("test_prompt", "test_llm", [Generation(text="test_response")])
        result = cache.lookup("test_prompt", "test_llm")
        assert len(result) == 1
        assert result[0].text == "test_response"
        cache.clear()
        assert cache.lookup("test_prompt", "test_llm") is None

    def test_redis_semantic_cache(self, redis_semantic_cache, openai_embeddings):
        cache_name = redis_semantic_cache.name()
        response1 = "The capital of France is Paris."
        response2 = "This should not be returned."
        llm = FakeListLLM(responses=[response1, response2])
        prompt1 = "What is the capital of France?"
        prompt2 = "Tell me about the capital of France."

        # Manually update the cache with the first prompt
        redis_semantic_cache.update(prompt1, str(llm), [Generation(text=response1)])

        # Lookup using the first prompt (should be an exact match)
        cached_result1 = redis_semantic_cache.lookup(prompt1, str(llm))
        assert cached_result1 is not None, "Cache lookup for exact match returned None"
        assert cached_result1[0].text == response1

        # Lookup using the second prompt (should be a semantic match)
        cached_result2 = redis_semantic_cache.lookup(prompt2, str(llm))
        assert (
            cached_result2 is not None
        ), "Cache lookup for semantic match returned None"
        assert cached_result2[0].text == response1

        llm_result = llm.generate([prompt2])
        assert llm_result.generations[0][0].text == response1

        # Clear the cache
        redis_semantic_cache.clear()
        assert (
            len(redis_semantic_cache.cache._index.client.keys(f"{cache_name}:*")) == 0
        )

        # Verify that after clearing the cache, we get the LLM response (which should be response1 again)
        cached_result3 = redis_semantic_cache.lookup(prompt2, str(llm))
        assert cached_result3 is None, "Cache should be empty after clearing"
