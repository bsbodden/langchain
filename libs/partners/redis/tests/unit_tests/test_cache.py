import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation
from langchain_redis import RedisCache, RedisSemanticCache
from redisvl.extensions.llmcache import SemanticCache as RedisVLSemanticCache


class MockRedisJSON:
    def __init__(self):
        self.data = {}

    def set(self, key, path, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)


class MockRedis:
    def __init__(self):
        self._json = MockRedisJSON()

    def json(self):
        return self._json

    def expire(self, key, ttl):
        pass  # We're not implementing TTL in the mock

    def scan(self, cursor, match=None, count=None):
        matching_keys = [
            k for k in self._json.data.keys() if match is None or k.startswith(match)
        ]
        return 0, matching_keys

    def delete(self, *keys):
        for key in keys:
            self._json.data.pop(key, None)


# Helper functions (make sure these match the ones in your actual implementation)
def _serialize_generations(generations):
    return json.dumps([gen.dict() for gen in generations])


def _deserialize_generations(generations_str):
    try:
        return [Generation(**gen) for gen in json.loads(generations_str)]
    except (json.JSONDecodeError, TypeError):
        return None


class MockRedisVLSemanticCache:
    def __init__(self):
        self.data = {}
        self.distance_threshold = 0.2  # Default value

    def check(self, vector):
        for stored_vector, stored_data in self.data.items():
            distance = np.linalg.norm(np.array(vector) - np.array(stored_vector))
            if distance <= self.distance_threshold:
                return stored_data
        return []

    def store(self, prompt, response, vector, metadata=None):
        self.data[tuple(vector)] = [{"response": response, "metadata": metadata}]

    def clear(self):
        self.data.clear()

    def _vectorize_prompt(self, prompt):
        # Simple mock implementation that returns different vectors for different prompts
        return [hash(prompt) % 10 * 0.1, hash(prompt) % 7 * 0.1, hash(prompt) % 5 * 0.1]


class TestRedisCache:
    @pytest.fixture
    def redis_cache(self):
        mock_redis = Mock()
        mock_redis.json.return_value = mock_redis
        mock_redis.set = Mock()
        mock_redis.get = Mock(return_value=None)
        mock_redis.expire = Mock()
        mock_redis.scan = Mock(return_value=(0, []))
        mock_redis.delete = Mock()

        with patch("langchain_redis.cache.Redis.from_url", return_value=mock_redis):
            cache = RedisCache(
                redis_url="redis://localhost:6379", ttl=3600
            )
            cache.redis = mock_redis
            return cache

    def test_update_and_lookup(self, redis_cache):
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]

        # Mock the set method to store the data
        def mock_set(key, path, value):
            redis_cache.redis.data = {key: value}

        redis_cache.redis.set.side_effect = mock_set

        # Mock the get method to retrieve the data
        def mock_get(key):
            return redis_cache.redis.data.get(key)

        redis_cache.redis.get.side_effect = mock_get

        redis_cache.update(prompt, llm_string, return_val)
        result = redis_cache.lookup(prompt, llm_string)

        assert result is not None, "Lookup result should not be None"
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert (
            result[0].text == "test response"
        ), f"Expected 'test response', got '{result[0].text}'"

    def test_clear(self, redis_cache):
        prompt1, prompt2 = "test prompt 1", "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_cache.update(prompt1, llm_string, return_val)
        redis_cache.update(prompt2, llm_string, return_val)

        redis_cache.clear()
        assert redis_cache.lookup(prompt1, llm_string) is None
        assert redis_cache.lookup(prompt2, llm_string) is None

    def test_ttl(self, redis_cache):
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]

        redis_cache.update(prompt, llm_string, return_val)

        key = redis_cache._key(prompt, llm_string)
        redis_cache.redis.expire.assert_called_once_with(
            key, 3600
        )


class TestRedisSemanticCache:
    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock(spec=Embeddings)
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        return embeddings

    @pytest.fixture
    def redis_semantic_cache(self, mock_embeddings):
        with patch(
            "langchain_redis.cache.RedisVLSemanticCache",
            return_value=MockRedisVLSemanticCache(),
        ):
            return RedisSemanticCache(
                redis_url="redis://localhost:6379", embedding=mock_embeddings
            )

    def test_update(self, redis_semantic_cache):
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt, llm_string, return_val)

        vector = redis_semantic_cache.cache._vectorize_prompt(prompt)
        assert redis_semantic_cache.cache.data[tuple(vector)] is not None

    def test_lookup(self, redis_semantic_cache):
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt, llm_string, return_val)

        result = redis_semantic_cache.lookup(prompt, llm_string)

        assert result is not None
        assert len(result) == 1
        assert result[0].text == "test response"

        # Test lookup with different llm_string
        different_result = redis_semantic_cache.lookup(prompt, "different_llm")
        assert different_result is None

    def test_clear(self, redis_semantic_cache):
        prompt1, prompt2 = "test prompt 1", "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt1, llm_string, return_val)
        redis_semantic_cache.update(prompt2, llm_string, return_val)

        redis_semantic_cache.clear()
        assert len(redis_semantic_cache.cache.data) == 0

    def test_distance_threshold(self, redis_semantic_cache):
        redis_semantic_cache.cache.distance_threshold = 0.1
        prompt1 = "test prompt 1"
        prompt2 = "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt1, llm_string, return_val)

        # Lookup with the same prompt should return the result
        result_same = redis_semantic_cache.lookup(prompt1, llm_string)
        assert result_same is not None
        assert len(result_same) == 1
        assert result_same[0].text == "test response"

        # Lookup with a different prompt should return None due to distance threshold
        result_different = redis_semantic_cache.lookup(prompt2, llm_string)
        assert result_different is None

        # Test with a higher distance threshold
        redis_semantic_cache.cache.distance_threshold = 1.0
        result_high_threshold = redis_semantic_cache.lookup(prompt2, llm_string)
        assert result_high_threshold is not None
        assert len(result_high_threshold) == 1
        assert result_high_threshold[0].text == "test response"
