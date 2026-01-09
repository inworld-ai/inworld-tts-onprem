"""Text generation load testing data structures."""

from typing import Optional, Any
from pydantic import BaseModel


class InferenceSettings(BaseModel):
    """Settings for inference."""

    max_generation_duration_sec: float
    max_tokens: int = 400
    # Optional timestamp control for public/internal TTS clients
    # Accepted values: "TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"
    timestamp_type: Optional[str] = None


class GenerationStats(BaseModel):
    """Stats for a single request."""

    e2e_latency_ms: float
    second_of_audio_generation_latency_ms: float
    first_chunk_latency_ms: Optional[float] = None
    forth_chunk_latency_ms: Optional[float] = None
    avg_chunk_latency_ms: Optional[float] = None
    forth_chunk_missing_count: int = 0


class LlmCompletionStats(BaseModel):
    """Stats for a single LLM completion request."""

    e2e_latency_ms: float
    response_data: Optional[dict[str, Any]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    first_chunk_latency_ms: Optional[float] = None
    forth_chunk_latency_ms: Optional[float] = None
    avg_chunk_latency_ms: Optional[float] = None
    forth_chunk_missing_count: int = 0

    @property
    def second_of_audio_generation_latency_ms(self) -> float:
        """Compatibility property for existing benchmarking code."""
        # For LLM, we can use tokens per second as a rough equivalent
        if self.total_tokens and self.total_tokens > 0:
            return self.e2e_latency_ms / self.total_tokens * 1000  # ms per 1000 tokens
        return self.e2e_latency_ms


class EmbeddingStats(BaseModel):
    """Stats for a single embedding request."""

    e2e_latency_ms: float
    response_data: Optional[dict[str, Any]] = None
    embedding_dimension: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def second_of_audio_generation_latency_ms(self) -> float:
        """Compatibility property for existing benchmarking code."""
        # For embeddings, we can use dimension per second as a rough equivalent
        if self.embedding_dimension and self.embedding_dimension > 0:
            return (
                self.e2e_latency_ms / self.embedding_dimension * 1000
            )  # ms per 1000 dimensions
        return self.e2e_latency_ms


class LoadTestingSettings(BaseModel):
    """Settings for load testing."""

    number_of_samples: int = 100
    max_tokens: int = 1700


class LoadTestResults(BaseModel):
    """Load testing results."""

    avg_qps: float
    e2e_latency_ms_percentiles: dict[int, float]
    second_of_audio_generation_latency_ms_percentiles: dict[int, float]
    first_chunk_latency_ms_percentiles: Optional[dict[int, float]] = None
    forth_chunk_latency_ms_percentiles: Optional[dict[int, float]] = None
    avg_chunk_latency_ms_percentiles: Optional[dict[int, float]] = None
    forth_chunk_missing_count: int
    token_stats: Optional[dict[str, float]] = None
