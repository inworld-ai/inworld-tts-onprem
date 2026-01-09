"""Embedding client implementations for embedding server APIs."""

import aiohttp
import click

from . import data_types
from .clients import EmbeddingClient
from .timer import PausableTimer


class HttpEmbeddingClient(EmbeddingClient):
    """Base class for HTTP-based embedding clients."""

    def __init__(
        self, server_url: str, model_id: str | None = None, verbose: bool = False
    ):
        self._server_url = server_url
        self._model_id = model_id
        self._verbose = verbose

    def _get_request_url(self) -> str:
        """Get the request URL for embeddings."""
        return f"{self._server_url.rstrip('/')}/v1/embeddings"

    def _format_request_data(
        self,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> dict:
        """Format request data for the embeddings API."""
        request_data = {
            "input": [text],  # API expects a list of strings
            "normalize": True,  # Enable normalization as shown in README example
        }

        # Add model if specified
        if self._model_id:
            request_data["model"] = self._model_id

        return request_data

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
        }

    def _extract_embedding_from_response(self, response_data: dict) -> list[float]:
        """Extract embedding vector from response."""
        data = response_data.get("data", [])
        if not data:
            raise ValueError("No embedding data in response")

        # Return the first embedding (since we send one text at a time)
        return data[0].get("embedding", [])

    def _extract_usage_from_response(self, response_data: dict) -> tuple[int, int]:
        """Extract token usage from response. Returns (prompt_tokens, total_tokens)."""
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get(
            "total_tokens", prompt_tokens
        )  # fallback to prompt_tokens
        return prompt_tokens, total_tokens

    async def embed(
        self,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> data_types.EmbeddingStats:
        """Embeds text and collects latency statistics."""
        if self._verbose:
            click.echo(f"Embedding text: {text[:50]}...")

        # Create a timer to measure the total time taken for the request
        timer = PausableTimer()

        # Pause during request preparation (client-side processing)
        timer.pause()
        request_data = self._format_request_data(text, inference_settings)
        headers = self._get_headers()
        timer.resume()

        # Use aiohttp for async HTTP requests
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=300, connect=30)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True,
            read_bufsize=1024 * 1024,  # 1MB read buffer
        ) as session:
            timer.start()
            url = self._get_request_url()
            if self._verbose:
                timer.pause()
                click.echo(f"Sending request to {url}")
                timer.resume()

            async with session.post(
                url, json=request_data, headers=headers
            ) as response:
                response.raise_for_status()
                generation_time = timer.stop()

                timer.pause()  # Pause during response processing
                response_data = await response.json()
                embedding = self._extract_embedding_from_response(response_data)
                prompt_tokens, total_tokens = self._extract_usage_from_response(
                    response_data
                )
                timer.resume()

                stats = data_types.EmbeddingStats(
                    e2e_latency_ms=generation_time * 1000.0,
                    response_data=response_data,
                    embedding_dimension=len(embedding),
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                )

                if self._verbose:
                    click.echo(f"Embedding generation time: {generation_time:.3f}s")
                    click.echo(f"Embedding dimension: {len(embedding)}")
                    click.echo(f"Tokens used: {total_tokens}")

                return stats


class EmbeddingServerClient(HttpEmbeddingClient):
    """Client for embedding server compatible with the Universe-AI embedding server."""

    def __init__(
        self, server_url: str, model_id: str | None = None, verbose: bool = False
    ):
        super().__init__(server_url, model_id, verbose)
