"""TTS client implementations for different TTS services."""

import base64
import json
import uuid

import aiohttp
import click

from . import data_types
from .clients import TextToSpeechClient
from .timer import PausableTimer


def _get_audio_duration_and_samples(
    audio_data: bytes, sample_rate: int, has_header: bool = True
) -> tuple[float, int]:
    """Returns the duration and number of samples from audio data assuming PCM audio 16-bit LE."""
    header_size = 44 if has_header else 0
    if len(audio_data) <= header_size:
        return 0.0, 0
    num_samples = (len(audio_data) - header_size) // 2
    duration = num_samples / sample_rate
    return duration, num_samples


class HttpTextToSpeechClient(TextToSpeechClient):
    """Base class for HTTP-based TTS clients."""

    def __init__(
        self, server_url: str, model_id: str | None = None, verbose: bool = False
    ):
        self._server_url = server_url
        self._model_id = model_id
        self._verbose = verbose

    def _get_request_url(self, stream: bool) -> str:
        """Get the request URL for the given stream mode. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_request_url")

    def _format_request_data(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> dict:
        """Format request data for the API. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _format_request_data")

    def _get_headers(self) -> dict:
        """Get request headers. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_headers")

    def _extract_audio_from_response(self, response_data: dict) -> bytes | None:
        """Extract audio data from non-streaming response. To be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _extract_audio_from_response"
        )

    def _extract_audio_from_chunk(self, chunk_data: dict) -> bytes | None:
        """Extract audio data from streaming chunk. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _extract_audio_from_chunk")

    async def synthesize_text(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
        stream: bool = False,
    ) -> data_types.GenerationStats:
        """Collects a single request latency statistics for HTTP API calls."""
        if self._verbose:
            click.echo(f"Synthesizing text: {text}")

        # Create a timer to measure the total time taken for the request
        timer = PausableTimer()

        # Pause during request preparation (client-side processing)
        timer.pause()
        request_data = self._format_request_data(speaker_id, text, inference_settings)
        headers = self._get_headers()

        sampling_rate = 24000  # Standard TTS sample rate
        num_samples = 0

        # Use aiohttp for async HTTP requests with increased chunk size limits
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        timer.resume()

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            # Increase max line size and max field size to handle large chunks
            connector_owner=True,
            read_bufsize=1024 * 1024,  # 1MB read buffer
        ) as session:
            if not stream:
                timer.start()
                url = self._get_request_url(stream=False)
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
                    audio_bytes = self._extract_audio_from_response(response_data)

                    if audio_bytes:
                        duration, num_samples = _get_audio_duration_and_samples(
                            audio_bytes, sampling_rate
                        )
                    else:
                        num_samples += int(
                            0.001 * sampling_rate
                        )  # About 24 samples - 1ms of audio

                    if sampling_rate <= 0:
                        raise ValueError(f"Invalid sample rate [{sampling_rate}].")
                    timer.resume()
            else:
                timer.start()
                first_chunk_latency = 0
                avg_chunk_latency = 0
                forth_chunk_latency = 0
                forth_chunk_missing_count = 0
                total_chunks = 0
                chunk_latencies = []
                last_chunk_elapsed = 0

                url = self._get_request_url(stream=True)
                async with session.post(
                    url, json=request_data, headers=headers
                ) as response:
                    response.raise_for_status()

                    i = 0
                    buffer = b""

                    # Process chunks with proper buffering
                    async for chunk in response.content.iter_chunked(
                        8192
                    ):  # 8KB chunks
                        if not chunk:
                            continue

                        timer.pause()  # Pause during client-side buffer processing
                        buffer += chunk

                        # Look for complete lines in the buffer
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line:
                                continue

                            try:
                                # Handle line-by-line streaming responses
                                line_text = line.decode("utf-8").strip()
                                if not line_text:
                                    continue
                                chunk_data = json.loads(line_text)
                                if self._verbose:
                                    click.echo(
                                        f"Received chunk data length: {len(line_text)}"
                                    )
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                # Skip lines that aren't valid JSON
                                click.echo(f"Skipping invalid JSON line: {e}", err=True)
                                continue

                            timer.resume()  # Resume to capture chunk timing
                            chunk_elapsed = timer.elapsed()
                            timer.pause()  # Pause again for processing

                            if i == 0:
                                first_chunk_latency = chunk_elapsed
                            elif i == 3:  # 4th chunk (0-indexed)
                                forth_chunk_latency = chunk_elapsed

                            # Record individual chunk latency (time between chunks)
                            individual_chunk_latency = (
                                chunk_elapsed - last_chunk_elapsed
                            )
                            chunk_latencies.append(individual_chunk_latency)
                            last_chunk_elapsed = chunk_elapsed

                            audio_bytes = self._extract_audio_from_chunk(chunk_data)
                            if audio_bytes:
                                duration, chunk_samples = (
                                    _get_audio_duration_and_samples(
                                        audio_bytes, sampling_rate
                                    )
                                )
                                num_samples += chunk_samples
                                total_chunks += 1
                            else:
                                click.echo("No audio content found.")
                                num_samples += int(
                                    0.001 * sampling_rate
                                )  # About 24 samples - 1ms of audio
                                total_chunks += 1

                            i += 1
                        timer.resume()  # Resume for next chunk

                generation_time = timer.stop()

                # Pause for final calculations
                timer.pause()
                # Calculate forth_chunk_missing_count correctly
                forth_chunk_missing_count = max(0, 4 - total_chunks)
                avg_chunk_latency = sum(chunk_latencies) / max(1, len(chunk_latencies))
                timer.resume()

        # Final calculations while paused
        timer.pause()
        generation_duration_sec = num_samples / max(1, sampling_rate)
        stats = data_types.GenerationStats(
            e2e_latency_ms=generation_time * 1000.0,
            second_of_audio_generation_latency_ms=(
                generation_time * 1000.0 / generation_duration_sec
            ),
        )
        if stream:
            stats.first_chunk_latency_ms = first_chunk_latency * 1000.0
            stats.forth_chunk_latency_ms = forth_chunk_latency * 1000.0
            stats.avg_chunk_latency_ms = avg_chunk_latency * 1000.0
            stats.forth_chunk_missing_count = forth_chunk_missing_count
        return stats


class TtsInternalClient(HttpTextToSpeechClient):
    """Client to talk to the TTS server using Internal API."""

    def _get_request_url(self, stream: bool) -> str:
        """Internal API uses the same URL for both streaming and non-streaming."""
        return self._server_url

    def _format_request_data(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> dict:
        """Format request data for Internal API."""
        request = {
            "text": text,
            "speaker_id": speaker_id,
            "top_k": 50,
            "max_tokens": inference_settings.max_tokens,
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            "top_p": 1.0,
            "seed": 42,
        }
        # Map timestamp_type to forced_alignment_type for internal API
        if inference_settings.timestamp_type:
            ts = inference_settings.timestamp_type.upper()
            mapping = {
                "TIMESTAMP_TYPE_UNSPECIFIED": "none",
                "WORD": "words",
                "CHARACTER": "chars",
            }
            forced_alignment = mapping.get(ts)
            if forced_alignment:
                request["forced_alignment_type"] = forced_alignment
        return request

    def _get_headers(self) -> dict:
        """Get headers for Internal API."""
        headers = {"Content-Type": "application/json"}
        if self._model_id:
            headers["model_id"] = self._model_id
        return headers

    def _extract_audio_from_response(self, response_data: dict) -> bytes | None:
        """Extract audio data from Internal API response."""
        audio_data = response_data.get("data")
        return base64.b64decode(audio_data) if audio_data else None

    def _extract_audio_from_chunk(self, chunk_data: dict) -> bytes | None:
        """Extract audio data from Internal API streaming chunk."""
        audio_data = chunk_data.get("data")
        return base64.b64decode(audio_data) if audio_data else None

    async def synthesize_text_with_metadata(
        self, speaker_id: str, text: str, metadata: tuple
    ):
        raise NotImplementedError(
            "Synthesize text with metadata is not supported for Internal API."
        )


class TtsPublicClient(HttpTextToSpeechClient):
    """Client to talk to the external Inworld TTS API through proxies."""

    def _get_request_url(self, stream: bool) -> str:
        """Get URL for Inworld API based on stream mode."""
        return self._server_url

    def _format_request_data(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> dict:
        """Format request data for Inworld API using new format."""
        request_data = {
            "text": text,
            "voice_id": speaker_id,
            "audio_config": {"audio_encoding": "LINEAR16"},
        }
        if self._model_id:
            request_data["model_id"] = self._model_id
        # Forward timestampType when requested
        if inference_settings.timestamp_type:
            request_data["timestampType"] = inference_settings.timestamp_type.upper()
        return request_data

    def _get_headers(self) -> dict:
        """Get headers for Inworld API."""
        return {"Content-Type": "application/json"}

    def _extract_audio_from_response(self, response_data: dict) -> bytes | None:
        """Extract audio data from Inworld API response."""
        audio_content = response_data.get("audioContent")
        return base64.b64decode(audio_content) if audio_content else None

    def _extract_audio_from_chunk(self, chunk_data: dict) -> bytes | None:
        """Extract audio data from Inworld API streaming chunk."""
        result = chunk_data.get("result")
        if result:
            audio_content = result.get("audioContent")
            return base64.b64decode(audio_content) if audio_content else None
        return None

    async def synthesize_text_with_metadata(
        self, speaker_id: str, text: str, metadata: tuple
    ):
        """Synthesize text with metadata is not supported for Inworld API."""
        raise NotImplementedError(
            "Synthesize text with metadata is not supported for Inworld API."
        )
