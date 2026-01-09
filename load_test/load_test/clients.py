"""Common client abstractions and factory for TTS and LLM load testing."""

import enum
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from . import data_types


class ClientType(enum.Enum):
    TTS_INTERNAL = "tts_internal"
    TTS_PUBLIC = "tts_public"
    EMBEDDING_HTTP = "embedding_http"


class TextToSpeechClient(ABC):
    """Abstract base class for TTS clients."""

    @abstractmethod
    async def synthesize_text(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
        stream: bool = False,
    ) -> data_types.GenerationStats:
        """Synthesizes text to speech and returns generation statistics."""
        pass

    @abstractmethod
    async def synthesize_text_with_metadata(
        self, speaker_id: str, text: str, metadata: tuple
    ):
        """Synthesize text with given metadata."""
        pass


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def embed(
        self,
        text: str,
        inference_settings: data_types.InferenceSettings,
    ) -> data_types.EmbeddingStats:
        """Embeds text and returns embedding statistics."""
        pass


# Union type for any client type
AnyClient = Union[TextToSpeechClient, EmbeddingClient]


class ClientFactory:
    """Factory to create a client to talk to the TTS server."""

    def __init__(
        self,
        server_address: str,
        client_type: ClientType,
        model_id: str | None = "finch",
        verbose: bool = False,
    ):
        self._server_address = server_address
        self._client_type = client_type
        self._model_id = model_id
        self._verbose = verbose

    @staticmethod
    def determine_client_config(
        mode: str,
        host: Optional[str],
        stream: bool = False,
    ) -> Tuple[ClientType, str, bool]:
        """
        Determine client type and server address based on mode and host.

        Args:
            mode: Client mode ('tts' or 'embedding')
            host: Base host URL
            stream: Whether streaming is enabled

        Returns:
            Tuple of (client_type, server_address, should_ignore_voice_ids)
        """
        if mode == "embedding":
            client_type = ClientType.EMBEDDING_HTTP
            base_address = host.rstrip("/") if host else ""
            server_address = base_address  # Embedding endpoint will be added in client
            return (
                client_type,
                server_address,
                True,
            )  # Ignore voice_ids in embedding mode
        else:
            # TTS mode - existing logic
            base_address = host.rstrip("/") if host else ""

            if "8081" in base_address:
                client_type = ClientType.TTS_PUBLIC
            else:
                client_type = ClientType.TTS_INTERNAL

            # Automatically append the correct endpoint based on client type and stream parameter
            if client_type == ClientType.TTS_PUBLIC:
                if stream:
                    server_address = f"{base_address}/tts/v1/voice:stream"
                else:
                    server_address = f"{base_address}/tts/v1/voice"
            else:  # TTS_INTERNAL
                if stream:
                    server_address = f"{base_address}/SynthesizeSpeechStream"
                else:
                    server_address = f"{base_address}/SynthesizeSpeech"

            return (
                client_type,
                server_address,
                False,
            )  # Don't ignore voice_ids in TTS mode

    def create(self) -> AnyClient:
        """Creates a client of the specified type."""
        if self._client_type == ClientType.TTS_INTERNAL:
            from .tts_clients import TtsInternalClient

            return TtsInternalClient(
                self._server_address, self._model_id, self._verbose
            )
        elif self._client_type == ClientType.TTS_PUBLIC:
            from .tts_clients import TtsPublicClient

            return TtsPublicClient(self._server_address, self._model_id, self._verbose)
        elif self._client_type == ClientType.EMBEDDING_HTTP:
            from .embedding_clients import EmbeddingServerClient

            return EmbeddingServerClient(
                self._server_address, self._model_id, self._verbose
            )
        else:
            raise ValueError(f"Unsupported client type: {self._client_type}")


def create_factory(
    server_address: str,
    client_type: ClientType,
    model_id: str | None = None,
    verbose: bool = False,
) -> ClientFactory:
    """Creates a client factory for the TTS server."""
    if not server_address:
        raise ValueError("Invalid server address.")
    return ClientFactory(server_address, client_type, model_id, verbose)


def create_factory_from_config(
    mode: str,
    host: Optional[str],
    stream: bool = False,
    model_id: str | None = None,
    verbose: bool = False,
) -> Tuple[ClientFactory, bool]:
    """
    Creates a client factory by determining configuration from mode and host.

            Args:
            mode: Client mode ('tts' or 'embedding')
            host: Base host URL
            stream: Whether streaming is enabled
            model_id: Model ID to use
            verbose: Enable verbose logging

    Returns:
        Tuple of (ClientFactory, should_ignore_voice_ids)
    """
    client_type, server_address, should_ignore_voice_ids = (
        ClientFactory.determine_client_config(mode, host, stream)
    )

    factory = create_factory(
        server_address=server_address,
        client_type=client_type,
        model_id=model_id,
        verbose=verbose,
    )

    return factory, should_ignore_voice_ids
