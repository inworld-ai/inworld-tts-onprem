"""Sample request runner for TTS clients with redacted output printing."""

import json
from typing import Optional, Sequence

import aiohttp
import click

from . import content, data_types


def _redact(obj):
    if isinstance(obj, dict):
        red = {}
        for k, v in obj.items():
            if k in ("audioContent", "data") and isinstance(v, (str, bytes)):
                red[k] = f"[redacted audio] len={len(v)}"
            else:
                red[k] = _redact(v)
        return red
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


async def run_sample_request(
    client_factory,
    should_ignore_voice_ids: bool,
    voice_ids: Sequence[str],
    text_samples_file: str,
    sample_format_enum,
    use_random_prefix: bool,
    stream: bool,
    max_tokens: int,
    timestamp: Optional[str],
):
    """Run a single TTS request and print a redacted response summary.

    Supports both HTTP (public/internal) and gRPC clients.
    """
    speaker_id = (
        content.get_random_speaker_id(client_factory._client_type, list(voice_ids))
        if not should_ignore_voice_ids
        else ""
    )
    text = content.get_random_text_sample(
        sample_file=text_samples_file,
        sample_format=sample_format_enum,
        use_random_prefix=use_random_prefix,
    )

    inference_settings = data_types.InferenceSettings(
        max_generation_duration_sec=0,
        max_tokens=max_tokens,
        timestamp_type=(timestamp.upper() if timestamp else None),
    )

    client = client_factory.create()

    # HTTP-based clients: have formatting and header helpers
    if hasattr(client, "_format_request_data") and hasattr(client, "_get_request_url"):
        req = client._format_request_data(speaker_id, text, inference_settings)
        headers = client._get_headers()
        url = client._get_request_url(stream=stream)
        async with aiohttp.ClientSession() as session:
            if not stream:
                async with session.post(url, json=req, headers=headers) as resp:
                    resp.raise_for_status()
                    body = await resp.json()
                    click.echo("Sample response (redacted):")
                    click.echo(json.dumps(_redact(body), indent=2))
                    # If timestamp requested, try to print alignment resource explicitly
                    if inference_settings.timestamp_type:
                        if isinstance(body, dict) and "timestampInfo" in body:
                            click.echo("Alignment (timestampInfo):")
                            click.echo(json.dumps(body.get("timestampInfo"), indent=2))
                        elif isinstance(body, dict) and "alignment" in body:
                            click.echo("Alignment (alignment):")
                            click.echo(json.dumps(body.get("alignment"), indent=2))
            else:
                click.echo("Streaming sample response (first 5 chunks, redacted):")
                async with session.post(url, json=req, headers=headers) as resp:
                    resp.raise_for_status()
                    i = 0
                    alignment_printed = False
                    async for chunk in resp.content.iter_chunked(8192):
                        if not chunk:
                            continue
                        for line in chunk.split(b"\n"):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line.decode("utf-8"))
                            except Exception:
                                continue
                            click.echo(json.dumps(_redact(item)))
                            if (
                                inference_settings.timestamp_type
                                and not alignment_printed
                            ):
                                if isinstance(item, dict) and "timestampInfo" in item:
                                    click.echo("Alignment (timestampInfo):")
                                    click.echo(
                                        json.dumps(item.get("timestampInfo"), indent=2)
                                    )
                                    alignment_printed = True
                                elif isinstance(item, dict) and "alignment" in item:
                                    click.echo("Alignment (alignment):")
                                    click.echo(
                                        json.dumps(item.get("alignment"), indent=2)
                                    )
                                    alignment_printed = True
                            i += 1
                            if i >= 5:
                                return
        return
