"""Text sample provider."""

import json
import random
import uuid
from typing import Set
from enum import Enum

import click

from .clients import ClientType

_TEXT_SAMPLES_FILE = "scripts/tts_load_testing/text_samples.json"
_TEXT_SAMPLES_CACHE = {}  # Changed from None to empty dict
_SEED = 42


class SampleFormat(Enum):
    """Supported sample formats."""

    SIMPLE_JSON = "simple-json"
    AXOLOTL_INPUT_OUTPUT = "axolotl-input-output"
    ID_PROMPT_JSON = "id-prompt-json"


def _load_simple_json_samples(text_samples_file: str) -> Set[str]:
    """Loads text samples from a simple JSON file with 'samples' key."""
    with open(text_samples_file) as f:
        text_samples = json.load(f)
        if "samples" not in text_samples:
            raise KeyError(
                "JSON file must contain a 'samples' key for simple-json format"
            )
        res = set(text_samples["samples"])
    return res


def _load_axolotl_samples(text_samples_file: str) -> Set[str]:
    """
    Loads text samples from an Axolotl-formatted dataset.

    Extracts all text segments where label is false (assistant responses).
    Expected format:
    [
        {
            "segments": [
                {
                    "label": true,
                    "text": "human text..."
                },
                {
                    "label": false,
                    "text": "assistant text..."
                }
            ]
        },
        ...
    ]
    """
    with open(text_samples_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract all text segments where label is false (assistant responses)
    samples = set()
    for conversation in raw_data:
        if "segments" not in conversation:
            continue
        for segment in conversation["segments"]:
            if not segment.get(
                "label", True
            ):  # label=false or missing (default to true)
                text = segment.get("text", "").strip()
                if text:
                    samples.add(text)

    return samples


def _load_id_prompt_json_samples(text_samples_file: str) -> Set[str]:
    """
    Loads text samples from an ID-prompt JSON format.

    Extracts prompt text from objects with 'id' and 'prompt' fields.
    Expected format:
    [
        {
            "id": 1,
            "prompt": "text content..."
        },
        {
            "id": 2,
            "prompt": "more text content..."
        },
        ...
    ]
    """
    with open(text_samples_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract prompt text from each object
    samples = set()
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        prompt_text = item.get("prompt", "").strip()
        if prompt_text:
            samples.add(prompt_text)

    return samples


def _load_text_samples(
    text_samples_file: str, sample_format: SampleFormat = SampleFormat.SIMPLE_JSON
) -> Set[str]:
    """Loads text samples from a file based on the specified format."""
    click.echo(
        f"Loading text samples from [{text_samples_file}] using format [{sample_format.value}]"
    )

    if sample_format == SampleFormat.SIMPLE_JSON:
        res = _load_simple_json_samples(text_samples_file)
    elif sample_format == SampleFormat.AXOLOTL_INPUT_OUTPUT:
        res = _load_axolotl_samples(text_samples_file)
    elif sample_format == SampleFormat.ID_PROMPT_JSON:
        res = _load_id_prompt_json_samples(text_samples_file)
    else:
        raise ValueError(f"Unsupported sample format: {sample_format}")

    click.echo("Loaded {} text samples.".format(len(res)))
    return res


def get_random_text_sample(
    sample_file: str = _TEXT_SAMPLES_FILE,
    sample_format: SampleFormat = SampleFormat.SIMPLE_JSON,
    use_random_prefix: bool = False,
) -> str:
    """Returns a random text sample.

    Args:
        sample_file: Path to the text samples file. Defaults to _TEXT_SAMPLES_FILE.
        sample_format: Format of the sample file. Defaults to SampleFormat.SIMPLE_JSON.
        use_random_prefix: If True, prepend a random prefix to prevent caching.

    Returns:
        A random text sample from the specified file, optionally with random prefix.
    """
    global _TEXT_SAMPLES_CACHE

    # Create cache key from file and format
    cache_key = (sample_file, sample_format)

    # Check if we need to reload the cache
    if cache_key not in _TEXT_SAMPLES_CACHE:
        samples = _load_text_samples(sample_file, sample_format)
        _TEXT_SAMPLES_CACHE[cache_key] = samples
        random.seed(_SEED)

    text_sample = random.choice(list(_TEXT_SAMPLES_CACHE[cache_key]))

    # Add random prefix if requested
    if use_random_prefix:
        random_prefix = generate_random_prefix()
        text_sample = random_prefix + text_sample

    return text_sample


def get_random_speaker_id(client_type: ClientType, voice_ids: list[str]) -> str:
    """Returns a random speaker ID."""
    if client_type == ClientType.EMBEDDING_HTTP:
        # For embedding mode, speaker_id is not used, return empty string
        return ""
    return random.choice(voice_ids)


def calculate_average_prompt_length(
    input_file: str, sample_format: SampleFormat = SampleFormat.SIMPLE_JSON
) -> float:
    """Calculate the average prompt length from a JSON file.

    Args:
        input_file: Path to the JSON file containing text samples.
        sample_format: Format of the sample file. Defaults to SampleFormat.SIMPLE_JSON.

    Returns:
        Average length of all text samples in the file.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        KeyError: If the JSON file doesn't have the expected format.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        samples = _load_text_samples(input_file, sample_format)

        if not samples:
            click.echo("Warning: No samples found in the file")
            return 0.0

        # Calculate character lengths of all samples
        sample_char_lengths = [len(sample) for sample in samples]
        average_char_length = sum(sample_char_lengths) / len(sample_char_lengths)

        # Calculate word counts of all samples
        sample_word_counts = [len(sample.split()) for sample in samples]
        average_word_count = sum(sample_word_counts) / len(sample_word_counts)

        # Calculate total statistics
        total_chars = sum(sample_char_lengths)
        total_words = sum(sample_word_counts)

        click.echo(f"Analyzed {len(samples)} text samples from {input_file}")
        click.echo("\n--- CHARACTER STATISTICS ---")
        click.echo(f"Total characters: {total_chars:,}")
        click.echo(f"Average characters per prompt: {average_char_length:.2f}")
        click.echo(f"Shortest prompt: {min(sample_char_lengths)} characters")
        click.echo(f"Longest prompt: {max(sample_char_lengths)} characters")

        click.echo("\n--- WORD STATISTICS ---")
        click.echo(f"Total words: {total_words:,}")
        click.echo(f"Average words per prompt: {average_word_count:.2f}")
        click.echo(f"Shortest prompt: {min(sample_word_counts)} words")
        click.echo(f"Longest prompt: {max(sample_word_counts)} words")

        return average_char_length

    except FileNotFoundError:
        click.echo(f"Error: File '{input_file}' not found", err=True)
        raise
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in file '{input_file}': {e}", err=True)
        raise
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise


def generate_random_prefix() -> str:
    """
    Generate a random prefix to prevent caching.

    Returns:
        A unique string containing random words and UUID that can be prepended to prompts.
    """
    # List of random words to choose from
    random_words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "theta",
        "lambda",
        "sigma",
        "omega",
        "phoenix",
        "nebula",
        "cosmic",
        "stellar",
        "quantum",
        "matrix",
        "vector",
        "plasma",
        "fusion",
        "crystal",
        "prism",
        "echo",
        "zenith",
        "vertex",
        "flux",
        "spiral",
        "orbit",
        "pulse",
        "wave",
        "surge",
    ]

    # Generate a random combination
    word1 = random.choice(random_words)
    word2 = random.choice(random_words)
    unique_id = uuid.uuid4().hex[:8]

    return f"Random prefix {word1} {word2} {unique_id}: "
