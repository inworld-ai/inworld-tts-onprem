r"""TTS load tester.

    ./scripts/tts-load-test --host http://localhost:8081 --stream \
        --min-qps 500.0 --max-qps 500.0 --qps-step 10.0 --burstiness 1.0 --number_of_samples 3000 \
        --text_samples_file tests-data/tts-load-test/text_samples.json \
        --voice-ids Olivia --benchmark_name turbo-torch-prod

    High QPS:
    ./scripts/tts-load-test --host http://localhost:8081 --stream \
        --min-qps 300.0 --max-qps 300.0 --qps-step 10.0 --burstiness 1.0 --number_of_samples 3000 \
        --text_samples_file tests-data/tts-load-test/text_samples.json \
        --voice-ids Olivia --benchmark_name turbo-torch-prod

Plot only mode (load existing results):
    ./scripts/tts-load-test --plot_only --benchmark_name turbo-torch-prod-non-stream

Analyze prompt lengths:
    ./scripts/tts-load-test --analyze_prompts --text_samples_file tests-data/tts-load-test/text_samples.json

Using ID-prompt JSON format:
    ./scripts/tts-load-test --host http://localhost:8081 \
        --text_samples_file tests-data/tts-load-test/text_embeding_sample.json \
        --sample-format id-prompt-json --benchmark_name id-prompt-test

Using embedding mode:
    ./scripts/tts-load-test --host http://localhost:8081 --mode embedding \
        --text_samples_file tests-data/tts-load-test/text_embeding_sample.json \
        --sample-format id-prompt-json --benchmark_name embedding-test

Available options:
    --host TEXT       Base address of the TTS server (endpoint will be auto-appended) [required]
    --number_of_samples INTEGER  Total number of text to synthesize [default: 1]
    --min-qps FLOAT                 Minimum requests per second [default: 1.0]
    --max-qps FLOAT                 Maximum requests per second [default: 10.0]
    --qps-step FLOAT                Step for QPS increments [default: 2.0]
    --burstiness FLOAT              Burstiness factor for request generation [default: 1.0]
    --benchmark_name TEXT           Name of the benchmark run. If not provided, a UUID will be generated
    --plot_only                     If set, only plot results from existing result file
    --stream                        If set, use streaming synthesis (/SynthesizeSpeechStream), otherwise use non-streaming (/SynthesizeSpeech)
    --text_samples_file TEXT        File to read the text samples from [default: scripts/tts_load_testing/text_samples.json]
    --max_tokens INTEGER            Maximum number of tokens to synthesize [default: 400]
    --model_id TEXT                 Model ID to use for TTS synthesis
    --mode TEXT                     Client mode: 'tts' for TTS synthesis or 'embedding' for text embeddings.
    --voice-ids TEXT                Voice IDs to use for TTS synthesis [default: Olivia, Remy]
    --sample-format TEXT            Format of text samples file [default: simple-json] (simple-json|axolotl-input-output|id-prompt-json)
    --analyze_prompts               Analyze prompt lengths in text samples file and exit
    --help                          Show this message and exit

Sample formats:
    simple-json:           Standard format with {"samples": ["text1", "text2", ...]}
    axolotl-input-output:  Axolotl dataset format extracting assistant responses from conversation segments
    id-prompt-json:        Format with [{"id": 1, "prompt": "text..."}, {"id": 2, "prompt": "text..."}, ...]

Note: The appropriate endpoint (/SynthesizeSpeech or /SynthesizeSpeechStream) will be automatically
appended to the server address based on the --stream flag.

"""

import asyncio
import json
import os
import uuid
from typing import Optional, AsyncGenerator
from collections.abc import Iterable

import click
import numpy as np
from tqdm.asyncio import tqdm

from tts_load_test import (
    clients,
    content,
    data_types,
    results_plot,
    results_printer,
)
from tts_load_test.timer import PausableTimer

# Default text samples file (bundled with the package)
_DEFAULT_TEXT_SAMPLES_FILE = os.path.join(
    os.path.dirname(__file__), "text_samples_single.json"
)

# Percentiles to measure the latency on.
_PERCENTILES = [50, 90, 95, 99]


def compute_percentiles(latencies: list[float]) -> dict[int, float]:
    """Computes percentiles for the given latencies."""
    if not latencies:
        return {pct: 0.0 for pct in _PERCENTILES}

    res = {}
    for pct in _PERCENTILES:
        res[pct] = np.percentile(latencies, pct)
    return res


async def get_request_with_timing(
    input_requests: list[tuple],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[tuple, None]:
    """
    Asynchronously generates requests at a specified rate with burstiness.

    Args:
        input_requests: A list of input request tuples
        request_rate: The rate at which requests are generated (requests/s)
        burstiness: The burstiness factor of the request generation.
            1.0 follows a Poisson process, <1.0 more bursty, >1.0 more uniform.
    """
    request_iterator: Iterable[tuple] = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate
    assert (
        burstiness > 0
    ), f"A positive burstiness factor is expected, but given {burstiness}."
    theta = 1.0 / (request_rate * burstiness)

    for request in request_iterator:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait
            continue

        # Sample the request interval from the gamma distribution
        # If burstiness is 1, it follows exponential distribution
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval
        await asyncio.sleep(interval)


class UserSimulator:
    """Simulates a conversation with the server."""

    def __init__(self, client: clients.AnyClient, client_type: clients.ClientType):
        self._client = client
        self._client_type = client_type

    async def run_simulation(
        self,
        speaker_id: str,
        text: str,
        inference_settings: data_types.InferenceSettings,
        stream: bool = False,
    ) -> (
        data_types.GenerationStats
        | data_types.LlmCompletionStats
        | data_types.EmbeddingStats
        | None
    ):
        stats = None

        try:
            if self._client_type == clients.ClientType.EMBEDDING_HTTP:
                # Embedding client uses embed() method
                from .clients import EmbeddingClient

                if isinstance(self._client, EmbeddingClient):
                    stats = await self._client.embed(
                        text=text,
                        inference_settings=inference_settings,
                    )
            else:
                # TTS clients use synthesize_text() method
                from .clients import TextToSpeechClient

                if isinstance(self._client, TextToSpeechClient):
                    stats = await self._client.synthesize_text(
                        speaker_id=speaker_id,
                        text=text,
                        inference_settings=inference_settings,
                        stream=stream,
                    )
        except Exception as e:
            click.echo(f"Couldn't sustain load: [{str(e)}]", err=True)

        return stats


class QpsLoadTester:
    """Load test that studies backend performance under specified QPS load."""

    def __init__(
        self,
        client_factory: clients.ClientFactory,
        load_testing_settings: data_types.LoadTestingSettings,
        client_type: clients.ClientType,
        sample_format: content.SampleFormat,
        use_random_prefix: bool = False,
        timestamp_type: Optional[str] = None,
    ):
        self._client_factory = client_factory
        self._load_testing_settings = load_testing_settings
        self._client_type = client_type
        self._sample_format = sample_format
        self._use_random_prefix = use_random_prefix
        self._timestamp_type = timestamp_type
        click.echo(f"Load testing settings: [{self._load_testing_settings}].")
        if use_random_prefix:
            click.echo("Random prefix mode enabled - caching will be prevented.")

    def _get_random_inference_settings(self) -> data_types.InferenceSettings:
        return data_types.InferenceSettings(
            max_generation_duration_sec=0,
            max_tokens=self._load_testing_settings.max_tokens,
            timestamp_type=self._timestamp_type,
        )

    async def _execute_request_async(
        self, request_params: tuple
    ) -> data_types.GenerationStats | data_types.LlmCompletionStats | None:
        """Execute a single request asynchronously."""
        speaker_id, text, inference_settings, stream = request_params

        # Create a new client for this request
        client = self._client_factory.create()
        user_simulator = UserSimulator(client=client, client_type=self._client_type)

        return await user_simulator.run_simulation(
            speaker_id=speaker_id,
            text=text,
            inference_settings=inference_settings,
            stream=stream,
        )

    async def run(
        self,
        request_rate: float,
        burstiness: float,
        text_samples_file: str,
        stream: bool,
        voice_ids: list[str],
    ) -> tuple[data_types.LoadTestResults, float]:
        """Runs the load test with specified QPS."""
        n_texts = self._load_testing_settings.number_of_samples

        # Prepare all request parameters upfront
        request_params_list = []
        for _ in range(n_texts):
            speaker_id = content.get_random_speaker_id(self._client_type, voice_ids)
            text = content.get_random_text_sample(
                sample_file=text_samples_file,
                sample_format=self._sample_format,
                use_random_prefix=self._use_random_prefix,
            )
            inference_settings = self._get_random_inference_settings()
            request_params_list.append((speaker_id, text, inference_settings, stream))

        click.echo(
            f"Starting QPS load test with rate: {request_rate} req/s, burstiness: {burstiness}"
        )

        # Create progress bar
        pbar = tqdm(total=n_texts, desc="Sending requests", unit="req")

        # Pure asyncio approach - no threads needed
        tasks = []
        timer = PausableTimer()
        timer.start()

        async for request_params in get_request_with_timing(
            request_params_list, request_rate, burstiness
        ):
            # Create and schedule async task for each request
            task = asyncio.create_task(self._execute_request_async(request_params))
            tasks.append(task)

            # Update progress bar when request is sent (not completed)
            pbar.update(1)

        # Close the sending progress bar and show completion status
        pbar.close()
        click.echo(f"All {len(tasks)} requests sent! Waiting for responses...")

        # Pause timer during result processing (client-side work)
        timer.pause()
        # Wait for all requests to complete using asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert them to None results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                click.echo(f"Request failed with exception: {result}", err=True)
                processed_results.append(None)
            else:
                processed_results.append(result)
        results = processed_results
        timer.resume()

        load_test_time = timer.stop()

        # Pause timer during results processing (client-side work)
        timer.pause()
        # Process results
        e2e_latencies = []
        second_of_audio_generation_latencies = []
        first_chunk_latencies = []
        forth_chunk_latencies = []
        avg_chunk_latencies = []
        forth_chunk_missing_counts = []
        total_number_of_requests = 0

        # Token statistics for LLM requests
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        llm_request_count = 0

        for stats in results:
            if stats:
                total_number_of_requests += 1
                e2e_latencies.append(stats.e2e_latency_ms)
                second_of_audio_generation_latencies.append(
                    stats.second_of_audio_generation_latency_ms
                )
                if stream:
                    first_chunk_latencies.append(stats.first_chunk_latency_ms)
                    forth_chunk_latencies.append(stats.forth_chunk_latency_ms)
                    avg_chunk_latencies.append(stats.avg_chunk_latency_ms)
                    forth_chunk_missing_counts.append(stats.forth_chunk_missing_count)

                # Collect token statistics for LLM requests
                if isinstance(stats, data_types.LlmCompletionStats):
                    llm_request_count += 1
                    if stats.prompt_tokens:
                        total_prompt_tokens += stats.prompt_tokens
                    if stats.completion_tokens:
                        total_completion_tokens += stats.completion_tokens
                    if stats.total_tokens:
                        total_tokens += stats.total_tokens

        click.echo(f"Completed [{total_number_of_requests}] synthesis requests.")

        # Calculate actual QPS achieved
        actual_qps = total_number_of_requests / load_test_time

        # Prepare token statistics for LLM requests
        token_stats = None
        if llm_request_count > 0:
            token_stats = {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_prompt_tokens": total_prompt_tokens / llm_request_count
                if llm_request_count > 0
                else 0,
                "avg_completion_tokens": total_completion_tokens / llm_request_count
                if llm_request_count > 0
                else 0,
                "avg_total_tokens": total_tokens / llm_request_count
                if llm_request_count > 0
                else 0,
            }

        load_test_results = data_types.LoadTestResults(
            avg_qps=actual_qps,
            e2e_latency_ms_percentiles=compute_percentiles(e2e_latencies),
            second_of_audio_generation_latency_ms_percentiles=compute_percentiles(
                second_of_audio_generation_latencies
            ),
            first_chunk_latency_ms_percentiles=compute_percentiles(
                first_chunk_latencies
            )
            if stream
            else None,
            forth_chunk_latency_ms_percentiles=compute_percentiles(
                forth_chunk_latencies
            )
            if stream
            else None,
            avg_chunk_latency_ms_percentiles=compute_percentiles(avg_chunk_latencies)
            if stream
            else None,
            forth_chunk_missing_count=sum(forth_chunk_missing_counts),
            token_stats=token_stats,
        )
        return load_test_results, load_test_time


async def collect_results_for_qps(
    client_factory: clients.ClientFactory,
    load_testing_settings: data_types.LoadTestingSettings,
    request_rate: float,
    burstiness: float,
    client_type: clients.ClientType,
    text_samples_file: str,
    stream: bool,
    voice_ids: list[str],
    sample_format: content.SampleFormat,
    results_file: str,
    use_random_prefix: bool = False,
    timestamp: Optional[str] = None,
) -> data_types.LoadTestResults:
    click.echo(
        f"Starting QPS load test for [{request_rate}] req/s with burstiness [{burstiness}]..."
    )
    load_test_results, load_test_time = await QpsLoadTester(
        client_factory=client_factory,
        load_testing_settings=load_testing_settings,
        client_type=client_type,
        sample_format=sample_format,
        use_random_prefix=use_random_prefix,
        timestamp_type=(timestamp.upper() if timestamp else None),
    ).run(
        request_rate=request_rate,
        burstiness=burstiness,
        text_samples_file=text_samples_file,
        stream=stream,
        voice_ids=voice_ids,
    )
    click.echo(f"Load test took [{load_test_time}] seconds.")
    click.echo(
        f"Target QPS: {request_rate}, Actual QPS: {load_test_results.avg_qps:.2f}"
    )
    results_printer.print_load_test_results(
        load_test_results,
        stream=stream,
        is_llm_mode=False,
        results_file=results_file,
    )
    return load_test_results


@click.command()
@click.option(
    "--host",
    required=False,
    help="Base address of the TTS server (endpoint will be auto-appended based on --stream flag).",
)
@click.option(
    "--number-of-samples",
    default=100,
    help="Total number of text to synthesize.",
)
@click.option("--min-qps", default=1.0, help="Minimum requests per second.")
@click.option("--max-qps", default=10.0, help="Maximum requests per second.")
@click.option("--qps-step", default=1.0, help="Step for QPS increments.")
@click.option(
    "--burstiness",
    default=1.0,
    help="Burstiness factor for request generation. "
    "1.0 follows Poisson process, <1.0 more bursty, >1.0 more uniform.",
)
@click.option(
    "--benchmark-name",
    default=None,
    help="Name of the benchmark run. If not provided, a UUID will be generated.",
)
@click.option(
    "--plot-only",
    is_flag=True,
    help="If True, only plot results from existing result file.",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Use streaming synthesis (/SynthesizeSpeechStream), otherwise use non-streaming (/SynthesizeSpeech).",
)
@click.option(
    "--text-samples-file",
    default=_DEFAULT_TEXT_SAMPLES_FILE,
    help="File to read the text samples from.",
)
@click.option(
    "--max-tokens",
    default=400,
    help="Maximum number of tokens to synthesize, 400 token is 8s audio by default 50 tokens/s rate.",
)
@click.option(
    "--model-id",
    default="inworld-tts-1",
    help="Model ID to use for TTS synthesis. If not provided, we will not set model_id in the request.",
)
@click.option(
    "--mode",
    type=click.Choice(["tts", "embedding"]),
    default="tts",
    help="Client mode: 'tts' for TTS synthesis or 'embedding' for text embeddings.",
)
@click.option(
    "--voice-ids",
    default=["Alex"],
    multiple=True,
    help="Voice IDs to use for TTS synthesis. Multiple values can be provided. Ignored in LLM and embedding modes.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="If True, print verbose output.",
)
@click.option(
    "--analyze-prompts",
    is_flag=True,
    help="If True, analyze the prompt lengths in the text samples file and exit.",
)
@click.option(
    "--sample-format",
    type=click.Choice(["simple-json", "axolotl-input-output", "id-prompt-json"]),
    default="simple-json",
    help='Format of the text samples file. \'simple-json\' expects {"samples": [...]}, \'axolotl-input-output\' expects Axolotl dataset format with segments, \'id-prompt-json\' expects [{"id": 1, "prompt": "text..."}, ...].',
)
@click.option(
    "--random",
    is_flag=True,
    help="If set, prepend random words to each prompt to prevent caching.",
)
@click.option(
    "--sample",
    is_flag=True,
    help="If set, send a single request and print the redacted response.",
)
@click.option(
    "--timestamp",
    type=click.Choice(
        ["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"], case_sensitive=False
    ),
    default=None,
    help="Controls timestamp metadata. Public API: sends timestampType. Internal: maps to forced_alignment_type.",
)
def main(
    host: Optional[str],
    number_of_samples: int,
    min_qps: float,
    max_qps: float,
    qps_step: float,
    burstiness: float,
    benchmark_name: Optional[str],
    plot_only: bool,
    stream: bool,
    text_samples_file: str,
    max_tokens: int,
    model_id: Optional[str],
    mode: str,
    voice_ids: tuple[str, ...],
    verbose: bool,
    analyze_prompts: bool,
    sample_format: str,
    random: bool,
    sample: bool,
    timestamp: Optional[str],
):
    """TTS load tester main function."""

    # Import SampleFormat enum to convert string parameter to enum
    from .content import SampleFormat

    sample_format_enum = SampleFormat(sample_format)

    # If analyze_prompts is True, analyze the text samples and exit
    if analyze_prompts:
        try:
            content.calculate_average_prompt_length(
                text_samples_file, sample_format_enum
            )
            return
        except Exception as e:
            click.echo(f"Failed to analyze prompts: {e}", err=True)
            return

    # If no host is provided, print the module docstring and exit
    if not host and not plot_only:
        click.echo(__doc__)
        return

    # Log all parameter values.
    click.echo("Load test starts with:")
    click.echo(f"  host: {host}")
    click.echo(f"  number_of_samples: {number_of_samples}")
    click.echo(f"  min_qps: {min_qps}")
    click.echo(f"  max_qps: {max_qps}")
    click.echo(f"  qps_step: {qps_step}")
    click.echo(f"  burstiness: {burstiness}")
    click.echo(f"  benchmark_name: {benchmark_name}")
    click.echo(f"  plot_only: {plot_only}")
    click.echo(f"  stream: {stream}")
    click.echo(f"  text_samples_file: {text_samples_file}")
    click.echo(f"  max_tokens: {max_tokens}")
    click.echo(f"  model_id: {model_id}")
    click.echo(f"  mode: {mode}")
    click.echo(f"  voice_ids: {list(voice_ids)}")
    click.echo(f"  verbose: {verbose}")
    click.echo(f"  sample_format: {sample_format}")
    click.echo(f"  random: {random}")
    click.echo(f"  sample: {sample}")
    click.echo(f"  timestamp: {timestamp}")
    # Generate or use benchmark name
    if not benchmark_name:
        benchmark_name = f"bench_{uuid.uuid4().hex[:8]}"

    # Set up result folder
    result_folder = f"benchmark_result/{benchmark_name}"
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, "result.json")

    if not plot_only:
        # Use the factory to determine client configuration
        client_factory, should_ignore_voice_ids = clients.create_factory_from_config(
            mode=mode,
            host=host,
            stream=stream,
            model_id=model_id,
            verbose=verbose,
        )

        # Get client type for downstream usage
        client_type = client_factory._client_type
        server_address = client_factory._server_address

        # Display mode information
        if mode == "embedding":
            click.echo(f"Using embedding mode with endpoint: {server_address}")
        else:
            click.echo(f"Using TTS mode with endpoint: {server_address}")

        # Handle voice_ids for LLM and embedding modes
        if should_ignore_voice_ids:
            voice_ids = ()  # Clear voice_ids for non-TTS modes, keeping tuple type

        # If sample mode is requested, perform a single request and print redacted response, then exit
        if sample:
            if mode != "tts":
                click.echo("--sample is only supported in TTS mode.")
                return

            from .sample_runner import run_sample_request

            asyncio.run(
                run_sample_request(
                    client_factory=client_factory,
                    should_ignore_voice_ids=should_ignore_voice_ids,
                    voice_ids=list(voice_ids),
                    text_samples_file=text_samples_file,
                    sample_format_enum=sample_format_enum,
                    use_random_prefix=random,
                    stream=stream,
                    max_tokens=max_tokens,
                    timestamp=timestamp,
                )
            )
            return

        default_settings = data_types.LoadTestingSettings(
            number_of_samples=number_of_samples,
            max_tokens=max_tokens,
        )

        results = {}

        # Generate QPS values to test
        qps_values = []
        current_qps = min_qps
        while current_qps <= max_qps:
            qps_values.append(current_qps)
            current_qps += qps_step

        # Make this function async and use asyncio.run() to run the QPS tests
        async def run_qps_tests():
            qps_results = {}
            for qps in qps_values:
                load_test_results = await collect_results_for_qps(
                    client_factory=client_factory,
                    load_testing_settings=default_settings,
                    request_rate=qps,
                    burstiness=burstiness,
                    client_type=client_type,
                    text_samples_file=text_samples_file,
                    stream=stream,
                    voice_ids=list(voice_ids),
                    sample_format=sample_format_enum,
                    results_file=benchmark_name,
                    use_random_prefix=random,
                    timestamp=timestamp,
                )
                qps_results[qps] = load_test_results
            return qps_results

        results = asyncio.run(run_qps_tests())

        click.echo(f"Saving results to [{result_file}]...")
        with open(result_file, "w") as f:
            # Convert results dict to serializable format
            serializable_results = {str(k): v.model_dump() for k, v in results.items()}
            json.dump(serializable_results, f)
    else:
        click.echo(f"Loading results from [{result_file}]...")
        with open(result_file, "r") as f:
            loaded_data = json.load(f)
            # Convert back to LoadTestResults objects
            results = {
                float(k): data_types.LoadTestResults.model_validate(v)
                for k, v in loaded_data.items()
            }

    # Create plot config based on stream mode
    plot_config = results_plot.PlotConfig()
    if not stream:
        # Remove streaming-specific metrics when not in stream mode
        plot_config.metrics_to_plot = {
            "e2e_latency_ms_percentiles",
            "second_of_audio_generation_latency_ms_percentiles",
        }

    results_printer.plot_load_test_results(
        results,
        results_file=benchmark_name,
        percentiles=_PERCENTILES,
        stream=stream,
        plot_config=plot_config,
    )


if __name__ == "__main__":
    main()
