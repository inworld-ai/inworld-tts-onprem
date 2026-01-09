"""Prints load test results."""

import os

import click
from tabulate import tabulate

from . import data_types, results_plot


def print_load_test_results(
    load_test_results: data_types.LoadTestResults,
    stream: bool = False,
    is_llm_mode: bool = False,
    results_file: str | None = None,
):
    log_lines: list[str] = []

    click.echo("\n--------End-to-End Latency Percentiles--------")
    log_lines.append("\n--------End-to-End Latency Percentiles--------")
    _table = tabulate(
        load_test_results.e2e_latency_ms_percentiles.items(),
        headers=["Percentile", "Latency (ms)"],
        tablefmt="fancy_grid",
    )
    print(_table)
    log_lines.append(_table)
    if stream:
        if load_test_results.first_chunk_latency_ms_percentiles:
            click.echo("\n--------First Chunk Latency Percentiles--------")
            log_lines.append("\n--------First Chunk Latency Percentiles--------")
            _table = tabulate(
                load_test_results.first_chunk_latency_ms_percentiles.items(),
                headers=["Percentile", "First Chunk Latency (ms)"],
                tablefmt="fancy_grid",
            )
            print(_table)
            log_lines.append(_table)

        if load_test_results.forth_chunk_latency_ms_percentiles:
            click.echo("\n--------Fourth Chunk Latency Percentiles--------")
            log_lines.append("\n--------Fourth Chunk Latency Percentiles--------")
            _table = tabulate(
                load_test_results.forth_chunk_latency_ms_percentiles.items(),
                headers=["Percentile", "Fourth Chunk Latency (ms)"],
                tablefmt="fancy_grid",
            )
            print(_table)
            log_lines.append(_table)

        if load_test_results.avg_chunk_latency_ms_percentiles:
            click.echo("\n--------Average Chunk Latency Percentiles--------")
            log_lines.append("\n--------Average Chunk Latency Percentiles--------")
            _table = tabulate(
                load_test_results.avg_chunk_latency_ms_percentiles.items(),
                headers=["Percentile", "Avg Chunk Latency (ms)"],
                tablefmt="fancy_grid",
            )
            print(_table)
            log_lines.append(_table)

    # Skip audio-second latency for LLM mode since it doesn't make sense
    if not is_llm_mode:
        click.echo("\n--------Average Latency per audio-second--------")
        log_lines.append("\n--------Average Latency per audio-second--------")
        _table = tabulate(
            load_test_results.second_of_audio_generation_latency_ms_percentiles.items(),
            headers=["Token", "Latency (ms)"],
            tablefmt="fancy_grid",
        )
        print(_table)
        log_lines.append(_table)
    else:
        # Show token statistics for LLM mode
        if hasattr(load_test_results, "token_stats") and load_test_results.token_stats:
            click.echo("\n--------Token Statistics--------")
            log_lines.append("\n--------Token Statistics--------")
            token_data = [
                [
                    "Total Input Tokens",
                    load_test_results.token_stats.get("total_prompt_tokens", 0),
                ],
                [
                    "Total Generated Tokens",
                    load_test_results.token_stats.get("total_completion_tokens", 0),
                ],
                ["Total Tokens", load_test_results.token_stats.get("total_tokens", 0)],
                [
                    "Avg Input Tokens per Request",
                    load_test_results.token_stats.get("avg_prompt_tokens", 0),
                ],
                [
                    "Avg Generated Tokens per Request",
                    load_test_results.token_stats.get("avg_completion_tokens", 0),
                ],
                [
                    "Avg Total Tokens per Request",
                    load_test_results.token_stats.get("avg_total_tokens", 0),
                ],
            ]
            _table = tabulate(
                token_data,
                headers=["Metric", "Value"],
                tablefmt="fancy_grid",
                floatfmt=".2f",
            )
            print(_table)
            log_lines.append(_table)

    click.echo("--------Average QPS--------")
    log_lines.append("--------Average QPS--------")
    click.echo(f"{load_test_results.avg_qps} qps")
    log_lines.append(f"{load_test_results.avg_qps} qps")
    click.echo("-----------------------\n\n")
    log_lines.append("-----------------------")

    # Persist to logs if results_file provided
    if results_file:
        logs_dir = os.path.join("benchmark_result", results_file)
        try:
            os.makedirs(logs_dir, exist_ok=True)
            logs_path = os.path.join(logs_dir, "logs.txt")
            with open(logs_path, "a") as f:
                f.write("\n".join(log_lines) + "\n")
        except Exception:
            # Don't fail the run if logging fails
            pass


def plot_load_test_results(
    results: dict[float, data_types.LoadTestResults],
    results_file: str,
    percentiles: list[int],
    stream: bool = False,
    plot_config: results_plot.PlotConfig = results_plot.PlotConfig(),
):
    """Plot load test results."""
    results_plot.plot_load_test_results(
        results, results_file, percentiles, stream, plot_config
    )
