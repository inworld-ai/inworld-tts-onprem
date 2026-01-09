"""Plotting functionality for load test results."""

import dataclasses
from typing import Set

from matplotlib import cm, pyplot

from . import data_types

# Plot labels and tooltips
PLOT_LABELS = {
    "e2e_latency_ms_percentiles": "E2E latency, ms",
    "second_of_audio_generation_latency_ms_percentiles": "Latency to generate audio-second, ms",
    "first_chunk_latency_ms_percentiles": "First Chunk Latency(100 tokens), ms",
    "forth_chunk_latency_ms_percentiles": "Fourth Chunk Latency(400 tokens), ms",
    "avg_chunk_latency_ms_percentiles": "Avg Chunk Latency, ms",
    "avg_qps": "Avg QPS",
}


@dataclasses.dataclass
class PlotConfig:
    """Configuration for which metrics to plot."""

    metrics_to_plot: Set[str] = dataclasses.field(
        default_factory=lambda: {
            "e2e_latency_ms_percentiles",
            "first_chunk_latency_ms_percentiles",
            # "forth_chunk_latency_ms_percentiles",
        }
    )


def plot_load_test_results(
    results: dict[float, data_types.LoadTestResults],
    results_file: str,
    percentiles: list[int],
    stream: bool = False,
    plot_config: PlotConfig = PlotConfig(),
):
    """Plot load test results."""

    def _plot_2x2_subplots():
        _, axs = pyplot.subplots(nrows=2, ncols=2, figsize=(30, 20), dpi=100)
        axs = axs.flatten()  # Flatten the array for easy indexing
        color_map = cm.get_cmap(
            "rainbow", len(plot_config.metrics_to_plot) + 1
        )  # +1 for QPS
        x_axis = list(results.keys())

        for p_idx, fixed_percentile in enumerate(percentiles):
            ax1 = axs[p_idx]
            color_idx = 0

            if "e2e_latency_ms_percentiles" in plot_config.metrics_to_plot:
                ax1.plot(
                    x_axis,
                    [
                        results[i].e2e_latency_ms_percentiles[fixed_percentile]
                        for i in x_axis
                    ],
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["e2e_latency_ms_percentiles"],
                )
                color_idx += 1

            if (
                "second_of_audio_generation_latency_ms_percentiles"
                in plot_config.metrics_to_plot
            ):
                ax1.plot(
                    x_axis,
                    [
                        results[i].second_of_audio_generation_latency_ms_percentiles[
                            fixed_percentile
                        ]
                        for i in x_axis
                    ],
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS[
                        "second_of_audio_generation_latency_ms_percentiles"
                    ],
                )
                color_idx += 1

            if stream:
                if "first_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                    ax1.plot(
                        x_axis,
                        [
                            results[i].first_chunk_latency_ms_percentiles[
                                fixed_percentile
                            ]
                            for i in x_axis
                        ],
                        marker="o",
                        linestyle="-",
                        color=color_map(color_idx),
                        label=PLOT_LABELS["first_chunk_latency_ms_percentiles"],
                    )
                    color_idx += 1

                if "forth_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                    ax1.plot(
                        x_axis,
                        [
                            results[i].forth_chunk_latency_ms_percentiles[
                                fixed_percentile
                            ]
                            for i in x_axis
                        ],
                        marker="o",
                        linestyle="-",
                        color=color_map(color_idx),
                        label=PLOT_LABELS["forth_chunk_latency_ms_percentiles"],
                    )
                    color_idx += 1

                if "avg_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                    ax1.plot(
                        x_axis,
                        [
                            results[i].avg_chunk_latency_ms_percentiles[
                                fixed_percentile
                            ]
                            for i in x_axis
                        ],
                        marker="o",
                        linestyle="-",
                        color=color_map(color_idx),
                        label=PLOT_LABELS["avg_chunk_latency_ms_percentiles"],
                    )
                    color_idx += 1

            # Plot avg_qps on the second y-axis.
            ax2 = ax1.twinx()
            ax2.plot(
                x_axis,
                [results[i].avg_qps for i in x_axis],
                marker="o",
                linestyle="-",
                color=color_map(color_idx),
                label=PLOT_LABELS["avg_qps"],
            )

            ax1.set_xlabel("Request Rate (QPS)")
            ax1.set_ylabel("Latency (ms)")
            ax2.set_ylabel("Actual QPS")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            ax1.set_title(
                "Load Test Results Metrics ({} percentile)".format(fixed_percentile)
            )
            ax1.grid()

        pyplot.tight_layout()
        pyplot.savefig(f"benchmark_result/{results_file}/plot_2x2.png")

    def _plot_50perc_plot():
        _, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=200)
        color_map = cm.get_cmap(
            "rainbow", len(plot_config.metrics_to_plot) + 1
        )  # +1 for QPS
        x_axis = list(results.keys())
        fixed_percentile = 50
        color_idx = 0

        if "e2e_latency_ms_percentiles" in plot_config.metrics_to_plot:
            ax.plot(
                x_axis,
                [
                    results[i].e2e_latency_ms_percentiles[fixed_percentile]
                    for i in x_axis
                ],
                marker="o",
                linestyle="-",
                color=color_map(color_idx),
                label=PLOT_LABELS["e2e_latency_ms_percentiles"],
            )
            color_idx += 1

        if (
            "second_of_audio_generation_latency_ms_percentiles"
            in plot_config.metrics_to_plot
        ):
            ax.plot(
                x_axis,
                [
                    results[i].second_of_audio_generation_latency_ms_percentiles[
                        fixed_percentile
                    ]
                    for i in x_axis
                ],
                marker="o",
                linestyle="-",
                color=color_map(color_idx),
                label=PLOT_LABELS["second_of_audio_generation_latency_ms_percentiles"],
            )
            color_idx += 1

        if stream:
            if "first_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                ax.plot(
                    x_axis,
                    [
                        results[i].first_chunk_latency_ms_percentiles[fixed_percentile]
                        for i in x_axis
                    ],
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["first_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

            if "forth_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                ax.plot(
                    x_axis,
                    [
                        results[i].forth_chunk_latency_ms_percentiles[fixed_percentile]
                        for i in x_axis
                    ],
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["forth_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

            if "avg_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                ax.plot(
                    x_axis,
                    [
                        results[i].avg_chunk_latency_ms_percentiles[fixed_percentile]
                        for i in x_axis
                    ],
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["avg_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

        # Plot avg_qps on the second y-axis.
        ax2 = ax.twinx()
        ax2.plot(
            x_axis,
            [results[i].avg_qps for i in x_axis],
            marker="o",
            linestyle="-",
            color=color_map(color_idx),
            label=PLOT_LABELS["avg_qps"],
        )

        ax.set_xlabel("Request Rate (QPS)")
        ax.set_ylabel("Latency (ms)")
        ax2.set_ylabel("Actual QPS")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.set_title(
            "Load Test Results Metrics ({} percentile)".format(fixed_percentile)
        )
        ax.grid()

        pyplot.tight_layout()
        pyplot.savefig(f"benchmark_result/{results_file}/plot_p50.png")

    def _plot_qps_vs_latency():
        """Plot QPS vs latency metrics for p50."""
        _, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=200)
        color_map = cm.get_cmap("rainbow", len(plot_config.metrics_to_plot))
        x_axis = [results[i].avg_qps for i in sorted(results.keys())]
        fixed_percentile = 50
        color_idx = 0

        if "e2e_latency_ms_percentiles" in plot_config.metrics_to_plot:
            y_axis = [
                results[i].e2e_latency_ms_percentiles[fixed_percentile]
                for i in sorted(results.keys())
            ]
            ax.plot(
                x_axis,
                y_axis,
                marker="o",
                linestyle="-",
                color=color_map(color_idx),
                label=PLOT_LABELS["e2e_latency_ms_percentiles"],
            )
            color_idx += 1

        if (
            "second_of_audio_generation_latency_ms_percentiles"
            in plot_config.metrics_to_plot
        ):
            y_axis = [
                results[i].second_of_audio_generation_latency_ms_percentiles[
                    fixed_percentile
                ]
                for i in sorted(results.keys())
            ]
            ax.plot(
                x_axis,
                y_axis,
                marker="o",
                linestyle="-",
                color=color_map(color_idx),
                label=PLOT_LABELS["second_of_audio_generation_latency_ms_percentiles"],
            )
            color_idx += 1

        if stream:
            if "first_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                y_axis = [
                    results[i].first_chunk_latency_ms_percentiles[fixed_percentile]
                    for i in sorted(results.keys())
                ]
                ax.plot(
                    x_axis,
                    y_axis,
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["first_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

            if "forth_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                y_axis = [
                    results[i].forth_chunk_latency_ms_percentiles[fixed_percentile]
                    for i in sorted(results.keys())
                ]
                ax.plot(
                    x_axis,
                    y_axis,
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["forth_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

            if "avg_chunk_latency_ms_percentiles" in plot_config.metrics_to_plot:
                y_axis = [
                    results[i].avg_chunk_latency_ms_percentiles[fixed_percentile]
                    for i in sorted(results.keys())
                ]
                ax.plot(
                    x_axis,
                    y_axis,
                    marker="o",
                    linestyle="-",
                    color=color_map(color_idx),
                    label=PLOT_LABELS["avg_chunk_latency_ms_percentiles"],
                )
                color_idx += 1

        ax.set_xlabel("QPS")
        ax.set_ylabel("Latency (ms)")
        ax.legend(loc="upper left")
        ax.set_title("QPS vs Latency Metrics (50th percentile)")
        ax.grid()

        pyplot.tight_layout()
        pyplot.savefig(f"benchmark_result/{results_file}/qps_v_latency.png")

    _plot_2x2_subplots()
    _plot_50perc_plot()
    _plot_qps_vs_latency()
