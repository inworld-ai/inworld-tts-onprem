#!/usr/bin/env python3
"""Chart comparison tool for TTS load test benchmark results.

Usage:
    chart_compare.py --benchmark modular-dense-2 --benchmark torch-dense-2 \
                     --metric avg_chunk_latency_ms_percentiles_p50 \
                     --metric first_chunk_latency_ms_percentiles_p50

This will create a chart with:
- avg_qps as x-axis
- latency metric values as y-axis
- A series per benchmark/metric combination
"""

import csv
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import click

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.interpolate import interp1d
    import numpy as np
except ImportError as e:
    missing_packages = []
    if "matplotlib" in str(e):
        missing_packages.append("matplotlib")
    if "scipy" in str(e):
        missing_packages.append("scipy")
    if "numpy" in str(e):
        missing_packages.append("numpy")

    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
    else:
        print(f"Import error: {e}")
    sys.exit(1)


def load_benchmark_results(benchmark_name: str) -> Dict:
    """Load benchmark results from result.json file."""
    result_file = f"benchmark_result/{benchmark_name}/result.json"
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, "r") as f:
        return json.load(f)


def parse_metric(metric_name: str) -> Tuple[str, Optional[str]]:
    """Parse metric name like 'avg_chunk_latency_ms_percentiles_p50' into base and percentile.

    Args:
        metric_name: Metric name like 'avg_chunk_latency_ms_percentiles_p50'

    Returns:
        Tuple of (base_metric, percentile) where percentile is '50', '90', '95', '99' or None
    """
    if metric_name.endswith(("_p50", "_p90", "_p95", "_p99")):
        base_metric = metric_name[:-4]  # Remove _pXX
        percentile = metric_name[-2:]  # Get XX from _pXX
        return base_metric, percentile
    else:
        return metric_name, None


def extract_metric_data(
    results: Dict, metric_name: str
) -> Tuple[List[float], List[float]]:
    """Extract QPS and metric values from results.

    Args:
        results: Dictionary of benchmark results
        metric_name: Name of metric to extract

    Returns:
        Tuple of (qps_values, metric_values) lists
    """
    base_metric, percentile = parse_metric(metric_name)

    qps_values = []
    metric_values = []

    for qps_key, result_data in results.items():
        qps = result_data.get("avg_qps")
        if qps is None:
            continue

        if percentile:
            # Handle percentile metrics like avg_chunk_latency_ms_percentiles_p50
            if base_metric in result_data and result_data[base_metric] is not None:
                metric_value = result_data[base_metric].get(percentile)
                if metric_value is not None:
                    qps_values.append(qps)
                    metric_values.append(metric_value)
        else:
            # Handle simple metrics
            if base_metric in result_data:
                metric_value = result_data[base_metric]
                if metric_value is not None:
                    qps_values.append(qps)
                    metric_values.append(metric_value)

    return qps_values, metric_values


def get_color_and_marker(index: int) -> Tuple[str, str]:
    """Get color and marker style for plot series."""
    colors = list(mcolors.TABLEAU_COLORS.values())
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    color = colors[index % len(colors)]
    marker = markers[index % len(markers)]

    return color, marker


def smooth_data(
    x_values: List[float], y_values: List[float], smooth_factor: float
) -> Tuple[List[float], List[float]]:
    """Apply smoothing to data points using interpolation.

    Args:
        x_values: X coordinates
        y_values: Y coordinates
        smooth_factor: Smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)

    Returns:
        Tuple of (smoothed_x, smoothed_y) lists
    """
    if smooth_factor == 0.0 or len(x_values) < 3:
        return x_values, y_values

    try:
        # Create more points for smoother curve
        x_min, x_max = min(x_values), max(x_values)
        num_points = max(50, len(x_values) * 3)  # At least 50 points for smooth curve
        x_smooth = np.linspace(x_min, x_max, num_points)

        # Apply smoothing - higher smooth_factor means more smoothing
        # Use cubic interpolation with smoothing
        if smooth_factor >= 1.0:
            # Maximum smoothing - use linear interpolation
            f = interp1d(x_values, y_values, kind="linear", fill_value="extrapolate")
        else:
            # Interpolate with cubic spline, then apply smoothing by reducing points
            f = interp1d(x_values, y_values, kind="cubic", fill_value="extrapolate")

        y_smooth = f(x_smooth)

        # Apply additional smoothing by reducing the number of interpolated points
        if smooth_factor > 0.0:
            # Reduce resolution based on smoothing factor
            step = max(1, int(smooth_factor * 10))
            x_smooth = x_smooth[::step]
            y_smooth = y_smooth[::step]

        return list(x_smooth), list(y_smooth)
    except Exception:
        # Fall back to original data if smoothing fails
        return x_values, y_values


def print_performance_improvements(all_data: List[Dict]) -> None:
    """Print percentage improvements between benchmarks at key data points."""
    if len(all_data) < 2:
        return

    # Group data by metric
    metric_groups = {}
    for data in all_data:
        metric = data["metric"]
        if metric not in metric_groups:
            metric_groups[metric] = []
        metric_groups[metric].append(data)

    for metric, metric_data in metric_groups.items():
        if len(metric_data) < 2:
            continue

        click.echo(f"\n📊 Performance Improvements for {metric}:")
        click.echo("=" * (50 + len(metric)))

        # Use the first benchmark as baseline
        baseline = metric_data[0]
        baseline_name = baseline["benchmark"]

        for comparison in metric_data[1:]:
            comparison_name = comparison["benchmark"]

            # Create interpolation points across the overlapping QPS range
            baseline_qps_min, baseline_qps_max = (
                min(baseline["qps"]),
                max(baseline["qps"]),
            )
            comparison_qps_min, comparison_qps_max = (
                min(comparison["qps"]),
                max(comparison["qps"]),
            )

            # Find overlapping range
            overlap_min = max(baseline_qps_min, comparison_qps_min)
            overlap_max = min(baseline_qps_max, comparison_qps_max)

            if overlap_min >= overlap_max:
                click.echo(
                    f"⚠️  No overlapping QPS range between {baseline_name} and {comparison_name}"
                )
                continue

            # Create interpolation points (5 evenly spaced points in the overlap range)
            interpolation_qps = [
                overlap_min,
                overlap_min + (overlap_max - overlap_min) * 0.25,
                overlap_min + (overlap_max - overlap_min) * 0.5,
                overlap_min + (overlap_max - overlap_min) * 0.75,
                overlap_max,
            ]

            click.echo(f"\n🔀 {comparison_name} vs {baseline_name} (baseline):")

            try:
                # Create interpolation functions
                baseline_interp = interp1d(
                    baseline["qps"],
                    baseline["values"],
                    kind="linear",
                    fill_value="extrapolate",
                )
                comparison_interp = interp1d(
                    comparison["qps"],
                    comparison["values"],
                    kind="linear",
                    fill_value="extrapolate",
                )

                improvements = []
                for qps in interpolation_qps:
                    # Get interpolated latency values
                    baseline_latency = float(baseline_interp(qps))
                    comparison_latency = float(comparison_interp(qps))

                    # Calculate percentage improvement (lower latency is better)
                    if baseline_latency > 0:
                        improvement = (
                            (baseline_latency - comparison_latency) / baseline_latency
                        ) * 100
                        improvements.append(
                            (qps, improvement, baseline_latency, comparison_latency)
                        )

            except Exception as e:
                click.echo(f"⚠️  Failed to interpolate data: {e}")
                continue

            # Sort by QPS and show key points
            improvements.sort()

            if improvements:
                # Show all interpolated points with descriptive names
                point_names = [
                    "Low QPS",
                    "Low-Mid QPS",
                    "Mid QPS",
                    "Mid-High QPS",
                    "High QPS",
                ]

                for i, (qps, improvement, baseline_lat, comparison_lat) in enumerate(
                    improvements
                ):
                    point_name = (
                        point_names[i] if i < len(point_names) else f"Point {i+1}"
                    )

                    if improvement > 0:
                        click.echo(
                            f"  📈 {point_name} ({qps:,.0f} QPS): "
                            f"{improvement:+5.1f}% faster "
                            f"({comparison_lat:.1f}ms vs {baseline_lat:.1f}ms)"
                        )
                    elif improvement < 0:
                        click.echo(
                            f"  📉 {point_name} ({qps:,.0f} QPS): "
                            f"{abs(improvement):5.1f}% slower "
                            f"({comparison_lat:.1f}ms vs {baseline_lat:.1f}ms)"
                        )
                    else:
                        click.echo(
                            f"  ➡️  {point_name} ({qps:,.0f} QPS): "
                            f"Same performance ({comparison_lat:.1f}ms)"
                        )

                # Show average improvement
                avg_improvement = sum(imp for _, imp, _, _ in improvements) / len(
                    improvements
                )
                if avg_improvement > 0:
                    click.echo(
                        f"  🎯 Average improvement: {avg_improvement:+5.1f}% faster"
                    )
                elif avg_improvement < 0:
                    click.echo(
                        f"  🎯 Average difference: {abs(avg_improvement):5.1f}% slower"
                    )
                else:
                    click.echo("  🎯 Average difference: Same performance")


def print_throughput_analysis(
    all_data: List[Dict], latency_limit: float = 500.0
) -> None:
    """Print throughput analysis based on latency constraint."""
    if len(all_data) < 2:
        return

    # Group data by metric
    metric_groups = {}
    for data in all_data:
        metric = data["metric"]
        if metric not in metric_groups:
            metric_groups[metric] = []
        metric_groups[metric].append(data)

    for metric, metric_data in metric_groups.items():
        if len(metric_data) < 2:
            continue

        click.echo(
            f"\n🚀 Throughput Analysis at {latency_limit}ms Latency Limit for {metric}:"
        )
        click.echo("=" * (60 + len(metric)))

        throughput_results = []

        for data in metric_data:
            benchmark_name = data["benchmark"]
            qps_values = data["qps"]
            latency_values = data["values"]

            # Find maximum QPS under latency limit
            max_qps_under_limit = 0.0

            try:
                # Check if any points are under the limit
                under_limit_points = [
                    (q, latency)
                    for q, latency in zip(qps_values, latency_values)
                    if latency <= latency_limit
                ]

                if under_limit_points:
                    # Get the highest QPS where latency is still under limit
                    max_qps_under_limit = max(q for q, _ in under_limit_points)
                else:
                    # If no points are under limit, try interpolation to find where it crosses
                    if len(qps_values) >= 2:
                        # Create interpolation function
                        latency_interp = interp1d(
                            qps_values,
                            latency_values,
                            kind="linear",
                            fill_value="extrapolate",
                        )

                        # Binary search to find QPS where latency = limit
                        min_qps, max_qps = min(qps_values), max(qps_values)

                        # Check if the minimum QPS already exceeds the limit
                        if latency_interp(min_qps) > latency_limit:
                            max_qps_under_limit = 0.0
                        else:
                            # Binary search for the crossing point
                            left, right = min_qps, max_qps
                            for _ in range(20):  # 20 iterations should be enough
                                mid = (left + right) / 2
                                mid_latency = float(latency_interp(mid))

                                if mid_latency <= latency_limit:
                                    left = mid
                                else:
                                    right = mid

                                if abs(right - left) < 0.1:  # Precision of 0.1 QPS
                                    break

                            max_qps_under_limit = left

                throughput_results.append((benchmark_name, max_qps_under_limit))

                if max_qps_under_limit > 0:
                    click.echo(f"  📊 {benchmark_name}: {max_qps_under_limit:,.1f} QPS")
                else:
                    click.echo(
                        f"  ⚠️  {benchmark_name}: Cannot achieve {latency_limit}ms latency"
                    )

            except Exception as e:
                click.echo(f"  ❌ {benchmark_name}: Error calculating throughput - {e}")
                throughput_results.append((benchmark_name, 0.0))

        # Calculate throughput improvements
        if len(throughput_results) >= 2:
            # Sort by throughput (highest first)
            throughput_results.sort(key=lambda x: x[1], reverse=True)

            best_benchmark, best_throughput = throughput_results[0]

            if best_throughput > 0:
                click.echo(
                    f"\n  🏆 Best performer: {best_benchmark} ({best_throughput:,.1f} QPS)"
                )

                for benchmark_name, throughput in throughput_results[1:]:
                    if throughput > 0 and best_throughput > 0:
                        improvement = (
                            (best_throughput - throughput) / throughput
                        ) * 100
                        throughput_ratio = best_throughput / throughput

                        # Calculate cost reduction
                        # Cost reduction = 1 - (before_qps / after_qps)
                        # Assuming cost is inversely proportional to throughput
                        cost_reduction = (1 - (throughput / best_throughput)) * 100

                        click.echo(
                            f"  📈 {best_benchmark} vs {benchmark_name}: "
                            f"{improvement:+.1f}% more throughput "
                            f"({throughput_ratio:.1f}x faster)"
                        )
                        click.echo(
                            f"      💰 Cost reduction: {cost_reduction:.1f}% "
                            f"(need {throughput_ratio:.1f}x fewer servers)"
                        )
                    elif throughput == 0:
                        click.echo(
                            f"  ♾️  {best_benchmark} vs {benchmark_name}: "
                            f"Infinite improvement (only {best_benchmark} can meet {latency_limit}ms limit)"
                        )
                        click.echo(
                            f"      💰 Cost reduction: ~100% (eliminates need for {benchmark_name})"
                        )
                    else:
                        click.echo(
                            f"  ➡️  {best_benchmark} vs {benchmark_name}: Same throughput"
                        )
                        click.echo(
                            "      💰 Cost reduction: 0% (same infrastructure needed)"
                        )

                # Add cost analysis summary
                click.echo("\n  💡 Cost Analysis Summary:")
                click.echo(
                    "     • Cost model: Assumes cost ∝ 1/throughput (higher QPS = fewer servers)"
                )
                click.echo(
                    "     • Formula: Cost Reduction = 1 - (baseline_QPS / improved_QPS)"
                )
                click.echo(f"     • Based on {latency_limit}ms latency constraint")

                if best_throughput > 0:
                    total_cost_savings = 0
                    valid_comparisons = 0
                    for benchmark_name, throughput in throughput_results[1:]:
                        if throughput > 0:
                            cost_reduction = (1 - (throughput / best_throughput)) * 100
                            total_cost_savings += cost_reduction
                            valid_comparisons += 1

                    if valid_comparisons > 0:
                        avg_cost_savings = total_cost_savings / valid_comparisons
                        click.echo(
                            f"     • Average cost reduction vs alternatives: {avg_cost_savings:.1f}%"
                        )


def save_chart_data_to_csv(all_data: List[Dict], save_path: str) -> None:
    """Save chart data to CSV file in the same directory as the chart."""
    # Determine CSV file path
    save_dir = os.path.dirname(save_path) if save_path else "benchmark_result"
    csv_filename = "comparison_chart.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["benchmark", "metric", "qps", "latency_ms"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            for data in all_data:
                benchmark = data["benchmark"]
                metric = data["metric"]
                qps_values = data["qps"]
                latency_values = data["values"]

                for qps, latency in zip(qps_values, latency_values):
                    writer.writerow(
                        {
                            "benchmark": benchmark,
                            "metric": metric,
                            "qps": qps,
                            "latency_ms": latency,
                        }
                    )

        click.echo(f"Chart data saved to: {csv_path}")

    except Exception as e:
        click.echo(f"Warning: Failed to save CSV data: {e}", err=True)


def create_comparison_chart(
    benchmarks: List[str],
    metrics: List[str],
    save_path: Optional[str],
    smooth: float,
    scale: float,
    throughput_latency_target: float,
) -> None:
    """Create comparison chart for multiple benchmarks and metrics."""
    # Scale the figure size
    base_width, base_height = 14, 10
    plt.figure(figsize=(base_width * scale, base_height * scale))

    series_index = 0
    all_data = []

    # Collect all data first for consistent styling
    for benchmark in benchmarks:
        try:
            click.echo(f"Loading benchmark: {benchmark}")
            results = load_benchmark_results(benchmark)

            for metric in metrics:
                qps_values, metric_values = extract_metric_data(results, metric)

                if qps_values and metric_values:
                    # Sort by QPS for proper line plotting
                    sorted_data = sorted(zip(qps_values, metric_values))
                    qps_sorted, metric_sorted = zip(*sorted_data)

                    all_data.append(
                        {
                            "benchmark": benchmark,
                            "metric": metric,
                            "qps": list(qps_sorted),
                            "values": list(metric_sorted),
                            "index": series_index,
                        }
                    )
                    series_index += 1
                else:
                    click.echo(f"Warning: No data found for {benchmark} - {metric}")

        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
        except Exception as e:
            click.echo(f"Error processing {benchmark}: {e}")

    if not all_data:
        click.echo("Error: No valid data found to plot")
        return

    # Save chart data to CSV
    csv_save_path = save_path if save_path else "benchmark_result/comparison_chart.png"
    save_chart_data_to_csv(all_data, csv_save_path)

    # Plot all series
    for data in all_data:
        color, marker = get_color_and_marker(data["index"])
        label = f"{data['benchmark']} - {data['metric']}"

        # Apply smoothing if requested
        if smooth > 0.0:
            x_plot, y_plot = smooth_data(data["qps"], data["values"], smooth)
            # For smoothed lines, reduce marker frequency to avoid clutter
            marker_every = max(1, len(x_plot) // 10) if len(x_plot) > 10 else 1
            plt.plot(
                x_plot,
                y_plot,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=4,
                markevery=marker_every,
            )
            # Add labels to original data points (not smoothed points) to avoid clutter
            for x, y in zip(data["qps"], data["values"]):
                plt.annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    alpha=0.7,
                )
            # Add benchmark name along the line (at the end point)
            if len(data["qps"]) > 0:
                end_x, end_y = data["qps"][-1], data["values"][-1]
                plt.annotate(
                    data["benchmark"],
                    (end_x, end_y),
                    textcoords="offset points",
                    xytext=(15, 0),
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    alpha=0.8,
                )
        else:
            plt.plot(
                data["qps"],
                data["values"],
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=6,
            )
            # Add value labels to each data point
            for x, y in zip(data["qps"], data["values"]):
                plt.annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    alpha=0.7,
                )
            # Add benchmark name along the line (at the end point)
            if len(data["qps"]) > 0:
                end_x, end_y = data["qps"][-1], data["values"][-1]
                plt.annotate(
                    data["benchmark"],
                    (end_x, end_y),
                    textcoords="offset points",
                    xytext=(15, 0),
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                    alpha=0.8,
                )

    plt.xlabel("Throughput (Average QPS)", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title("Benchmark Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        click.echo(f"Chart saved to: {save_path}")

    # Print percentage improvements
    print_performance_improvements(all_data)

    # Print throughput analysis based on latency constraint
    print_throughput_analysis(all_data, latency_limit=throughput_latency_target)

    plt.show()


@click.command()
@click.option(
    "--benchmark",
    multiple=True,
    help="Benchmark name (can be specified multiple times). "
    "Examples: modular-dense-2, torch-dense-2",
)
@click.option(
    "--metric",
    multiple=True,
    default=["first_chunk_latency_ms_percentiles_p50"],
    help="Metric to plot (can be specified multiple times). "
    "Default: first_chunk_latency_ms_percentiles_p50. "
    "Examples: avg_chunk_latency_ms_percentiles_p50, "
    "e2e_latency_ms_percentiles_p99, "
    "second_of_audio_generation_latency_ms_percentiles_p95",
)
@click.option(
    "--save",
    default="benchmark_result/comparison_chart.png",
    help="Save chart to file. Default: benchmark_result/comparison_chart.png",
)
@click.option(
    "--list-benchmarks", is_flag=True, help="List available benchmarks and exit"
)
@click.option(
    "--smooth",
    default=0.0,
    type=float,
    help="Smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing). Default: 0.0",
)
@click.option(
    "--scale",
    default=1.0,
    type=float,
    help="Scale factor for chart size (1.0 = default size). Default: 1.0",
)
@click.option(
    "--throughput-latency-target",
    default=1000.0,
    type=float,
    help="Latency target in ms for throughput analysis. Default: 1000.0ms",
)
def main(
    benchmark: tuple[str, ...],
    metric: tuple[str, ...],
    save: Optional[str],
    list_benchmarks: bool,
    smooth: float,
    scale: float,
    throughput_latency_target: float,
):
    """Compare benchmark results across multiple runs and metrics.

    Examples:

    \b
    chart_compare.py --benchmark modular-dense-2 --benchmark torch-dense-2 \\
                     --metric avg_chunk_latency_ms_percentiles_p50 \\
                     --metric avg_chunk_latency_ms_percentiles_p99

    \b
    chart_compare.py --benchmark test1 --benchmark test2 \\
                     --metric e2e_latency_ms_percentiles_p95 \\
                     --smooth 0.5 --scale 1.5 \\
                     --throughput-latency-target 500.0 \\
                     --save comparison_chart.png
    """

    # List available benchmarks if requested
    if list_benchmarks:
        benchmark_dir = "benchmark_result"
        if os.path.exists(benchmark_dir):
            benchmarks = [
                d
                for d in os.listdir(benchmark_dir)
                if os.path.isdir(os.path.join(benchmark_dir, d))
                and os.path.exists(os.path.join(benchmark_dir, d, "result.json"))
            ]
            if benchmarks:
                click.echo("Available benchmarks:")
                for bench in sorted(benchmarks):
                    click.echo(f"  {bench}")
            else:
                click.echo("No benchmarks found in benchmark_result/")
        else:
            click.echo("benchmark_result/ directory not found")
        return

    # Validate required arguments when not listing benchmarks
    if not benchmark:
        click.echo("Error: --benchmark is required", err=True)
        click.echo("Use --help for usage information")
        sys.exit(1)

    if not metric:
        click.echo("Error: --metric is required", err=True)
        click.echo("Use --help for usage information")
        sys.exit(1)

    # Validate that benchmark_result directory exists
    if not os.path.exists("benchmark_result"):
        click.echo(
            "Error: benchmark_result/ directory not found. Make sure you're running from the correct directory.",
            err=True,
        )
        sys.exit(1)

    # Validate smooth parameter
    if smooth < 0.0 or smooth > 1.0:
        click.echo("Error: --smooth must be between 0.0 and 1.0", err=True)
        sys.exit(1)

    # Validate scale parameter
    if scale <= 0.0:
        click.echo("Error: --scale must be greater than 0.0", err=True)
        sys.exit(1)

    # Validate throughput latency target parameter
    if throughput_latency_target <= 0.0:
        click.echo(
            "Error: --throughput-latency-target must be greater than 0.0", err=True
        )
        sys.exit(1)

    click.echo("Creating comparison chart for:")
    click.echo(f"  Benchmarks: {', '.join(benchmark)}")
    click.echo(f"  Metrics: {', '.join(metric)}")
    click.echo(f"  Smoothing: {smooth}")
    click.echo(f"  Scale: {scale}x")
    click.echo(f"  Throughput latency target: {throughput_latency_target}ms")

    create_comparison_chart(
        list(benchmark), list(metric), save, smooth, scale, throughput_latency_target
    )


if __name__ == "__main__":
    main()
