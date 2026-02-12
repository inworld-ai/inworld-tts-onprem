# TTS Load Test Tool

A comprehensive load testing tool for Text-to-Speech (TTS) services that measures performance metrics including latency, throughput, and streaming characteristics across different QPS (Queries Per Second) loads.

## Overview

This tool simulates realistic TTS workloads by sending requests at specified rates with configurable burstiness patterns. It measures:
- End-to-end latency
- Audio generation latency per second
- Streaming metrics (first chunk, 4th chunk, average chunk latencies)
- Request success rates
- Server performance under different load conditions

## Quick Start

```bash
# Install dependencies
pip install -e .

# Basic streaming load test
python -m load_test.main \
    --host http://localhost:8081 \
    --model-id inworld-tts-1.5-mini \
    --stream \
    --min-qps 1.0 \
    --max-qps 7.0 \
    --qps-step 2.0 \
    --number-of-samples 300
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|---|---|---|
| `--host` | Base address of the On-Premise TTS server (endpoint auto-appended) | `http://localhost:8081` |
| `--model-id` | Model ID to use for TTS synthesis | `inworld-tts-1.5-mini`, `inworld-tts-1.5-max` |

### Load Configuration

| Parameter | Default | Description |
|---|---:|---|
| `--min-qps` | `1.0` | Minimum requests per second to test |
| `--max-qps` | `10.0` | Maximum requests per second to test |
| `--qps-step` | `1.0` | Step size for QPS increments |
| `--number-of-samples` | `100` | Total number of texts to synthesize per QPS level |
| `--burstiness` | `1.0` | Request timing pattern (`1.0` = Poisson, `< 1.0` = bursty, `> 1.0` = uniform) |

### TTS Configuration

| Parameter | Default | Description |
|---|---:|---|
| `--stream` | `False` | Use streaming synthesis vs non-streaming |
| `--max-tokens` | `400` | Maximum tokens to synthesize (~8s audio at 50 tokens/s) |
| `--voice-ids` | `["Alex"]` | Voice IDs to use (can specify multiple: `--voice-ids Alex --voice-ids Craig`) |
| `--text-samples-file` | bundled `text_samples_single.json` | File containing text samples |
| `--mode` | `tts` | Client mode: `tts` for TTS synthesis or `embedding` for text embeddings |
| `--sample-format` | `simple-json` | Format of text samples file (`simple-json`, `axolotl-input-output`, `id-prompt-json`) |
| `--random` | `False` | Prepend random words to each prompt to prevent caching |
| `--timestamp` | `None` | Controls timestamp metadata (`TIMESTAMP_TYPE_UNSPECIFIED`, `WORD`, `CHARACTER`) |

### Output & Analysis

| Parameter | Default | Description |
|---|---:|---|
| `--benchmark-name` | auto-generated | Name for the benchmark run (affects output files) |
| `--plot-only` | `False` | Only generate plots from existing results (skip testing) |
| `--verbose` | `False` | Enable verbose output for debugging |
| `--sample` | `False` | Send a single request and print the redacted response |
| `--analyze-prompts` | `False` | Analyze prompt lengths in text samples file and exit |

## Examples

### Streaming vs Non-Streaming Comparison
```bash
# Non-streaming test
python -m load_test.main \
    --host http://localhost:8081 \
    --model-id inworld-tts-1.5-mini \
    --min-qps 10.0 \
    --max-qps 50.0 \
    --qps-step 10.0 \
    --number-of-samples 500 \
    --benchmark-name non-streaming-test

# Streaming test
python -m load_test.main \
    --host http://localhost:8081 \
    --model-id inworld-tts-1.5-mini \
    --stream \
    --min-qps 10.0 \
    --max-qps 50.0 \
    --qps-step 10.0 \
    --number-of-samples 500 \
    --benchmark-name streaming-test
```

### Sample Request Mode
```bash
# Send a single request and inspect the redacted response
python -m load_test.main \
    --host http://localhost:8081 \
    --model-id inworld-tts-1.5-mini \
    --sample
```

### Plot-Only Mode
```bash
# Generate plots from existing results
python -m load_test.main \
    --plot-only \
    --benchmark-name prod-stress-test
```

### Analyze Prompt Lengths
```bash
python -m load_test.main \
    --analyze-prompts \
    --text-samples-file path/to/text_samples.json
```

## Understanding Results

The tool generates comprehensive metrics for each QPS level:

### Latency Metrics
- **E2E Latency**: Complete request-response time
- **Audio Generation Latency**: Time per second of generated audio
- **First Chunk Latency**: Time to first audio chunk (streaming only)
- **4th Chunk Latency**: Time to 4th audio chunk (streaming only)
- **Average Chunk Latency**: Mean time between chunks (streaming only)

### Percentiles
Results include P50, P90, P95, and P99 percentiles for all latency metrics.

### Output Files
Results are saved in `benchmark_result/{benchmark_name}/`:
- `result.json`: Raw performance data
- `plot_2x2.png`: 2x2 grid of percentile charts
- `plot_p50.png`: P50 latency chart
- `qps_v_latency.png`: QPS vs latency chart
- `logs.txt`: Text log of results

## Sample Formats

| Format | Description | Schema |
|---|---|---|
| `simple-json` | Default format | `{"samples": ["text1", "text2", ...]}` |
| `axolotl-input-output` | Axolotl dataset format | Extracts assistant responses from conversation segments |
| `id-prompt-json` | ID-prompt pairs | `[{"id": 1, "prompt": "text..."}, ...]` |

## Burstiness Parameter

The burstiness parameter controls request timing distribution:
- **1.0**: Poisson process (natural randomness)
- **< 1.0**: More bursty (requests come in clusters)
- **> 1.0**: More uniform (evenly spaced requests)

## Performance Tips

1. **Start small**: Begin with low QPS and small sample sizes
2. **Use appropriate text samples**: Match your production text length distribution
3. **Monitor server resources**: Watch CPU, memory, and network during tests
4. **Consider burstiness**: Real-world traffic is often bursty (try 0.7-0.9)
5. **Test both modes**: Compare streaming vs non-streaming for your use case

## Troubleshooting

### Common Issues
- **400 Bad Request**: Verify `--model-id` matches the model loaded in your server (e.g. `inworld-tts-1.5-mini`)
- **Connection errors**: Verify server address and network connectivity
- **High latency**: Check server load and network conditions
- **Memory issues**: Reduce `--number-of-samples` for high QPS tests

### Debug Mode
Use `--verbose` flag for detailed request/response logging:
```bash
python -m load_test.main --verbose --host http://localhost:8081 --model-id inworld-tts-1.5-mini --stream
```

## Architecture

The tool uses:
- **Async/await**: Efficient concurrent request handling
- **Pausable timers**: Accurate server-only timing measurements
- **HTTP REST API**: Supports both public and internal TTS endpoints
- **Configurable clients**: Pluggable client architecture for TTS and embedding modes
- **Real-time progress**: Live progress bars and status updates
