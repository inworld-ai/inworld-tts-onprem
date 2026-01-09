# TTS Load Test Tool

A comprehensive load testing tool for Text-to-Speech (TTS) services that measures performance metrics including latency, throughput, and streaming characteristics across different QPS (Queries Per Second) loads.

## Overview

This tool simulates realistic TTS workloads by sending requests at specified rates with configurable burstiness patterns. It measures:
- End-to-end latency
- Audio generation latency per second
- Streaming metrics (first chunk, 4th chunk, average chunk latencies)
- Request success rates
- Server performance under different load conditions

## Features

- **Multiple Client Types**: Supports gRPC, Internal API, and Inworld API protocols
- **Streaming & Non-Streaming**: Tests both real-time streaming and batch synthesis
- **Configurable Load Patterns**: Control QPS, burstiness, and request distribution
- **Accurate Timing**: Uses pausable timers to exclude client-side processing from server metrics
- **Comprehensive Metrics**: Detailed latency percentiles and performance statistics
- **Visualization**: Automatic plotting of results with customizable charts
- **Async Architecture**: Efficient concurrent request handling


## Quick Start

```bash
# Basic load test with streaming
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.dev.oc.inworld.dev \
    --stream \
    --min-qps 1.0 \
    --max-qps 10.0 \
    --qps-step 2.0 \
    --number-of-samples 100

# High-throughput production test
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.prod.oc.inworld.dev \
    --stream \
    --min-qps 300.0 \
    --max-qps 500.0 \
    --qps-step 50.0 \
    --number-of-samples 3000 \
    --voice-ids Olivia \
    --benchmark_name prod-high-load
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--host` | Base address of the TTS server (endpoint auto-appended) | `http://tts-v3-turbo-torch.dev.oc.inworld.dev` |

### Load Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-qps` | `1.0` | Minimum requests per second to test |
| `--max-qps` | `10.0` | Maximum requests per second to test |
| `--qps-step` | `2.0` | Step size for QPS increments |
| `--number-of-samples` | `1` | Total number of texts to synthesize per QPS level |
| `--burstiness` | `1.0` | Request timing pattern (1.0=Poisson, <1.0=bursty, >1.0=uniform) |

### TTS Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stream` | `False` | Use streaming synthesis (`/SynthesizeSpeechStream`) vs non-streaming (`/SynthesizeSpeech`) |
| `--max_tokens` | `400` | Maximum tokens to synthesize (~8s audio at 50 tokens/s) |
| `--voice-ids` | `["Olivia", "Remy"]` | Voice IDs to use (can specify multiple) |
| `--model_id` | `None` | Model ID for TTS synthesis (optional) |
| `--text_samples_file` | `scripts/tts_load_testing/text_samples.json` | File containing text samples |

### Output & Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--benchmark_name` | `auto-generated` | Name for the benchmark run (affects output files) |
| `--plot_only` | `False` | Only generate plots from existing results (skip testing) |
| `--verbose` | `False` | Enable verbose output for debugging |

## Server Endpoints

The tool automatically appends the correct endpoint based on the `--stream` flag:

### Development Servers
- `http://tts-v3-turbo-torch.dev.oc.inworld.dev` - Dev 1B model
- `http://tts-v3-hd-torch.dev.oc.inworld.dev` - Dev HD model

### Production Servers
- `http://tts-v3-turbo-torch.prod.oc.inworld.dev` - Prod 1B model
- `http://tts-v3-hd-torch.prod.oc.inworld.dev` - Prod HD model

### External APIs
- `https://api.inworld.ai/v1/tts/synthesize-sync` - Inworld API (non-streaming)
- `https://api.inworld.ai/v1/tts/synthesize` - Inworld API (streaming)

## Usage Examples

### Development Testing
```bash
# Light load test for development
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.dev.oc.inworld.dev \
    --min-qps 1.0 \
    --max-qps 5.0 \
    --qps-step 1.0 \
    --number-of-samples 50 \
    --benchmark_name dev-light-load
```

### Production Load Testing
```bash
# High QPS production test
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.prod.oc.inworld.dev \
    --stream \
    --min-qps 100.0 \
    --max-qps 600.0 \
    --qps-step 100.0 \
    --burstiness 0.8 \
    --number-of-samples 2000 \
    --text_samples_file tests-data/tts-load-test/text_samples_gt_30_words.json \
    --voice-ids Olivia Remy Marcus \
    --benchmark_name prod-stress-test
```

### Streaming vs Non-Streaming Comparison
```bash
# Non-streaming test
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.dev.oc.inworld.dev \
    --min-qps 10.0 \
    --max-qps 50.0 \
    --qps-step 10.0 \
    --number-of-samples 500 \
    --benchmark_name non-streaming-test

# Streaming test
./scripts/tts-load-test \
    --host http://tts-v3-turbo-torch.dev.oc.inworld.dev \
    --stream \
    --min-qps 10.0 \
    --max-qps 50.0 \
    --qps-step 10.0 \
    --number-of-samples 500 \
    --benchmark_name streaming-test
```

### External API Testing
```bash
# Test Inworld API with authentication
export INWORLD_API_KEY="your-api-key-here"
./scripts/tts-load-test \
    --host https://api.inworld.ai/v1/tts/synthesize-sync \
    --min-qps 1.0 \
    --max-qps 10.0 \
    --number-of-samples 100 \
    --voice-ids "en-US-Studio-O" \
    --benchmark_name inworld-api-test
```

### Plot-Only Mode
```bash
# Generate plots from existing results
./scripts/tts-load-test \
    --plot_only \
    --benchmark_name prod-stress-test
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
- `{benchmark_name}_*.png`: Performance charts

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
- **Connection errors**: Verify server address and network connectivity
- **Authentication errors**: Set `INWORLD_API_KEY` for external APIs
- **High latency**: Check server load and network conditions
- **Memory issues**: Reduce `number-of-samples` for high QPS tests

### Debug Mode
Use `--verbose` flag for detailed request/response logging:
```bash
./scripts/tts-load-test --verbose --host ... # other params
```

## Architecture

The tool uses:
- **Async/await**: Efficient concurrent request handling
- **Pausable timers**: Accurate server-only timing measurements
- **Multiple protocols**: gRPC, HTTP REST API support
- **Configurable clients**: Pluggable client architecture
- **Real-time progress**: Live progress bars and status updates
