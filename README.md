# TTS On-Prem

## Env Requirements

- **NVIDIA 1xH100 SXM5 GPU Node**
- NVIDIA Container Toolkit
- Docker with GPU support
- ~40GB disk space for the image download

## Quick Start

### Pull the image

```bash
# Authenticate to GCP Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the on-prem image
docker pull us-central1-docker.pkg.dev/inworld-ai-registry/backend/tts-onprem-h100:<commit-sha>
```

### Run the container

```bash
docker run -d \
  --gpus all \
  --name tts-onprem \
  -p 8081:8081 \
  -p 9030:9030 \
  us-central1-docker.pkg.dev/inworld-ai-registry/backend/tts-onprem-h100:<commit-sha>
```

### Wait for startup

The ML model takes ~2-3 minutes to load. Check readiness:

```bash
# Check container health
docker ps | grep tts-onprem

# Check all services are running
docker exec tts-onprem supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status

```

## Ports

| Port | Service | Protocol | Exposed | Access |
|------|---------|----------|---------|--------|
| **8081** | **grpc-gateway** | **HTTP** | **Yes** | **External API (recommended)** |
| **9030** | **w-proxy** | **gRPC (h2c)** | **Yes** | **External API (gRPC clients)** |
| 9090 | public-tts-service | gRPC | No | Internal only |
| 50051 | ML Server | HTTP/gRPC | No | Internal only |
| 50073 | Text Normalization | gRPC | No | Internal only |

**Recommended:** Use port 8081 (HTTP) for curl/REST access, matching the Inworld cloud API format.

**Note:** Internal ports (9090, 50051, 50073) are not exposed by default. For debugging, you can manually expose them with `-p 9090:9090 -p 50051:50051`.

## Test Commands

### HTTP REST API (Recommended - matches Inworld cloud API)

Basic TTS request via curl:

```bash
curl -X POST http://localhost:8081/tts/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the on-premise TTS system.",
    "voice_id": "Craig",
    "model_id": "inworld-tts-1.5-mini",
    "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
  }'
```

Save audio to file:

```bash
curl -X POST http://localhost:8081/tts/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world! This is a test of the TTS system.",
    "voice_id": "Craig",
    "model_id": "inworld-tts-1.5-mini",
    "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
  }' | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('output.wav','wb').write(base64.b64decode(d['audioContent'])); print(f'Saved: {len(d[\"audioContent\"])//1000}KB')"
```

List available voices:

```bash
curl http://localhost:8081/tts/v1/voices
```

### gRPC API (Alternative)

For gRPC clients, use port 9030 via w-proxy:

```bash
grpcurl -plaintext -d '{
  "text": "Hello, this is a test of the on-premise TTS system.",
  "voice_id": "Craig",
  "model_id": "inworld-tts-1.5-mini",
  "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
}' localhost:9030 ai.inworld.tts.v1.TextToSpeech/SynthesizeSpeech
```

Note: gRPC reflection is not available through w-proxy. Use pre-compiled proto files or protosets.

### Direct access (debugging only)

For debugging, you can expose internal ports manually and access public-tts-service directly:

```bash
# Run with internal ports exposed for debugging
docker run -d --gpus all --name tts-onprem \
  -p 8081:8081 -p 9030:9030 -p 9090:9090 -p 50051:50051 \
  us-central1-docker.pkg.dev/inworld-ai-registry/backend/tts-onprem-h100:<commit-sha>

# Access public-tts-service directly
grpcurl -plaintext -d '{
  "text": "Direct test.",
  "voice_id": "Craig",
  "model_id": "inworld-tts-1.5-mini",
  "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
}' localhost:9090 ai.inworld.tts.v1.TextToSpeech/SynthesizeSpeech
```

## Available Voices

List all available voices via HTTP:

```bash
curl http://localhost:8081/tts/v1/voices
```

Or via gRPC:

```bash
grpcurl -plaintext -d '{}' localhost:9030 ai.inworld.tts.v1.TextToSpeech/ListVoices
```

Common voices: `Craig`, `Dennis`, `Alex`, `Sarah`, `Olivia`

## Model

Use `inworld-tts-1.5-mini` as the model ID for all requests.

## Logs

View service logs inside the container:

```bash
# All services
docker exec tts-onprem tail -f /var/log/supervisord.log

# grpc-gateway (HTTP API)
docker exec tts-onprem tail -f /var/log/grpc-gateway.log

# w-proxy (gRPC proxy)
docker exec tts-onprem tail -f /var/log/w-proxy.log

# ML server
docker exec tts-onprem tail -f /var/log/tts-v3-trtllm.log

# Normalization
docker exec tts-onprem tail -f /var/log/tts-normalization.log

# Public TTS API
docker exec tts-onprem tail -f /var/log/public-tts-service.log
```

## Image Overview

Container package:
- **grpc-gateway** - HTTP REST API (matches Inworld cloud API format)
- **w-proxy** - gRPC API proxy (routing and metrics)
- **public-tts-service** - TTS service layer
- **tts-normalization** - Text normalization service
- **tts-v3-trtllm** - ML inference server (TensorRT-LLM, H100)

**Architecture:** `HTTP/curl → grpc-gateway → w-proxy → public-tts-service → ML Server`

## Source Images

| Image | Tag | Comment |
|-------|-----|---------|
| `trt-speech-synthesizer-onprem` | `v1-h100` | ML inference server for H100 |
| `tts-normalization` | `2ca1d1d3` | Text normalization service |
| `public-tts-service` | `776a9dda` | TTS service layer (minimal profile) |
| `w-proxy` | `21b0c10b` | gRPC API proxy |
| `grpc-gateway` | `21b0c10b` | HTTP to gRPC transcoding |

Registry: `us-central1-docker.pkg.dev/inworld-ai-registry/backend/`

## Benchmarking

We provide a comprehensize load-test facility and reference benchmark results. Follow up tts-load-test-simple/README.md#quick-start to get the benchmark results for your deployment and compare with REF-BENCHMARK.md result.
