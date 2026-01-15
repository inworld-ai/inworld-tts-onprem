<div align="center">

![Inworld Logo](https://github.com/inworld-ai/inworld-tts-onprem/blob/main/assets/cover.jpg?raw=true)

# TTS _On-Premise_

</div>

## Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA H100 SXM5 (80GB) |
| **RAM** | 64GB+ system memory |
| **CPU** | 8+ cores |
| **Disk** | 50GB free space |
| **OS** | Ubuntu 22.04 LTS |
| **Software** | Docker + NVIDIA Container Toolkit |
| **Software** | Google Cloud SDK (gcloud CLI) |

## Authentication

To pull images from GCP Artifact Registry, you need to configure google service account Docker authentication using gcloud CLI.

For detailed instructions on authentication methods including service account configuration, see the official documentation:
[Configure authentication to Artifact Registry for Docker](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper)

## Quick Start

```bash
# Authenticate to GCP Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the image (replace <version> with latest release version, e.g., 1.0.0)
docker pull us-central1-docker.pkg.dev/inworld-ai-registry/tts-onprem/tts-1.5-mini-h100-onprem:<version>

# Run the container
docker run -d \
  --gpus all \
  --name inworld-tts-onprem \
  -p 8081:8081 \
  -p 9030:9030 \
  us-central1-docker.pkg.dev/inworld-ai-registry/tts-onprem/tts-1.5-mini-h100-onprem:<version>
```

The ML model takes ~2-3 minutes to load. Check readiness:

```bash
docker exec inworld-tts-onprem supervisorctl -s unix:///tmp/supervisor.sock status
```

## API

| Port | Protocol | Description |
|------|----------|-------------|
| **8081** | HTTP | REST API (recommended) |
| **9030** | gRPC | For gRPC clients |

### Test Request

```bash
curl -X POST http://localhost:8081/tts/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice_id": "Craig",
    "model_id": "inworld-tts-1.5-mini",
    "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
  }'
```

For full API documentation, see [Synthesize Speech](https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech).

### List Voices

```bash
curl http://localhost:8081/tts/v1/voices
```

## Logs

```bash
# Supervisor (all services)
docker exec inworld-tts-onprem tail -f /var/log/supervisord.log

# ML server
docker exec inworld-tts-onprem tail -f /var/log/tts-v3-trtllm.log
```

## Troubleshooting

```bash
# Restart container
docker restart inworld-tts-onprem

# Check service status
docker exec inworld-tts-onprem supervisorctl -s unix:///tmp/supervisor.sock status

# Export logs to folder
d=tts-logs-$(date +%Y%m%d_%H%M%S) && mkdir $d && docker exec inworld-tts-onprem sh -c 'cd /var/log && tar cf - supervisord.log tts-normalization.log tts-v3-trtllm.log grpc-gateway.log w-proxy.log public-tts-service.log' | tar xf - -C $d && ls -lh $d
```

Still having issues? Contact Inworld support with the exported logs file.

## Benchmarking

See [load_test](https://github.com/inworld-ai/inworld-tts-onprem/tree/main/load_test#quick-start) for performance testing.
