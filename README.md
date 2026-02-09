<div align="center">

![Inworld Logo](https://github.com/inworld-ai/inworld-tts-onprem/blob/main/assets/cover.jpg?raw=true)

# TTS _On-Premise_

</div>

## Prerequisites

### Hardware

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA H100 SXM5 (80GB) |
| **RAM** | 64GB+ system memory |
| **CPU** | 8+ cores |
| **Disk** | 50GB free space |
| **OS** | Ubuntu 22.04 LTS |

### Software

| Software | Installation |
|----------|-------------|
| **Docker** | [Install Docker Engine](https://docs.docker.com/engine/install/) |
| **NVIDIA Container Toolkit** | [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **Google Cloud SDK** | [Install gcloud CLI](https://cloud.google.com/sdk/docs/install) |

Optionally, add your user to the `docker` group so you can run Docker without `sudo`:
[Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)

### Verification

Verify that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

You should see your H100 GPU listed in the output.

## Latest Version
**Version:** 20260208-7cf208d1

## Quick Start

### 1. Create a GCP service account

Create a service account in your GCP project and generate a key file:

```bash
# Create the service account
gcloud iam service-accounts create inworld-tts-onprem \
  --project=<YOUR_GCP_PROJECT> \
  --display-name="Inworld TTS On-Prem" \
  --description="Service account for Inworld TTS on-prem container"

# Create a key file
gcloud iam service-accounts keys create service-account-key.json \
  --iam-account=inworld-tts-onprem@<YOUR_GCP_PROJECT>.iam.gserviceaccount.com \
  --project=<YOUR_GCP_PROJECT>
```

### 2. Share the service account email with Inworld

Send the service account email (e.g., `inworld-tts-onprem@<YOUR_GCP_PROJECT>.iam.gserviceaccount.com`) to your Inworld contact. Inworld will:
- Provide your **Customer ID**

### 3. Authenticate to the container registry

```bash
gcloud auth activate-service-account \
  --key-file=service-account-key.json

gcloud auth configure-docker us-central1-docker.pkg.dev
```

For more authentication options, see [Configure authentication to Artifact Registry for Docker](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper).

### 4. Configure

```bash
cp onprem.env.example onprem.env
```

Edit `onprem.env` with your values:

```bash
INWORLD_CUSTOMER_ID=<your-customer-id>
TTS_IMAGE=us-central1-docker.pkg.dev/inworld-ai-registry/tts-onprem/tts-1.5-mini-h100-onprem:[<version>](https://github.com/inworld-ai/inworld-tts-onprem?tab=readme-ov-file#latest-version)
KEY_FILE=./service-account-key.json
```

### 5. Start

```bash
./run.sh
```

The script will:
1. Check prerequisites (Docker, GPU, NVIDIA Container Toolkit)
2. Validate your configuration
3. Fix key file permissions if needed
4. Pull the Docker image
5. Start the container
6. Wait for services to be ready (~3 minutes)

> **Note:** The ML model takes approximately 3 minutes to load on first startup. This is normal.

## Lifecycle Commands

```bash
./run.sh              # Start the container
./run.sh stop         # Stop and remove the container
./run.sh status       # Check container and service health
./run.sh logs         # Show recent logs from all services
./run.sh logs -f      # Tail all service logs live
./run.sh logs export  # Export all logs to a timestamped folder
./run.sh restart      # Restart the container
```

## API

| Port | Protocol | Description |
|------|----------|-------------|
| **8081** | HTTP | REST API (recommended) |
| **9030** | gRPC | For gRPC clients |

### Health Check

```bash
curl http://localhost:8081/tts/v1/voices
```

### Test Request

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

### Save Audio to File

```bash
curl -X POST http://localhost:8081/tts/v1/voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world! This is a test of the TTS system.",
    "voice_id": "Craig",
    "model_id": "inworld-tts-1.5-mini",
    "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
  }' | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
open('output.wav', 'wb').write(base64.b64decode(d['audioContent']))
print(f'Saved: {len(d[\"audioContent\"])//1000}KB')
"
```

### List Voices

```bash
curl http://localhost:8081/tts/v1/voices
```

### gRPC

For gRPC clients, use port 9030:

```bash
grpcurl -plaintext -d '{
  "text": "Hello, this is a test.",
  "voice_id": "Craig",
  "model_id": "inworld-tts-1.5-mini",
  "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
}' localhost:9030 ai.inworld.tts.v1.TextToSpeech/SynthesizeSpeech
```

For full API documentation, see [Synthesize Speech](https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech).

## Available Images

| Image | Model | GPU |
|-------|-------|-----|
| `tts-1.5-mini-h100-onprem` | 1B (mini) | H100 |
| `tts-1.5-max-h100-onprem` | 8B (max) | H100 |

Registry: `us-central1-docker.pkg.dev/inworld-ai-registry/tts-onprem/`

## Configuration

### onprem.env

| Variable | Required | Description |
|----------|----------|-------------|
| `INWORLD_CUSTOMER_ID` | Yes | Your customer ID |
| `TTS_IMAGE` | Yes | Docker image URL (see [Available Images](#available-images)) |
| `KEY_FILE` | Yes | Path to your GCP service account key file |

## Logs

```bash
# Show recent logs from all services (last 20 lines each)
./run.sh logs

# Tail all service logs live
./run.sh logs -f

# Export all logs to a timestamped folder
./run.sh logs export
```

Individual service logs:

```bash
docker exec inworld-tts-onprem tail -f /var/log/tts-v3-trtllm.log        # ML server
docker exec inworld-tts-onprem tail -f /var/log/tts-normalization.log     # Text normalization
docker exec inworld-tts-onprem tail -f /var/log/public-tts-service.log    # TTS service
docker exec inworld-tts-onprem tail -f /var/log/grpc-gateway.log          # HTTP gateway
docker exec inworld-tts-onprem tail -f /var/log/w-proxy.log               # gRPC proxy
docker exec inworld-tts-onprem tail -f /var/log/supervisord.log           # Supervisor
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "INWORLD_CUSTOMER_ID is required" | Set `INWORLD_CUSTOMER_ID` in `onprem.env` |
| "GCP credentials file not found" | Check that `KEY_FILE` in `onprem.env` points to a valid file |
| "Credentials file is not readable" | Fix permissions on host: `chmod 644 <your-key-file>.json` |
| "Topic not found" | Verify your `INWORLD_CUSTOMER_ID` matches the PubSub topic name |
| "Permission denied for topic" | Ensure Inworld has granted your service account publish access |
| Slow startup (~3 min) | Normal -- text processing grammars take time to initialize |

### Check service status

```bash
docker exec inworld-tts-onprem supervisorctl -s unix:///tmp/supervisor.sock status
```

### Export logs for support

```bash
./run.sh logs export
```

This creates a timestamped folder with all service logs. Share this with Inworld support when reporting issues.

## Advanced: Manual Docker Run

For users who prefer to run Docker directly without `run.sh`:

```bash
docker run -d \
  --gpus all \
  --name inworld-tts-onprem \
  -p 8081:8081 \
  -p 9030:9030 \
  -e INWORLD_CUSTOMER_ID=<your-customer-id> \
  -v $(pwd)/service-account-key.json:/app/gcp-credentials/service-account.json:ro \
  us-central1-docker.pkg.dev/inworld-ai-registry/tts-onprem/tts-1.5-mini-h100-onprem:<version>
```

**Notes:**
- Ensure your key file has 644 permissions: `chmod 644 service-account-key.json`
- The container exposes port 8081 (HTTP) and 9030 (gRPC)
- Use `docker ps` to check container health -- STATUS will show `healthy` when ready

```bash
# Stop and remove
docker stop inworld-tts-onprem && docker rm inworld-tts-onprem

# View logs
docker logs inworld-tts-onprem

# Check service status
docker exec inworld-tts-onprem supervisorctl -s unix:///tmp/supervisor.sock status
```

## Benchmarking

See [load_test](https://github.com/inworld-ai/inworld-tts-onprem/tree/main/load_test#quick-start) for performance testing tools and instructions.
