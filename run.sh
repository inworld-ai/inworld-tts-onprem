#!/bin/bash
# TTS On-Prem Launcher
# Single-command setup and lifecycle management for the TTS container.
#
# Usage:
#   ./run.sh                  Start the container (reads config from onprem.env)
#   ./run.sh stop             Stop the container
#   ./run.sh status           Check container status and health
#   ./run.sh logs             Show recent logs from all services
#   ./run.sh logs -f          Tail all service logs live
#   ./run.sh logs export      Export all logs to a timestamped folder
#   ./run.sh restart          Restart the container
#
# All configuration is read from onprem.env. CLI flags override env file values:
#   ./run.sh --customer-id <id> --image <url> --key <path>

set -e

CONTAINER_NAME="inworld-tts-onprem"
ENV_FILE="onprem.env"
HEALTH_URL="http://localhost:8081/tts/v1/voices"
HEALTH_TIMEOUT=360  # 6 minutes max wait

# =============================================================================
# Colors and output helpers
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*"; }
check() { echo -e "${GREEN}[check]${NC} $*"; }

# =============================================================================
# Subcommands: stop, status, logs, restart
# =============================================================================
case "${1:-}" in
    stop)
        info "Stopping $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME" 2>/dev/null && docker rm "$CONTAINER_NAME" 2>/dev/null
        ok "Container stopped and removed."
        exit 0
        ;;
    status)
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
            ok "Container is running (health: $HEALTH)"
            docker exec "$CONTAINER_NAME" supervisorctl -s unix:///tmp/supervisor.sock status 2>/dev/null || true
        else
            warn "Container is not running."
        fi
        exit 0
        ;;
    logs)
        LOG_FILES="supervisord.log tts-v3-trtllm.log tts-normalization.log forced-alignment.log public-tts-service.log w-proxy.log grpc-gateway.log"
        if [ "${2:-}" = "export" ]; then
            DIR="tts-logs-$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$DIR"
            docker exec "$CONTAINER_NAME" sh -c "cd /var/log && tar cf - $LOG_FILES" | tar xf - -C "$DIR"
            ok "Logs exported to $DIR/"
            ls -lh "$DIR"
        elif [ "${2:-}" = "-f" ]; then
            docker exec "$CONTAINER_NAME" tail -f \
                /var/log/supervisord.log \
                /var/log/tts-v3-trtllm.log \
                /var/log/tts-normalization.log \
                /var/log/forced-alignment.log \
                /var/log/public-tts-service.log \
                /var/log/w-proxy.log \
                /var/log/grpc-gateway.log
        else
            for LOG in $LOG_FILES; do
                echo ""
                echo "=== $LOG (last 20 lines) ==="
                docker exec "$CONTAINER_NAME" tail -20 "/var/log/$LOG" 2>/dev/null || echo "(not available)"
            done
        fi
        exit 0
        ;;
    restart)
        info "Restarting $CONTAINER_NAME..."
        docker restart "$CONTAINER_NAME"
        ok "Container restarted."
        exit 0
        ;;
    -*)
        # Flag -- fall through to start logic
        ;;
    "")
        # No args -- fall through to start logic
        ;;
    *)
        err "Unknown command: $1"
        echo "Usage: $0 [start|stop|status|logs|restart]"
        exit 1
        ;;
esac

# =============================================================================
# Load config from env file
# =============================================================================
if [ -f "$ENV_FILE" ]; then
    info "Reading config from $ENV_FILE"
    # Source env file (skip comments and empty lines)
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$')
    set +a
fi

# =============================================================================
# Parse CLI flags (override env file values)
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --customer-id) INWORLD_CUSTOMER_ID="$2"; shift 2 ;;
        --image)       TTS_IMAGE="$2"; shift 2 ;;
        --key)         KEY_FILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# =============================================================================
# Prerequisite checks
# =============================================================================
echo ""
info "Checking prerequisites..."

# Docker
if ! command -v docker &>/dev/null; then
    err "Docker is not installed. See https://docs.docker.com/engine/install/"
    exit 1
fi
check "Docker: $(docker --version | head -1)"

# NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    check "NVIDIA GPU: $GPU_NAME"
else
    warn "nvidia-smi not found. Make sure NVIDIA Container Toolkit is installed."
    warn "See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# NVIDIA Container Toolkit
if docker info 2>/dev/null | grep -q "nvidia"; then
    check "NVIDIA Container Toolkit: OK"
else
    warn "NVIDIA runtime not detected in Docker. GPU support may not work."
fi

# =============================================================================
# Validate configuration
# =============================================================================
echo ""
info "Validating configuration..."

ERRORS=0

if [ -z "$INWORLD_CUSTOMER_ID" ]; then
    err "INWORLD_CUSTOMER_ID is not set. Edit $ENV_FILE or pass --customer-id <id>"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$TTS_IMAGE" ]; then
    err "TTS_IMAGE is not set. Edit $ENV_FILE or pass --image <url>"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$KEY_FILE" ]; then
    err "KEY_FILE is not set. Edit $ENV_FILE or pass --key <path>"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    err "Fix the above errors and try again."
    echo "  Tip: cp onprem.env.example onprem.env && vi onprem.env"
    exit 1
fi

# Validate key file exists
if [ ! -f "$KEY_FILE" ]; then
    err "Key file not found: $KEY_FILE"
    exit 1
fi

check "Customer ID: $INWORLD_CUSTOMER_ID"
check "Image: $TTS_IMAGE"
check "Key file: $KEY_FILE"

# =============================================================================
# Check if container already exists
# =============================================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container $CONTAINER_NAME is already running."
        echo "  Use '$0 stop' to stop it first, or '$0 restart' to restart."
        exit 1
    else
        info "Removing stopped container $CONTAINER_NAME..."
        docker rm "$CONTAINER_NAME" >/dev/null
    fi
fi

# =============================================================================
# Pull image if not present
# =============================================================================
echo ""
if ! docker image inspect "$TTS_IMAGE" &>/dev/null; then
    info "Pulling image: $TTS_IMAGE"
    docker pull "$TTS_IMAGE"
else
    info "Image already available locally."
fi

# =============================================================================
# Start container
# =============================================================================
echo ""
info "Starting container..."

# shellcheck disable=SC2086
docker run -d \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -p 8081:8081 \
    -p 9030:9030 \
    -e INWORLD_CUSTOMER_ID="$INWORLD_CUSTOMER_ID" \
    -v "$(realpath "$KEY_FILE"):/app/gcp-credentials/.mounted-key.json:ro" \
    ${DOCKER_EXTRA_ARGS:-} \
    "$TTS_IMAGE" >/dev/null

ok "Container started."

# =============================================================================
# Wait for health
# =============================================================================
echo ""
info "Waiting for TTS services to be ready (this takes ~3 minutes)..."

ELAPSED=0
INTERVAL=10
while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        echo ""
        ok "TTS On-Prem is ready!"
        echo ""
        info "Test it:"
        echo "  curl -X POST http://localhost:8081/tts/v1/voice \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"text\":\"Hello world\",\"voice_id\":\"Craig\",\"model_id\":\"inworld-tts-1.5-mini\",\"audio_config\":{\"audio_encoding\":\"LINEAR16\",\"sample_rate_hertz\":48000}}'"
        echo ""
        info "List voices: curl http://localhost:8081/tts/v1/voices"
        info "View logs:   $0 logs"
        info "Stop:        $0 stop"
        exit 0
    fi

    # Check if container is still running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo ""
        err "Container stopped unexpectedly. Check logs:"
        echo "  docker logs $CONTAINER_NAME"
        exit 1
    fi

    printf "."
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
warn "Container is running but health check timed out after ${HEALTH_TIMEOUT}s."
warn "The ML model may still be loading. Check status:"
echo "  $0 status"
echo "  docker logs $CONTAINER_NAME"
exit 1
