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
#   ./run.sh diagnose         Print system diagnostics (GPU, drivers, OS, etc.)
#
# All configuration is read from onprem.env. CLI flags override env file values:
#   ./run.sh --customer-id <id> --image <url> --key <path>
#   ./run.sh --api-key <value> --customer-id <id> --image <url> --key <path> [--api-endpoint <url>] [--enable-auth-validation]

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
# Subcommands: stop, status, logs, restart, diagnose
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
    diagnose)
        echo "========================================"
        echo " Inworld TTS On-Prem Diagnostics"
        echo "========================================"
        echo ""

        # -- Kernel & OS --
        info "Kernel & OS"
        echo "  Kernel:   $(uname -r)"
        echo "  Arch:     $(uname -m)"
        if [ -f /etc/os-release ]; then
            OS_NAME=$(. /etc/os-release && echo "${NAME:-unknown}")
            OS_VERSION=$(. /etc/os-release && echo "${VERSION:-unknown}")
            echo "  OS:       $OS_NAME $OS_VERSION"
        elif command -v lsb_release &>/dev/null; then
            echo "  OS:       $(lsb_release -ds 2>/dev/null)"
        else
            echo "  OS:       $(uname -s) (no /etc/os-release)"
        fi
        echo ""

        # -- CPU --
        info "CPU"
        if command -v lscpu &>/dev/null; then
            CPU_MODEL=$(lscpu | grep -m1 'Model name' | sed 's/.*:\s*//')
            CPU_CORES=$(lscpu | grep -m1 '^CPU(s):' | sed 's/.*:\s*//')
            echo "  Model:    $CPU_MODEL"
            echo "  CPUs:     $CPU_CORES"
        else
            echo "  $(uname -p)"
        fi
        echo ""

        # -- Memory --
        info "Memory"
        if command -v free &>/dev/null; then
            free -h | awk '/^Mem:/ {printf "  Total: %s   Used: %s   Available: %s\n", $2, $3, $7}'
        else
            warn "  free command not available"
        fi
        echo ""

        # -- Disk --
        info "Disk"
        echo "  Root partition:"
        df -h / 2>/dev/null | awk 'NR==2 {printf "    Size: %s   Used: %s   Avail: %s   Use%%: %s\n", $2, $3, $4, $5}'
        DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "/var/lib/docker")
        if [ "$DOCKER_ROOT" != "/" ]; then
            echo "  Docker partition ($DOCKER_ROOT):"
            df -h "$DOCKER_ROOT" 2>/dev/null | awk 'NR==2 {printf "    Size: %s   Used: %s   Avail: %s   Use%%: %s\n", $2, $3, $4, $5}'
        fi
        echo ""

        # -- NVIDIA GPU & Drivers --
        info "NVIDIA GPU & Drivers"
        if command -v nvidia-smi &>/dev/null; then
            DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9.]+' | head -1)
            GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
            KMOD_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oP 'Kernel Module\s+\K[0-9.]+' || modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}' || echo "unknown")
            echo "  Driver:   $DRIVER_VER"
            echo "  Kernel:   $KMOD_VER"
            echo "  CUDA:     ${CUDA_VER:-unknown}"
            echo "  GPU(s):   $GPU_COUNT"
            echo ""
            nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu \
                --format=csv,noheader 2>/dev/null | while IFS=',' read -r IDX NAME MEM_TOT MEM_USED TEMP UTIL; do
                echo "  [$IDX] $NAME"
                echo "        Memory: $MEM_USED /$MEM_TOT   Temp: ${TEMP}C   Util: $UTIL"
            done
        else
            err "nvidia-smi not found -- cannot query GPU or driver info"
        fi
        echo ""

        # -- Container Runtime --
        info "Container Runtime"
        if command -v docker &>/dev/null; then
            echo "  $(docker --version)"
            DOCKER_SERVER_VER=$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo "unknown")
            STORAGE_DRIVER=$(docker info --format '{{.Driver}}' 2>/dev/null || echo "unknown")
            CGROUP_DRIVER=$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo "unknown")
            echo "  Server:   $DOCKER_SERVER_VER"
            echo "  Storage:  $STORAGE_DRIVER"
            echo "  Cgroup:   $CGROUP_DRIVER"
            if docker info 2>/dev/null | grep -q "nvidia"; then
                ok "  NVIDIA runtime: detected"
            else
                warn "  NVIDIA runtime: not detected"
            fi
        else
            warn "Docker is not installed"
        fi
        if command -v containerd &>/dev/null; then
            echo "  containerd: $(containerd --version 2>/dev/null | awk '{print $3}')"
        fi
        if command -v crio &>/dev/null; then
            echo "  CRI-O:    $(crio --version 2>/dev/null | awk '/^crio version/{print $3}')"
        fi
        echo ""

        # -- NVIDIA Container Toolkit --
        info "NVIDIA Container Toolkit"
        if command -v nvidia-container-cli &>/dev/null; then
            echo "  $(nvidia-container-cli --version 2>/dev/null | head -1)"
        elif command -v nvidia-ctk &>/dev/null; then
            echo "  $(nvidia-ctk --version 2>/dev/null | head -1)"
        else
            warn "nvidia-container-cli / nvidia-ctk not found in PATH"
        fi
        echo ""

        echo "========================================"
        echo " End of diagnostics"
        echo "========================================"
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
        echo "Usage: $0 [start|stop|status|logs|restart|diagnose]"
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
        --customer-id)              INWORLD_CUSTOMER_ID="$2"; shift 2 ;;
        --image)                    TTS_IMAGE="$2"; shift 2 ;;
        --key)                      KEY_FILE="$2"; shift 2 ;;
        --api-key)                  INWORLD_API_KEY="$2"; shift 2 ;;
        --api-endpoint)             INWORLD_API_ENDPOINT="$2"; shift 2 ;;
        --enable-auth-validation)   INWORLD_ENABLE_AUTH_VALIDATION="true"; shift ;;
        *) shift ;;
    esac
done

# Defaults for API key mode
INWORLD_API_ENDPOINT="${INWORLD_API_ENDPOINT:-https://api.inworld.ai}"
INWORLD_ENABLE_AUTH_VALIDATION="${INWORLD_ENABLE_AUTH_VALIDATION:-false}"

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
# Determine metering mode: API key (dual) or customer ID only (legacy)
# =============================================================================
echo ""
info "Validating configuration..."

if [ -n "${INWORLD_API_KEY:-}" ]; then
    METERING_MODE="apikey"
else
    METERING_MODE="legacy"
fi

ERRORS=0

# TTS_IMAGE is always required
if [ -z "$TTS_IMAGE" ]; then
    err "TTS_IMAGE is not set. Edit $ENV_FILE or pass --image <url>"
    ERRORS=$((ERRORS + 1))
fi

# Customer ID and GCP credentials are required for ALL modes.
# In API key mode the container runs dual metering (gRPC + PubSub) for
# BigQuery comparison during the verification period.
if [ -z "${INWORLD_CUSTOMER_ID:-}" ]; then
    if [ "$METERING_MODE" = "apikey" ]; then
        err "INWORLD_CUSTOMER_ID is required alongside INWORLD_API_KEY during the verification period."
        err "  Both are needed: gRPC metering (new) + PubSub metering (legacy, for BigQuery comparison)."
    else
        err "INWORLD_CUSTOMER_ID is not set. Edit $ENV_FILE or pass --customer-id <id>"
    fi
    ERRORS=$((ERRORS + 1))
fi

if [ -z "${KEY_FILE:-}" ]; then
    if [ "$METERING_MODE" = "apikey" ]; then
        err "KEY_FILE is required alongside INWORLD_API_KEY during the verification period."
        err "  GCP credentials are needed for PubSub-based metering (BigQuery comparison)."
    else
        err "KEY_FILE is not set. Edit $ENV_FILE or pass --key <path>"
    fi
    ERRORS=$((ERRORS + 1))
elif [ ! -f "$KEY_FILE" ]; then
    err "Key file not found: $KEY_FILE"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    err "Fix the above errors and try again."
    echo "  Tip: cp onprem.env.example onprem.env && vi onprem.env"
    exit 1
fi

# Display configuration summary
check "Image: $TTS_IMAGE"
check "Metering mode: $METERING_MODE"
check "Customer ID: $INWORLD_CUSTOMER_ID"
check "Key file: $KEY_FILE"

if [ "$METERING_MODE" = "apikey" ]; then
    check "API endpoint: $INWORLD_API_ENDPOINT"
    check "Auth validation: $INWORLD_ENABLE_AUTH_VALIDATION"

    # Validate API key at startup if auth validation is enabled
    if [ "$INWORLD_ENABLE_AUTH_VALIDATION" = "true" ]; then
        info "Validating API key against $INWORLD_API_ENDPOINT..."
        HTTP_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" \
            -H "Authorization: Basic $INWORLD_API_KEY" \
            "$INWORLD_API_ENDPOINT/tts/v1/voices" 2>/dev/null) || HTTP_STATUS="000"

        if [ "$HTTP_STATUS" = "200" ]; then
            check "API key: valid"
        elif [ "$HTTP_STATUS" = "401" ] || [ "$HTTP_STATUS" = "403" ]; then
            err "API key validation failed (HTTP $HTTP_STATUS). Check your INWORLD_API_KEY."
            exit 1
        else
            warn "API key validation returned HTTP $HTTP_STATUS (endpoint may be unreachable). Proceeding anyway."
        fi
    else
        check "API key: configured (auth validation disabled)"
    fi
fi

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

# Build docker run arguments based on metering mode
DOCKER_ENV_ARGS="-e INWORLD_CUSTOMER_ID=$INWORLD_CUSTOMER_ID"

if [ "$METERING_MODE" = "apikey" ]; then
    # Dual-metering (verification period): pass both API key and legacy credentials.
    # The container entrypoint runs gRPC metering (new) + PubSub metering (legacy)
    # in parallel for BigQuery comparison.
    DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e INWORLD_API_KEY=$INWORLD_API_KEY"
    DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e INWORLD_API_ENDPOINT=$INWORLD_API_ENDPOINT"
    DOCKER_ENV_ARGS="$DOCKER_ENV_ARGS -e INWORLD_METERING_MODE=grpc"
fi

# shellcheck disable=SC2086
docker run -d \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -p 8081:8081 \
    -p 9030:9030 \
    $DOCKER_ENV_ARGS \
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
