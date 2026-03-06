# inworld-tts Helm Chart

Deploys the Inworld TTS On-Prem service to a Kubernetes cluster.

## Available variants

Inworld provides two model sizes and supports multiple GPU types. Contact your Inworld
representative to confirm which image is included in your agreement.

| Variant | Model | `image.repository` |
|---------|-------|--------------------|
| Mini | inworld-tts-1.5-mini | `inworld-ai-registry/backend/tts-1.5-mini-h100-onprem` |
| Max | inworld-tts-1.5-max | `inworld-ai-registry/backend/tts-1.5-max-h100-onprem` |

GPU types other than H100 are available by request.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Kubernetes | 1.25+ |
| Helm | 3.10+ |
| GPU | As specified in your Inworld agreement |
| NVIDIA device plugin | [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) running in the cluster |
| NVIDIA Container Toolkit | Installed on GPU nodes (`nvidia-container-toolkit`) |
| GCP service account key | Your own GCP service account key (Inworld grants it the required permissions) |
| Outbound HTTPS | GPU nodes must reach `*.googleapis.com:443` |

> **OpenShift / restricted PSA**: the container starts as root briefly to set up credentials,
> then drops to an unprivileged user. You must allow the `baseline` Pod Security
> Admission policy on the namespace, or grant the `anyuid` SCC on OpenShift.

---

## Required values

Three values **must** be set — the chart will refuse to render without them.

| Value | Description | Example |
|-------|-------------|---------|
| `image.tag` | Image tag provided by Inworld for your release | `20240301-ab12cd34` |
| `image.repository` | Set to match your variant (see table above) | `inworld-ai-registry/backend/tts-1.5-mini-h100-onprem` |
| `config.customerId` | Your customer ID, provided by Inworld | `onprem-metering-acme-corp` |
| `credentials.inlineKey` **or** `credentials.existingSecret` | GCP service account key (see below) | — |

---

## Preparing the GCP credentials

Inworld grants your GCP service account the permissions needed to use the metering
API. You supply your own GCP service account key (`sa-key.json`) and make it
available in Kubernetes.

Choose Option A (simplest) or Option B (recommended for production) below.

---

## Installation

> All commands below should be run from the **repository root** (`inworld-tts-onprem/`).

### Option A — Chart manages the Secret (simplest)

```bash
helm install inworld-tts ./helm \
  --set image.tag=20240301-ab12cd34 \
  --set config.customerId=onprem-metering-acme-corp \
  --set-file credentials.inlineKey=./sa-key.json
```

The chart creates a Kubernetes Secret containing the key. The key is also stored
(base64-encoded) in the Helm release history. If your security policy prohibits
secrets in Helm history, use Option B instead.

### Option B — Pre-existing Secret (recommended for production)

Create a Kubernetes Secret from your GCP service account key file:

```bash
kubectl create secret generic inworld-tts-gcp-creds \
  --from-file=key.json=./sa-key.json
```

Then install the chart referencing that Secret:

```bash
helm install inworld-tts ./helm \
  --set image.tag=20240301-ab12cd34 \
  --set config.customerId=onprem-metering-acme-corp \
  --set credentials.existingSecret=inworld-tts-gcp-creds
```

### Namespace

Install into a dedicated namespace (recommended):

```bash
kubectl create namespace inworld-tts
helm install inworld-tts ... --namespace inworld-tts
```

---

## Targeting GPU nodes

Kubernetes will not schedule the pod on a node without an available GPU
(the resource request ensures this). You only need additional scheduling
configuration in these cases:

- **`nodeSelector`** — if your cluster has multiple GPU types and you need to
  target H100 nodes specifically
- **`tolerations`** — if your GPU nodes carry a `NoSchedule` taint to keep
  other workloads off them

Examples:

```bash
# By node label
helm install inworld-tts ... \
  --set nodeSelector."cloud\.google\.com/gke-accelerator"=nvidia-h100-80gb

# By taint (common on dedicated GPU node pools)
helm install inworld-tts ... \
  --set "tolerations[0].key=nvidia.com/gpu" \
  --set "tolerations[0].operator=Exists" \
  --set "tolerations[0].effect=NoSchedule"
```

Or put these in a `values.yaml` file:

```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-h100-80gb

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

---

## Topology Manager (recommended)

Enabling the Kubernetes [Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)
on your GPU nodes ensures that the CPU cores, memory, and GPU assigned to the pod
all reside on the same NUMA node. For inference workloads this matters because:

- **Lower latency** — data transferred between the GPU and CPUs on the same NUMA
  node avoids crossing the inter-socket interconnect
- **Higher PCIe bandwidth** — the GPU is directly attached to one CPU socket;
  cross-NUMA access goes through a slower path
- **Predictable performance** — without topology alignment, inference latency can
  vary significantly depending on which NUMA node the scheduler happens to assign

To enable it, set `--topology-manager-policy=single-numa-node` in the kubelet
configuration on your GPU nodes. No changes to the Helm chart are needed — the
pod's resource requests (GPU + CPU + memory) are sufficient for the Topology
Manager to apply NUMA alignment automatically.

---

## Startup time

The container loads TensorRT models on startup, which takes **~6 minutes**.
`helm install --wait` will block until the service is ready:

```bash
helm install inworld-tts ... --wait --timeout 10m
```

The pod will show `0/1 Running` during model load — this is expected.

---

## Verifying the deployment

```bash
# Check pod is ready
kubectl get pod -l app.kubernetes.io/name=inworld-tts

# Run the built-in connectivity test
helm test inworld-tts --timeout 5m

# Hit the API directly via port-forward
kubectl port-forward svc/inworld-tts 8081:8081 9030:9030
curl http://localhost:8081/tts/v1/voices
```

---

## Upgrading

```bash
helm upgrade inworld-tts ./helm \
  --reuse-values \
  --set image.tag=20240401-cd56ef78
```

The chart uses `strategy: Recreate`, so the old pod is terminated before the new
one starts. Expect ~6 minutes of downtime during the model reload.

If you rotate the GCP service account key, update the Secret and then restart
the pod to pick up the new credentials:

```bash
# Option A (chart-managed Secret)
helm upgrade inworld-tts ... --set-file credentials.inlineKey=./new-sa-key.json

# Option B (pre-existing Secret)
kubectl create secret generic inworld-tts-gcp-creds \
  --from-file=key.json=./new-sa-key.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/inworld-tts
```

---

## Values reference

| Value | Default | Description |
|-------|---------|-------------|
| `image.registry` | `us-central1-docker.pkg.dev` | Container registry |
| `image.repository` | `inworld-ai-registry/backend/tts-1.5-mini-h100-onprem` | Image repository |
| `image.tag` | `""` | **Required.** Image tag provided by Inworld |
| `image.pullPolicy` | `IfNotPresent` | Kubernetes image pull policy |
| `imagePullSecrets` | `[]` | Registry pull secrets (if your cluster can't reach GCP directly) |
| `config.customerId` | `""` | **Required.** Your Inworld customer ID |
| `credentials.inlineKey` | `""` | GCP SA key JSON (use `--set-file`). Creates a chart-managed Secret. |
| `credentials.existingSecret` | `""` | Name of a pre-existing Secret containing the key. Takes precedence over `inlineKey`. |
| `credentials.existingSecretKey` | `key.json` | Key name inside the Secret |
| `service.type` | `ClusterIP` | Kubernetes Service type (`ClusterIP`, `LoadBalancer`, `NodePort`) |
| `service.httpPort` | `8081` | HTTP REST port |
| `service.grpcPort` | `9030` | gRPC port (h2c / plaintext HTTP/2) |
| `resources.limits` | 1 GPU, 128Gi RAM, 20 CPU | Pod resource limits. Sized for a standard 1×H100 node (e.g. Azure NC40ads_H100_v5: 40 vCPU, 320 GiB). Adjust if your node has different capacity. |
| `resources.requests` | 1 GPU, 128Gi RAM, 20 CPU | Pod resource requests (equal to limits — Guaranteed QoS). |
| `nodeSelector` | `{}` | Node selector for GPU node targeting |
| `tolerations` | `[]` | Tolerations for GPU node taints |
| `affinity` | `{}` | Pod affinity/anti-affinity rules |
| `replicaCount` | `1` | Number of replicas — requires one H100 per replica |
| `podAnnotations` | `{}` | Extra annotations on the pod |
| `podLabels` | `{}` | Extra labels on the pod |
| `serviceAccount.create` | `false` | Create a ServiceAccount |
| `serviceAccount.name` | `""` | ServiceAccount name to use or create |
| `nameOverride` | `""` | Override the chart name |
| `fullnameOverride` | `""` | Override the full release name |

---

## Exposing the service externally

The chart creates a `ClusterIP` Service by default. To expose it:

**LoadBalancer** (cloud clusters):
```bash
helm upgrade inworld-tts ... --set service.type=LoadBalancer
```

**Ingress** — note that the gRPC port (9030) uses plaintext HTTP/2 (h2c).
Your Ingress controller must support h2c backends. With NGINX:
```yaml
nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
nginx.ingress.kubernetes.io/grpc-backend: "true"
```

---

## Uninstalling

```bash
helm uninstall inworld-tts

# If you used Option A, the Secret is removed with the release.
# If you used Option B, delete the Secret separately:
kubectl delete secret inworld-tts-gcp-creds
```
