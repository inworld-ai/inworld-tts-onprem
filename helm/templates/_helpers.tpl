{{/*
Expand the name of the chart.
*/}}
{{- define "inworld-tts.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "inworld-tts.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label.
*/}}
{{- define "inworld-tts.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "inworld-tts.labels" -}}
helm.sh/chart: {{ include "inworld-tts.chart" . }}
{{ include "inworld-tts.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "inworld-tts.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inworld-tts.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "inworld-tts.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "inworld-tts.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Full image reference: registry/repository:tag
*/}}
{{- define "inworld-tts.image" -}}
{{- printf "%s/%s:%s" .Values.image.registry .Values.image.repository (required "image.tag is required" .Values.image.tag) }}
{{- end }}

{{/*
Resolve the name of the Secret holding the GCP credentials.
Precedence: existingSecret > chart-managed secret (derived from fullname).
*/}}
{{- define "inworld-tts.credentialSecretName" -}}
{{- if .Values.credentials.existingSecret }}
{{- .Values.credentials.existingSecret }}
{{- else }}
{{- printf "%s-gcp-credentials" (include "inworld-tts.fullname" .) }}
{{- end }}
{{- end }}
