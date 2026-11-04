#!/usr/bin/env bash
# Put the try-it server on Cloud Run. It needs the repo's code and nothing else: no adapters, no runs, no data.
#   PROJECT=my-project SERVICE_ACCOUNT=try@my-project.iam.gserviceaccount.com SECRET=baseten-key BUCKET=my-meter \
#   ALLOW_ORIGINS=https://me.github.io bash demo/deploy.sh
set -euo pipefail
: "${PROJECT:?}" "${SERVICE_ACCOUNT:?}" "${SECRET:?}" "${BUCKET:?}" "${ALLOW_ORIGINS:?}"
root="$(cd "$(dirname "$0")/.." && pwd)"
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/demo"
cp -r "$root/workalike" "$root/testbed" "$ctx/"
cp "$root"/demo/*.py "$root/demo/tickets.jsonl" "$ctx/demo/"
cp "$root/demo/Dockerfile" "$ctx/Dockerfile"
find "$ctx" -name __pycache__ -type d -prune -exec rm -rf {} +
# One instance: the limits live in it, and the spending count in the bucket. It waits on the model, so one small machine serves everyone.
gcloud run deploy workalike-try --project "$PROJECT" --region "${REGION:-us-central1}" --source "$ctx" \
  --service-account "$SERVICE_ACCOUNT" --set-secrets "BASETEN_API_KEY=$SECRET:latest" \
  --set-env-vars "^@^ALLOW_ORIGINS=$ALLOW_ORIGINS@METER_BUCKET=$BUCKET" \
  --min-instances 1 --max-instances 1 --concurrency 40 --cpu 1 --memory 512Mi --timeout 180 \
  --allow-unauthenticated --quiet
