#!/usr/bin/env bash
# Upload Kestra flows (*.yml) via the REST API import endpoint.
# Works on Linux, macOS and Git Bash/WSL on Windows. Requires curl.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

KESTRA_URL="${KESTRA_URL:-http://localhost:8082}"
KESTRA_USERNAME="${KESTRA_BASIC_AUTH_USERNAME:-}"
KESTRA_PASSWORD="${KESTRA_BASIC_AUTH_PASSWORD:-}"
FLOW_FILE=""
FLOWS_DIR="$ROOT_DIR/kestra/flows"

usage() {
  cat <<EOF
Usage: $0 [options]

Uploads all flows from kestra/flows/*.yml (or a single file) to a running
Kestra server via POST /api/v1/flows/import (multipart, idempotent upsert).

Options:
  --file <path>       Upload a single flow file instead of the whole directory
  --url <url>         Kestra base URL (default: \$KESTRA_URL or http://localhost:8082)
  --username <user>   Basic auth username (default: \$KESTRA_BASIC_AUTH_USERNAME or .env)
  --password <pass>   Basic auth password (default: \$KESTRA_BASIC_AUTH_PASSWORD or .env)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) FLOW_FILE="$2"; shift 2 ;;
    --url) KESTRA_URL="$2"; shift 2 ;;
    --username) KESTRA_USERNAME="$2"; shift 2 ;;
    --password) KESTRA_PASSWORD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

env_value() {
  local key="$1" fallback="$2"
  local val="$fallback"
  if [[ -f "$ENV_FILE" ]]; then
    local found
    found="$(sed -nE "s/^${key}=(\"?)(.*)\1$/\2/p" "$ENV_FILE" | head -n1)"
    [[ -n "$found" ]] && val="$found"
  fi
  printf '%s' "$val"
}

KESTRA_URL="$(env_value KESTRA_URL "$KESTRA_URL")"
KESTRA_USERNAME="$(env_value KESTRA_BASIC_AUTH_USERNAME "$KESTRA_USERNAME")"
KESTRA_PASSWORD="$(env_value KESTRA_BASIC_AUTH_PASSWORD "$KESTRA_PASSWORD")"

if [[ -z "$KESTRA_USERNAME" || -z "$KESTRA_PASSWORD" ]]; then
  echo "ERROR: KESTRA_BASIC_AUTH_USERNAME and KESTRA_BASIC_AUTH_PASSWORD are required (env vars or $ENV_FILE)" >&2
  exit 1
fi

import_flow() {
  local file="$1"
  local curl_path="$file"
  if command -v cygpath >/dev/null 2>&1; then
    curl_path="$(cygpath -m "$file")"
  fi
  echo "Importing: $file"
  curl --silent --show-error --fail-with-body \
    -u "$KESTRA_USERNAME:$KESTRA_PASSWORD" \
    -F "fileUpload=@$curl_path;type=application/yaml" \
    "$KESTRA_URL/api/v1/flows/import"
  echo
}

if [[ -n "$FLOW_FILE" ]]; then
  import_flow "$FLOW_FILE"
else
  shopt -s nullglob
  for file in "$FLOWS_DIR"/*.yml; do
    import_flow "$file"
  done
fi

echo "Done."
