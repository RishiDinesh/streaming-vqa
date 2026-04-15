#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/backup_streaming_outputs_to_gdrive.sh sync
  scripts/backup_streaming_outputs_to_gdrive.sh watch

Required environment:
  RCLONE_REMOTE   Name of the configured rclone remote, e.g. "gdrive"

Optional environment:
  REMOTE_SUBDIR=streaming-vqa/evaluations_streaming
  SOURCE_DIR=outputs/evaluations_streaming
  INCLUDE_FEATURE_CACHE=0
  INTERVAL_SEC=300
  DELETE=0
  RCLONE_EXTRA_ARGS=""

Examples:
  RCLONE_REMOTE=gdrive bash scripts/backup_streaming_outputs_to_gdrive.sh sync

  RCLONE_REMOTE=gdrive REMOTE_SUBDIR=research/streaming-vqa/results \
    INTERVAL_SEC=120 bash scripts/backup_streaming_outputs_to_gdrive.sh watch

Notes:
  - Excludes feature_cache by default because it is very large.
  - Uses rclone copy by default, so local files are never deleted from source.
  - Set DELETE=1 to mirror deletions with rclone sync instead.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

MODE=$1
RCLONE_REMOTE=${RCLONE_REMOTE:-}
REMOTE_SUBDIR=${REMOTE_SUBDIR:-streaming-vqa/evaluations_streaming}
SOURCE_DIR=${SOURCE_DIR:-outputs/evaluations_streaming}
INCLUDE_FEATURE_CACHE=${INCLUDE_FEATURE_CACHE:-0}
INTERVAL_SEC=${INTERVAL_SEC:-300}
DELETE=${DELETE:-0}
RCLONE_EXTRA_ARGS=${RCLONE_EXTRA_ARGS:-}

case "${MODE}" in
  sync|watch)
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

if [[ -z "${RCLONE_REMOTE}" ]]; then
  echo "RCLONE_REMOTE is required, e.g. RCLONE_REMOTE=gdrive" >&2
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is not installed. Install/configure rclone first." >&2
  exit 1
fi

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

REMOTE_PATH="${RCLONE_REMOTE}:${REMOTE_SUBDIR}"

declare -a RCLONE_ARGS=(
  --create-empty-src-dirs
  --transfers=4
  --checkers=8
  --fast-list
  --exclude "*.tmp"
  --exclude "*.json.tmp"
)

if [[ "${INCLUDE_FEATURE_CACHE}" != "1" ]]; then
  RCLONE_ARGS+=(--exclude "feature_cache/**")
fi

if [[ -n "${RCLONE_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${RCLONE_EXTRA_ARGS} )
  RCLONE_ARGS+=("${EXTRA[@]}")
fi

run_backup() {
  local started_at cmd
  started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "[backup] ${started_at} source=${SOURCE_DIR} remote=${REMOTE_PATH}"
  if [[ "${DELETE}" == "1" ]]; then
    cmd="sync"
  else
    cmd="copy"
  fi
  if rclone "${cmd}" "${SOURCE_DIR}" "${REMOTE_PATH}" "${RCLONE_ARGS[@]}"; then
    echo "[backup] completed $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  else
    echo "[backup] pass failed $(date -u +"%Y-%m-%dT%H:%M:%SZ"), will retry on next cycle"
  fi
}

run_backup

if [[ "${MODE}" == "watch" ]]; then
  while true; do
    sleep "${INTERVAL_SEC}"
    run_backup
  done
fi
