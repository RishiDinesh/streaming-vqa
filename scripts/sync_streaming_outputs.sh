#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_streaming_outputs.sh sync <source> <destination>
  scripts/sync_streaming_outputs.sh watch <source> <destination>

Examples:
  # Run on your laptop: keep pulling results from the remote droplet every 2 minutes.
  INTERVAL_SEC=120 scripts/sync_streaming_outputs.sh watch \
    user@droplet:/workspace/streaming-vqa/outputs/evaluations_streaming/ \
    ./backups/evaluations_streaming/

  # Run once on the remote machine: copy results to another mounted disk.
  scripts/sync_streaming_outputs.sh sync \
    outputs/evaluations_streaming/ \
    /mnt/persistent/streaming-vqa/evaluations_streaming/

Behavior:
  - Uses rsync so only changed files are transferred after the first sync.
  - Excludes feature_cache by default because it is very large.
  - Preserves partial JSONs, plots, and qualitative bundles.

Environment overrides:
  INTERVAL_SEC=300          Poll interval for watch mode.
  INCLUDE_FEATURE_CACHE=0   Set to 1 to include outputs/evaluations_streaming/feature_cache.
  DELETE=0                  Set to 1 to delete files at destination that no longer exist at source.
  RSYNC_EXTRA_ARGS=""       Extra flags passed to rsync.
EOF
}

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 1
fi

MODE=$1
SOURCE=$2
DESTINATION=$3
INTERVAL_SEC=${INTERVAL_SEC:-300}
INCLUDE_FEATURE_CACHE=${INCLUDE_FEATURE_CACHE:-0}
DELETE=${DELETE:-0}
RSYNC_EXTRA_ARGS=${RSYNC_EXTRA_ARGS:-}

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in PATH." >&2
  exit 1
fi

case "${MODE}" in
  sync|watch)
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

declare -a RSYNC_ARGS=(
  -az
  --partial
  --info=stats1,progress2
  --exclude=.DS_Store
  --exclude=*.tmp
  --exclude=*.json.tmp
)

if [[ "${INCLUDE_FEATURE_CACHE}" != "1" ]]; then
  RSYNC_ARGS+=(--exclude=feature_cache/)
fi

if [[ "${DELETE}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ -n "${RSYNC_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${RSYNC_EXTRA_ARGS} )
  RSYNC_ARGS+=("${EXTRA[@]}")
fi

ensure_local_parent() {
  local target=$1
  if [[ "${target}" == *:* ]]; then
    return 0
  fi
  mkdir -p "${target}"
}

run_sync() {
  local started_at
  started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "[sync] ${started_at} source=${SOURCE} dest=${DESTINATION}"
  ensure_local_parent "${DESTINATION}"
  rsync "${RSYNC_ARGS[@]}" "${SOURCE}" "${DESTINATION}"
  echo "[sync] completed $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}

run_sync

if [[ "${MODE}" == "watch" ]]; then
  while true; do
    sleep "${INTERVAL_SEC}"
    run_sync
  done
fi
