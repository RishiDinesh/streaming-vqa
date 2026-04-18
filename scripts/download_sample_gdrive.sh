#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./download_sample_from_gdrive.sh <gdrive-file-id-or-url> [output_path]
#   GDRIVE_FILE_ID=<gdrive-file-id> ./download_sample_from_gdrive.sh
#
# Before using this, make sure the Google Drive file is shared so anyone with
# the link can download it.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_PATH="${SCRIPT_DIR}/sample.mp4"

extract_file_id() {
    local input="$1"

    if [[ "$input" =~ ^https?:// ]]; then
        if [[ "$input" =~ /file/d/([^/]+) ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi

        if [[ "$input" =~ [\?\&]id=([^&]+) ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi

        return 1
    fi

    printf '%s\n' "$input"
}

INPUT_REF="${1:-${GDRIVE_FILE_ID:-}}"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}}"

if [[ -z "${INPUT_REF}" ]]; then
    cat <<EOF
Usage:
  ./download_sample_from_gdrive.sh <gdrive-file-id-or-url> [output_path]

Examples:
  ./download_sample_from_gdrive.sh 1AbCdEfGhIjKlMnOpQrStUvWxYz
  ./download_sample_from_gdrive.sh "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
  GDRIVE_FILE_ID=1AbCdEfGhIjKlMnOpQrStUvWxYz ./download_sample_from_gdrive.sh
EOF
    exit 1
fi

if ! FILE_ID="$(extract_file_id "${INPUT_REF}")"; then
    echo "Could not extract a Google Drive file ID from: ${INPUT_REF}" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_PATH}")"

DOWNLOAD_URL="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t"

echo "Downloading Google Drive file ${FILE_ID}..."

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "${DOWNLOAD_URL}" -o "${OUTPUT_PATH}"
elif command -v wget >/dev/null 2>&1; then
    wget --progress=bar:force -O "${OUTPUT_PATH}" "${DOWNLOAD_URL}"
elif command -v python3 >/dev/null 2>&1; then
    python3 - "${DOWNLOAD_URL}" "${OUTPUT_PATH}" <<'PY'
import shutil
import sys
import urllib.request

url, output_path = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as response, open(output_path, "wb") as output_file:
    shutil.copyfileobj(response, output_file)
PY
else
    echo "No downloader available. Install curl, wget, or python3." >&2
    exit 1
fi

echo "Saved to ${OUTPUT_PATH}"