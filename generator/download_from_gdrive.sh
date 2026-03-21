#!/usr/bin/env bash
# Usage: ./download_from_gdrive.sh <GDrive_File_ID>
# This script requires `gdown`. Install via: pip install gdown

set -euo pipefail

GDRIVE_ID=${1:-"1XxhjLtyfRkY9FhGwc8KSeBXJQ1wc6lah"}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/unedited_500"
ZIP_PATH="${OUTPUT_DIR}/unedited_videos.zip"

mkdir -p "${OUTPUT_DIR}"
echo "Downloading Google Drive ZIP file (ID: $GDRIVE_ID)..."

# Download using gdown
gdown --id "${GDRIVE_ID}" -O "${ZIP_PATH}"

echo "Extracting..."
unzip -q -o "${ZIP_PATH}" -d "${OUTPUT_DIR}"

# Some archives contain an extra top-level `unedited_500` directory.
# Flatten it so videos always end up directly in ${OUTPUT_DIR}.
if [ -d "${OUTPUT_DIR}/unedited_500" ]; then
    shopt -s dotglob nullglob
    nested_files=("${OUTPUT_DIR}/unedited_500"/*)
    if [ ${#nested_files[@]} -gt 0 ]; then
        mv -f "${OUTPUT_DIR}/unedited_500"/* "${OUTPUT_DIR}/"
    fi
    shopt -u dotglob nullglob
    rmdir "${OUTPUT_DIR}/unedited_500" 2>/dev/null || true
fi

rm -f "${ZIP_PATH}"

echo "Done! Videos are ready in ${OUTPUT_DIR}"
