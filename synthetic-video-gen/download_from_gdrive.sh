#!/bin/bash
# Usage: ./download_from_gdrive.sh <GDrive_File_ID>
# This script requires `gdown`. Install via: pip install gdown

GDRIVE_ID=${1:-"1XxhjLtyfRkY9FhGwc8KSeBXJQ1wc6lah"}
OUTPUT_DIR="source_videos/unedited_500"

mkdir -p $OUTPUT_DIR
echo "Downloading Google Drive ZIP file (ID: $GDRIVE_ID)..."

# Download using gdown
gdown --id $GDRIVE_ID -O unedited_videos.zip

echo "Extracting..."
unzip -q unedited_videos.zip -d $OUTPUT_DIR
rm unedited_videos.zip

echo "Done! Videos are ready in $OUTPUT_DIR"
