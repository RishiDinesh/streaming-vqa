#!/usr/bin/env bash
set -euo pipefail

# =========================
# Edit these variables
# =========================
MODEL_ID="llava-hf/llava-onevision-qwen2-7b-ov-hf"
OUTPUT_DIR="models/"
REVISION="main"   # branch/tag/commit
TOKEN=""          # optional; leave empty for public models
DRY_RUN=0          # 1 = print file list only, 0 = download

if [[ -z "$MODEL_ID" ]]; then
  echo "Error: MODEL_ID is empty." >&2
  exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: wget not found. Install wget first." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python not found. Python is required to parse HF metadata." >&2
  exit 1
fi

DEST_DIR="${OUTPUT_DIR}/$(echo "$MODEL_ID" | tr '/' '-')"
mkdir -p "$DEST_DIR"

API_URL="https://huggingface.co/api/models/${MODEL_ID}"
META_JSON="$(mktemp)"
MATCHED_TXT="$(mktemp)"
trap 'rm -f "$META_JSON" "$MATCHED_TXT"' EXIT

WGET_COMMON=(--quiet)
if [[ -n "$TOKEN" ]]; then
  WGET_COMMON+=(--header "Authorization: Bearer ${TOKEN}")
fi

if ! wget "${WGET_COMMON[@]}" -O "$META_JSON" "$API_URL"; then
  echo "Error: failed to fetch model metadata from $API_URL" >&2
  exit 1
fi

python - "$META_JSON" > "$MATCHED_TXT" <<'PY'
import json
import pathlib
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)

siblings = data.get("siblings", [])
files = [item.get("rfilename", "") for item in siblings if item.get("rfilename")]

required = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "model_index.json",
}

tok_suffixes = (".model", ".json", ".txt", ".tiktoken", ".bpe")
matched = set()

for p in files:
    name = pathlib.Path(p).name.lower()
    lower = p.lower()

    if lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
        matched.add(p)
        continue

    if name in required:
        matched.add(p)
        continue

    if name.startswith("tokenizer") and name.endswith(tok_suffixes):
        matched.add(p)

for p in sorted(matched):
    print(p)
PY

COUNT="$(wc -l < "$MATCHED_TXT" | tr -d ' ')"
if [[ "$COUNT" == "0" ]]; then
  echo "No matching files found." >&2
  exit 1
fi

echo "Model: $MODEL_ID"
echo "Revision: $REVISION"
echo "Destination: $DEST_DIR"
echo "Matched files: $COUNT"
cat "$MATCHED_TXT" | sed 's/^/  - /'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

download_one() {
  local rel="$1"
  local encoded
  encoded="$(python - "$rel" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe='/'))
PY
)"

  local out="$DEST_DIR/$rel"
  mkdir -p "$(dirname "$out")"

  local url="https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/${encoded}?download=true"
  wget "${WGET_COMMON[@]}" -c -O "$out" "$url"
}

while IFS= read -r relpath; do
  [[ -z "$relpath" ]] && continue
  download_one "$relpath"
done < "$MATCHED_TXT"

echo "Done."
