#!/usr/bin/env bash
# Upload simulation batches to Kaggle (macOS / Linux).
set -euo pipefail

cd "$(dirname "$0")"

KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "${KAGGLE_JSON}" ]]; then
  echo "kaggle.json not found at ${KAGGLE_JSON}" >&2
  echo "Download your API token from kaggle.com/settings and save it there." >&2
  exit 1
fi

username="$(python3 -c "import json; print(json.load(open('${KAGGLE_JSON}'))['username'])")"
echo "Using Kaggle account: ${username}"

mkdir -p dataset
python3 - <<PY
import json
from pathlib import Path

meta = {
    "title": "wildfire-drone-full",
    "id": "${username}/wildfire-drone-full",
    "licenses": [{"name": "CC0-1.0"}],
}
Path("dataset/dataset-metadata.json").write_text(json.dumps(meta, indent=2))
print("dataset-metadata.json written")
PY

echo "Starting upload..."
python3 upload_helper.py
