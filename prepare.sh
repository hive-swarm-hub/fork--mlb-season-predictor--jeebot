#!/usr/bin/env bash
# Download and extract the MLB Season Predictor data bundle.
#
# The bundle contains real MLB stats 2010-2025 with anonymized identifiers
# (team names, season labels, divisions, leagues) so memorization-based
# shortcuts cannot win — only structured-stat reasoning helps. The public
# bundle contains frozen features only. Public task labels live in
# eval/frozen_labels.csv in this repository.
set -euo pipefail
cd "$(dirname "$0")"

DATA_URL="${MLB_DATA_URL:-https://github.com/jeewoo-lee/mlb-season-predictor/releases/download/mlb-data-v2/mlb_season_data_v2.zip}"
DATA_SHA256="${MLB_DATA_SHA256:-c8fe8f0cce236e1a86e4fe960f0f06a3be38aeb0a7c20b4c79f57442d2775011}"
CACHE="${MLB_CACHE_DIR:-$HOME/.cache/mlb-season-predictor}"
ZIP="$CACHE/mlb_season_data_v2.zip"

mkdir -p "$CACHE" data/train data/val eval/test_data

# If MLB_DATA_LOCAL_ZIP is set, use that instead of downloading (used by the
# task author for local testing before the GitHub release is published).
if [ -n "${MLB_DATA_LOCAL_ZIP:-}" ]; then
    echo "Using local zip: $MLB_DATA_LOCAL_ZIP"
    cp "$MLB_DATA_LOCAL_ZIP" "$ZIP"
elif [ ! -f "$ZIP" ]; then
    echo "Downloading data bundle from $DATA_URL ..."
    curl -L -f -o "$ZIP.tmp" "$DATA_URL"
    mv "$ZIP.tmp" "$ZIP"
fi

echo "Verifying SHA256 ..."
ACTUAL=$(shasum -a 256 "$ZIP" | awk '{print $1}')
if [ "$ACTUAL" != "$DATA_SHA256" ]; then
    echo "Checksum mismatch!" >&2
    echo "  expected: $DATA_SHA256" >&2
    echo "  actual:   $ACTUAL" >&2
    rm -f "$ZIP"
    exit 1
fi

echo "Extracting ..."
EXTRACTED="$CACHE/extracted"
rm -rf "$EXTRACTED"
mkdir -p "$EXTRACTED"
unzip -q "$ZIP" -d "$EXTRACTED"

cp "$EXTRACTED/train/"*.csv  data/train/
cp "$EXTRACTED/val/"*.csv    data/val/
cp "$EXTRACTED/frozen/"*.csv eval/test_data/

echo "Done."
echo "  data/train/:    $(ls data/train/ | wc -l | tr -d ' ') files"
echo "  data/val/:      $(ls data/val/ | wc -l | tr -d ' ') files"
echo "  eval/test_data/: $(ls eval/test_data/ | wc -l | tr -d ' ') files"
