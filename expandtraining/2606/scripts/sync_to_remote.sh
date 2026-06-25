#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REMOTE="${1:?usage: sync_to_remote.sh user@host:/workspace/stylevec}"

required_files=(
  "data/genshin/genshin.db"
  "data/paratarget/speakers_top100.json"
  "data/paratarget/generations_pairwise_validation_2p5cny.jsonl"
  "data/datasets/msynthstel/data/train-00000-of-00001.parquet"
  "data/datasets/msynthstel/data/test-00000-of-00001.parquet"
  "data/datasets/corpora/bookraw/books_cleaned.parquet"
  "artifacts/base-models/qwen-3-0.6b/config.json"
  "artifacts/base-models/qwen-3-0.6b/tokenizer.json"
  "artifacts/base-models/qwen-3-0.6b/model.safetensors"
)

for path in "${required_files[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

rsync -a --human-readable --info=progress2 \
  --exclude outputs/ \
  --exclude tf-logs/ \
  --exclude __pycache__/ \
  --exclude '*.py[cod]' \
  expandtraining/2606/ "${REMOTE%/}/expandtraining/2606/"

rsync -aR --human-readable --info=progress2 \
  data/genshin/genshin.db \
  data/paratarget/speakers_top100.json \
  data/paratarget/generations_pairwise_validation_2p5cny.jsonl \
  "${REMOTE%/}/"

rsync -aR --human-readable --info=progress2 \
  data/datasets/msynthstel/data/ \
  data/datasets/corpora/bookraw/books_cleaned.parquet \
  "${REMOTE%/}/"

rsync -aR --human-readable --info=progress2 \
  artifacts/base-models/qwen-3-0.6b/ \
  "${REMOTE%/}/"
