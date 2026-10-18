#!/usr/bin/env bash
# Push code, training inputs and the base model to a remote box over rsync.
#
#   sync_to_remote.sh user@host:/root/stylevec [dataset ...]
#
# dataset: a directory name under expandtraining/2606/outputs/datasets/ (prebuilt triplets).
#
# Code goes to the remote root on the instance disk. Data (inputs, base model, datasets, bookraw)
# goes to DATA_ROOT, by default AutoDL's autodl-fs, which survives instance changes within a
# region. The remote root then links data/, artifacts/ and outputs/datasets/ to it, so configs
# keep their repo-relative paths.
# Env:
#   DATA_ROOT=/path  where data lives on the remote (default /root/autodl-fs/stylevec);
#                    set it to the remote root to keep everything on the instance disk
#   SSH_PORT=12345   AutoDL's ssh port (or use a Host alias in ~/.ssh/config instead)
#   BOOKRAW=1        also push books_cleaned.parquet from the data volume to data/bookraw/
#   CHECKSUM=1       compare file contents instead of size+mtime (repairs a corrupt remote copy)
#
# Integrity is rsync's own: every transfer is checksummed, and CHECKSUM=1 hashes whole files. Both use
# xxh128 (then xxh3, xxh64) when both ends have xxhash, which rsync >= 3.2 normally does; md5 otherwise.
#
# Re-runs are cheap: unchanged files are skipped, and an interrupted transfer resumes from
# .rsync-partial/ instead of leaving a truncated file at the real path.
set -euo pipefail

cd "$(dirname "$0")/../../.."

usage="usage: sync_to_remote.sh user@host:/remote/root [dataset ...]"
REMOTE="${1:?$usage}"
shift
[[ "$REMOTE" == *:* ]] || { echo "$usage" >&2; exit 2; }
HOST="${REMOTE%%:*}"
ROOT="${REMOTE#*:}"
ROOT="${ROOT%/}"
[[ -n "$ROOT" ]] || { echo "remote root is empty: $usage" >&2; exit 2; }
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/stylevec}"
DATA_ROOT="${DATA_ROOT%/}"
# Paths under the remote root that hold data; each becomes a link into DATA_ROOT.
linked=(data artifacts expandtraining/2606/outputs/datasets)

BOOKRAW_SRC="/Volumes/EMC/stylevec_data/manifests/bookraw/epub20_legacy/books_cleaned.parquet"
BOOKRAW_DST="data/bookraw/books_cleaned.parquet"

ssh_cmd=(ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
[[ -n "${SSH_PORT:-}" ]] && ssh_cmd+=(-p "$SSH_PORT")

rsync_base=(
  rsync -a --human-readable --info=progress2
  --partial-dir=.rsync-partial
  --compress --compress-choice=zstd  # the paratarget jsonl shrinks ~9x, genshin.db ~3x
  --exclude '._*' --exclude .DS_Store
  -e "${ssh_cmd[*]}"
)
[[ "${CHECKSUM:-0}" == 1 ]] && rsync_base+=(--checksum)
export RSYNC_CHECKSUM_LIST="${RSYNC_CHECKSUM_LIST:-xxh128 xxh3 xxh64 md5}"

inputs=(
  data/genshin/genshin.db
  data/paratarget/speakers_top100.json
  data/paratarget/generations_pairwise_validation_2p5cny.jsonl
  data/datasets/msynthstel/data/train-00000-of-00001.parquet
  data/datasets/msynthstel/data/test-00000-of-00001.parquet
)
model=artifacts/base-models/qwen-3-0.6b

missing=0
for path in "${inputs[@]}" "$model/config.json" "$model/tokenizer.json" "$model/model.safetensors"; do
  [[ -s "$path" ]] || { echo "missing required file: $path" >&2; missing=1; }
done
datasets=()
for name in "$@"; do
  dir="expandtraining/2606/outputs/datasets/$name"
  [[ -s "$dir/triplets.jsonl" && -s "$dir/manifest.json" ]] \
    || { echo "dataset $name lacks triplets.jsonl or manifest.json: $dir" >&2; missing=1; }
  datasets+=("$dir")
done
if [[ "${BOOKRAW:-0}" == 1 && ! -s "$BOOKRAW_SRC" ]]; then
  echo "BOOKRAW=1 but the data volume file is missing: $BOOKRAW_SRC" >&2
  missing=1
fi
(( missing == 0 )) || exit 1

if [[ "$DATA_ROOT" != "$ROOT" ]]; then
  echo "== link $ROOT/{${linked[*]}} -> $DATA_ROOT"
  # Refuses to replace a real directory: move its contents into DATA_ROOT by hand first.
  "${ssh_cmd[@]}" "$HOST" "set -e
    [ -d '$(dirname "$DATA_ROOT")' ] || { echo 'DATA_ROOT parent missing (is autodl-fs mounted?): $DATA_ROOT' >&2; exit 1; }
    for rel in ${linked[*]}; do
      if [ -e '$ROOT'/\$rel ] && [ ! -L '$ROOT'/\$rel ]; then
        echo \"$ROOT/\$rel is a real directory; move it into $DATA_ROOT/\$rel first\" >&2; exit 1
      fi
    done
    for rel in ${linked[*]}; do
      mkdir -p '$DATA_ROOT'/\$rel \"\$(dirname '$ROOT'/\$rel)\"
      ln -sfn '$DATA_ROOT'/\$rel '$ROOT'/\$rel
    done"
fi
"${ssh_cmd[@]}" "$HOST" "mkdir -p '$ROOT/expandtraining/2606' '$DATA_ROOT/data/bookraw'"

echo "== code"
# --delete keeps the remote tree identical to ours; excluded paths (run outputs) are never touched.
"${rsync_base[@]}" --delete \
  --exclude outputs/ --exclude tf-logs/ --exclude __pycache__/ --exclude '*.py[cod]' \
  --exclude .venv/ --exclude .pytest_cache/ --exclude .ruff_cache/ \
  expandtraining/2606/ "$HOST:$ROOT/expandtraining/2606/"
# The uv project lives at the repo root; remote_setup.sh builds the environment from it.
"${rsync_base[@]}" pyproject.toml uv.lock "$HOST:$ROOT/"

echo "== inputs"
"${rsync_base[@]}" -R "${inputs[@]}" "$HOST:$DATA_ROOT/"

echo "== base model"
"${rsync_base[@]}" -R "$model/" "$HOST:$DATA_ROOT/"

if (( ${#datasets[@]} )); then
  echo "== datasets: $*"
  "${rsync_base[@]}" -R "${datasets[@]/%//}" "$HOST:$DATA_ROOT/"
fi

if [[ "${BOOKRAW:-0}" == 1 ]]; then
  echo "== bookraw -> $BOOKRAW_DST"
  "${rsync_base[@]}" "$BOOKRAW_SRC" "$HOST:$DATA_ROOT/$BOOKRAW_DST"
fi
