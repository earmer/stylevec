#!/bin/bash
set -euo pipefail

SYNC_ONLY=false
FETCH=false
SYNC_MODEL=false
STOP_EXISTING=false
BATCH_SIZE=128
MIN_BATCH_SIZE=8
MAX_ROWS=""
LANGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synconly|-s) SYNC_ONLY=true; shift ;;
        --fetch|-f) FETCH=true; shift ;;
        --sync-model) SYNC_MODEL=true; shift ;;
        --stop-existing) STOP_EXISTING=true; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --min-batch-size) MIN_BATCH_SIZE="$2"; shift 2 ;;
        --max-rows) MAX_ROWS="$2"; shift 2 ;;
        --langs) IFS=' ' read -r -a LANGS <<< "$2"; shift 2 ;;
        --help|-h)
            cat << 'EOF'
Usage: bash data/datasets/msynthstel/remote_score_ppl.sh [options] [lang ...]

Options:
  --synconly, -s          Sync files only, do not launch remote scoring.
  --fetch, -f            Fetch remote scored JSONL files back to local scored/.
  --sync-model           Also rsync artifacts/base-models/qwen-3-0.6b to remote.
  --stop-existing        Kill existing remote 04_score_ppl.py jobs before launch.
  --batch-size N         Remote initial batch size. Default: 128.
  --min-batch-size N     OOM fallback floor. Default: 8.
  --max-rows N           Score at most N rows per language.
  --langs "en la"        Language list. Positional language args also work.

Examples:
  bash data/datasets/msynthstel/remote_score_ppl.sh en la --max-rows 1000
  bash data/datasets/msynthstel/remote_score_ppl.sh --batch-size 128
  bash data/datasets/msynthstel/remote_score_ppl.sh --fetch
EOF
            exit 0
            ;;
        -*)
            echo "Unknown flag: $1"
            exit 1
            ;;
        *)
            LANGS+=("$1")
            shift
            ;;
    esac
done

LOCAL_DIR="/Users/earmercarey/stylevec"
REMOTE_DIR="/root/autodl-tmp/stylevec"
REMOTE_HOST="root@connect.westd.seetacloud.com"
SSH_PORT=34492

LOCAL_DELTA="$LOCAL_DIR/data/datasets/msynthstel/pipeline/datadelta"
REMOTE_DELTA="$REMOTE_DIR/data/datasets/msynthstel/pipeline/datadelta"
SSH=(ssh -p "$SSH_PORT")
RSYNC=(rsync -avzP -e "ssh -p $SSH_PORT")

if $FETCH; then
    echo "=== Fetching remote scored JSONL files ==="
    mkdir -p "$LOCAL_DELTA/scored"
    "${RSYNC[@]}" "$REMOTE_HOST:$REMOTE_DELTA/scored/" "$LOCAL_DELTA/scored/"
    echo "=== Scored files fetched to $LOCAL_DELTA/scored ==="
    exit 0
fi

echo "=== Preparing remote directories ==="
"${SSH[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DELTA/deduped' '$REMOTE_DELTA/scored'"

echo "=== Syncing scorer and project metadata ==="
"${RSYNC[@]}" "$LOCAL_DELTA/04_score_ppl.py" "$REMOTE_HOST:$REMOTE_DELTA/04_score_ppl.py"
for f in pyproject.toml uv.lock; do
    if [[ -f "$LOCAL_DIR/$f" ]]; then
        "${RSYNC[@]}" "$LOCAL_DIR/$f" "$REMOTE_HOST:$REMOTE_DIR/$f"
    fi
done

if $SYNC_MODEL; then
    echo "=== Syncing Qwen model ==="
    "${SSH[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/artifacts/base-models'"
    "${RSYNC[@]}" "$LOCAL_DIR/artifacts/base-models/qwen-3-0.6b/" "$REMOTE_HOST:$REMOTE_DIR/artifacts/base-models/qwen-3-0.6b/"
else
    echo "=== Checking remote Qwen model ==="
    "${SSH[@]}" "$REMOTE_HOST" "test -d '$REMOTE_DIR/artifacts/base-models/qwen-3-0.6b' || { echo 'ERROR: missing remote artifacts/base-models/qwen-3-0.6b. Rerun with --sync-model.'; exit 1; }"
fi

echo "=== Syncing deduped input JSONL ==="
if [[ ${#LANGS[@]} -eq 0 ]]; then
    "${RSYNC[@]}" "$LOCAL_DELTA/deduped/" "$REMOTE_HOST:$REMOTE_DELTA/deduped/"
else
    for lang in "${LANGS[@]}"; do
        src="$LOCAL_DELTA/deduped/$lang.jsonl"
        if [[ ! -f "$src" ]]; then
            echo "ERROR: missing local deduped file: $src"
            exit 1
        fi
        "${RSYNC[@]}" "$src" "$REMOTE_HOST:$REMOTE_DELTA/deduped/$lang.jsonl"
    done
fi

if $SYNC_ONLY; then
    echo "=== Sync-only mode: skipping remote scoring ==="
    exit 0
fi

REMOTE_CMD=(uv run python data/datasets/msynthstel/pipeline/datadelta/04_score_ppl.py)
if [[ ${#LANGS[@]} -gt 0 ]]; then
    REMOTE_CMD+=("${LANGS[@]}")
fi
REMOTE_CMD+=(--batch-size "$BATCH_SIZE" --min-batch-size "$MIN_BATCH_SIZE" --progress-every 5000)
if [[ -n "$MAX_ROWS" ]]; then
    REMOTE_CMD+=(--max-rows "$MAX_ROWS")
fi
REMOTE_CMD_QUOTED=$(printf "%q " "${REMOTE_CMD[@]}")

echo "=== Launching remote perplexity scoring ==="
"${SSH[@]}" "$REMOTE_HOST" bash -s << ENDSSH
set -euo pipefail
export PATH="\$HOME/.local/bin:/root/.local/bin:/root/.cargo/bin:\$PATH"
cd "$REMOTE_DIR"
uv sync
if $STOP_EXISTING; then
  pkill -9 -f "data/datasets/msynthstel/pipeline/datadelta/04_score_ppl.py" 2>/dev/null || true
  sleep 2
fi
mkdir -p "$REMOTE_DELTA/scored"
nohup $REMOTE_CMD_QUOTED > "$REMOTE_DELTA/score_ppl.log" 2>&1 &
echo \$! > "$REMOTE_DELTA/score_ppl.pid"
echo "pid=\$(cat "$REMOTE_DELTA/score_ppl.pid")"
echo "log=$REMOTE_DELTA/score_ppl.log"
sleep 3
tail -40 "$REMOTE_DELTA/score_ppl.log" || true
ENDSSH
