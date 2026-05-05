#!/bin/bash
set -e

SYNC_ONLY=false
FETCH_LOGS=false
WHITE_PERCENT=""
CONTINUE_LATEST=false
ANALYZE_GENSHIN=false
FETCH_GENSHIN_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synconly|-s) SYNC_ONLY=true; shift ;;
        --fetchlogs|--fl) FETCH_LOGS=true; shift ;;
        --continue-latest) CONTINUE_LATEST=true; shift ;;
        --analyze-genshin) ANALYZE_GENSHIN=true; shift ;;
        --fetch-genshin-analysis) FETCH_GENSHIN_ANALYSIS=true; shift ;;
        --white-percent)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --white-percent"
                exit 1
            fi
            WHITE_PERCENT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

LOCAL_DIR="/Users/earmercarey/stylevec"
REMOTE_DIR="/root/autodl-tmp/stylevec"
REMOTE_HOST="root@connect.westd.seetacloud.com"
SSH_PORT=34492
# ── Rsync exclude file ────────────────────────────────────────────────────────
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" << 'EOF'
.git/
__pycache__/
*.pyc
.DS_Store
.venv/
checkpoints/
*.pt
*.pth
*.log
.cache/
bookraw/
ao3_zh_kudos_crawl/
ao3_random_subset/
datadelta/
tf-logs/
logs/
target/
genshin/embedding_analysis/
EOF

# ── Sync ──────────────────────────────────────────────────────────────────────
rsync -avzP --delete --exclude-from="$EXCLUDE_FILE" \
    -e "ssh -p $SSH_PORT" \
    "$LOCAL_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/"
rm "$EXCLUDE_FILE"

# ── Fetch logs from remote ──────────────────────────────────────────────────────
if $FETCH_LOGS; then
    echo "=== Fetching logs from $REMOTE_HOST:/root/tf-logs/ ==="
    LOCAL_LOG_DIR="$LOCAL_DIR/paper_replication/tf-logs"
    mkdir -p "$LOCAL_LOG_DIR"
    rsync -avzP \
        -e "ssh -p $SSH_PORT" \
        "$REMOTE_HOST:/root/tf-logs/" \
        "$LOCAL_LOG_DIR/"
    echo "=== Logs fetched to $LOCAL_LOG_DIR ==="
    echo "  tensorboard --logdir $LOCAL_LOG_DIR"
    exit 0
fi

# ── Fetch Genshin embedding analysis outputs ───────────────────────────────────
if $FETCH_GENSHIN_ANALYSIS; then
    echo "=== Fetching Genshin embedding analysis outputs ==="
    LOCAL_ANALYSIS_DIR="$LOCAL_DIR/genshin/embedding_analysis"
    mkdir -p "$LOCAL_ANALYSIS_DIR"
    rsync -avzP \
        -e "ssh -p $SSH_PORT" \
        "$REMOTE_HOST:$REMOTE_DIR/genshin/embedding_analysis/" \
        "$LOCAL_ANALYSIS_DIR/"
    echo "=== Analysis outputs fetched to $LOCAL_ANALYSIS_DIR ==="
    exit 0
fi

# ── Remote setup & train ──────────────────────────────────────────────────────
if $SYNC_ONLY; then
    echo "=== Sync-only mode: skipping remote setup & training ==="
    exit 0
fi

# ── Remote Genshin embedding analysis ─────────────────────────────────────────
if $ANALYZE_GENSHIN; then
ssh -p "$SSH_PORT" "$REMOTE_HOST" 'bash -s' << 'ENDSSH'
  set -e
  export PATH="/root/.local/bin:$PATH"
  cd /root/autodl-tmp/stylevec

  OUT_DIR="genshin/embedding_analysis/latest_20260505-005138"
  RUN_LOG="$OUT_DIR/run.log"

  echo "=== Installing dependencies from synced uv.lock ==="
  uv sync

  if pgrep -af "genshin/analyze_latest_embeddings.py" >/dev/null; then
    echo "=== Genshin embedding analysis is already running ==="
    pgrep -af "genshin/analyze_latest_embeddings.py"
    echo "Monitor: tail -f /root/autodl-tmp/stylevec/$RUN_LOG"
    exit 0
  fi

  if [[ ! -f paper_replication/checkpoints/latest/training_state.pt ]]; then
    echo "ERROR: Missing paper_replication/checkpoints/latest/training_state.pt"
    exit 1
  fi

  mkdir -p "$OUT_DIR"
  echo "=== Latest checkpoint state ==="
  .venv/bin/python - << 'PY'
from pathlib import Path
import torch

state = torch.load("paper_replication/checkpoints/latest/training_state.pt", map_location="cpu", weights_only=False)
print(f"global_step={state.get('global_step')}")
print(f"epoch={state.get('epoch')}")
PY

  JOBS=23
  echo "=== Starting Genshin embedding analysis with $JOBS CPU workers ==="
  nohup env \
    OMP_NUM_THREADS="$JOBS" \
    MKL_NUM_THREADS="$JOBS" \
    OPENBLAS_NUM_THREADS="$JOBS" \
    NUMEXPR_NUM_THREADS="$JOBS" \
    .venv/bin/python genshin/analyze_latest_embeddings.py \
      --db genshin/genshin.db \
      --checkpoint paper_replication/checkpoints/latest \
      --model-name base-models/xlm-roberta-base \
      --out-dir "$OUT_DIR" \
      --batch-size 1024 \
      --jobs "$JOBS" \
      --no-tsne \
      --umap-neighbors 15 \
      --umap-components 30 \
      --umap-epochs 200 \
      > "$RUN_LOG" 2>&1 &

  echo "Monitor: tail -f /root/autodl-tmp/stylevec/$RUN_LOG"
ENDSSH
exit 0
fi

ssh -p "$SSH_PORT" "$REMOTE_HOST" WHITE_PERCENT="$WHITE_PERCENT" CONTINUE_LATEST="$CONTINUE_LATEST" 'bash -s' << 'ENDSSH'
  set -e
  export PATH="/root/.local/bin:$PATH"
  cd /root/autodl-tmp/stylevec

  RESUME_RUN="20260505-005138"
  if [[ "$CONTINUE_LATEST" == "true" ]]; then
    RESUME_PREFIX="latest"
    RESUME_DIR="paper_replication/checkpoints/latest"
  else
    RESUME_PREFIX=$(date -d "${RESUME_RUN:0:8} ${RESUME_RUN:9:2}:${RESUME_RUN:11:2}:${RESUME_RUN:13:2}" +%s)
    RESUME_DIR=$(find paper_replication/checkpoints -maxdepth 1 -type d -name "${RESUME_PREFIX}-step-*" \
      | sort -t- -k3,3n \
      | tail -1)
  fi
  if [[ ! -d "$RESUME_DIR" || ! -f "$RESUME_DIR/training_state.pt" ]]; then
    echo "ERROR: Missing resume checkpoint for run $RESUME_RUN (prefix $RESUME_PREFIX)"
    exit 1
  fi
  echo "=== Resuming from $RESUME_DIR ==="
  .venv/bin/python - "$RESUME_DIR" "$RESUME_RUN" << 'PY'
import sys
from pathlib import Path

import torch

ckpt_dir = Path(sys.argv[1])
resume_run = sys.argv[2]
state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False)
print(f"checkpoint_global_step={state.get('global_step')}")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = Path("/root/tf-logs") / resume_run
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    max_step = None
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            max_step = event.step if max_step is None else max(max_step, event.step)
    print(f"tensorboard_max_step={max_step}")
except Exception as exc:
    print(f"tensorboard_max_step=unavailable ({exc})")
PY

  echo "=== Installing dependencies ==="
  uv sync

  echo "=== Stopping any existing training ==="
  pkill -9 -f "train.py" 2>/dev/null || true
  sleep 2   # wait for GPU memory release

  echo "=== Starting training ==="
  mkdir -p /root/tf-logs
  WHITE_ARGS=()
  if [[ -n "$WHITE_PERCENT" ]]; then
    WHITE_ARGS=(--white-percent "$WHITE_PERCENT")
    echo "=== Using --white-percent $WHITE_PERCENT ==="
  fi
  RESUME_ARGS=(--resume-from "$RESUME_RUN" --epochs 30)
  if [[ "$CONTINUE_LATEST" == "true" ]]; then
    RESUME_ARGS=(--resume --resume-next-epoch --log-run-name "$RESUME_RUN" --epochs 35)
    echo "=== Continuing checkpoints/latest for 5 more epochs ==="
  fi
  nohup uv run python paper_replication/train.py \
    --use-local-data \
    "${RESUME_ARGS[@]}" \
    --max-pairs-per-feature 32 \
    --val-every 250 \
    --save-every 1500 \
    "${WHITE_ARGS[@]}" \
    --log-dir /root/tf-logs \
    > paper_replication/train.log 2>&1 &

  echo "=== Waiting for training to start (up to 30s) ==="
  for i in $(seq 1 30); do
    if grep -Eq "Resumed from|Epoch" paper_replication/train.log 2>/dev/null; then
      echo "Training started successfully."
      echo "Monitor: tail -f /root/autodl-tmp/stylevec/paper_replication/train.log"
      exit 0
    fi
    sleep 1
  done

  echo "ERROR: Training did not start within 30 seconds. Last log lines:"
  tail -20 paper_replication/train.log
  exit 1
ENDSSH
