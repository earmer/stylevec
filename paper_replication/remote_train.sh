#!/bin/bash
set -e

SYNC_ONLY=false
FETCH_LOGS=false
WHITE_PERCENT=""
TYPE_C_RATIO=""
CROSS_LANG_TRAIT_RATIO=""
SAME_LANG_HARD_RATIO=""
EPOCHS=50
RESUME_RUN_REQUESTED=false
RESUME_RUN=""
CONTINUE_LATEST=false
ANALYZE_GENSHIN=false
FETCH_GENSHIN_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synconly|-s) SYNC_ONLY=true; shift ;;
        --fetchlogs|--fl) FETCH_LOGS=true; shift ;;
        --resume-run)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --resume-run (format: YYYYMMDD-HHMMSS)"
                exit 1
            fi
            RESUME_RUN_REQUESTED=true; RESUME_RUN="$2"; shift 2 ;;
        --continue-latest) CONTINUE_LATEST=true; shift ;;
        --analyze-genshin) ANALYZE_GENSHIN=true; shift ;;
        --fetch-genshin-analysis) FETCH_GENSHIN_ANALYSIS=true; shift ;;
        --white-percent)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --white-percent"
                exit 1
            fi
            WHITE_PERCENT="$2"; shift 2 ;;
        --epochs)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --epochs"
                exit 1
            fi
            EPOCHS="$2"; shift 2 ;;
        --type-c-ratio)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --type-c-ratio"
                exit 1
            fi
            TYPE_C_RATIO="$2"; shift 2 ;;
        --cross-lang-trait-ratio)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --cross-lang-trait-ratio"
                exit 1
            fi
            CROSS_LANG_TRAIT_RATIO="$2"; shift 2 ;;
        --same-lang-hard-ratio)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --same-lang-hard-ratio"
                exit 1
            fi
            SAME_LANG_HARD_RATIO="$2"; shift 2 ;;
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
*.pt
*.pth
*.log
.cache/
artifacts/
data/datasets/msynthstel/corpora/
data/datasets/msynthstel/analysis/
data/datasets/msynthstel/pipeline/
data/datasets/msynthstel/.cache/
data/datasets/msynthstel/.claude/
target/
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
    LOCAL_LOG_DIR="$LOCAL_DIR/artifacts/paper_replication/checkpoints/tf-logs"
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
    LOCAL_ANALYSIS_DIR="$LOCAL_DIR/artifacts/genshin/embedding_analysis"
    mkdir -p "$LOCAL_ANALYSIS_DIR"
    rsync -avzP \
        -e "ssh -p $SSH_PORT" \
        "$REMOTE_HOST:$REMOTE_DIR/artifacts/genshin/embedding_analysis/" \
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

  OUT_DIR="artifacts/genshin/embedding_analysis/latest_20260505-005138"
  RUN_LOG="$OUT_DIR/run.log"

  echo "=== Installing dependencies from synced uv.lock ==="
  uv sync

  if pgrep -af "genshin/analyze_latest_embeddings.py" >/dev/null; then
    echo "=== Genshin embedding analysis is already running ==="
    pgrep -af "genshin/analyze_latest_embeddings.py"
    echo "Monitor: tail -f /root/autodl-tmp/stylevec/$RUN_LOG"
    exit 0
  fi

  if [[ ! -f artifacts/paper_replication/checkpoints/latest/training_state.pt ]]; then
    echo "ERROR: Missing artifacts/paper_replication/checkpoints/latest/training_state.pt"
    exit 1
  fi

  mkdir -p "$OUT_DIR"
  echo "=== Latest checkpoint state ==="
  .venv/bin/python - << 'PY'
from pathlib import Path
import torch

state = torch.load("artifacts/paper_replication/checkpoints/latest/training_state.pt", map_location="cpu", weights_only=False)
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
      --db data/genshin/genshin.db \
      --checkpoint artifacts/paper_replication/checkpoints/latest \
      --model-name artifacts/base-models/xlm-roberta-base \
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

ssh -p "$SSH_PORT" "$REMOTE_HOST" \
  WHITE_PERCENT="$WHITE_PERCENT" \
  TYPE_C_RATIO="$TYPE_C_RATIO" \
  CROSS_LANG_TRAIT_RATIO="$CROSS_LANG_TRAIT_RATIO" \
  SAME_LANG_HARD_RATIO="$SAME_LANG_HARD_RATIO" \
  EPOCHS="$EPOCHS" \
  RESUME_RUN_REQUESTED="$RESUME_RUN_REQUESTED" \
  RESUME_RUN="$RESUME_RUN" \
  CONTINUE_LATEST="$CONTINUE_LATEST" \
  'bash -s' << 'ENDSSH'
  set -e
  export PATH="/root/.local/bin:$PATH"
  cd /root/autodl-tmp/stylevec

  echo "=== Installing dependencies ==="
  uv sync

  if [[ "$RESUME_RUN_REQUESTED" == "true" && "$CONTINUE_LATEST" == "true" ]]; then
    echo "ERROR: Use only one of --resume-run or --continue-latest"
    exit 1
  fi

  if [[ "$CONTINUE_LATEST" == "true" ]]; then
    # Derive RESUME_RUN by matching latest/ to the actual step directory it was copied from
    LATEST_STEP=$(cd artifacts/paper_replication/checkpoints && uv run python -c "
import torch
state = torch.load('latest/training_state.pt', map_location='cpu', weights_only=False)
print(state['global_step'])
")
    LATEST_DIR=$(find artifacts/paper_replication/checkpoints -maxdepth 1 -type d -name "*-step-$(printf '%06d' $LATEST_STEP)" \
      | grep -v '/latest$' \
      | head -1)
    if [[ -z "$LATEST_DIR" ]]; then
      echo "ERROR: Could not find original checkpoint dir for step=$LATEST_STEP"
      exit 1
    fi
    TS=$(basename "$LATEST_DIR" | cut -d- -f1)
    RESUME_RUN=$(date -d "@$TS" +%Y%m%d-%H%M%S)
    echo "=== Derived RESUME_RUN=$RESUME_RUN from $LATEST_DIR (step=$LATEST_STEP) ==="
    RESUME_PREFIX="latest"
    RESUME_DIR="artifacts/paper_replication/checkpoints/latest"
  elif [[ "$RESUME_RUN_REQUESTED" == "true" ]]; then
    RESUME_PREFIX=$(date -d "${RESUME_RUN:0:8} ${RESUME_RUN:9:2}:${RESUME_RUN:11:2}:${RESUME_RUN:13:2}" +%s)
    RESUME_DIR=$(find artifacts/paper_replication/checkpoints -maxdepth 1 -type d -name "${RESUME_PREFIX}-step-*" \
      | sort -t- -k3,3n \
      | tail -1)
  fi
  if [[ "$RESUME_RUN_REQUESTED" == "true" || "$CONTINUE_LATEST" == "true" ]]; then
    if [[ ! -d "$RESUME_DIR" || ! -f "$RESUME_DIR/training_state.pt" ]]; then
      echo "ERROR: Missing resume checkpoint for run $RESUME_RUN (prefix $RESUME_PREFIX)"
      exit 1
    fi
    echo "=== Resuming from $RESUME_DIR ==="
    uv run python - "$RESUME_DIR" "$RESUME_RUN" << 'PY'
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
  else
    echo "=== Starting clean training from base model (no resume) ==="
  fi

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
  CURRICULUM_ARGS=()
  if [[ -n "$TYPE_C_RATIO" ]]; then
    CURRICULUM_ARGS+=(--type-c-ratio "$TYPE_C_RATIO")
    echo "=== Using --type-c-ratio $TYPE_C_RATIO ==="
  fi
  if [[ -n "$CROSS_LANG_TRAIT_RATIO" ]]; then
    CURRICULUM_ARGS+=(--cross-lang-trait-ratio "$CROSS_LANG_TRAIT_RATIO")
    echo "=== Using --cross-lang-trait-ratio $CROSS_LANG_TRAIT_RATIO ==="
  fi
  if [[ -n "$SAME_LANG_HARD_RATIO" ]]; then
    CURRICULUM_ARGS+=(--same-lang-hard-ratio "$SAME_LANG_HARD_RATIO")
    echo "=== Using --same-lang-hard-ratio $SAME_LANG_HARD_RATIO ==="
  fi
  TRAIN_ARGS=(--epochs "$EPOCHS")
  if [[ "$RESUME_RUN_REQUESTED" == "true" ]]; then
    TRAIN_ARGS=(--resume-from "$RESUME_RUN" --epochs "$EPOCHS")
    echo "=== Resuming requested run $RESUME_RUN ==="
  fi
  if [[ "$CONTINUE_LATEST" == "true" ]]; then
    TRAIN_ARGS=(--resume --resume-next-epoch --log-run-name "$RESUME_RUN" --epochs 35)
    echo "=== Continuing artifacts/paper_replication/checkpoints/latest for 5 more epochs ==="
  fi
  mkdir -p artifacts/paper_replication
  nohup uv run python paper_replication/train.py \
    --use-local-data \
    "${TRAIN_ARGS[@]}" \
    --max-pairs-per-feature 32 \
    --val-every 250 \
    --save-every 1500 \
    "${WHITE_ARGS[@]}" \
    "${CURRICULUM_ARGS[@]}" \
    --log-dir /root/tf-logs \
    > artifacts/paper_replication/train.log 2>&1 &

  echo "=== Waiting for training to start (up to 30s) ==="
  for i in $(seq 1 30); do
    if grep -Eq "Resumed from|Epoch" artifacts/paper_replication/train.log 2>/dev/null; then
      echo "Training started successfully."
      echo "Monitor: tail -f /root/autodl-tmp/stylevec/artifacts/paper_replication/train.log"
      exit 0
    fi
    sleep 1
  done

  echo "ERROR: Training did not start within 30 seconds. Last log lines:"
  tail -20 artifacts/paper_replication/train.log
  exit 1
ENDSSH
