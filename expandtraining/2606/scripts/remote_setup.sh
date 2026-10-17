#!/usr/bin/env bash
# Build the uv environment on an AutoDL box from uv.lock, through the region's pip mirror.
#
#   bash expandtraining/2606/scripts/remote_setup.sh        (run on the instance, from anywhere)
#
# `uv sync --frozen` downloads from the URLs recorded in uv.lock (files.pythonhosted.org), which
# crawls from AutoDL. Exporting the lock to a hash-pinned requirements file and installing that from
# the mirror gives the same versions and the same hashes. The uv cache lives on autodl-fs, so the
# next instance in the region installs from it instead of the network.
#
# Unpacking ~7 GB of small files out of the autodl-fs cache is slow (torch alone has thousands of
# headers), so a finished .venv is also saved there as one tar per uv.lock. A new box in the region
# restores that tar in a sequential read, and `uv pip sync` then only verifies it.
# Env:
#   INDEX_URL=...    package index (default: the mirror in /etc/pip.conf, else Aliyun)
#   UV_CACHE_DIR=... (default /root/autodl-fs/uv-cache when autodl-fs is mounted)
#   VENV_SNAPSHOTS=/path  where .venv tars live (default /root/autodl-fs/venvs; empty disables)
#   SKIP_GPU_CHECK=1 import check only, no CUDA (re-running while a job holds the GPU's memory)
set -euo pipefail

cd "$(dirname "$0")/../../.."

if ! command -v uv >/dev/null 2>&1; then
  for candidate in /root/miniconda3/bin/uv "$HOME/.local/bin/uv"; do
    [[ -x "$candidate" ]] && PATH="$(dirname "$candidate"):$PATH" && break
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  /root/miniconda3/bin/pip install -q uv
  PATH="/root/miniconda3/bin:$PATH"
fi

if [[ -z "${INDEX_URL:-}" ]]; then
  INDEX_URL="$(sed -n 's/^index-url *= *//p' /etc/pip.conf 2>/dev/null | head -1)"
  INDEX_URL="${INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
fi
INDEX_URL="${INDEX_URL/#http:/https:}"
if [[ -z "${UV_CACHE_DIR:-}" && -d /root/autodl-fs ]]; then
  export UV_CACHE_DIR=/root/autodl-fs/uv-cache
fi
export UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300

python_bin="${PYTHON:-/root/miniconda3/bin/python}"
lock_sha="$(sha256sum uv.lock | cut -d' ' -f1)"
snapshots="${VENV_SNAPSHOTS-$([[ -d /root/autodl-fs ]] && echo /root/autodl-fs/venvs)}"
# The venv hard-codes its own path and interpreter, so a tar is only valid at the same location.
snapshot="${snapshots:+$snapshots/venv-${lock_sha:0:16}-$(echo "$PWD" | tr / _).tar}"
if [[ ! -x .venv/bin/python && -n "$snapshot" && -s "$snapshot" ]]; then
  echo "== restoring .venv from $snapshot ($(du -h "$snapshot" | cut -f1))"
  start=$(date +%s)
  tar -xf "$snapshot"
  echo "restored in $(( $(date +%s) - start ))s"
fi
[[ -x .venv/bin/python ]] || uv venv --python "$python_bin" .venv

requirements="$(mktemp)"
trap 'rm -f "$requirements"' EXIT
uv export --frozen --no-dev --no-emit-project --format requirements-txt -o "$requirements" >/dev/null
echo "== installing $(grep -c '==' "$requirements") pinned packages from $INDEX_URL (cache: ${UV_CACHE_DIR:-default})"
start=$(date +%s)
uv pip sync --python .venv/bin/python --index-url "$INDEX_URL" "$requirements"
echo "installed in $(( $(date +%s) - start ))s"

# remote_train.sh and validate.sh see this marker and pass --no-sync to uv run, which would otherwise
# re-sync from uv.lock's URLs.
echo "$lock_sha" > .venv/.remote-setup-lock

.venv/bin/python - <<'EOF'
import torch, transformers, accelerate, peft
import os
cuda = torch.cuda.is_available() and os.environ.get("SKIP_GPU_CHECK") != "1"
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, "
      f"accelerate {accelerate.__version__}, peft {peft.__version__}; "
      f"GPUs: {'check skipped' if os.environ.get('SKIP_GPU_CHECK') == '1' else names or 'none visible'}")
if cuda:
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise SystemExit(f"this torch build has no kernels for {arch}: {torch.cuda.get_arch_list()}")
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print(f"bf16 matmul ok on {arch}: {float((x @ x).float().abs().mean()):.3f}")
EOF

# Only a venv that passed the import and GPU check above is worth restoring elsewhere.
if [[ -n "$snapshot" && ! -s "$snapshot" ]]; then
  echo "== saving .venv to $snapshot for the next box in this region"
  mkdir -p "$snapshots"
  start=$(date +%s)
  tar -cf "$snapshot.partial" .venv && mv "$snapshot.partial" "$snapshot"
  echo "saved $(du -h "$snapshot" | cut -f1) in $(( $(date +%s) - start ))s"
fi
