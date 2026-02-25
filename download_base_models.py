import argparse

DEST = "base-models/qwen-3-0.6b"

parser = argparse.ArgumentParser()
parser.add_argument("--modelscope", action="store_true", help="Use ModelScope mirror (for China)")
args = parser.parse_args()

if args.modelscope:
    from modelscope import snapshot_download
    snapshot_download("qwen/Qwen3-0.6B", local_dir=DEST)
else:
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-0.6B", local_dir=DEST)
