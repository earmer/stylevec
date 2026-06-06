# simlar

Naive batch similarity library for short text pairs. Computes the average of
char n-gram Jaccard similarity and normalized Levenshtein similarity. Intended to be called
from Python via `pyo3` (no optimizations).
Batch computation uses `rayon` for parallelism.

## Rust API

- `batch_avg_similarity_rs(pairs, n)` -> `Vec<f64>`

## Python build (example)

Use `maturin` or another pyo3-compatible build tool to build a wheel. Example with maturin:

```
pip install maturin
maturin develop
```

## Python usage

```
import simlar

pairs = [("我喜欢吃苹果", "我爱吃苹果")]
print(simlar.batch_avg_similarity(pairs, 3))
```
