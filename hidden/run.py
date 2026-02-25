"""主入口：串联数据准备 → 隐藏层提取 → 分类器 → 评估。"""

import numpy as np
from data_prep import prepare_data
from extract_hidden import extract_and_cache, load_cached, detect_n_layers, POOL_METHODS
from classifiers import run_lda, train_mlp_arcface, extract_mlp_vecs, ARCHS, STYLE_DIMS
from evaluate import evaluate_all, silhouette, consistency


def main():
    # ── Step 1: 数据准备 ──
    print("=" * 60)
    print("Step 1: Data preparation")
    print("=" * 60)
    data = prepare_data()

    n_train_classes = len(set(data["train"]["labels"].tolist()))
    n_gen_classes = len(set(data["gen"]["labels"].tolist()))

    # 合并所有文本用于提取（顺序：train, val, gen）
    all_texts = data["train"]["texts"] + data["val"]["texts"] + data["gen"]["texts"]
    n_train = len(data["train"]["texts"])
    n_val = len(data["val"]["texts"])
    n_gen = len(data["gen"]["texts"])

    # ── Step 2: 隐藏层提取 ──
    print("\n" + "=" * 60)
    print("Step 2: Hidden state extraction")
    print("=" * 60)
    extract_and_cache(all_texts)

    n_layers = detect_n_layers()
    print(f"Detected {n_layers} layers from cache.")

    # ── Step 3: 逐层逐池化评估 ──
    print("\n" + "=" * 60)
    print("Step 3: Evaluation (pool × layer × method)")
    print("=" * 60)

    results = []  # list of dicts for final summary

    for pool in POOL_METHODS:
        print(f"\n{'─' * 50}")
        print(f"Pool: {pool}")
        print(f"{'─' * 50}")

        for layer in range(n_layers):
            embs = load_cached(layer, pool)  # (1200, 1024)
            train_vecs = embs[:n_train]
            val_vecs = embs[n_train:n_train + n_val]
            gen_vecs = embs[n_train + n_val:]

            train_labels = data["train"]["labels"]
            val_labels = data["val"]["labels"]
            gen_labels = data["gen"]["labels"]

            # ── LDA ──
            lda_train, lda_val, lda_gen = run_lda(
                train_vecs, train_labels, val_vecs, gen_vecs)
            lda_metrics = evaluate_all(
                lda_train, train_labels, n_train_classes,
                lda_val, val_labels, n_train_classes,
                lda_gen, gen_labels, n_gen_classes)
            lda_metrics.update(pool=pool, layer=layer, method="LDA")
            results.append(lda_metrics)

            # ── MLP + ArcFace variants ──
            for arch_name in ARCHS:
                for dim in STYLE_DIMS:
                    tag = f"MLP-{arch_name}-d{dim}"
                    backbone = train_mlp_arcface(
                        train_vecs, train_labels, n_train_classes,
                        arch_name, dim)
                    mlp_train = extract_mlp_vecs(backbone, train_vecs)
                    mlp_val = extract_mlp_vecs(backbone, val_vecs)
                    mlp_gen = extract_mlp_vecs(backbone, gen_vecs)

                    mlp_metrics = evaluate_all(
                        mlp_train, train_labels, n_train_classes,
                        mlp_val, val_labels, n_train_classes,
                        mlp_gen, gen_labels, n_gen_classes)
                    mlp_metrics.update(pool=pool, layer=layer, method=tag)
                    results.append(mlp_metrics)

                    del backbone

            if layer % 7 == 0:
                _print_layer_summary(pool, layer, results)

    # ── Step 4: 汇总 ──
    print("\n" + "=" * 60)
    print("Step 4: Full Results Summary")
    print("=" * 60)
    _print_full_table(results)

    # ── Step 5: 高亮最佳组合 ──
    print("\n" + "=" * 60)
    print("Step 5: Best combinations (by val silhouette)")
    print("=" * 60)
    _print_best(results)


def _print_layer_summary(pool, layer, results):
    """打印单层的简要指标。"""
    layer_results = [r for r in results if r["pool"] == pool and r["layer"] == layer]
    print(f"\n  L{layer:02d} | ", end="")
    for r in layer_results:
        print(f"{r['method']:20s} val_sil={r['val_sil']:+.4f} gen_sil={r['gen_sil']:+.4f} | ", end="")
    print()


def _print_full_table(results):
    """打印完整指标表格。"""
    header = (f"{'Pool':<10} {'Layer':>5} {'Method':<22} "
              f"{'tr_sil':>7} {'tr_con':>7} "
              f"{'va_sil':>7} {'va_con':>7} "
              f"{'ge_sil':>7} {'ge_con':>7}")
    print(header)
    print("─" * len(header))
    for r in results:
        print(f"{r['pool']:<10} {r['layer']:>5} {r['method']:<22} "
              f"{r['train_sil']:+.4f} {r['train_cons']:+.4f} "
              f"{r['val_sil']:+.4f} {r['val_cons']:+.4f} "
              f"{r['gen_sil']:+.4f} {r['gen_cons']:+.4f}")


def _print_best(results, top_k=20):
    """按 val_sil 降序打印 top-k 组合。"""
    sorted_r = sorted(results, key=lambda r: r["val_sil"], reverse=True)
    print(f"\nTop {top_k} by val silhouette:")
    header = (f"{'#':>3} {'Pool':<10} {'Layer':>5} {'Method':<22} "
              f"{'va_sil':>7} {'va_con':>7} "
              f"{'ge_sil':>7} {'ge_con':>7}")
    print(header)
    print("─" * len(header))
    for i, r in enumerate(sorted_r[:top_k]):
        marker = " ***" if r["val_sil"] > 0 and r["gen_sil"] > 0 else ""
        print(f"{i+1:>3} {r['pool']:<10} {r['layer']:>5} {r['method']:<22} "
              f"{r['val_sil']:+.4f} {r['val_cons']:+.4f} "
              f"{r['gen_sil']:+.4f} {r['gen_cons']:+.4f}{marker}")


if __name__ == "__main__":
    main()
