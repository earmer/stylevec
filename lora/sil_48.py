"""48人子集 silhouette：合并「少女」为哥伦比娅。"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_samples

EMB_CACHE = Path(__file__).resolve().parent / "analysis_embeddings.pkl"

SPEAKERS_48 = [
    "娜维娅","纳西妲","温迪","阿贝多","茜特菈莉","八重神子","玛拉妮","芙宁娜",
    "玛薇卡","赛诺","艾尔海森","林尼","提纳里","荒泷一斗","那维莱特","钟离",
    "枫原万叶","迪希雅","宵宫","恰斯卡","希诺宁","胡桃","莫娜","莱欧斯利",
    "夜兰","奈芙尔","凯亚","神里绫华","妮露","刻晴","柯莱","菈乌玛",
    "珊瑚宫心海","安柏","琴","「少女」","可莉","香菱","卡维","卡齐娜",
    "欧洛伦","基尼奇","砂糖","烟绯","夏洛蒂","魈","雅珂达","菲林斯",
]

with open(EMB_CACHE, "rb") as f:
    all_emb = pickle.load(f)

# 构建子集，「少女」→ 哥伦比娅
vecs_list, labels_list, names = [], [], []
for i, spk in enumerate(SPEAKERS_48):
    display = "哥伦比娅" if spk == "「少女」" else spk
    names.append(display)
    v = all_emb[spk]
    vecs_list.append(v)
    labels_list.extend([i] * len(v))

X = np.concatenate(vecs_list)
labels = np.array(labels_list)
print(f"说话人: {len(names)}, 总样本: {X.shape[0]}")

sample_sil = silhouette_samples(X, labels, metric="cosine", n_jobs=-1)

results = []
for i, name in enumerate(names):
    mask = labels == i
    s = sample_sil[mask]
    results.append((name, int(mask.sum()), float(s.mean())))

results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'说话人':<12} {'N':>4} {'mean_sil':>10}")
print("-" * 30)
for name, n, mean in results:
    print(f"{name:<12} {n:>4} {mean:>10.4f}")

pos = sum(1 for r in results if r[2] > 0)
print(f"\nsilhouette > 0: {pos}/{len(results)}")
print(f"全局 silhouette: {sample_sil.mean():.4f}")
