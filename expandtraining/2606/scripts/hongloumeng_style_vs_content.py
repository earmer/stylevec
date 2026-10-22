"""Style or content? Re-run the single change-point properly, then explain what separates the segments."""
import numpy as np, pandas as pd, re
C = "outputs/hongloumeng/results/cache"
P = pd.read_parquet(f"{C}/passages.parquet")
ch = P.chapter.to_numpy(); L = np.log(P.length.to_numpy(float))
texts = pd.read_json("outputs/hongloumeng/results/passages.json").text.to_numpy()

def chap(E, lc=True):
    E = E.astype(float)
    if lc:
        Z = np.c_[L, np.ones(len(L))]; E = E - Z @ np.linalg.lstsq(Z, E, rcond=None)[0]
    V = np.stack([E[ch == c].mean(0) for c in range(1, 121)])
    return V / np.linalg.norm(V, axis=1, keepdims=True)

def scan(V, first=1):
    """Split chapters first..120 into [first..k] and [k+1..120]; return best k, its score, and the score curve."""
    W = V[first - 1:]; n = len(W); sc = {}
    for k in range(2, n - 1):
        a, b = W[:k].mean(0), W[k:].mean(0)
        sc[first - 1 + k] = 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    k = max(sc, key=sc.get); return k, sc

def perm_p(V, first, obs, n=300, rng=np.random.default_rng(0)):
    W = V[first - 1:]
    return np.mean([max(scan(W[rng.permutation(len(W))], 1)[1].values()) >= obs for _ in range(n)])

# per-chapter register features (no content words)
def rate(pat):
    r = re.compile(pat); return np.array([sum(len(r.findall(t)) for t in texts[ch == c]) / sum(len(t) for t in texts[ch == c]) for c in range(1, 121)])
feat = pd.DataFrame({
    "classical(之乎者也矣焉哉)": rate("[之乎者也矣焉哉]"),
    "vernacular(了的着呢麼罷)": rate("[了的著着呢麼么罷罢]"),
    "dialogue(道：/說：/“)": rate("道：|說：|“|「"),
    "verse_lines(5/7字，)": rate(r"(?:^|[，。．])[一-鿿]{5}[，。．]|(?:^|[，。．])[一-鿿]{7}[，。．]"),
    "mean_passage_len": np.array([P.length[ch == c].mean() for c in range(1, 121)]),
}, index=range(1, 121))

def stylo():
    fc = list("之其或亦方即皆仍故尚呢了的着著吗嗎么麼罢罷吧便就也矣乎焉哉而以於于乃则則但却卻又且都将將把被叫让讓")
    M = np.array([[t.count(f) / max(len(t), 1) for f in fc] for t in texts]); return (M - M.mean(0)) / (M.std(0) + 1e-9)

methods = {"trained_256": np.load(f"{C}/trained_full.npy"), "trained_256_masked": np.load(f"{C}/trained_full_masked.npy"),
           "base_qwen": np.load(f"{C}/base_qwen3.npy"), "function_chars": stylo()}
print("best single split, chapters assigned to the first segment are 1..k (length-corrected)")
for name, E in methods.items():
    V = chap(E)
    for first in (1, 6, 12):
        k, sc = scan(V, first); top = sorted(sc, key=sc.get, reverse=True)[:4]
        p = perm_p(V, first, sc[k], n=200)
        print(f"  {name:20s} chapters {first:>2}-120: split after {k:>3}  p={p:.3f}  next best {top[1:]}  rank of 80: {sorted(sc, key=sc.get, reverse=True).index(80)+1}/{len(sc)}")
    # register regressed out of chapter vectors
    F = (feat - feat.mean()) / feat.std(); Z = np.c_[F.to_numpy(), np.ones(120)]
    Vr = V - Z @ np.linalg.lstsq(Z, V, rcond=None)[0]; Vr /= np.linalg.norm(Vr, axis=1, keepdims=True)
    for first in (1, 6):
        k, sc = scan(Vr, first)
        print(f"  {name:20s} register removed, chapters {first}-120: split after {k}  rank of 80: {sorted(sc, key=sc.get, reverse=True).index(80)+1}/{len(sc)}")
print("\nregister features: chapters 1-5 / 6-11 / 12-80 / 81-120")
seg = pd.cut(feat.index, [0, 5, 11, 80, 120], labels=["1-5", "6-11", "12-80", "81-120"])
print(feat.groupby(seg, observed=True).mean().T.round(4).to_string())

print("\n--- permutation p after removing register; and a content-only baseline ---")
def resid(V):
    F = (feat - feat.mean()) / feat.std(); Z = np.c_[F.to_numpy(), np.ones(120)]
    R = V - Z @ np.linalg.lstsq(Z, V, rcond=None)[0]; return R / np.linalg.norm(R, axis=1, keepdims=True)
for name in ("trained_256", "trained_256_masked", "base_qwen"):
    Vr = resid(chap(methods[name])); k, sc = scan(Vr, 1)
    print(f"  {name:20s} register removed: split after {k}, p={perm_p(Vr, 1, sc[k], n=500):.3f}, score80={sc[80]:.4f}, next {sorted(sc, key=sc.get, reverse=True)[1:4]}")
# content only: frequent content characters (no function chars), names already in the text
from collections import Counter
fcs = set("之其或亦方即皆仍故尚呢了的着著吗嗎么麼罢罷吧便就也矣乎焉哉而以於于乃则則但却卻又且都将將把被叫让讓是不我你他這这那一有在來来去說说道人")
cnt = Counter(c for t in texts for c in t if '一' <= c <= '鿿' and c not in fcs)
voc = [c for c, _ in cnt.most_common(600)]; idx = {c: i for i, c in enumerate(voc)}
M = np.zeros((len(texts), len(voc)))
for i, t in enumerate(texts):
    for c in t:
        j = idx.get(c)
        if j is not None: M[i, j] += 1
M = np.log1p(M / np.maximum(P.length.to_numpy()[:, None], 1) * 1000)
Vc = chap(M, lc=True); k, sc = scan(Vc, 1)
print(f"  content chars (top 600, no function chars): split after {k}, next {sorted(sc, key=sc.get, reverse=True)[1:4]}, rank of 80 {sorted(sc, key=sc.get, reverse=True).index(80)+1}")
for name in ("trained_256",):
    V = chap(methods[name]); cc = [np.corrcoef((V @ V.T)[np.triu_indices(120, 1)], (Vc @ Vc.T)[np.triu_indices(120, 1)])[0, 1]]
    print(f"  correlation of chapter-similarity matrices, trained vs content chars: {cc[0]:.2f}")
    print(f"  ... after removing register from trained: {np.corrcoef((resid(V) @ resid(V).T)[np.triu_indices(120,1)], (Vc @ Vc.T)[np.triu_indices(120,1)])[0,1]:.2f}")

print("\n--- placebo AUC (chapter-level logistic), after removing register and/or content ---")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
def placebo(V):
    out = {}
    for b in range(41, 81):
        tr = list(range(b - 40, b - 10)) + list(range(b + 11, b + 41)); te = list(range(b - 10, b + 11))
        y = np.array([c > b for c in tr]); yt = np.array([c > b for c in te])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(V[[c - 1 for c in tr]], y)
        out[b] = roc_auc_score(yt, clf.predict_proba(V[[c - 1 for c in te]])[:, 1])
    nul = [out[b] for b in range(41, 70)]
    return out[80], max(nul), np.mean(nul), np.std(nul)
def remove(V, Z):
    Z = np.c_[Z, np.ones(len(Z))]; R = V - Z @ np.linalg.lstsq(Z, V, rcond=None)[0]; return R / np.linalg.norm(R, axis=1, keepdims=True)
Freg = ((feat - feat.mean()) / feat.std()).to_numpy()
Mn = M.copy()  # content chars per passage -> chapter means, top PCs
Cc = np.stack([Mn[ch == c].mean(0) for c in range(1, 121)]); Cpc = PCA(10, random_state=0).fit_transform((Cc - Cc.mean(0)) / (Cc.std(0) + 1e-9))
print(f"  {'variant':42s} AUC@80  null max  null mean±sd   z")
for name in ("trained_256", "trained_256_masked", "base_qwen", "function_chars"):
    V = chap(methods[name])
    for lab, Vv in [("as is", V), ("- register", remove(V, Freg)), ("- content PCs(10)", remove(V, Cpc)), ("- register - content", remove(V, np.c_[Freg, Cpc]))]:
        a, mx, mu, sd = placebo(Vv); print(f"  {name+' '+lab:42s} {a:.2f}    {mx:.2f}     {mu:.2f}±{sd:.2f}   {(a-mu)/sd:+.1f}")
a, mx, mu, sd = placebo(Vc); print(f"  {'content chars only':42s} {a:.2f}    {mx:.2f}     {mu:.2f}±{sd:.2f}   {(a-mu)/sd:+.1f}")
