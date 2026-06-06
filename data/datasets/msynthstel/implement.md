## 完整流程

如下所有的工作均使用uv（不能直接使用python3，要使用UV！）, node完成。

核心流水线脚本和中间数据放在pipeline/datadelta中；原始语料放在corpora/，翻译输出放在data/translated/，分析结果放在analysis/。

---

### 0. 数据源
语言： en, zh, ru, es, de, fr, ja, ko, it, pt, th, da, sv, tr, nl, pl, hu, ro, hbs, id, bg, el, ar, nb, fi, he, uk, cs, fa, ms, sk, ca, vi, hi, bn, lt, sl, la, et, az, lv, ur, ta, gl, sq, ne, mk, af, tl, sw, eu, is, ka, hy, my, nn, ml, mn, be, uz, mr, si, te, kk, mt, so, gu, kn, cy, ga, tt, pa, eo, ps, ky
**Viet-Mistral/CulturaY**（HuggingFace，JSONL 格式）已带语言标签，无需从零做语言检测。

按照语言码随机读取20k文档（主要语言，前6种）以及1k文档（非主要语言），保存到本地。

中文除外，中文从Skywork/SkyPile-150B中读取20k，不读取CulturaY的。

---

### 1. 分句

**工具**：`Intl.Segmenter`（Node 原生）

```javascript
const segmenter = new Intl.Segmenter(langCode, { granularity: 'sentence' });
const sentences = [...segmenter.segment(text)].map(s => s.segment);
```

低资源语言标点体系规范，规则级精度在这里够用。无额外依赖。

---

### 2. 长度过滤

**工具**：tiktoken

```
下界：15 token
上界：512 token（或对应约 120 token）
```

过宽过窄的句子此步淘汰，后续无需再处理。

---

### 3. 启发式规则过滤

**工具**：自写，参考 Gopher / C4 规则集，**阈值分语言调**

核心检查项：
- 句末无终止标点（`.!?。！？……` 等，按语言配置，要对于小语言支持好）
- 数字或特殊字符（注意小语言的情况）占比 > 30%
- 重复词占比 > 30%（词表长度 / 总词数）
- 含 URL、HTML 残留
- 行内多次 `|`、`//`、`>>` 等模板残留特征

---

### 4. MinHash 近重复去重

**工具**：[`mnemonist`](https://github.com/Yomguithereal/mnemonist)（Node，内置 MinHash）

```javascript
import { MinHash } from 'mnemonist';
// 5-gram 字符 shingle，Jaccard 阈值 0.75
```

字符级 shingle 而非词级，对多语言更稳定。这步去掉模板句、重复模式，保留面貌相近但不同语言的句子（它们 shingle 不重叠）。

---

### 5. 质量打分

分两条路：

**覆盖语言（前 40 种）**
→ **LLM**，../../../artifacts/base-models/qwen-3-0.6b 
计算Perplexity，归一化后去掉首尾。

**尾部语言（其余尾部）**
→ 直接用 **LiteLLM API（调用模型参照translate.py）** 打分，prompt 要求返回 0–1 质量分，1k 句子 token 消耗可忽略

```
判断维度：语法完整性、信息密度、是否为自然语言句子（要宽松）
```

---

### 6. 语义去重

**工具**：本地Ollama模型——granite-embedding:278m

对每句生成 384 维 embedding，内存内做余弦相似度矩阵，贪心去重（相似度 > 0.85 保留较早一句）。

20k 量级完全可以内存内运算，无需向量数据库。

---

### 7. 分桶采样

**工具**：自写

按 token 数分桶，各桶按目标比例采样，避免中等长度句子主导最终分布：

```
桶 A：15–40 字符
桶 B：40–120 字符  
桶 C：120–256 字符
桶 D：256–512 字符

比例建议：25 / 40 / 25 / 10
```

最终按语言配额截取（主体语言 20k，尾部 1k）。

---

### 8. 输出

Parquet格式，存储在pipeline/datadelta/multilang.parquet，scheme：

```json
{
  "text": "...",
  "lang": "zh",
  "length_chars": 42,
  "perplexity": 128.3,
  "quality_score": 0.87,
  "bucket": "B"
}
```

保留元数据字段，方便后续按需重新过滤，不必重跑整条管道。

---
