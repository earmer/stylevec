# 扫描书籍 OCR 与严格文本集构建技术报告

**日期：** 2026-09-30  
**输出数据：** [`books_scanned.parquet`](books_scanned.parquet)  
**数据说明：** [`README_books_scanned.md`](README_books_scanned.md)

## 摘要

本报告记录从无文本层扫描 PDF 到 StyleVec 严格正文段落表的处理过程。工作覆盖 20 本书、10,316 页，先比较 OCR 与页面布局推理配置，再对全语料执行 OCR，随后审查跨页段落续接、清理重复扫描页及版面噪声，最后按照 `books_work_primary_strict.parquet` 的 Arrow schema 导出并验证 `books_scanned.parquet`。

最终表有 **40,838 个段落**，覆盖 **20 本书、8 位作者、24 个作品或卷级记录**。保存结果校验确认其字段和类型与参考表完全一致，40,838 个行级 block ID 唯一，覆盖 46,342 个唯一来源 OCR fragment。边界与正文角色是基于页面范围和扫描图像的 GPT-6 Luna 辅助复核结果，不代表逐字人工校对。另有一个因原始扫描缺损而无法判断的跨页边界，继续留在 review 队列并排除在严格数据表之外。

## 目标与约束

目标是在不改变现有 PDF OCR 主模型和原始结果的前提下，提高大批扫描页处理速度，改善跨页续接的可靠性，并产出与现有严格正文数据集可直接对齐的 Parquet 表。源 PDF、原始 OCR 页结果和已审查版本均作为来源证据保留；重新识别或清理产生独立 revision 与 sidecar，避免覆盖原始材料。

无文本层的 PDF 纳入 OCR。文件名标记“有文本层”的 PDF 按约定排除，未读取其文本层。macOS `._` 元数据文件也不计入语料。

## 流水线

处理分为以下阶段：

1. **页面盘点与抽样。** 盘点 20 本书、10,316 页；从每本书约 15%、50%、85% 深度选取相邻页对，形成 120 页的宽样本，观察版面和续页差异。
2. **页面渲染。** 将输入页渲染为 200 DPI RGB PNG，供版面分析、OCR 和源图复核使用。
3. **布局分析与 OCR 并行。** 主配置将 PP-DocLayoutV3 布局检测放在独立 CPU worker，将 OCR 工作分派给两个 CPU worker；worker 间通过有界队列协作，以减少串行空等并限制图像积压。主脚本和 RapidOCR/RapidLayout 模型配置保持不变。
4. **页内文本块与跨页段落构建。** 根据布局块、行框和文本顺序建立页内 fragment，再对相邻页候选边界作 join、separate 或 review 决策。原自动结果保留，图像复核结果单独记录。
5. **扫描清理与严格角色筛选。** 清理结果保留 fragment 来源 ID；删除或排除重复扫描、出版元数据、目录、页眉页码等有图像或结构证据支持的内容。再按作者正文范围、文本角色和置信度筛选严格表。
6. **导出与独立验证。** 导出脚本重组清理后的文本及来源链，写入与参考表完全相同的 Arrow schema；只读 verifier 检查保存好的 Parquet 和 sidecar，不通过重新导出掩盖问题。

页码定位使用一基 PDF 页序号，并写作 `source_href=pdf-page:<book_id>:<page>`；`source_dom_position` 是对应 revision 的 `paragraphs.json` 中零基行序。稳定 `source_block_id` 由书籍 ID 和组成该段落的 fragment ID 摘要生成。

## 性能实验与硬件判断

实验设备为 Apple M2 Pro、32 GiB 内存、12 个逻辑 CPU 核。各配置使用相同渲染图像；性能计时包括 PNG 解码、阶段协调、段落构建和页 JSON 写入，不包括初始 PDF 渲染。峰值 RSS 汇总进程树。宽样本使用 120 页；不同配置间串行运行，避免并发基准互相争用。

| 配置 | 样本页数 | 秒/页 | 峰值进程树 RSS |
| --- | ---: | ---: | ---: |
| CPU 串行，默认 ORT 线程 | 40 | 1.372 | 2.97 GiB |
| CPU 串行，布局 4 / OCR 2 线程 | 40 | 2.114 | 2.31 GiB |
| CPU 布局 worker + 1 OCR worker | 40 | 1.525 | 3.07 GiB |
| CPU 布局 worker + 2 OCR workers | 40 | 0.893 | 4.91 GiB |
| CPU 串行，各阶段 1 线程 | 40 | 4.606 | 2.20 GiB |
| 选定 worker 配置，较宽样本 | 120 | 0.925 | 4.84 GiB |
| CPU 布局 + CoreML ALL OCR | 6 | 6.731 | 13.73 GiB |
| CPU 布局 + CoreML CPU/GPU OCR | 6 | 6.830 | 14.84 GiB |
| CoreML NeuralNetwork 布局 + 2 个 CPU OCR workers | 40 | 0.821 | 9.04 GiB |

在 40 页对照中，两个 OCR worker 相对默认线程 CPU 串行配置快约 **35%**，相对串行 4/2 线程配置快约 **58%**。40 页上，CPU 串行和 worker 配置的文本、块、行框与置信度完全一致。120 页样本的选定配置实测 0.925 秒/页，另测页面渲染平均约 0.093 秒/页。直接线性外推 10,316 页约 2.92 小时；将长短书差异、磁盘波动、热降频和最终导出计入后，预留 **4–6 小时**较稳妥。达到 24 小时目标允许平均约 8.38 秒/页，因此基于此机器的样本速度，CPU 方案有充分余量。该时长是样本外推，不是全量作业墙钟实测。

Apple GPU/CoreML 的尝试没有成为默认配置：MLProgram 布局模型因 MaxPool `ceil_mode` 与 `same` padding 不兼容而不能编译；CoreML OCR 在小样本中明显慢于 CPU，并将峰值内存推高到约 14 GiB。CoreML NeuralNetwork 布局在 40 页小样本中表现较快，但峰值约 9 GiB，未改变整体选择。结论是这批作业无需等待或租用 RTX 5090：已验证的 CPU pipeline 预计一天内完成，GPU 试验未显示出更好的整体成本与稳定性。

## 跨页续接与异常审查

跨页决策并不只看相邻页文本是否“读起来连贯”。候选边界会结合 fragment 来源、页面位置、行框和扫描图像判断：句子或段落在页底明显未完且下一页紧接续写时 join；新段、诗行/诗节、说话者切换或页面结构显示独立正文时 separate；证据不足时保留 review。对诗歌特别检查断行、韵律单元和说话者，避免把有意分行误并成散文段落。

对全量首次审查中标记的 751 个边界，Luna 团队逐项复核；其后对 193 个案例做一致性复核，修正 169 个早期 join 为 separate，主要是保留原有诗行或诗节边界。最终 10,296 个边界决策为 5,727 joins、4,568 separations、1 review；其余 9,545 个未标记自动边界保持不变。所有 751 个被标记边界都有最终处置。

唯一仍未解决的是《周立波文集》第 5 卷 PDF 第 19–20 页：第 19 页扫描图像在一行中途被裁断，缺失部分无法从 OCR 或下一页可靠恢复。因此两侧 fragment 继续分开并留在 `review.json`，不虚构缺文，也不进入严格表。对于不确定的续接，review 输出同时保留两侧原始片段，供以后获取更好扫描或人工查书时复核。

质量审计对 10,316 页标记了 1,505 页供后续检查，包括低置信度行、重叠行框、没有保留内容的页面、疑似页码及疑似截断等。类别可以重叠，这些是复查提示，不是已确认错误数。五个截断候选中，四个经源图检查为日期或标点造成的误报，一个是上述确实缺损页。

## PP-OCRv5 server 的定向复识别

针对《英华之歌》等扫描或识别质量存疑的材料，单独建立 PP-OCRv5 server revision。布局模型保持 PP-DocLayoutV3；server 版检测与识别在 CPU、200 DPI、960 像素检测上限运行。共处理 659 页、5 本书：完整重识别《英华之歌》653 页，另从 4 本书挑选 6 页。完整 653 页运行耗时 **45 分 21 秒**，进程树峰值 RSS 约 **3.55 GiB**，两个 ONNX 权重约 **165 MB**。没有使用付费 GPU。

server revision 的 79 个新边界均由 Luna 查看扫描页图后复核：57 joins、22 separations、0 个新 unresolved。12 个候选端点在图像复核后重映射，因为初始候选落在书眉、页边标记等非正文区域。四本部分升级书的未选页面仍引用原已复核 mobile revision，不将局部 server 输出冒充整本重识别。

v5 server 并非所有页面都更好。例如 `bc87fbaddf81` PDF 第 44 页中很小的“上。”被 mobile OCR 识成“0”，server OCR 则漏检。两名独立复核者依据源图支持派生转录 overlay，在汇总段落中恢复“……草滩小径上。”，同时保留 mobile/server 原始 JSON 不变。这个例子说明更换 OCR 模型仍可能产生召回退化；置信度不是字符准确率，扫描缺损内容也无法靠 OCR 恢复。

## 清洗与 `books_scanned.parquet`

清洗侧保留 78,553 个权威来源 fragment，其中 78,435 个进入清理后来源视图；118 个完整 fragment 因确定的相邻重复页或档案元数据而排除，原因和来源 ID 均记入审计。四处印刷页码行在有扫描图证据时从行内文本移除，fragment ID 保持不变，修改行及文本记录在 sidecar。三组相邻页重复副本经 200 DPI 全页渲染哈希确认，只排除第二份拷贝。目录/元数据、独立标题、采访提问及非作者正文等可在严格训练表排除，但 all-clean 视图保留其 OCR fragment，便于追溯。

20 本书的作者正文范围经页面图像抽样审查；作品集在扫描不能可靠区分单篇边界时以卷或合集作为 `work_id` 粒度。作者签名序、日记、书信、正文中的诗歌/对话/表格等，按来源证据保留。孤立标题和采访者提问等从 strict 表排除。稀有布局标签审查了 26 个图像样例：10 个为正文上下文而保留，16 个孤立块排除；另有 14 个页边/旁注/印刷页码端点不纳入严格行。排除为严格表范围，不表示删除原 OCR。

导出结果为 **40,838 行、27 个字段**，schema 与参考 `books_work_primary_strict.parquet` 逐字段、逐类型相同。表含 20 本书、8 位作者、24 个卷级/作品级 work ID；包含 46,342 个唯一来源 fragment。至少 30 字、中文、正文角色、角色与边界置信度为 high，是 strict 行的硬条件。段落行通过 lineage 指回来源书、PDF 页、assembled 段落位置、OCR revision 与贡献 fragment，并记录文本哈希。

## 验证结果与复现入口

保存结果 verifier 报告 `verified: true`。它检查 schema 指纹与参考表一致、manifest 中所有书与来源 PDF 存在、block ID 与 fragment 唯一、行文本哈希和 lineage 一致、页码/来源链接有效、正文语言/角色/长度/置信度约束成立、sidecar 覆盖所有书，并确认 20 本书的原始 OCR 页文件哈希与复核前基线一致。完整复核版另验证 78,553 个 fragment 的恰好一次覆盖、所有 10,296 个边界决策及 review 文件；v5 revision 验证五本书组装结果、页与 fragment lineage、79 项决策和转录 overlay 的来源记录。

主要代码入口：

- `scripts/pdf_ocr/`：扫描 PDF 主 OCR、并行流水线、状态和质量审计工具。
- `scripts/bookraw/prepare_server_clean_fragments.py`：准备 server revision 的清理 fragment。
- `scripts/bookraw/export_scanned.py`：按 reviewed body ranges 导出严格数据表。
- `scripts/bookraw/verify_scanned_export.py`：只读验证保存好的表与 provenance sidecar。
- `/Volumes/EMC/stylevec_data/experiments/pdf_ocr/report.md`：性能实验、全量审查与 v5 revision 的细节和审计链接。
- `reports/bookraw/README_books_scanned.md`：20 本书范围表、字段语义、限制和 sidecar 索引。
- `reports/bookraw/books_scanned_verification.json`：最终 Parquet 的机器验证结果。

重新执行时使用仓库 `uv` 环境，按本地脚本 `--help` 传入输入与输出目录；不要覆盖 raw OCR revision。全语料质量标记、review JSON、清理/排除审计、body range manifest、lineage 与 verification 文件应与 Parquet 一起保留。

## 限制与使用建议

本数据适用于从扫描书籍建立作者正文段落级研究语料，并支持按页与 OCR fragment 回查。它不是经过逐字人工校对的定本：Luna 的角色和边界判断是模型辅助图像审查，OCR 行置信度也不是准确率。低清、污损、裁切和缺页仍会带来漏字或错字；唯一确认缺损的续页在严格表中被排除。24 个 work ID 中有合集按卷聚合，不能据此假定已准确切分到单篇小说。开展敏感的字符级语言学分析、引文校勘或单篇作品级统计前，应针对目标页面另做人工核验。
