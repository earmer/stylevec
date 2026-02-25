# 对话改写使用指南

## 概述

`paraphrase_dialogues.py` 批量发送对话到兼容 OpenRouter 的大语言模型进行改写。处理所有 `dialogues` 表中 `para_text IS NULL` 的行，将结果写回数据库后停止。重新运行可继续处理。

## 前置要求

- Python 3.7+ 及 `requests` 库（`pip install requests`）
- OpenRouter API 密钥（https://openrouter.ai）
- SQLite 数据库，包含 `dialogues` 表（字段：`id`、`speaker`、`origin_text`、`para_text`）

## 配置

编辑 `conf.json`（默认位于脚本同目录）：

```json
{
    "api": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": "sk-or-v1-YOUR_REAL_KEY_HERE",
        "model": "anthropic/claude-sonnet-4",
        "temperature": 0.7,
        "max_tokens": 4096
    },
    "processing": {
        "batch_size": 50,
        "max_retries": 3,
        "retry_delay": 5,
        "concurrency": 3
    },
    "prompt": {
        "system": "You are a text paraphrasing assistant...",
        "user_template": "{lines}"
    }
}
```

必需的键：`api.base_url`、`api.api_key`、`api.model`、`prompt.system`。其他所有键都有合理的默认值（如上所示）。

## 命令行界面

```
paraphrase_dialogues.py [-h] [-d] [-c CONFIG] database
```

| 参数 | 说明 |
|----------|-------------|
| `database` | SQLite 数据库文件路径（必需） |
| `-d, --dry-run` | 仅预览格式 — 不进行 API 调用，不写入数据库 |
| `--damp-run [N]` | 测试运行 N 个批次（默认 30）并生成质量和成本指标报告，不写入数据库 |
| `-c, --config` | conf.json 路径（默认：脚本目录下的 `conf.json`） |

## 使用方法

### 干运行（预览格式，不进行 API 调用）

```bash
python3 paraphrase_dialogues.py genshin.db --dry-run
```

```
DRY RUN - Format Preview (no API calls, no DB writes)
Remaining records: 264,221

Batch 1: 50 records, IDs 1..50
  | 派蒙：从风魔龙手里成功抢下一座遗迹了呢。
  | 旅行者：那，我们就先走咯！
  | 派蒙：呼，可算是全部处理完了。
  | ... (47 more lines)
```

### 湿运行（测试 N 个批次，生成质量和成本报告，不写入数据库）

```bash
python3 paraphrase_dialogues.py genshin.db --damp-run 10
```

测试指定数量的批次（默认 30），生成成功率、相似度、长度变化、令牌估计和时间估计等指标。

### 正常处理

```bash
python3 paraphrase_dialogues.py genshin.db
```

```
Remaining records: 264,221
Concurrency: 3 workers

Batch 1: 50 records, IDs 1..50 submitted
Batch 2: 50 records, IDs 51..100 submitted
Batch 3: 50 records, IDs 101..150 submitted
  Batch 1 OK
  Batch 3 OK
  Batch 2 FAILED (skipped)

==================================================
PROCESSING SUMMARY
==================================================
  Batches processed: 2
  Batches failed:    1
  Records written:   100
  Remaining:         264,121 / 264,221
```

### 继续处理

重新运行相同的命令。脚本会查询 `WHERE para_text IS NULL`，因此会从中断处继续。失败的批次会被重试。`Ctrl+C` 是安全的 — 每个批次成功时都会提交到数据库。

## 工作流程

1. `python3 paraphrase_dialogues.py genshin.db --dry-run` — 验证格式
2. `python3 paraphrase_dialogues.py genshin.db --damp-run 10` — 测试质量和成本
3. `python3 paraphrase_dialogues.py genshin.db` — 正式运行
4. `sqlite3 genshin.db "SELECT id, origin_text, para_text FROM dialogues WHERE para_text IS NOT NULL LIMIT 5"` — 抽查结果

## 故障排除

### "Config error: api.api_key must be set to a real key"

**问题：** `conf.json` 中的占位符 API 密钥尚未被替换。

**解决方案：** 编辑 `conf.json` 并将 `sk-or-v1-xxxxxxxxxxxx` 替换为你的真实 OpenRouter API 密钥。

### "Line count mismatch: expected 50, got 48"

**问题：** API 返回的行数少于预期（模型没有遵循指令）。

**解决方案：**
- 脚本会自动重试（最多 `max_retries` 次）
- 如果持续失败，该批次会被跳过，下次运行时仍为 NULL
- 尝试调整 `conf.json` 中的系统提示词，使其更明确
- 考虑使用不同的模型

### "Rate limited (429)"

**问题：** 向 OpenRouter API 发送了过多请求。

**解决方案：**
- 脚本会自动等待并使用指数退避重试
- 在 `conf.json` 中减小 `batch_size` 以发送更小的请求
- 等待几分钟后重试
