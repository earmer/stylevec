# 原神对话 Mask 映射

根据 ReSearchDiary.md 中的规则，将原神对话中的内容信息 mask 掉，保留风格信息。

## 文件说明

- **words-2.json** - 原神词库数据（包含中文、日文、英文等多语言）
- **cn_word_to_mask.json** - 中文词汇到 mask 的映射字典（5021个词）
- **generate_mask.py** - 生成 mask 映射的脚本
- **demo.py** - 从数据库中提取句子并展示 mask 效果的演示脚本

## Mask 规则

使用中文 mask，共 12 种：

| 原词类型 | Mask | 数量 |
|---------|------|------|
| 角色/称号 | ta | 1594 |
| 任务 | 任务 | 609 |
| 地点/设施 | 那里 | 535 |
| 物品/宝石 | 东西 | 418 |
| 敌人 | 敌人 | 417 |
| 食物 | 食物 | 407 |
| 掉落物/材料 | 战利品 | 364 |
| 武器 | 武器 | 235 |
| 圣遗物 | 圣遗物 | 177 |
| 生物 | 动物 | 110 |
| 秘境 | 秘境 | 83 |
| 组织 | 他们 | 72 |

**覆盖率**: 82.9% (4975/6050)

**过滤策略**: 移除了 46 个常用词（如"苹果"、"蘑菇"等通用食物名），但保留所有游戏专有名词（角色、称号、组织、敌人、秘境等），即使它们是常用词（如"博士"、"少女"、"公子"）。

## 使用方法

```python
import json

# 加载映射
with open('cn_word_to_mask.json', 'r', encoding='utf-8') as f:
    word_to_mask = json.load(f)

# 应用 mask
def apply_mask(text, word_to_mask):
    masked_text = text
    # 按词长度从长到短排序，避免短词覆盖长词
    sorted_words = sorted(word_to_mask.items(), key=lambda x: len(x[0]), reverse=True)
    for word, mask in sorted_words:
        if word in masked_text:
            masked_text = masked_text.replace(word, mask)
    return masked_text

# 示例
text = "派蒙在蒙德遇到了迪奥娜"
masked = apply_mask(text, word_to_mask)
print(masked)  # "ta在那里遇到了ta"
```

## 演示

运行 `python demo.py` 可以从数据库中随机提取30个句子并展示 mask 效果。
