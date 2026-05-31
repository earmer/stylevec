#!/usr/bin/env python3
"""从数据库中提取句子并展示 mask 效果（使用正确的保护-mask-恢复流程）"""
import sqlite3
import json

def apply_mask_correctly(text, word_to_mask, common_words):
    """
    正确的 mask 流程：
    1. 保护常用词（替换为占位符）
    2. 应用 mask 映射
    3. 恢复常用词
    """
    if not text:
        return text

    # 步骤1: 保护常用词
    protected_text = text
    protection_map = {}  # 占位符 -> 原词

    # 按词长度从长到短排序，避免短词覆盖长词
    sorted_common = sorted(common_words, key=len, reverse=True)

    for idx, word in enumerate(sorted_common):
        if word in protected_text:
            placeholder = f"__COMMON_{idx}__"
            protected_text = protected_text.replace(word, placeholder)
            protection_map[placeholder] = word

    # 步骤2: 应用 mask 映射
    masked_text = protected_text
    sorted_mask = sorted(word_to_mask.items(), key=lambda x: len(x[0]), reverse=True)

    for word, mask in sorted_mask:
        if word in masked_text:
            masked_text = masked_text.replace(word, mask)

    # 步骤3: 恢复常用词
    for placeholder, original_word in protection_map.items():
        masked_text = masked_text.replace(placeholder, original_word)

    return masked_text


# 读取常用词列表
common_words = set()
with open('现代汉语常用词.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            word = parts[0]
            common_words.add(word)

# 读取中文词汇的 mask 映射
with open('cn_word_to_mask.json', 'r', encoding='utf-8') as f:
    cn_word_to_mask = json.load(f)

# 连接数据库
conn = sqlite3.connect('../../genshin/genshin.db')
cursor = conn.cursor()

# 随机提取30条对话
cursor.execute("""
    SELECT speaker, origin_text
    FROM dialogues
    WHERE origin_text IS NOT NULL
    AND length(origin_text) > 15
    AND length(origin_text) < 150
    ORDER BY RANDOM()
    LIMIT 30
""")
dialogues = cursor.fetchall()
conn.close()

# 展示效果
print("=" * 80)
print("原文 vs Mask 后效果对比（30个句子）")
print("=" * 80)

for i, (speaker, text) in enumerate(dialogues, 1):
    masked_text = apply_mask_correctly(text, cn_word_to_mask, common_words)

    # 计算 mask 率
    if text != masked_text:
        mask_indicator = "✓"
    else:
        mask_indicator = " "

    print(f"\n【{i}】{mask_indicator} {speaker}")
    print(f"原文: {text}")
    print(f"Mask: {masked_text}")

print("\n" + "=" * 80)
print("说明: ✓ 表示该句子有词汇被 mask")
