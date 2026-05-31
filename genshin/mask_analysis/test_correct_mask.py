#!/usr/bin/env python3
"""正确的 mask 应用方式：只保护不在 mask 映射中的常用词"""
import json

def apply_mask_correctly(text, word_to_mask, common_words):
    """
    正确的 mask 流程：
    1. 保护常用词（但排除在 mask 映射中的词）
    2. 应用 mask 映射
    3. 恢复常用词
    """
    if not text:
        return text

    # 步骤1: 保护常用词（但排除在 mask 映射中的词）
    protected_text = text
    protection_map = {}  # 占位符 -> 原词

    # 只保护那些不在 mask 映射中的常用词
    words_to_protect = common_words - set(word_to_mask.keys())

    # 按词长度从长到短排序，避免短词覆盖长词
    sorted_protect = sorted(words_to_protect, key=len, reverse=True)

    for idx, word in enumerate(sorted_protect):
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

print(f"常用词数量: {len(common_words)}")

# 读取 mask 映射
with open('cn_word_to_mask.json', 'r', encoding='utf-8') as f:
    word_to_mask = json.load(f)

print(f"Mask 映射数量: {len(word_to_mask)}")

# 计算需要保护的常用词数量
words_to_protect = common_words - set(word_to_mask.keys())
print(f"需要保护的常用词数量: {len(words_to_protect)}")

# 测试示例
test_cases = [
    "刚结束了「识藏日」的仪式，终于清静了一些。但相比平时，也有些太安静了。",
    "派蒙在蒙德遇到了迪奥娜，她们一起去了猫尾酒馆。",
    "博士是愚人众的执行官，少女也是。",
    "旅行者和派蒙在璃月港吃了很多美食。",
    "灰灰硬硬的女孩子，风晶蝶味道的人，猫叔叔的女儿…",
]

print("\n=== 测试正确的 Mask 流程 ===\n")
for i, text in enumerate(test_cases, 1):
    masked = apply_mask_correctly(text, word_to_mask, common_words)
    print(f"【{i}】")
    print(f"原文: {text}")
    print(f"Mask: {masked}")
    if text != masked:
        print("  ✓ 已 mask")
    else:
        print("  ✗ 未 mask")
    print()
