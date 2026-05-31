#!/usr/bin/env python3
"""使用 words-2.json 生成中文词汇的 mask 映射"""

import json
from collections import Counter

# 读取 words-2.json
with open("words-2.json", "r", encoding="utf-8") as f:
    words_data = json.load(f)

print(f"总词条数: {len(words_data)}\n")

# 根据 ReSearchDiary.md 定义的映射规则
priority_rules = [
    ("CHARACTER", "ta", {"character-main", "character-sub"}),
    ("TITLE", "ta", {"title", "how-to-call"}),
    ("DOMAIN", "秘境", {"domain"}),
    ("LOCATION", "那里", {"location", "facility"}),
    (
        "QUEST",
        "任务",
        {
            "quest-world",
            "quest-daily",
            "quest-story",
            "quest-archon",
            "quest-selenic",
            "quest-random",
            "quest-tribal",
        },
    ),
    ("ENEMY", "敌人", {"enemy", "enemy-boss", "enemy-legend"}),
    ("FOOD", "食物", {"food"}),
    ("DROP", "战利品", {"drop"}),
    ("ITEM", "东西", {"item", "gemstone"}),
    ("WEAPON", "武器", {"weapon", "sword", "bow", "catalyst", "claymore", "polearm"}),
    ("ARTIFACT", "圣遗物", {"artifact", "artifact-piece"}),
    ("MATERIAL", "战利品", {"weapon-material", "talent-material", "drop-boss"}),
    ("ORGANIZATION", "他们", {"organization", "fatui"}),
    ("CREATURE", "动物", {"living-being"}),
]

# 生成中文词汇到 mask 的映射
cn_word_to_mask = {}
mask_distribution = Counter()

for entry in words_data:
    cn_word = entry.get("zhCN", "")
    tags = set(entry.get("tags", []))

    if not cn_word or not tags:
        continue

    # 按优先级匹配
    for rule_name, mask, rule_tags in priority_rules:
        if tags & rule_tags:
            cn_word_to_mask[cn_word] = mask
            mask_distribution[mask] += 1
            break

print(f"生成的中文词汇映射数: {len(cn_word_to_mask)}\n")

print("=== Mask 分布 ===")
for mask in [
    "ta",
    "那里",
    "秘境",
    "任务",
    "敌人",
    "食物",
    "战利品",
    "东西",
    "武器",
    "圣遗物",
    "他们",
    "动物",
]:
    count = mask_distribution[mask]
    if count > 0:
        print(f"{mask:10s} {count:5d} 条")

# 保存映射
with open("cn_word_to_mask.json", "w", encoding="utf-8") as f:
    json.dump(cn_word_to_mask, f, ensure_ascii=False, indent=2)

print(f"\n中文词汇映射已保存到 cn_word_to_mask.json")
