#!/usr/bin/env python3
"""
Dry run cleaning script - outputs candidates to candidate.txt without deleting
"""

import sqlite3
import re
from pathlib import Path
from collections import defaultdict

DB_FILE = Path(__file__).parent / "genshin.db"
OUTPUT_FILE = Path(__file__).parent / "candidate.txt"


class DryRunCleaner:
    """Analyzes data and outputs candidates for removal"""

    def __init__(self):
        self.conn = sqlite3.connect(str(DB_FILE))
        self.cursor = self.conn.cursor()
        self.removal_stats = defaultdict(int)
        self.candidates = []

    def is_pure_ellipsis(self, text):
        """Check if text is only ellipsis (……)"""
        if not text:
            return False
        return text.strip() == "……"

    def is_pure_emdash(self, text):
        """Check if text is only em-dash (——)"""
        if not text:
            return False
        return text.strip() == "——"

    def count_chars(self, text):
        """Count meaningful characters (excluding punctuation)"""
        if not text:
            return 0

        text = text.strip()
        if not text:
            return 0

        # Remove common punctuation and whitespace
        cleaned = re.sub(r'[，。！？；：""''（）【】《》、·…—\s]', '', text)

        return len(cleaned)

    def should_remove(self, text):
        """Determine if record should be removed and return reason"""
        if not text or text.strip() == "":
            self.removal_stats["empty"] += 1
            return "empty"

        if self.is_pure_ellipsis(text):
            self.removal_stats["pure_ellipsis"] += 1
            return "pure_ellipsis"

        if self.is_pure_emdash(text):
            self.removal_stats["pure_emdash"] += 1
            return "pure_emdash"

        # Check text length (including punctuation)
        text_length = len(text.strip())
        if text_length <= 3:
            self.removal_stats["too_short"] += 1
            return f"too_short({text_length})"

        return None

    def analyze(self):
        """Analyze data and collect candidates"""
        print("=" * 60)
        print("DRY RUN ANALYSIS")
        print("=" * 60)
        print()

        self.cursor.execute('SELECT id, speaker, origin_text FROM dialogues')

        for row_id, speaker, text in self.cursor.fetchall():
            reason = self.should_remove(text)
            if reason:
                self.candidates.append((row_id, speaker, text, reason))

        print(f"Removal breakdown:")
        print(f"  Empty text: {self.removal_stats['empty']:,}")
        print(f"  Pure ellipsis (……): {self.removal_stats['pure_ellipsis']:,}")
        print(f"  Pure em-dash (——): {self.removal_stats['pure_emdash']:,}")
        print(f"  Single word or less: {self.removal_stats['single_word']:,}")
        print(f"  Total candidates: {len(self.candidates):,}")
        print()

    def write_candidates(self):
        """Write candidates to file"""
        print(f"Writing candidates to {OUTPUT_FILE}...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Total candidates for removal: {len(self.candidates)}\n")
            f.write(f"Removal rate: {(len(self.candidates) / (len(self.candidates) + (283972 - len(self.candidates))) * 100):.2f}%\n")
            f.write("=" * 80 + "\n\n")

            for row_id, speaker, text, reason in self.candidates:
                f.write(f"[ID: {row_id}] [{reason}] {speaker}: {text}\n")

        print(f"✓ Written {len(self.candidates):,} candidates to {OUTPUT_FILE}")
        print()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Dry Run: Genshin Data Cleaning".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    cleaner = DryRunCleaner()
    cleaner.analyze()
    cleaner.write_candidates()
    cleaner.close()

    print("✓ Dry run completed!")
    print()


if __name__ == '__main__':
    main()
