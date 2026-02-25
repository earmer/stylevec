#!/usr/bin/env python3
"""
Genshin Dialogue Data Cleaning Script
Removes low-quality records: empty, pure ellipsis, em-dash, or ≤3 characters
"""

import sqlite3
import re
from pathlib import Path
from collections import defaultdict

DB_FILE = Path(__file__).parent / "genshin.db"


class DataCleaner:
    """Cleans dialogue data by removing low-quality records"""

    def __init__(self):
        self.conn = sqlite3.connect(str(DB_FILE))
        self.cursor = self.conn.cursor()
        self.removal_stats = defaultdict(int)

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

    def should_remove(self, text):
        """Determine if record should be removed"""
        if not text or text.strip() == "":
            self.removal_stats["empty"] += 1
            return True

        if self.is_pure_ellipsis(text):
            self.removal_stats["pure_ellipsis"] += 1
            return True

        if self.is_pure_emdash(text):
            self.removal_stats["pure_emdash"] += 1
            return True

        # Check text length (including punctuation)
        text_length = len(text.strip())
        if text_length <= 3:
            self.removal_stats["too_short"] += 1
            return True

        return False

    def analyze(self):
        """Analyze data to identify records for removal"""
        print("=" * 60)
        print("DATA CLEANING ANALYSIS")
        print("=" * 60)

        self.cursor.execute('SELECT id, origin_text FROM dialogues')
        records_to_remove = []

        for row_id, text in self.cursor.fetchall():
            if self.should_remove(text):
                records_to_remove.append(row_id)

        print(f"\nRecords to remove:")
        print(f"  Empty text: {self.removal_stats['empty']:,}")
        print(f"  Pure ellipsis (……): {self.removal_stats['pure_ellipsis']:,}")
        print(f"  Pure em-dash (——): {self.removal_stats['pure_emdash']:,}")
        print(f"  Too short (≤3 chars): {self.removal_stats['too_short']:,}")
        print(f"  Total to remove: {len(records_to_remove):,}")
        print()

        return records_to_remove

    def clean(self, records_to_remove):
        """Remove identified records"""
        print("=" * 60)
        print("CLEANING DATA")
        print("=" * 60)

        if not records_to_remove:
            print("No records to remove")
            return

        # Delete in batches
        batch_size = 1000
        for i in range(0, len(records_to_remove), batch_size):
            batch = records_to_remove[i:i + batch_size]
            placeholders = ','.join('?' * len(batch))
            self.cursor.execute(f'DELETE FROM dialogues WHERE id IN ({placeholders})', batch)
            self.conn.commit()
            progress = min(i + batch_size, len(records_to_remove))
            print(f"Deleted: {progress:,} / {len(records_to_remove):,}")

        print()

    def report(self):
        """Generate final report"""
        print("=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)

        self.cursor.execute('SELECT COUNT(*) FROM dialogues')
        remaining = self.cursor.fetchone()[0]

        total_removed = sum(self.removal_stats.values())

        print(f"\nRecords removed: {total_removed:,}")
        print(f"Records remaining: {remaining:,}")
        print(f"Removal rate: {(total_removed / (total_removed + remaining) * 100):.2f}%")
        print()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Genshin Dialogue Data Cleaning".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    cleaner = DataCleaner()

    # Step 1: Analyze
    records_to_remove = cleaner.analyze()

    # Step 2: Clean
    cleaner.clean(records_to_remove)

    # Step 3: Report
    cleaner.report()

    cleaner.close()
    print("✓ Cleaning completed successfully!")
    print()


if __name__ == '__main__':
    main()
