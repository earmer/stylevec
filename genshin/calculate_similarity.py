#!/usr/bin/env python3
"""
Calculate Similarity for Genshin Dialogue
Computes similarity between origin_text and para_text using simlar library.
Processes records in batches and updates the similarity column.
"""

import sqlite3
import sys
import time
from pathlib import Path

try:
    import simlar
except ImportError:
    print("ERROR: simlar library not found. Please install it first.")
    print("  cd ../simlar && maturin develop")
    sys.exit(1)

DB_FILE = Path(__file__).parent / "genshin.db"
BATCH_SIZE = 10000
NGRAM_SIZE = 3


class DatabaseManager:
    """SQLite operations for similarity calculation"""

    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()

    def count_pending(self):
        """Count records needing similarity calculation"""
        self.cursor.execute(
            'SELECT COUNT(*) FROM dialogues '
            'WHERE para_text IS NOT NULL AND similarity IS NULL'
        )
        return self.cursor.fetchone()[0]

    def fetch_batch(self, batch_size):
        """Fetch next batch of records needing similarity calculation"""
        self.cursor.execute(
            'SELECT id, origin_text, para_text FROM dialogues '
            'WHERE para_text IS NOT NULL AND similarity IS NULL '
            'ORDER BY id LIMIT ?',
            (batch_size,)
        )
        return self.cursor.fetchall()

    def update_batch(self, updates):
        """Update similarity values. updates = [(similarity, id), ...]"""
        self.cursor.executemany(
            'UPDATE dialogues SET similarity = ? WHERE id = ?',
            updates
        )
        self.conn.commit()

    def get_statistics(self):
        """Get similarity statistics"""
        self.cursor.execute(
            'SELECT COUNT(*), AVG(similarity), MIN(similarity), MAX(similarity) '
            'FROM dialogues WHERE similarity IS NOT NULL'
        )
        return self.cursor.fetchone()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class SimilarityCalculator:
    """Calculate similarity using simlar library"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.processed_count = 0
        self.skipped_count = 0

    def process_all(self):
        """Process all pending records in batches"""
        pending = self.db.count_pending()
        print(f"Records to process: {pending:,}")
        print(f"Batch size: {BATCH_SIZE:,}")
        print(f"N-gram size: {NGRAM_SIZE}")
        print()

        if pending == 0:
            print("No records to process.")
            return

        start_time = time.time()

        while True:
            batch = self.db.fetch_batch(BATCH_SIZE)
            if not batch:
                break

            batch_start = time.time()
            success = self._process_batch(batch)
            batch_elapsed = time.time() - batch_start

            if success:
                self.processed_count += len(batch)
                progress = (self.processed_count / pending) * 100
                speed = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
                print(f"Progress: {self.processed_count:,}/{pending:,} "
                      f"({progress:.1f}%) - {speed:.0f} records/sec")

        elapsed = time.time() - start_time
        self._print_summary(pending, elapsed)

    def _process_batch(self, batch):
        """Process a single batch of records"""
        ids = []
        pairs = []

        for row_id, origin_text, para_text in batch:
            # Skip if either text is empty
            if not origin_text or not origin_text.strip():
                self.skipped_count += 1
                continue
            if not para_text or not para_text.strip():
                self.skipped_count += 1
                continue

            ids.append(row_id)
            pairs.append((origin_text, para_text))

        if not pairs:
            return True

        try:
            # Calculate similarities using simlar
            scores = simlar.batch_avg_similarity(pairs, NGRAM_SIZE)

            # Prepare updates
            updates = list(zip(scores, ids))

            # Write to database
            self.db.update_batch(updates)

            return True

        except Exception as e:
            print(f"ERROR: Batch processing failed - {e}")
            return False

    def _print_summary(self, total_pending, elapsed):
        """Print final processing summary"""
        count, avg, min_val, max_val = self.db.get_statistics()

        print()
        print("=" * 60)
        print("SIMILARITY CALCULATION SUMMARY")
        print("=" * 60)
        print(f"Records processed:     {self.processed_count:,}")
        print(f"Records skipped:       {self.skipped_count:,}")
        print(f"Total with similarity: {count:,}")
        print(f"Processing time:       {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average speed:         {self.processed_count / elapsed:.0f} records/sec")
        print()
        print("Similarity Statistics:")
        if avg is not None:
            print(f"  Average: {avg:.4f}")
            print(f"  Minimum: {min_val:.4f}")
            print(f"  Maximum: {max_val:.4f}")
        else:
            print("  No statistics available")
        print()


def main():
    """Main execution flow"""
    print()
    print("=" * 60)
    print("Genshin Dialogue Similarity Calculator")
    print("=" * 60)
    print()

    # Check database exists
    if not DB_FILE.exists():
        print(f"ERROR: Database not found: {DB_FILE}")
        sys.exit(1)

    # Open database
    db = DatabaseManager(DB_FILE)

    try:
        # Process all records
        calculator = SimilarityCalculator(db)
        calculator.process_all()

        print("✓ Similarity calculation completed!")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == '__main__':
    main()
