#!/usr/bin/env python3
"""
Genshin Dialogue Data Import Script
Imports genshin.json (38.6MB, 283,972 records) into SQLite database
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
import time

# Configuration
JSON_FILE = Path(__file__).parent / "genshin.json"
DB_FILE = Path(__file__).parent / "genshin.db"
BATCH_SIZE = 1000


class DataValidator:
    """Validates JSON data and generates statistics"""

    def __init__(self):
        self.i_values = defaultdict(int)
        self.empty_text_count = 0
        self.total_records = 0

    def validate(self):
        """Check I field uniqueness and text statistics"""
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.total_records = len(data)

            for record in data:
                i_value = record.get('I')
                self.i_values[i_value] += 1

                text = record.get('T', '')
                if not text or text.strip() == '':
                    self.empty_text_count += 1

            # Count duplicates
            duplicates = sum(1 for count in self.i_values.values() if count > 1)
            duplicate_records = sum(count - 1 for count in self.i_values.values() if count > 1)

            print(f"Total records: {self.total_records:,}")
            print(f"Unique I values: {len(self.i_values):,}")
            print(f"Duplicate I values: {duplicates:,}")
            print(f"Total duplicate records: {duplicate_records:,}")
            print(f"Empty text records: {self.empty_text_count:,}")
            print()

            return True

        except Exception as e:
            print(f"ERROR: Validation failed - {e}")
            return False


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def create_database(self):
        """Create database and table structure"""
        try:
            # Remove existing database if present
            if DB_FILE.exists():
                DB_FILE.unlink()
                print(f"Removed existing database: {DB_FILE}")

            self.conn = sqlite3.connect(str(DB_FILE))
            self.cursor = self.conn.cursor()

            # Create table
            self.cursor.execute('''
                CREATE TABLE dialogues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    origin_text TEXT,
                    para_text TEXT,
                    similarity REAL
                )
            ''')

            self.conn.commit()
            print(f"Database created: {DB_FILE}")
            print("Table 'dialogues' created with schema")
            print()

            return True

        except Exception as e:
            print(f"ERROR: Database creation failed - {e}")
            return False

    def batch_insert(self, records):
        """Insert batch of records"""
        try:
            data = [
                (record.get('S'), record.get('T'), None, None)
                for record in records
            ]

            self.cursor.executemany(
                'INSERT INTO dialogues (speaker, origin_text, para_text, similarity) VALUES (?, ?, ?, ?)',
                data
            )

            self.conn.commit()

        except Exception as e:
            print(f"ERROR: Batch insert failed - {e}")
            self.conn.rollback()
            raise

    def create_index(self):
        """Create index on speaker column"""
        try:
            self.cursor.execute('CREATE INDEX idx_speaker ON dialogues(speaker)')
            self.conn.commit()
            print("Index created on 'speaker' column")

        except Exception as e:
            print(f"ERROR: Index creation failed - {e}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class DataImporter:
    """Handles data import process"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.imported_count = 0

    def import_data(self):
        """Stream import data from JSON file"""
        print("=" * 60)
        print("DATA IMPORT PROCESS")
        print("=" * 60)

        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total = len(data)
            batch = []

            for idx, record in enumerate(data, 1):
                batch.append(record)

                if len(batch) >= BATCH_SIZE:
                    self.db_manager.batch_insert(batch)
                    self.imported_count += len(batch)
                    progress = (idx / total) * 100
                    print(f"Progress: {self.imported_count:,} / {total:,} ({progress:.1f}%)")
                    batch = []

            # Insert remaining records
            if batch:
                self.db_manager.batch_insert(batch)
                self.imported_count += len(batch)
                print(f"Progress: {self.imported_count:,} / {total:,} (100.0%)")

            print()
            return True

        except Exception as e:
            print(f"ERROR: Import failed - {e}")
            return False


class ReportGenerator:
    """Generates import statistics report"""

    def __init__(self, db_manager, imported_count, elapsed_time):
        self.db_manager = db_manager
        self.imported_count = imported_count
        self.elapsed_time = elapsed_time

    def generate(self):
        """Generate final report"""
        print("=" * 60)
        print("IMPORT SUMMARY REPORT")
        print("=" * 60)

        try:
            # Get database statistics
            self.db_manager.cursor.execute('SELECT COUNT(*) FROM dialogues')
            total_rows = self.db_manager.cursor.fetchone()[0]

            self.db_manager.cursor.execute('SELECT COUNT(DISTINCT speaker) FROM dialogues')
            unique_speakers = self.db_manager.cursor.fetchone()[0]

            self.db_manager.cursor.execute('SELECT COUNT(*) FROM dialogues WHERE origin_text IS NULL OR origin_text = ""')
            null_text = self.db_manager.cursor.fetchone()[0]

            print(f"Total records imported: {total_rows:,}")
            print(f"Unique speakers: {unique_speakers:,}")
            print(f"Empty origin_text records: {null_text:,}")
            print(f"Import time: {self.elapsed_time:.2f} seconds")
            print(f"Records per second: {total_rows / self.elapsed_time:.0f}")
            print()
            print(f"Database file: {DB_FILE}")
            print(f"Database size: {DB_FILE.stat().st_size / (1024*1024):.2f} MB")
            print()

        except Exception as e:
            print(f"ERROR: Report generation failed - {e}")


def main():
    """Main execution flow"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Genshin Dialogue Data Import to SQLite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Step 1: Validate data
    validator = DataValidator()
    if not validator.validate():
        sys.exit(1)

    # Step 2: Create database
    db_manager = DatabaseManager()
    if not db_manager.create_database():
        sys.exit(1)

    # Step 3: Import data
    start_time = time.time()
    importer = DataImporter(db_manager)
    if not importer.import_data():
        db_manager.close()
        sys.exit(1)

    # Step 4: Create index
    db_manager.create_index()
    print()

    # Step 5: Generate report
    elapsed_time = time.time() - start_time
    report = ReportGenerator(db_manager, importer.imported_count, elapsed_time)
    report.generate()

    db_manager.close()
    print("✓ Import completed successfully!")
    print()


if __name__ == '__main__':
    main()
