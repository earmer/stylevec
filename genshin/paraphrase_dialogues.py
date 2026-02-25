#!/usr/bin/env python3
"""
Batch Paraphrase Script for Genshin Dialogue
Sends batches of origin_text to OpenRouter API and writes paraphrased results back.
Processes all records sequentially by ID with concurrent API calls.
Supports resumability (WHERE para_text IS NULL), dry-run and damp-run mode.
"""

import argparse
import concurrent.futures
import json
import math
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

import requests

try:
    import simlar
    HAS_SIMLAR = True
except ImportError:
    HAS_SIMLAR = False

DEFAULT_CONFIG = Path(__file__).parent / "conf.json"


class Config:
    """Load and validate conf.json, expose settings as attributes"""

    def __init__(self, config_path):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # API settings (base_url, model, api_key are required — KeyError if missing)
        api = data['api']
        self.base_url = api['base_url']
        self.api_key = api['api_key']
        self.model = api['model']
        self.temperature = api.get('temperature', 0.7)
        self.max_tokens = api.get('max_tokens', 4096)
        self.reasoning = api.get('reasoning')

        if not self.api_key or self.api_key == 'sk-or-v1-xxxxxxxxxxxx':
            raise ValueError("api.api_key must be set to a real key in conf.json")

        # Processing settings (all optional with defaults)
        proc = data.get('processing', {})
        self.batch_size = proc.get('batch_size', 50)
        self.max_retries = proc.get('max_retries', 3)
        self.retry_delay = proc.get('retry_delay', 5)
        self.concurrency = proc.get('concurrency', 3)

        # Prompt settings (system is required — KeyError if missing)
        prompt = data['prompt']
        self.system_prompt = prompt['system']
        self.user_template = prompt.get('user_template', '{lines}')
        self.strict_prompt = prompt.get('strict', '')


class DatabaseManager:
    """SQLite operations: fetch batch, write batch, count remaining"""

    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()

    def count_remaining(self):
        """Count all records with para_text IS NULL"""
        with self._lock:
            self.cursor.execute(
                'SELECT COUNT(*) FROM dialogues WHERE para_text IS NULL'
            )
            return self.cursor.fetchone()[0]

    def fetch_batch(self, batch_size, offset=0):
        """Fetch next batch of records needing paraphrase, ordered by ID"""
        with self._lock:
            self.cursor.execute(
                'SELECT id, speaker, origin_text FROM dialogues '
                'WHERE para_text IS NULL '
                'ORDER BY id LIMIT ? OFFSET ?',
                (batch_size, offset)
            )
            return self.cursor.fetchall()

    def write_batch(self, updates):
        """Write paraphrased text back. updates = list of (para_text, id)"""
        with self._lock:
            self.cursor.executemany(
                'UPDATE dialogues SET para_text = ? WHERE id = ?',
                updates
            )
            self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class APIClient:
    """POST to OpenRouter, extract response content, raise on errors"""

    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
        })

    def paraphrase(self, lines_text, strict=False):
        """Send lines to API and return dict with content, usage, and elapsed time"""
        user_content = self.config.user_template.replace('{lines}', lines_text)
        user_content = user_content.replace('{strict}', self.config.strict_prompt if strict else '')

        payload = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'messages': [
                {'role': 'system', 'content': self.config.system_prompt},
                {'role': 'user', 'content': user_content},
            ],
        }

        if self.config.reasoning:
            payload['reasoning'] = self.config.reasoning

        t0 = time.time()
        resp = self.session.post(self.config.base_url, json=payload, timeout=120)
        elapsed = time.time() - t0

        if resp.status_code == 429:
            raise RateLimitError("Rate limited (429)")

        resp.raise_for_status()

        data = resp.json()
        content = data['choices'][0]['message']['content']
        return {
            'content': content.strip(),
            'usage': data.get('usage'),
            'elapsed': elapsed,
        }


class RateLimitError(Exception):
    pass


class SpeakerMismatchError(Exception):
    """Raised when API response speaker doesn't match input speaker"""
    pass


class BatchProcessor:
    """Core loop: fetch batches → dispatch to thread pool → collect results"""

    def __init__(self, config, db, api_client, error_log_path=None):
        self.config = config
        self.db = db
        self.api = api_client
        self._error_log_path = error_log_path
        self._stats_lock = threading.Lock()
        self.stats = {
            'batches_processed': 0,
            'batches_failed': 0,
            'records_written': 0,
        }

    def _prepare_lines(self, speakers, texts):
        """Format lines as 'speaker：text', stripping internal newlines"""
        lines = []
        for speaker, text in zip(speakers, texts):
            cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()
            if not speaker or not speaker.strip():
                raise ValueError(f"Empty speaker for text: {cleaned_text[:50]}...")
            formatted_line = f"{speaker.strip()}：{cleaned_text}"
            lines.append(formatted_line)
        return '\n'.join(lines)

    def _parse_response(self, response_text, expected_speakers):
        """Parse response, validate speakers, return paraphrased text only"""
        lines = [l for l in response_text.split('\n') if l.strip()]
        expected_count = len(expected_speakers)

        if len(lines) != expected_count:
            raise ValueError(
                f"Line count mismatch: expected {expected_count}, got {len(lines)}"
            )

        paraphrased_texts = []
        for i, (line, expected_speaker) in enumerate(zip(lines, expected_speakers)):
            prefix = f"{expected_speaker.strip()}："
            if not line.startswith(prefix):
                raise SpeakerMismatchError(
                    f"Line {i+1} speaker mismatch: expected prefix '{prefix}', "
                    f"got '{line[:len(prefix)+10]}'"
                )
            paraphrased_text = line[len(prefix):].strip()

            if not paraphrased_text:
                raise ValueError(
                    f"Line {i+1} has empty paraphrased text for speaker "
                    f"'{expected_speaker}'"
                )

            paraphrased_texts.append(paraphrased_text)

        return paraphrased_texts

    def _log_error(self, batch_label, attempt, error, response_text):
        """Append LLM response to error log for post-mortem analysis"""
        log_path = self._error_log_path
        if not log_path:
            return
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{ts}] {batch_label} attempt {attempt}: {error}\n")
                f.write(f"--- LLM response ---\n{response_text}\n")
                f.write(f"--- end ---\n\n")
        except OSError:
            pass

    def _process_one_batch(self, batch, batch_label):
        """Process a single batch with retries. Returns True on success."""
        ids = [row[0] for row in batch]
        speakers = [row[1] for row in batch]
        texts = [row[2] for row in batch]
        lines_text = self._prepare_lines(speakers, texts)
        use_strict = False

        for attempt in range(1, self.config.max_retries + 1):
            response_text = None
            try:
                result = self.api.paraphrase(lines_text, strict=use_strict)
                response_text = result['content']
                parsed = self._parse_response(response_text, speakers)
                updates = list(zip(parsed, ids))
                self.db.write_batch(updates)
                with self._stats_lock:
                    self.stats['records_written'] += len(updates)
                return True
            except RateLimitError:
                delay = self.config.retry_delay * attempt
                print(f"  {batch_label} rate limited, waiting {delay}s "
                      f"(attempt {attempt}/{self.config.max_retries})")
                time.sleep(delay)
            except (SpeakerMismatchError, ValueError) as e:
                print(f"  {batch_label} {e} "
                      f"(attempt {attempt}/{self.config.max_retries})")
                if response_text:
                    self._log_error(batch_label, attempt, e, response_text)
                use_strict = True
                time.sleep(self.config.retry_delay)
            except requests.RequestException as e:
                print(f"  {batch_label} HTTP error: {e} "
                      f"(attempt {attempt}/{self.config.max_retries})")
                time.sleep(self.config.retry_delay)

        return False

    def run(self):
        """Main processing loop with producer-consumer pattern"""
        remaining = self.db.count_remaining()
        print(f"Remaining records: {remaining:,}")
        print(f"Concurrency: {self.config.concurrency} workers")
        print()

        if remaining == 0:
            print("Nothing to process.")
            return

        work_queue = queue.Queue(maxsize=self.config.concurrency * 2)
        batch_num_lock = threading.Lock()
        batch_num = 0

        def producer():
            """Continuously fetch batches and enqueue them"""
            offset = 0
            while True:
                batch = self.db.fetch_batch(self.config.batch_size, offset)
                if not batch:
                    break
                work_queue.put(batch)
                offset += len(batch)

            # Send stop signal to all workers
            for _ in range(self.config.concurrency):
                work_queue.put(None)

        def worker():
            """Fetch batches from queue and process them"""
            nonlocal batch_num
            while True:
                batch = work_queue.get()
                if batch is None:  # Stop signal
                    break

                with batch_num_lock:
                    batch_num += 1
                    num = batch_num

                ids = [row[0] for row in batch]
                label = f"Batch {num}"
                print(f"{label}: {len(batch)} records, IDs {ids[0]}..{ids[-1]}")

                success = self._process_one_batch(batch, label)

                with self._stats_lock:
                    if success:
                        self.stats['batches_processed'] += 1
                        print(f"  {label} OK")
                    else:
                        self.stats['batches_failed'] += 1
                        print(f"  {label} FAILED (skipped)")

        # Start producer thread (daemon mode)
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Start worker threads
        workers = [
            threading.Thread(target=worker)
            for _ in range(self.config.concurrency)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        self._print_summary(remaining)

    def _print_summary(self, initial_remaining):
        """Print final processing stats"""
        now_remaining = self.db.count_remaining()
        print()
        print("=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"  Batches processed: {self.stats['batches_processed']}")
        print(f"  Batches failed:    {self.stats['batches_failed']}")
        print(f"  Records written:   {self.stats['records_written']:,}")
        print(f"  Remaining:         {now_remaining:,} / {initial_remaining:,}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch paraphrase Genshin dialogue via OpenRouter API'
    )
    parser.add_argument(
        'database',
        help='Path to the SQLite database file'
    )
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Format preview only (no API calls, no DB writes)'
    )
    parser.add_argument(
        '--damp-run',
        nargs='?', const=30, type=int, metavar='N',
        help='Test run N batches (default 30) with metrics report, no DB writes'
    )
    parser.add_argument(
        '-c', '--config',
        default=str(DEFAULT_CONFIG),
        help='Path to conf.json (default: conf.json in script dir)'
    )
    return parser.parse_args()


def _dry_run(config, db):
    """Show a few batches for format verification, no API calls"""
    remaining = db.count_remaining()
    print(f"DRY RUN - Format Preview (no API calls, no DB writes)")
    print(f"Remaining records: {remaining:,}")
    print()

    processor = BatchProcessor(config, db, None)
    for i in range(3):
        batch = db.fetch_batch(config.batch_size, offset=i * config.batch_size)
        if not batch:
            break
        ids = [row[0] for row in batch]
        speakers = [row[1] for row in batch]
        texts = [row[2] for row in batch]
        print(f"Batch {i + 1}: {len(batch)} records, IDs {ids[0]}..{ids[-1]}")
        lines_text = processor._prepare_lines(speakers, texts)
        preview = lines_text.split('\n')[:3]
        for line in preview:
            print(f"  | {line}")
        if len(batch) > 3:
            print(f"  | ... ({len(batch) - 3} more lines)")
        print()

    if remaining > config.batch_size * 3:
        print(f"  ... (preview limited to 3 batches)")


class DampRunner:
    """Test run: call API on N batches, report quality and cost metrics"""

    def __init__(self, config, db, api_client, num_batches):
        self.config = config
        self.db = db
        self.api_client = api_client
        self.num_batches = num_batches
        self.processor = BatchProcessor(config, db, api_client)
        self.batch_results = []
        self.success_count = 0
        self.fail_count = 0
        self.total_records = 0

    def run(self):
        remaining = self.db.count_remaining()
        self.total_records = remaining
        sample_size = self.num_batches * self.config.batch_size
        print(f"DAMP RUN - Testing {self.num_batches} batches "
              f"({sample_size} records, no DB writes)")
        print(f"Remaining records: {remaining:,}")
        print()

        if remaining == 0:
            print("Nothing to process.")
            return

        sample_size = min(sample_size, remaining)
        records = self.db.fetch_batch(sample_size)
        if not records:
            print("No records fetched.")
            return

        batches = [
            records[i:i + self.config.batch_size]
            for i in range(0, len(records), self.config.batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            self._process_batch(batch_idx, batch, len(batches))

        self._report()

    def _process_batch(self, batch_idx, batch, total_batches):
        ids = [row[0] for row in batch]
        speakers = [row[1] for row in batch]
        texts = [row[2] for row in batch]
        label = f"Batch {batch_idx + 1}/{total_batches}"

        try:
            lines_text = self.processor._prepare_lines(speakers, texts)
        except ValueError as e:
            print(f"  {label} prepare error: {e}")
            self.fail_count += len(batch)
            return

        print(f"  {label}: {len(batch)} records, IDs {ids[0]}..{ids[-1]} ...",
              end='', flush=True)

        try:
            result = self.api_client.paraphrase(lines_text)
            parsed = self.processor._parse_response(result['content'], speakers)
            print(f" OK ({result['elapsed']:.1f}s)")
            self.batch_results.append({
                'batch_idx': batch_idx,
                'texts': texts,
                'speakers': speakers,
                'parsed': parsed,
                'usage': result['usage'],
                'elapsed': result['elapsed'],
            })
            self.success_count += len(batch)
        except (RateLimitError, SpeakerMismatchError, ValueError,
                requests.RequestException) as e:
            print(f" FAILED: {e}")
            self.fail_count += len(batch)

    def _report(self):
        total_tested = self.success_count + self.fail_count
        print()
        print("=" * 55)
        print(f"DAMP RUN REPORT ({total_tested} records, "
              f"{len(self.batch_results)} successful batches)")
        print("=" * 55)

        rate = self.success_count / total_tested * 100 if total_tested else 0
        print(f"  Success rate:      {self.success_count}/{total_tested} "
              f"({rate:.1f}%)")

        if not self.batch_results:
            print("  No successful batches — cannot compute metrics.")
            print()
            return

        all_origins = []
        all_paras = []
        for br in self.batch_results:
            for text, para in zip(br['texts'], br['parsed']):
                all_origins.append(text)
                all_paras.append(para)

        # Similarity
        if HAS_SIMLAR:
            pairs = list(zip(all_origins, all_paras))
            scores = simlar.batch_avg_similarity(pairs, 3)
            avg_sim = sum(scores) / len(scores)
            print(f"  Avg similarity:    {avg_sim:.4f}")
        else:
            scores = None
            print(f"  Avg similarity:    N/A (simlar not installed)")

        # Length change
        avg_origin_len = sum(len(t) for t in all_origins) / len(all_origins)
        avg_para_len = sum(len(t) for t in all_paras) / len(all_paras)
        len_change_pct = (avg_para_len - avg_origin_len) / avg_origin_len * 100
        sign = '+' if len_change_pct >= 0 else ''
        print(f"  Avg length change: {sign}{len_change_pct:.1f}% "
              f"({avg_origin_len:.1f} -> {avg_para_len:.1f} chars)")

        self._token_report()
        self._print_samples(scores)
        print()

    def _token_report(self):
        total_input = 0
        total_output = 0
        total_elapsed = 0.0
        has_usage = False

        for br in self.batch_results:
            total_elapsed += br['elapsed']
            usage = br.get('usage')
            if usage:
                has_usage = True
                total_input += usage.get('prompt_tokens', 0)
                total_output += usage.get('completion_tokens', 0)

        print()
        print("  --- Extrapolation to full dataset ---")
        print(f"  Total records:     {self.total_records:,}")

        if not has_usage:
            print("  Token info:        N/A (API did not return usage)")
        else:
            scale = (self.total_records / self.success_count
                     if self.success_count else 0)
            est_input = int(total_input * scale)
            est_output = int(total_output * scale)

            sys_words = len(self.config.system_prompt.split())
            user_chars = sum(
                len(text) for br in self.batch_results for text in br['texts']
            )
            denom = sys_words + user_chars
            cache_rate = sys_words / denom * 100 if denom else 0

            print(f"  Est. input tokens: {est_input:,} "
                  f"(cache rate: {cache_rate:.1f}%)")
            print(f"  Est. output tokens:{est_output:,}")

        avg_batch_time = total_elapsed / len(self.batch_results)
        total_batches = math.ceil(
            self.total_records / self.config.batch_size
        )
        est_seconds = avg_batch_time * total_batches
        hours = int(est_seconds // 3600)
        minutes = int((est_seconds % 3600) // 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m {int(est_seconds % 60)}s"
        print(f"  Est. API time:     {time_str} (sequential, no parallelism)")

    def _print_samples(self, scores):
        if not self.batch_results:
            return

        all_items = []
        for br in self.batch_results:
            for text, para in zip(br['texts'], br['parsed']):
                all_items.append(
                    (br['batch_idx'], len(all_items), text, para)
                )

        num_samples = min(10, len(all_items))
        num_batches = len(self.batch_results)
        per_batch = max(1, math.ceil(num_samples / num_batches))

        selected = []
        for br in self.batch_results:
            batch_items = [
                x for x in all_items if x[0] == br['batch_idx']
            ]
            selected.extend(batch_items[:per_batch])
            if len(selected) >= num_samples:
                break
        selected = selected[:num_samples]

        print()
        print(f"  --- Sample Comparisons ({len(selected)} from "
              f"{num_batches} requests) ---")

        for i, (bidx, fidx, origin, para) in enumerate(selected):
            sim_str = f", sim={scores[fidx]:.2f}" if scores else ""
            print(f"  [{i+1}] (batch {bidx+1}{sim_str})")
            o = origin.replace('\n', ' ')[:80]
            p = para.replace('\n', ' ')[:80]
            print(f"    原: {o}")
            print(f"    改: {p}")


def main():
    args = parse_args()

    # Load config
    try:
        config = Config(args.config)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Config error: {e}")
        sys.exit(1)

    # Open database
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    db = DatabaseManager(db_path)

    try:
        if args.dry_run:
            _dry_run(config, db)
            return

        api_client = APIClient(config)

        if args.damp_run is not None:
            DampRunner(config, db, api_client, args.damp_run).run()
            return

        error_log = db_path.with_name('paraphrase_errors.log')
        processor = BatchProcessor(config, db, api_client, error_log)
        processor.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress has been saved.")
    finally:
        db.close()


if __name__ == '__main__':
    main()
