import unittest

from export_scanned import (
    SCANNED_SCHEMA,
    paragraph_ocr_revision,
    reassemble_cleaned_paragraphs,
    selected_server_pages,
    selection_reason,
    strict_row,
)


class ExportScannedTest(unittest.TestCase):
    def setUp(self):
        self.body = {
            "start_page": 10,
            "end_page": 20,
            "author": "作者甲",
            "work_title": "作品甲",
            "genre": "小说",
            "role": "primary_text",
            "role_confidence": "high",
            "boundary_confidence": "high",
            "reviewer_model": "gpt-6-luna",
            "evidence": "目录与正文署名核实。",
        }
        self.roles = [{"start_page": 1, "end_page": 9, "role": "paratext", "reason": "封面和目录"}]

    def paragraph(self, text="这是一段长度足够的中文正文内容，应该进入严格主文本语料。这里再加一些文字。", pages=(10,)):
        return {
            "id": f"book:{pages[0]}:0:0",
            "label": "text",
            "text": text,
            "sources": [{"id": f"book:{page}:0:0", "page": page} for page in pages],
        }

    def test_only_paragraphs_inside_one_reviewed_primary_range_are_selected(self):
        row, reason, _ = selection_reason(self.paragraph(pages=(10, 20)), [self.body], self.roles, set())
        self.assertIsNotNone(row)
        self.assertEqual(reason, "selected")

        row, reason, _ = selection_reason(self.paragraph(pages=(9, 10)), [self.body], self.roles, set())
        self.assertIsNone(row)
        self.assertEqual(reason, "outside_one_reviewed_primary_body_range")

    def test_footnotes_short_fragments_and_reviewed_unknowns_are_excluded(self):
        footnote = self.paragraph()
        footnote["label"] = "footnote"
        self.assertEqual(selection_reason(footnote, [self.body], self.roles, set())[1], "ocr_layout_role:footnote")

        short = self.paragraph(text="短文")
        self.assertEqual(selection_reason(short, [self.body], self.roles, set())[1], "strict_chinese_minimum_length_under_30")

        unresolved = {"book:10:0:0"}
        self.assertEqual(selection_reason(self.paragraph(), [self.body], self.roles, unresolved)[1], "boundary_reidentification_requires_review")

        low_range = {**self.body, "boundary_confidence": "medium"}
        self.assertEqual(selection_reason(self.paragraph(), [low_range], self.roles, set())[1], "body_range_not_high_confidence")

        handwritten = {**self.body, "strict_exclusion_reason": "handwritten_ocr_unverified"}
        self.assertEqual(
            selection_reason(self.paragraph(), [handwritten], self.roles, set())[1],
            "reviewed_primary_text_excluded:handwritten_ocr_unverified",
        )

    def test_work_id_is_stable_across_paragraphs_in_same_work(self):
        book = {"book_id": "book", "source_pdf": "/scan.pdf"}
        first = strict_row(self.paragraph(pages=(10,)), self.body, book, "source-a", [10], 4, "ranges.jsonl")
        later = strict_row(self.paragraph(pages=(20,)), self.body, book, "source-b", [20], 9, "ranges.jsonl")
        self.assertEqual(first["work_id"], later["work_id"])
        self.assertNotEqual(first["source_href"], later["source_href"])
        self.assertEqual(first["source_dom_position"], 4)

    def test_schema_columns_follow_primary_strict_contract(self):
        self.assertEqual(len(SCANNED_SCHEMA), 27)
        self.assertEqual(SCANNED_SCHEMA.names[0], "text")
        self.assertEqual(SCANNED_SCHEMA.names[-1], "reference_ids")

    def test_cleanup_rebuilds_text_without_changing_saved_join_chains(self):
        paragraphs = [{
            "id": "joined",
            "label": "text",
            "text": "甲乙丙页码",
            "sources": [
                {"id": "book:10:0:0", "page": 10},
                {"id": "book:11:0:0", "page": 11},
            ],
        }]
        cleaned = {
            "book:10:0:0": {"id": "book:10:0:0", "page": 10, "text": "甲乙"},
            "book:11:0:0": {"id": "book:11:0:0", "page": 11, "text": "丙"},
        }
        rebuilt = reassemble_cleaned_paragraphs(paragraphs, cleaned)
        self.assertEqual(rebuilt[0]["id"], "joined")
        self.assertEqual(rebuilt[0]["sources"], paragraphs[0]["sources"])
        self.assertEqual(rebuilt[0]["text"], "甲乙丙")

    def test_server_selection_covers_full_and_selected_partial_pages(self):
        pages = selected_server_pages(
            {
                "full_books": [{"book": "full", "pages": 3}],
                "selected_pages": [{"book": "part", "page": 2}],
            },
            {
                "full_books": [{"book": "full", "expected_pages": 3}],
                "partial_books": [{
                    "book": "part", "pages": 4,
                    "server_reprocessed_pages": [2],
                    "page_reference_coverage": True,
                    "paragraph_fragment_coverage_verified": True,
                }],
            },
        )
        self.assertEqual(pages, {"full": {1, 2, 3}, "part": {2}})

    def test_partial_server_paragraph_revision_is_per_page(self):
        revisions = {"part": {12}}
        self.assertEqual(paragraph_ocr_revision(self.paragraph(pages=(10,)), "part", revisions), "pdf_ocr-reviewed-v1")
        self.assertEqual(paragraph_ocr_revision(self.paragraph(pages=(12,)), "part", revisions), "pdf_ocr-server-v1")
        self.assertEqual(
            paragraph_ocr_revision(self.paragraph(pages=(11, 12)), "part", revisions),
            "pdf_ocr-reviewed-v1+pdf_ocr-server-v1",
        )


if __name__ == "__main__":
    unittest.main()
