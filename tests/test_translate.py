#!/usr/bin/env python3
"""Unit tests for translate.py — markdown and JSON (Docling) code paths.

Asian languages are emphasised throughout: Simplified Chinese, Traditional
Chinese, Japanese, Korean, Vietnamese, and Thai are all exercised explicitly.

Run with:
    python -m pytest tests/
  or
    python -m unittest discover tests/
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from translate import (
    _collect_paths,
    _strip_code_fences,
    _try_parse_json,
    split_markdown,
    translate_json,
    translate_markdown,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resp(content: str) -> MagicMock:
    """Minimal mock of a single OpenAI chat-completion response."""
    m = MagicMock()
    m.choices[0].message.content = content
    return m


def _client(*contents: str) -> MagicMock:
    """Mock OpenAI client whose create() calls return *contents* in order."""
    c = MagicMock()
    c.chat.completions.create.side_effect = [_resp(s) for s in contents]
    return c


def _docling(*text_items: str, label: str = "paragraph") -> dict:
    """Return a minimal but structurally valid Docling v1.9 document.

    Each positional argument becomes a TextItem.  The *label* applies to
    every item (fine for unit-test fixtures).
    """
    return {
        "schema_name": "DoclingDocument",
        "version": "1.9.0",
        "name": "test",
        "origin": None,
        "body": {
            "self_ref": "#/body",
            "parent": None,
            "children": [f"#/texts/{i}" for i in range(len(text_items))],
            "label": "unspecified",
        },
        "furniture": {
            "self_ref": "#/furniture",
            "parent": None,
            "children": [],
            "label": "unspecified",
        },
        "groups": [],
        "texts": [
            {
                "self_ref": f"#/texts/{i}",
                "parent": "#/body",
                "children": [],
                "label": label,
                "prov": [],
                "orig": t,
                "text": t,
                "content_layer": "body",
                "formatting": None,
                "hyperlink": None,
            }
            for i, t in enumerate(text_items)
        ],
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "form_items": [],
        "pages": {},
    }


def _docling_with_table(*header_texts: str, row_texts: list[str] = ()) -> dict:
    """Return a Docling document that contains one table.

    All header_texts become column-header cells; all row_texts become
    data cells on a single row beneath them.
    """
    cells = []
    for col, t in enumerate(header_texts):
        cells.append({
            "text": t,
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 1,
            "start_col_offset_idx": col,
            "end_col_offset_idx": col + 1,
            "row_span": 1,
            "col_span": 1,
            "column_header": True,
            "row_header": False,
        })
    for col, t in enumerate(row_texts):
        cells.append({
            "text": t,
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "start_col_offset_idx": col,
            "end_col_offset_idx": col + 1,
            "row_span": 1,
            "col_span": 1,
            "column_header": False,
            "row_header": False,
        })
    doc = _docling()
    doc["tables"] = [{
        "self_ref": "#/tables/0",
        "label": "table",
        "parent": "#/body",
        "children": [],
        "content_layer": "body",
        "prov": [],
        "data": {
            "table_cells": cells,
            "num_rows": 2 if row_texts else 1,
            "num_cols": max(len(header_texts), len(row_texts), 1),
        },
    }]
    return doc


# ---------------------------------------------------------------------------
# split_markdown
# ---------------------------------------------------------------------------

class TestSplitMarkdown(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(split_markdown("", 500), [])

    def test_single_paragraph_no_header(self):
        self.assertEqual(split_markdown("Hello world.", 500), ["Hello world."])

    def test_splits_long_content_at_blank_lines(self):
        para = "x" * 100
        content = f"{para}\n\n{para}\n\n{para}"
        chunks = split_markdown(content, 150)
        self.assertGreater(len(chunks), 1)

    def test_single_header_with_body_fits_in_one_chunk(self):
        content = "# Title\n\nSome body text."
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("# Title", chunks[0])

    def test_two_header_sections_produce_two_chunks(self):
        content = "# One\n\nBody one.\n\n# Two\n\nBody two."
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(any("One" in c for c in chunks))
        self.assertTrue(any("Two" in c for c in chunks))

    def test_preamble_before_first_header_is_kept(self):
        content = "Preamble text.\n\n# Section\n\nBody."
        chunks = split_markdown(content, 500)
        self.assertTrue(any("Preamble" in c for c in chunks))
        self.assertTrue(any("Section" in c for c in chunks))

    def test_header_without_body(self):
        content = "# Lone Header"
        chunks = split_markdown(content, 500)
        self.assertEqual(chunks, ["# Lone Header"])

    # Asian-language content — chunking logic is encoding-agnostic but we
    # verify multi-byte characters do not confuse byte-count vs char-count.

    def test_simplified_chinese_with_header(self):
        content = "# 标题\n\n这是正文内容。"
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("标题", chunks[0])

    def test_japanese_two_sections(self):
        content = "# はじめに\n\n導入文です。\n\n# まとめ\n\n結論です。"
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(any("はじめに" in c for c in chunks))
        self.assertTrue(any("まとめ" in c for c in chunks))

    def test_korean_with_header(self):
        content = "# 제목\n\n한국어 본문입니다."
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("제목", chunks[0])

    def test_thai_with_header(self):
        content = "# หัวข้อ\n\nเนื้อหาภาษาไทย"
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 1)
        self.assertIn("หัวข้อ", chunks[0])

    def test_vietnamese_two_sections(self):
        content = "# Giới thiệu\n\nNội dung phần đầu.\n\n# Kết luận\n\nNội dung phần cuối."
        chunks = split_markdown(content, 500)
        self.assertEqual(len(chunks), 2)


# ---------------------------------------------------------------------------
# _collect_paths
# ---------------------------------------------------------------------------

class TestCollectPaths(unittest.TestCase):

    def test_no_filter_collects_all_strings(self):
        data = {"a": "x", "b": "y", "c": 1}
        values = {v for _, v in _collect_paths(data)}
        self.assertEqual(values, {"x", "y"})

    def test_field_filter_only_matching_key(self):
        data = {"text": "hello", "orig": "hello", "label": "paragraph"}
        result = list(_collect_paths(data, field="text"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "hello")

    def test_empty_string_excluded(self):
        data = {"text": "", "orig": "something"}
        self.assertEqual(list(_collect_paths(data, field="text")), [])

    def test_nested_dict(self):
        data = {"outer": {"inner": {"text": "deep"}}}
        result = list(_collect_paths(data, field="text"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "deep")

    def test_list_of_dicts_skips_empty(self):
        data = [{"text": "a"}, {"text": ""}, {"text": "b"}]
        values = [v for _, v in _collect_paths(data, field="text")]
        self.assertEqual(values, ["a", "b"])

    def test_docling_texts_array_only_text_field(self):
        doc = _docling("Hello", "World", "")
        values = [v for _, v in _collect_paths(doc, field="text")]
        self.assertCountEqual(values, ["Hello", "World"])
        # Structural / meta strings must not leak through
        for excluded in ("paragraph", "DoclingDocument", "#/texts/0", "#/body"):
            self.assertNotIn(excluded, values)

    def test_docling_table_cells_collected(self):
        doc = _docling_with_table("Name", "Age", row_texts=["Alice", "30"])
        values = [v for _, v in _collect_paths(doc, field="text")]
        self.assertCountEqual(values, ["Name", "Age", "Alice", "30"])

    def test_simplified_chinese(self):
        data = {"text": "你好世界"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "你好世界")

    def test_traditional_chinese(self):
        data = {"text": "你好世界（繁體）"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "你好世界（繁體）")

    def test_japanese(self):
        data = {"text": "こんにちは世界"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "こんにちは世界")

    def test_korean(self):
        data = {"text": "안녕하세요 세계"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "안녕하세요 세계")

    def test_thai(self):
        data = {"text": "สวัสดีโลก"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "สวัสดีโลก")

    def test_vietnamese(self):
        data = {"text": "Xin chào thế giới"}
        self.assertEqual(list(_collect_paths(data, field="text"))[0][1], "Xin chào thế giới")

    def test_custom_field_name(self):
        data = {"label": "Click here", "text": "ignore me"}
        result = list(_collect_paths(data, field="label"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "Click here")


# ---------------------------------------------------------------------------
# _strip_code_fences  /  _try_parse_json
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    # _strip_code_fences
    def test_strip_json_fence(self):
        self.assertEqual(_strip_code_fences('```json\n{"a":1}\n```'), '{"a":1}')

    def test_strip_plain_fence(self):
        self.assertEqual(_strip_code_fences('```\n{"a":1}\n```'), '{"a":1}')

    def test_no_fence_unchanged(self):
        self.assertEqual(_strip_code_fences('{"a":1}'), '{"a":1}')

    def test_whitespace_stripped(self):
        self.assertEqual(_strip_code_fences('  text  '), 'text')

    # _try_parse_json
    def test_valid_object(self):
        self.assertEqual(_try_parse_json('{"a":1}'), {"a": 1})

    def test_valid_array(self):
        self.assertEqual(_try_parse_json('[1,2]'), [1, 2])

    def test_plain_string_returns_none(self):
        self.assertIsNone(_try_parse_json("hello"))

    def test_number_string_returns_none(self):
        self.assertIsNone(_try_parse_json("42"))

    def test_invalid_json_returns_none(self):
        self.assertIsNone(_try_parse_json("{not json}"))

    def test_nested_object(self):
        obj = {"k": {"a": "b"}}
        self.assertEqual(_try_parse_json(json.dumps(obj)), obj)


# ---------------------------------------------------------------------------
# translate_markdown
# Patch _detect_lang → None so langdetect never runs; this prevents retries
# and makes every test deterministic regardless of fixture string length.
# ---------------------------------------------------------------------------

@patch("translate._detect_lang", return_value=None)
class TestTranslateMarkdown(unittest.TestCase):

    def _run(self, content, target_lang, *mock_texts):
        c = _client(*mock_texts)
        result, retries = translate_markdown(
            c, "model", content, max_chars=4000, target_lang=target_lang
        )
        return result, retries, c

    # --- English → Asian ---

    def test_english_to_simplified_chinese(self, _d):
        zh = "你好，世界！这是一个测试文档。"
        result, retries, _ = self._run(
            "Hello, world! This is a test document.", "Chinese", zh
        )
        self.assertEqual(result, zh)
        self.assertEqual(retries, 0)

    def test_english_to_traditional_chinese(self, _d):
        zh = "你好，世界！這是一個測試文件。"
        result, _, _ = self._run(
            "Hello, world! This is a test document.", "Chinese (Traditional)", zh
        )
        self.assertEqual(result, zh)

    def test_english_to_japanese(self, _d):
        ja = "こんにちは、世界！これはテスト文書です。"
        result, _, _ = self._run(
            "Hello, world! This is a test document.", "Japanese", ja
        )
        self.assertEqual(result, ja)

    def test_english_to_korean(self, _d):
        ko = "안녕하세요, 세계! 이것은 테스트 문서입니다."
        result, _, _ = self._run(
            "Hello, world! This is a test document.", "Korean", ko
        )
        self.assertEqual(result, ko)

    def test_english_to_vietnamese(self, _d):
        vi = "Xin chào, thế giới! Đây là tài liệu thử nghiệm."
        result, _, _ = self._run(
            "Hello, world! This is a test document.", "Vietnamese", vi
        )
        self.assertEqual(result, vi)

    def test_english_to_thai(self, _d):
        th = "สวัสดี โลก! นี่คือเอกสารทดสอบ"
        result, _, _ = self._run(
            "Hello, world! This is a test document.", "Thai", th
        )
        self.assertEqual(result, th)

    # --- Asian → English ---

    def test_simplified_chinese_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run("你好，世界！这是一个测试文档。", "English", en)
        self.assertEqual(result, en)

    def test_traditional_chinese_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run("你好，世界！這是一個測試文件。", "English", en)
        self.assertEqual(result, en)

    def test_japanese_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run(
            "こんにちは、世界！これはテスト文書です。", "English", en
        )
        self.assertEqual(result, en)

    def test_korean_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run(
            "안녕하세요, 세계! 이것은 테스트 문서입니다.", "English", en
        )
        self.assertEqual(result, en)

    def test_vietnamese_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run(
            "Xin chào, thế giới! Đây là tài liệu thử nghiệm.", "English", en
        )
        self.assertEqual(result, en)

    def test_thai_to_english(self, _d):
        en = "Hello, world! This is a test document."
        result, _, _ = self._run(
            "สวัสดี โลก! นี่คือเอกสารทดสอบ", "English", en
        )
        self.assertEqual(result, en)

    # --- Multi-chunk behaviour ---

    def test_two_sections_make_two_llm_calls(self, _d):
        content = "# Intro\n\nIntroduction text.\n\n# Body\n\nBody text."
        zh1, zh2 = "# 介绍\n\n介绍文本。", "# 正文\n\n正文内容。"
        result, _, c = self._run(content, "Chinese", zh1, zh2)
        self.assertEqual(c.chat.completions.create.call_count, 2)
        self.assertIn("介绍", result)
        self.assertIn("正文", result)

    def test_chinese_multi_section_to_english(self, _d):
        content = "# 介绍\n\n这是介绍。\n\n# 结论\n\n这是结论。"
        en1 = "# Introduction\n\nThis is the introduction."
        en2 = "# Conclusion\n\nThis is the conclusion."
        result, _, c = self._run(content, "English", en1, en2)
        self.assertEqual(c.chat.completions.create.call_count, 2)
        self.assertIn("Introduction", result)
        self.assertIn("Conclusion", result)

    def test_japanese_multi_section_to_english(self, _d):
        content = "# はじめに\n\n導入です。\n\n# まとめ\n\n結論です。"
        en1 = "# Introduction\n\nThis is the introduction."
        en2 = "# Conclusion\n\nThis is the conclusion."
        result, _, _ = self._run(content, "English", en1, en2)
        self.assertIn("Introduction", result)
        self.assertIn("Conclusion", result)

    def test_korean_multi_section_to_english(self, _d):
        content = "# 소개\n\n소개 텍스트입니다.\n\n# 결론\n\n결론 텍스트입니다."
        en1 = "# Introduction\n\nIntroduction text."
        en2 = "# Conclusion\n\nConclusion text."
        result, _, _ = self._run(content, "English", en1, en2)
        self.assertIn("Introduction", result)
        self.assertIn("Conclusion", result)

    def test_markdown_syntax_preserved_in_mock_response(self, _d):
        content = "## Section Title\n\nParagraph content."
        mock_resp = "## 章节标题\n\n段落内容。"
        result, _, _ = self._run(content, "Chinese", mock_resp)
        self.assertIn("##", result)

    def test_zero_retries_on_clean_translation(self, _d):
        _, retries, _ = self._run("Simple paragraph.", "Japanese", "シンプルな段落。")
        self.assertEqual(retries, 0)


# ---------------------------------------------------------------------------
# translate_json — Docling format
# Same langdetect patch strategy as above.
# ---------------------------------------------------------------------------

@patch("translate._detect_lang", return_value=None)
class TestTranslateJsonDocling(unittest.TestCase):

    def _run(self, doc, *batch_dicts, target_lang="Chinese", field="text", max_chars=4000):
        """Run translate_json with a mock client returning *batch_dicts* in order."""
        c = _client(*[json.dumps(b) for b in batch_dicts])
        result, retries = translate_json(
            c, "model", doc, max_chars=max_chars,
            target_lang=target_lang, field=field,
        )
        return result, retries, c

    # --- English → Asian ---

    def test_single_item_english_to_simplified_chinese(self, _d):
        doc = _docling("Hello, world!")
        result, _, _ = self._run(doc, {"0": "你好，世界！"})
        self.assertEqual(result["texts"][0]["text"], "你好，世界！")

    def test_multiple_items_english_to_simplified_chinese(self, _d):
        doc = _docling("Introduction", "Body content.", "Conclusion")
        result, _, _ = self._run(doc, {"0": "介绍", "1": "正文内容。", "2": "结论"})
        self.assertEqual(result["texts"][0]["text"], "介绍")
        self.assertEqual(result["texts"][1]["text"], "正文内容。")
        self.assertEqual(result["texts"][2]["text"], "结论")

    def test_multiple_items_english_to_traditional_chinese(self, _d):
        doc = _docling("Introduction", "Body content.")
        result, _, _ = self._run(
            doc, {"0": "介紹", "1": "正文內容。"}, target_lang="Chinese (Traditional)"
        )
        self.assertEqual(result["texts"][0]["text"], "介紹")
        self.assertEqual(result["texts"][1]["text"], "正文內容。")

    def test_multiple_items_english_to_japanese(self, _d):
        doc = _docling("Introduction", "Body content.")
        result, _, _ = self._run(
            doc, {"0": "はじめに", "1": "本文の内容。"}, target_lang="Japanese"
        )
        self.assertEqual(result["texts"][0]["text"], "はじめに")
        self.assertEqual(result["texts"][1]["text"], "本文の内容。")

    def test_multiple_items_english_to_korean(self, _d):
        doc = _docling("Introduction", "Body content.")
        result, _, _ = self._run(
            doc, {"0": "소개", "1": "본문 내용。"}, target_lang="Korean"
        )
        self.assertEqual(result["texts"][0]["text"], "소개")
        self.assertEqual(result["texts"][1]["text"], "본문 내용。")

    def test_multiple_items_english_to_vietnamese(self, _d):
        doc = _docling("Introduction", "Body content.")
        result, _, _ = self._run(
            doc, {"0": "Giới thiệu", "1": "Nội dung chính."}, target_lang="Vietnamese"
        )
        self.assertEqual(result["texts"][0]["text"], "Giới thiệu")
        self.assertEqual(result["texts"][1]["text"], "Nội dung chính.")

    def test_multiple_items_english_to_thai(self, _d):
        doc = _docling("Introduction", "Body content.")
        result, _, _ = self._run(
            doc, {"0": "บทนำ", "1": "เนื้อหาหลัก"}, target_lang="Thai"
        )
        self.assertEqual(result["texts"][0]["text"], "บทนำ")
        self.assertEqual(result["texts"][1]["text"], "เนื้อหาหลัก")

    # --- Asian → English ---

    def test_simplified_chinese_to_english(self, _d):
        doc = _docling("这是测试文档。", "本文档有多个段落。")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")
        self.assertEqual(result["texts"][1]["text"], "This document has multiple paragraphs.")

    def test_traditional_chinese_to_english(self, _d):
        doc = _docling("這是測試文件。", "此文件有多個段落。")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")

    def test_japanese_to_english(self, _d):
        doc = _docling("これはテスト文書です。", "この文書には複数の段落があります。")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")

    def test_korean_to_english(self, _d):
        doc = _docling("이것은 테스트 문서입니다.", "이 문서는 여러 단락을 가지고 있습니다.")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")

    def test_vietnamese_to_english(self, _d):
        doc = _docling("Đây là tài liệu thử nghiệm.", "Tài liệu này có nhiều đoạn văn.")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")

    def test_thai_to_english(self, _d):
        doc = _docling("นี่คือเอกสารทดสอบ", "เอกสารนี้มีหลายย่อหน้า")
        result, _, _ = self._run(
            doc,
            {"0": "This is a test document.", "1": "This document has multiple paragraphs."},
            target_lang="English",
        )
        self.assertEqual(result["texts"][0]["text"], "This is a test document.")

    # --- Structural integrity ---

    def test_orig_field_untouched(self, _d):
        doc = _docling("Hello, world!")
        result, _, _ = self._run(doc, {"0": "你好，世界！"})
        self.assertEqual(result["texts"][0]["orig"], "Hello, world!")

    def test_label_field_untouched(self, _d):
        doc = _docling("Hello")
        result, _, _ = self._run(doc, {"0": "你好"})
        self.assertEqual(result["texts"][0]["label"], "paragraph")

    def test_self_ref_untouched(self, _d):
        doc = _docling("Hello")
        result, _, _ = self._run(doc, {"0": "你好"})
        self.assertEqual(result["texts"][0]["self_ref"], "#/texts/0")

    def test_schema_name_untouched(self, _d):
        doc = _docling("Hello")
        result, _, _ = self._run(doc, {"0": "你好"})
        self.assertEqual(result["schema_name"], "DoclingDocument")

    def test_version_untouched(self, _d):
        doc = _docling("Hello")
        result, _, _ = self._run(doc, {"0": "你好"})
        self.assertEqual(result["version"], "1.9.0")

    def test_original_document_not_mutated(self, _d):
        doc = _docling("Hello")
        original = doc["texts"][0]["text"]
        self._run(doc, {"0": "你好"})
        self.assertEqual(doc["texts"][0]["text"], original)

    def test_empty_text_skipped_and_preserved(self, _d):
        doc = _docling("Hello", "", "World")
        result, _, c = self._run(doc, {"0": "你好", "1": "世界"})
        # Only 2 items should reach the LLM
        payload = json.loads(
            c.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
        )
        self.assertEqual(len(payload), 2)
        # Empty slot preserved as-is
        self.assertEqual(result["texts"][1]["text"], "")
        self.assertEqual(result["texts"][0]["text"], "你好")
        self.assertEqual(result["texts"][2]["text"], "世界")

    def test_batching_with_small_chunk_size(self, _d):
        """max_chars=5 forces each item into its own batch → two LLM calls."""
        doc = _docling("Hello", "World")
        result, _, c = self._run(
            doc, {"0": "你好"}, {"0": "世界"}, max_chars=5
        )
        self.assertEqual(c.chat.completions.create.call_count, 2)
        self.assertEqual(result["texts"][0]["text"], "你好")
        self.assertEqual(result["texts"][1]["text"], "世界")

    def test_custom_field_name(self, _d):
        """--json-field=label translates label values, leaves text untouched."""
        doc = {
            "items": [
                {"label": "Click here", "text": "do not touch"},
                {"label": "Submit",     "text": "do not touch"},
            ]
        }
        result, _, _ = self._run(
            doc, {"0": "点击这里", "1": "提交"}, field="label"
        )
        self.assertEqual(result["items"][0]["label"], "点击这里")
        self.assertEqual(result["items"][1]["label"], "提交")
        self.assertEqual(result["items"][0]["text"], "do not touch")
        self.assertEqual(result["items"][1]["text"], "do not touch")

    # --- Table cell translation ---

    def test_table_cell_text_translated_english_to_chinese(self, _d):
        doc = _docling_with_table("Name", "Age", row_texts=["Alice", "Engineer"])
        result, _, _ = self._run(doc, {"0": "姓名", "1": "年龄", "2": "爱丽丝", "3": "工程师"})
        cells = result["tables"][0]["data"]["table_cells"]
        translated = {c["text"] for c in cells}
        self.assertEqual(translated, {"姓名", "年龄", "爱丽丝", "工程师"})

    def test_table_cell_text_translated_japanese_to_english(self, _d):
        doc = _docling_with_table("名前", "年齢", row_texts=["田中", "エンジニア"])
        result, _, _ = self._run(
            doc,
            {"0": "Name", "1": "Age", "2": "Tanaka", "3": "Engineer"},
            target_lang="English",
        )
        cells = result["tables"][0]["data"]["table_cells"]
        translated = {c["text"] for c in cells}
        self.assertEqual(translated, {"Name", "Age", "Tanaka", "Engineer"})

    def test_table_cell_text_translated_korean_to_english(self, _d):
        doc = _docling_with_table("이름", "직업", row_texts=["김민준", "개발자"])
        result, _, _ = self._run(
            doc,
            {"0": "Name", "1": "Occupation", "2": "Kim Minjun", "3": "Developer"},
            target_lang="English",
        )
        cells = result["tables"][0]["data"]["table_cells"]
        translated = {c["text"] for c in cells}
        self.assertIn("Name", translated)
        self.assertIn("Developer", translated)

    # --- Embedded JSON in text field ---

    def test_embedded_json_object_translated_to_chinese(self, _d):
        """A 'text' value that is itself JSON: its string values are translated."""
        inner = {"title": "Hello", "body": "World"}
        doc = _docling(json.dumps(inner))
        # Recursive call translates inner values: "Hello"→"你好", "World"→"世界"
        result, _, _ = self._run(doc, {"0": "你好", "1": "世界"})
        translated = json.loads(result["texts"][0]["text"])
        self.assertEqual(translated["title"], "你好")
        self.assertEqual(translated["body"], "世界")

    def test_embedded_json_array_translated_to_chinese(self, _d):
        inner = ["Hello", "World"]
        doc = _docling(json.dumps(inner))
        result, _, _ = self._run(doc, {"0": "你好", "1": "世界"})
        translated = json.loads(result["texts"][0]["text"])
        self.assertEqual(translated[0], "你好")
        self.assertEqual(translated[1], "世界")

    def test_embedded_json_asian_source_to_english(self, _d):
        inner = {"title": "你好", "body": "世界"}
        doc = _docling(json.dumps(inner))
        result, _, _ = self._run(doc, {"0": "Hello", "1": "World"}, target_lang="English")
        translated = json.loads(result["texts"][0]["text"])
        self.assertEqual(translated["title"], "Hello")
        self.assertEqual(translated["body"], "World")

    def test_embedded_json_japanese_source_to_english(self, _d):
        inner = {"title": "はじめに", "summary": "これは要約です。"}
        doc = _docling(json.dumps(inner))
        result, _, _ = self._run(
            doc, {"0": "Introduction", "1": "This is a summary."}, target_lang="English"
        )
        translated = json.loads(result["texts"][0]["text"])
        self.assertEqual(translated["title"], "Introduction")
        self.assertEqual(translated["summary"], "This is a summary.")

    def test_mixed_plain_and_embedded_json(self, _d):
        """One plain text item and one embedded-JSON item in the same document."""
        inner = {"title": "Intro"}
        doc = _docling("Plain paragraph", json.dumps(inner))
        # Embedded JSON is processed first (call 0: "Intro"→"你好"),
        # then the plain-text batch (call 1: "Plain paragraph"→"普通段落").
        result, _, c = self._run(doc, {"0": "你好"}, {"0": "普通段落"})
        self.assertEqual(c.chat.completions.create.call_count, 2)
        translated_inner = json.loads(result["texts"][1]["text"])
        self.assertEqual(translated_inner["title"], "你好")
        self.assertEqual(result["texts"][0]["text"], "普通段落")

    def test_doc_with_only_empty_text_fields_makes_no_llm_call(self, _d):
        doc = _docling("", "", "")
        result, _, c = self._run(doc)
        c.chat.completions.create.assert_not_called()
        for item in result["texts"]:
            self.assertEqual(item["text"], "")


if __name__ == "__main__":
    unittest.main()
