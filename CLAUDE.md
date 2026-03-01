# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python src/translate.py <input> --url <api-base-url> --key <api-key> --model <model-name> [--target-lang <lang>] [--chunk-size N] [-o output]
```

Example with Mistral:
```bash
python src/translate.py doc.md --url https://api.mistral.ai/v1 --key $MISTRAL_KEY --model ministral-3b-2410 --target-lang French
```

## Architecture

Everything lives in `src/translate.py` — there is no package structure, build step, or test suite.

**Markdown translation:** `split_markdown()` divides content at headers and blank lines into chunks ≤ `max_chars`. Each chunk is sent to the LLM separately via `_translate_chunk()`, then reassembled with `_ensure_spacing()` to maintain correct blank-line separation between chunks.

**JSON translation:** `_collect_paths()` recursively yields all `(path_tuple, string_value)` leaf pairs. These are grouped into batches by `_batch_paths()`. Each batch is serialized as a flat JSON object `{"0": "val0", "1": "val1", ...}` and sent as one LLM call. The LLM returns a JSON object with translated values; `_set_path()` writes them back into the deep-copied result.

**Key design decisions:**
- The `openai` SDK is used with a configurable `base_url`, making it compatible with any OpenAI-compatible API (Mistral, local models, etc.)
- JSON batches use integer string keys (`"0"`, `"1"`, ...) rather than original keys to avoid ambiguity with nested structures
- `_strip_code_fences()` handles LLMs that wrap JSON responses in markdown fences
- Output filename defaults to `<stem>_<lang-code>.<ext>` using the ISO 639-1 code from `_LANG_CODES`
