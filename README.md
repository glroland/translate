# translate

A CLI tool for translating Markdown and JSON files using any OpenAI-compatible LLM API (Mistral, local models, etc.).

## Features

- Translates `.md` / `.markdown` files while preserving all markdown syntax
- Translates `.json` files, targeting only a configurable key (default: `text`) at any depth in the document
- Works with any OpenAI-compatible API endpoint
- Automatically detects the source language
- Retries when the model responds in the wrong language or returns unparseable JSON

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/translate.py <input> --url <api-base-url> --key <api-key> --model <model-name> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `<stem>_<lang-code>.<ext>` | Output file path |
| `--url` | *(required)* | API base URL (e.g. `https://api.mistral.ai/v1`) |
| `--key` | *(required)* | API key |
| `--model` | *(required)* | Model name (e.g. `ministral-3b-2410`) |
| `--target-lang` | `English` | Target language for translation |
| `--chunk-size` | `2000` | Max characters per translation chunk |
| `--retries` | `5` | Max retries on wrong-language or JSON parse errors |
| `--json-field` | `text` | JSON key whose string values are translated |

### Examples

Translate a Markdown file to French using Mistral:

```bash
python src/translate.py doc.md \
  --url https://api.mistral.ai/v1 \
  --key $MISTRAL_KEY \
  --model ministral-3b-2410 \
  --target-lang French
```

Translate a JSON file to Spanish (only `"text"` fields):

```bash
python src/translate.py strings.json \
  --url https://api.mistral.ai/v1 \
  --key $MISTRAL_KEY \
  --model ministral-3b-2410 \
  --target-lang Spanish
```

Translate a JSON file using a different field name:

```bash
python src/translate.py data.json \
  --url https://api.mistral.ai/v1 \
  --key $MISTRAL_KEY \
  --model ministral-3b-2410 \
  --target-lang German \
  --json-field label
```

Specify an explicit output file:

```bash
python src/translate.py doc.md \
  --url https://api.mistral.ai/v1 \
  --key $MISTRAL_KEY \
  --model ministral-3b-2410 \
  --target-lang Japanese \
  -o doc_ja.md
```

## Output

If no output path is given, the translated file is written to `<stem>_<lang-code>.<ext>` in the same directory as the input. For example, translating `readme.md` to French produces `readme_fr.md`.

## How It Works

### Markdown

The file is split into chunks at headers and paragraph boundaries, keeping each chunk within `--chunk-size` characters. Each chunk is translated independently and then reassembled with correct blank-line spacing.

### JSON

All string values in the document whose immediate key matches `--json-field` (default: `text`) are collected recursively, regardless of nesting depth. Empty strings are skipped. Values are grouped into batches and sent to the LLM as flat JSON objects; the translated values are written back into a deep copy of the original structure, leaving all other keys untouched.
