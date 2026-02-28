#!/usr/bin/env python3
"""Translate Markdown or JSON files to English via an OpenAI-compatible LLM."""

import copy
import json
import re
from pathlib import Path

import click
from openai import OpenAI

def _markdown_system_prompt(target_lang: str) -> str:
    return (
        f"You are a professional translator. Translate the provided text to {target_lang}. "
        "Preserve all markdown syntax exactly (headers, bold, italic, code blocks, links, lists). "
        "Return only the translated text with no additional commentary."
    )


def _json_system_prompt(target_lang: str) -> str:
    return (
        "You are a professional translator. You will receive a JSON object whose values are strings. "
        f"Translate each string value to {target_lang}. "
        "Return a valid JSON object with the same keys and translated values. "
        "Do not add any commentary or markdown fences."
    )


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text at blank lines into chunks not exceeding max_chars."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).lstrip("\n") if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds max_chars, keep it as-is
            current = para
    if current:
        chunks.append(current)
    return chunks


def split_markdown(content: str, max_chars: int) -> list[str]:
    """Split markdown into translatable chunks respecting headers and paragraphs."""
    header_pattern = re.compile(r"^#{1,6} .+", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    if not matches:
        # No headers — split by paragraphs
        return _split_paragraphs(content, max_chars)

    chunks: list[str] = []

    # Preamble before first header
    preamble = content[: matches[0].start()].rstrip()
    if preamble:
        chunks.extend(_split_paragraphs(preamble, max_chars))

    for i, match in enumerate(matches):
        header_text = match.group()
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[section_start:section_end].strip()

        section = (header_text + "\n" + body).strip() if body else header_text

        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Keep header with first sub-chunk, split body at blank lines
            body_chunks = _split_paragraphs(body, max_chars)
            if body_chunks:
                first = (header_text + "\n\n" + body_chunks[0]).strip()
                # If even first sub-chunk is too big, emit it anyway
                chunks.append(first)
                chunks.extend(body_chunks[1:])
            else:
                chunks.append(header_text)

    return [c for c in chunks if c.strip()]


def _ensure_spacing(a: str, b: str) -> str:
    """Join two markdown chunks, ensuring exactly one blank line between them."""
    if a.endswith("\n\n"):
        return a + b
    if a.endswith("\n"):
        return a + "\n" + b
    return a + "\n\n" + b


# ---------------------------------------------------------------------------
# JSON chunking
# ---------------------------------------------------------------------------

def _collect_paths(obj, path=()):
    """Recursively yield (path_tuple, string_value) for all string leaves."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _collect_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _collect_paths(v, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


def _set_path(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _batch_paths(paths_values: list, max_chars: int) -> list[list]:
    """Group (path, value) pairs into batches where total chars ≤ max_chars."""
    batches = []
    current_batch = []
    current_size = 0
    for path, value in paths_values:
        size = len(value)
        if current_batch and current_size + size > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append((path, value))
        current_size += size
    if current_batch:
        batches.append(current_batch)
    return batches


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Translation calls
# ---------------------------------------------------------------------------

def _translate_chunk(client: OpenAI, model: str, system: str, chunk: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": chunk},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def translate_markdown(client: OpenAI, model: str, content: str, max_chars: int, target_lang: str) -> str:
    chunks = split_markdown(content, max_chars)
    translated: list[str] = []
    system = _markdown_system_prompt(target_lang)
    for i, chunk in enumerate(chunks, 1):
        click.echo(f"  Translating chunk {i}/{len(chunks)} ({len(chunk)} chars)…", err=True)
        result = _translate_chunk(client, model, system, chunk)
        translated.append(result.strip())

    if not translated:
        return ""

    joined = translated[0]
    for part in translated[1:]:
        joined = _ensure_spacing(joined, part)
    return joined


def translate_json(client: OpenAI, model: str, data: dict | list, max_chars: int, target_lang: str) -> dict | list:
    result = copy.deepcopy(data)
    paths_values = list(_collect_paths(result))

    if not paths_values:
        return result

    batches = _batch_paths(paths_values, max_chars)
    system = _json_system_prompt(target_lang)
    click.echo(f"  Translating {len(paths_values)} strings in {len(batches)} batch(es)…", err=True)

    for i, batch in enumerate(batches, 1):
        click.echo(f"  Batch {i}/{len(batches)}…", err=True)
        payload = {str(j): value for j, (_, value) in enumerate(batch)}
        user_content = json.dumps(payload, ensure_ascii=False)
        raw = _translate_chunk(client, model, system, user_content)
        raw = _strip_code_fences(raw)
        try:
            translated_map: dict = json.loads(raw)
        except json.JSONDecodeError as e:
            click.echo(f"  Warning: failed to parse LLM response for batch {i}: {e}", err=True)
            click.echo(f"  Raw response: {raw[:200]}", err=True)
            continue
        for j, (path, _) in enumerate(batch):
            key = str(j)
            if key in translated_map and isinstance(translated_map[key], str):
                _set_path(result, path, translated_map[key])

    return result


# ISO 639-1 codes for common languages; fallback is first two letters lowercased.
_LANG_CODES: dict[str, str] = {
    "afrikaans": "af", "albanian": "sq", "arabic": "ar", "armenian": "hy",
    "azerbaijani": "az", "basque": "eu", "belarusian": "be", "bengali": "bn",
    "bosnian": "bs", "bulgarian": "bg", "catalan": "ca", "chinese": "zh",
    "croatian": "hr", "czech": "cs", "danish": "da", "dutch": "nl",
    "english": "en", "estonian": "et", "finnish": "fi", "french": "fr",
    "galician": "gl", "georgian": "ka", "german": "de", "greek": "el",
    "gujarati": "gu", "haitian creole": "ht", "hebrew": "he", "hindi": "hi",
    "hungarian": "hu", "icelandic": "is", "indonesian": "id", "irish": "ga",
    "italian": "it", "japanese": "ja", "kannada": "kn", "kazakh": "kk",
    "korean": "ko", "latvian": "lv", "lithuanian": "lt", "macedonian": "mk",
    "malay": "ms", "maltese": "mt", "marathi": "mr", "nepali": "ne",
    "norwegian": "no", "persian": "fa", "polish": "pl", "portuguese": "pt",
    "punjabi": "pa", "romanian": "ro", "russian": "ru", "serbian": "sr",
    "slovak": "sk", "slovenian": "sl", "spanish": "es", "swahili": "sw",
    "swedish": "sv", "tamil": "ta", "telugu": "te", "thai": "th",
    "turkish": "tr", "ukrainian": "uk", "urdu": "ur", "uzbek": "uz",
    "vietnamese": "vi", "welsh": "cy",
}


def _lang_code(target_lang: str) -> str:
    return _LANG_CODES.get(target_lang.lower(), target_lang[:2].lower())


def _default_output(input_path: Path, target_lang: str) -> Path:
    code = _lang_code(target_lang)
    return input_path.with_stem(f"{input_path.stem}_{code}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None, help="Output file path (default: <input>_<lang-code>.<ext>).")
@click.option("--url", required=True, help="API base URL (e.g. https://api.mistral.ai/v1).")
@click.option("--key", required=True, help="API key.")
@click.option("--model", required=True, help="Model name (e.g. ministral-3b-2410).")
@click.option("--chunk-size", default=2000, show_default=True, metavar="N", help="Max characters per chunk.")
@click.option("--target-lang", default="English", show_default=True, help="Target language for translation.")
def main(input, output, url, key, model, chunk_size, target_lang):
    """Translate a Markdown or JSON file via an OpenAI-compatible LLM."""
    input_path = Path(input)
    output_path = Path(output) if output else _default_output(input_path, target_lang)

    suffix = input_path.suffix.lower()
    if suffix not in {".md", ".markdown", ".json"}:
        raise click.BadParameter(
            f"unsupported file type '{suffix}'. Use .md, .markdown, or .json.",
            param_hint="INPUT",
        )

    client = OpenAI(api_key=key, base_url=url)

    if suffix in {".md", ".markdown"}:
        content = input_path.read_text(encoding="utf-8")
        click.echo(f"Translating markdown: {input_path} ({len(content)} chars) → {target_lang}", err=True)
        translated = translate_markdown(client, model, content, chunk_size, target_lang)
        output_path.write_text(translated + "\n", encoding="utf-8")
    else:
        content = input_path.read_text(encoding="utf-8")
        click.echo(f"Translating JSON: {input_path} → {target_lang}", err=True)
        data = json.loads(content)
        translated_data = translate_json(client, model, data, chunk_size, target_lang)
        output_path.write_text(
            json.dumps(translated_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    click.echo(f"Done. Output written to: {output_path}", err=True)


if __name__ == "__main__":
    main()
