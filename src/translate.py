#!/usr/bin/env python3
"""Translate Markdown or JSON files to English via an OpenAI-compatible LLM."""

import copy
import json
import re
from pathlib import Path

import click
from langdetect import detect, LangDetectException, DetectorFactory
from openai import OpenAI

DetectorFactory.seed = 0  # make langdetect deterministic

_MIN_DETECT_LEN = 30  # langdetect is unreliable below this many characters

def _markdown_system_prompt(target_lang: str, source_lang: str | None = None) -> str:
    direction = f"from {source_lang} to {target_lang}" if source_lang else f"to {target_lang}"
    return (
        f"You are a professional translator. Translate the provided text {direction}. "
        "Preserve all markdown syntax exactly (headers, bold, italic, code blocks, links, lists). "
        "Return only the translated text with no additional commentary."
    )


def _json_system_prompt(target_lang: str, source_lang: str | None = None) -> str:
    direction = f"from {source_lang} to {target_lang}" if source_lang else f"to {target_lang}"
    return (
        "You are a professional translator. You will receive a JSON object whose values are strings. "
        f"Translate each string value {direction}. "
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

def _collect_paths(obj, path=(), field: str | None = None):
    """Recursively yield (path_tuple, string_value) for string leaves.

    If *field* is set, only yield leaves whose immediate dict key matches it.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _collect_paths(v, path + (k,), field)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _collect_paths(v, path + (i,), field)
    elif isinstance(obj, str) and obj:
        if field is None or (path and path[-1] == field):
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


def _try_parse_json(s: str) -> dict | list | None:
    """Return the parsed object if s is a JSON object or array string, else None."""
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, (dict, list)) else None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Translation calls
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str | None:
    """Return ISO 639-1 code for the detected language, or None if detection fails."""
    try:
        return detect(text)
    except LangDetectException:
        return None


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


def _translate_with_retry(
    client: OpenAI, model: str, system: str, chunk: str,
    expected_lang: str, max_retries: int, target_lang: str,
) -> tuple[str, int]:
    """Translate a chunk, retrying up to max_retries times if the result is in the wrong language.

    On each retry the system prompt is escalated with an explicit correction.
    Returns (translated_text, retries_used).
    """
    result = _translate_chunk(client, model, system, chunk)
    retries_used = 0

    for attempt in range(max_retries):
        if len(result.strip()) < _MIN_DETECT_LEN:
            break
        detected = _detect_lang(result)
        if detected is None or detected == expected_lang:
            break
        click.echo(
            f"  Language detected as '{detected}', expected '{expected_lang}';"
            f" retrying ({attempt + 1}/{max_retries})…",
            err=True,
        )
        escalated = (
            system
            + f"\n\nIMPORTANT: Your previous response was detected as '{detected}', not {target_lang}."
            f" You MUST translate the text to {target_lang} only."
        )
        result = _translate_chunk(client, model, escalated, chunk)
        retries_used += 1
    else:
        # All retries exhausted — warn if the final result is still wrong
        detected = _detect_lang(result)
        if detected and detected != expected_lang:
            click.echo(
                f"  Warning: translation still detected as '{detected}'"
                f" after {max_retries} retries.",
                err=True,
            )

    return result, retries_used


def translate_markdown(client: OpenAI, model: str, content: str, max_chars: int, target_lang: str, retries: int = 0) -> tuple[str, int]:
    chunks = split_markdown(content, max_chars)
    translated: list[str] = []
    expected_lang = _lang_code(target_lang)
    total_retries = 0
    for i, chunk in enumerate(chunks, 1):
        chunk_lang = _detect_source_lang(chunk) if len(chunk) >= _MIN_DETECT_LEN else None
        system = _markdown_system_prompt(target_lang, chunk_lang)
        click.echo(f"  Translating chunk {i}/{len(chunks)} ({len(chunk)} chars, detected: {chunk_lang or 'unknown'})…", err=True)
        result, retries_used = _translate_with_retry(client, model, system, chunk, expected_lang, retries, target_lang)
        total_retries += retries_used
        translated.append(result.strip())

    if not translated:
        return "", total_retries

    joined = translated[0]
    for part in translated[1:]:
        joined = _ensure_spacing(joined, part)
    return joined, total_retries


def translate_json(client: OpenAI, model: str, data: dict | list, max_chars: int, target_lang: str, retries: int = 0, field: str | None = None) -> tuple[dict | list, int]:
    result = copy.deepcopy(data)
    paths_values = list(_collect_paths(result, field=field))

    if not paths_values:
        return result, 0

    total_retries = 0

    # Split: values that are themselves JSON objects/arrays are translated recursively;
    # plain strings go into the normal batching pipeline.
    embedded = [(path, value, _try_parse_json(value)) for path, value in paths_values]
    json_paths = [(path, value, parsed) for path, value, parsed in embedded if parsed is not None]
    regular = [(path, value) for path, value, parsed in embedded if parsed is None]

    for path, _, parsed in json_paths:
        click.echo(f"  Translating embedded JSON at path {path}…", err=True)
        translated_inner, r = translate_json(client, model, parsed, max_chars, target_lang, retries, field=None)
        total_retries += r
        _set_path(result, path, json.dumps(translated_inner, ensure_ascii=False))

    if not regular:
        return result, total_retries

    batches = _batch_paths(regular, max_chars)
    expected_lang = _lang_code(target_lang)
    click.echo(f"  Translating {len(regular)} strings in {len(batches)} batch(es)…", err=True)

    for i, batch in enumerate(batches, 1):
        batch_text = " ".join(value for _, value in batch)
        batch_lang = _detect_source_lang(batch_text) if len(batch_text) >= _MIN_DETECT_LEN else None
        system = _json_system_prompt(target_lang, batch_lang)
        click.echo(f"  Batch {i}/{len(batches)} (detected: {batch_lang or 'unknown'})…", err=True)
        payload = {str(j): value for j, (_, value) in enumerate(batch)}
        user_content = json.dumps(payload, ensure_ascii=False)

        translated_map: dict | None = None
        for attempt in range(retries + 1):
            raw, retries_used = _translate_with_retry(client, model, system, user_content, expected_lang, retries, target_lang)
            total_retries += retries_used
            raw = _strip_code_fences(raw)
            try:
                translated_map = json.loads(raw)
                break
            except json.JSONDecodeError as e:
                click.echo(f"  Warning: failed to parse LLM response for batch {i}: {e}", err=True)
                click.echo(f"  Input: {user_content}", err=True)
                click.echo(f"  Raw response: {raw[:200]}", err=True)
                if attempt < retries:
                    click.echo(f"  Retrying ({attempt + 1}/{retries})…", err=True)

        if translated_map is None:
            click.echo(f"  Skipping batch {i} after {retries} retries.", err=True)
            continue

        for j, (path, _) in enumerate(batch):
            key = str(j)
            if key in translated_map and isinstance(translated_map[key], str):
                _set_path(result, path, translated_map[key])

    return result, total_retries


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


_CODE_LANGS: dict[str, str] = {code: name.title() for name, code in _LANG_CODES.items()}


def _lang_code(target_lang: str) -> str:
    return _LANG_CODES.get(target_lang.lower(), target_lang[:2].lower())


def _detect_source_lang(text: str) -> str | None:
    """Detect the language of text and return a display name (e.g. 'French'), or None."""
    code = _detect_lang(text)
    return _CODE_LANGS.get(code) if code else None


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
@click.option("--retries", default=5, show_default=True, metavar="N", help="Max retries when translated chunk is detected in the wrong language.")
@click.option("--json-field", default="text", show_default=True, help="JSON key whose values are translated (other keys are left untouched).")
def main(input, output, url, key, model, chunk_size, target_lang, retries, json_field):
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
        translated, total_retries = translate_markdown(client, model, content, chunk_size, target_lang, retries)
        output_path.write_text(translated + "\n", encoding="utf-8")
    else:
        content = input_path.read_text(encoding="utf-8")
        data = json.loads(content)
        click.echo(f"Translating JSON: {input_path} → {target_lang}", err=True)
        translated_data, total_retries = translate_json(client, model, data, chunk_size, target_lang, retries, field=json_field)
        output_path.write_text(
            json.dumps(translated_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    click.echo(f"Retries: {total_retries}", err=True)
    click.echo(f"Done. Output written to: {output_path}", err=True)


if __name__ == "__main__":
    main()
