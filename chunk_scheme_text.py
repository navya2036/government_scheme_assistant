"""Clean extracted PDF text and create deterministic semantic chunks."""

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Optional


DEFAULT_TEXT_DIR = Path("data/extracted_text")
DEFAULT_METADATA = Path("data/scheme_metadata.csv")
DEFAULT_MANIFEST = Path("data/manifest_all.csv")
DEFAULT_OUTPUT = Path("data/chunks.jsonl")
EXPECTED_UNIQUE_PDF_COUNT = 2066
MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 80

PAGE_MARKER = re.compile(r"^===== PAGE (\d+) =====\s*$", re.MULTILINE)
SECTION_HEADINGS = (
    "Frequently Asked Questions",
    "Documents Required",
    "Application Process",
    "Sources And References",
    "Target Beneficiaries",
    "Eligible Beneficiaries",
    "Implementation Details",
    "Eligibility Criteria",
    "Who Can Apply",
    "How To Apply",
    "Additional Information",
    "Beneficiaries",
    "Eligibility",
    "Exclusions",
    "Objectives",
    "Objective",
    "Benefits",
    "Implementation",
    "Components",
    "Details",
)


def clean_page_text(text: str) -> str:
    """Remove repeated site chrome and normalize whitespace without summarizing."""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\x00", "").replace("\r", "")
    navigation = re.search(
        r"Are\s+you\s+sure\s+you\s+want\s+to\s+sign\s+out\?",
        text,
        re.IGNORECASE,
    )
    check = re.search(
        r"Check\s+Eligibility",
        text[navigation.end() :] if navigation else text,
        re.IGNORECASE,
    )
    if check:
        check_start = (navigation.end() if navigation else 0) + check.start()
        check_end = (navigation.end() if navigation else 0) + check.end()
        details = re.search(r"Details", text[check_end:], re.IGNORECASE)
        before_navigation = text[: navigation.start()] if navigation else text[:check_start]
        if details:
            text = before_navigation + " " + text[check_end + details.end() :]
        else:
            text = before_navigation
    text = re.sub(
        r"Was\s+this\s+helpful\?.*$",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"Something\s+went\s+wrong.*?(?:Ok|$)",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"Was\s+this\s+helpful\?.*$", " ", text, flags=re.IGNORECASE | re.DOTALL)
    for _ in range(3):
        updated = re.sub(
            r"(?:Ok|Manual|Form-\d+|Application Status)\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if updated == text:
            break
        text = updated
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def parse_pages(raw_text: str) -> list[tuple[int, str]]:
    matches = list(PAGE_MARKER.finditer(raw_text))
    if not matches:
        cleaned = clean_page_text(raw_text)
        return [(1, cleaned)] if cleaned else []

    pages = []
    for index, marker in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        page_number = int(marker.group(1))
        cleaned = clean_page_text(raw_text[marker.end() : end])
        if re.search(r"[A-Za-z]{2,}", cleaned):
            pages.append((page_number, cleaned))
    return pages


def split_sections(page_text: str) -> list[tuple[str, str]]:
    heading_pattern = "|".join(
        re.escape(heading) for heading in sorted(SECTION_HEADINGS, key=len, reverse=True)
    )
    heading_re = re.compile(
        rf"(?<![A-Za-z])(?P<heading>{heading_pattern})"
        r"(?=\s*[:\-]?\s*(?:[A-Z0-9]|$))",
        re.IGNORECASE,
    )
    matches = list(heading_re.finditer(page_text))
    if not matches:
        return [("", page_text)] if page_text else []

    sections = []
    first_start = matches[0].start()
    if page_text[:first_start].strip():
        sections.append(("", page_text[:first_start].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
        heading = match.group("heading").strip()
        body = page_text[match.end() : end].strip(" :-")
        if body:
            sections.append((heading, body))
    merged_sections = []
    for heading, body in sections:
        section_size = len(f"{heading}: {body}") if heading else len(body)
        if merged_sections and section_size < 80:
            previous_heading, previous_body = merged_sections[-1]
            suffix = f" {heading}: {body}" if heading else f" {body}"
            merged_sections[-1] = (previous_heading, f"{previous_body}{suffix}")
        else:
            merged_sections.append((heading, body))
    return merged_sections


def sentence_units(text: str) -> list[str]:
    units = re.split(r"(?<=[.!?;])\s*(?=[A-Z0-9])", text)
    return [unit.strip() for unit in units if unit.strip()]


def bounded_word_units(text: str, limit: int) -> list[str]:
    words = text.split()
    units = []
    current = ""
    for word in words:
        if len(word) > limit:
            if current:
                units.append(current)
                current = ""
            for start in range(0, len(word), limit):
                units.append(word[start : start + limit])
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            units.append(current)
            current = word
        else:
            current = candidate
    if current:
        units.append(current)
    return units


def split_section(text: str, heading: str) -> list[str]:
    prefix = f"{heading}: " if heading else ""
    full_text = f"{prefix}{text}".strip()
    if len(full_text) <= MAX_CHUNK_CHARS:
        return [full_text] if full_text else []

    units = []
    body_limit = max(1, MAX_CHUNK_CHARS - len(prefix))
    for unit in sentence_units(text):
        if len(unit) > body_limit:
            units.extend(bounded_word_units(unit, body_limit))
        else:
            units.append(unit)

    chunks = []
    current = prefix
    for unit in units:
        candidate = f"{current}{unit}" if current.endswith(": ") else f"{current} {unit}"
        if current.strip() and len(candidate) > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            current = f"{prefix}{unit}" if prefix else unit
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def coalesce_small_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge heading fragments into neighboring content on the same page."""
    if len(chunks) <= 1:
        return chunks
    result = []
    pending = []
    for chunk in chunks:
        if len(str(chunk["text"])) < MIN_CHUNK_CHARS:
            pending.append(chunk)
            continue
        if pending:
            prefix = " ".join(str(item["text"]) for item in pending)
            combined = f"{prefix} {chunk['text']}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                chunk = dict(chunk)
                chunk["text"] = combined
            else:
                result.extend(pending)
            pending = []
        result.append(chunk)
    if pending:
        if result:
            combined = f"{result[-1]['text']} {' '.join(str(item['text']) for item in pending)}".strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                result[-1] = dict(result[-1])
                result[-1]["text"] = combined
            # Otherwise the short tail is usually extraction/footer noise;
            # retain it only when it is the page's sole available content.
        else:
            result.extend(pending)
    return result


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    metadata = {row["sha256"]: row for row in rows}
    return metadata


def load_expected_hashes(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["sha256"] for row in rows}


def chunk_document(
    text_path: Path, metadata: dict[str, str]
) -> list[dict[str, object]]:
    raw_text = text_path.read_text(encoding="utf-8")
    pages = parse_pages(raw_text)
    chunks: list[dict[str, object]] = []
    for page_number, page_text in pages:
        page_chunks = []
        for heading, section_text in split_sections(page_text):
            for chunk_text in split_section(section_text, heading):
                if chunk_text.strip():
                    page_chunks.append(
                        {
                            "sha256": metadata["sha256"],
                            "scheme_name": metadata["scheme_name"],
                            "department": metadata["department"],
                            "state": metadata["state"],
                            "beneficiaries": metadata["beneficiaries"],
                            "source_file": metadata["relative_path"],
                            "page_start": page_number,
                            "page_end": page_number,
                            "text": chunk_text,
                        }
                    )
        chunks.extend(coalesce_small_chunks(page_chunks))
    if len(chunks) > 1:
        substantive = [
            chunk for chunk in chunks if len(str(chunk["text"])) >= MIN_CHUNK_CHARS
        ]
        if substantive:
            chunks = substantive
    return chunks


def write_chunks(
    text_dir: Path,
    metadata_by_sha: dict[str, dict[str, str]],
    output_path: Path,
) -> tuple[list[dict[str, object]], set[str]]:
    text_files = sorted(text_dir.glob("*.txt"), key=lambda path: path.stem)
    chunks: list[dict[str, object]] = []
    processed_hashes = set()
    for text_path in text_files:
        sha256 = text_path.stem
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Invalid extracted-text filename: {text_path.name}")
        if sha256 not in metadata_by_sha:
            raise ValueError(f"Missing metadata for SHA-256: {sha256}")
        processed_hashes.add(sha256)
        document_chunks = chunk_document(text_path, metadata_by_sha[sha256])
        for ordinal, chunk in enumerate(document_chunks, start=1):
            chunk["chunk_id"] = f"{sha256}_{ordinal:06d}"
            chunks.append(chunk)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                **{key: value for key, value in chunk.items() if key != "chunk_id"},
            }
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    return chunks, processed_hashes


def print_summary(
    chunks: list[dict[str, object]],
    processed_hashes: set[str],
    expected_hashes: set[str],
    output_path: Path,
) -> None:
    sizes = [len(str(chunk["text"])) for chunk in chunks]
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    missing_sha = sum(not chunk["sha256"] for chunk in chunks)
    missing_scheme = sum(not chunk["scheme_name"] for chunk in chunks)
    invalid_pages = sum(
        not isinstance(chunk["page_start"], int)
        or not isinstance(chunk["page_end"], int)
        or chunk["page_start"] < 1
        or chunk["page_end"] < chunk["page_start"]
        for chunk in chunks
    )
    chunk_hashes = {str(chunk["sha256"]) for chunk in chunks}
    zero_chunk_hashes = expected_hashes - chunk_hashes
    print(f"unique PDFs processed: {len(processed_hashes)}")
    print(f"total chunks created: {len(chunks)}")
    print(f"minimum chunk size: {min(sizes) if sizes else 0}")
    print(f"maximum chunk size: {max(sizes) if sizes else 0}")
    print(f"average chunk size: {mean(sizes) if sizes else 0:.2f}")
    print(f"median chunk size: {median(sizes) if sizes else 0:.2f}")
    print(f"PDFs producing zero chunks: {len(zero_chunk_hashes)}")
    print(f"chunks with missing sha256: {missing_sha}")
    print(f"chunks with missing scheme_name: {missing_scheme}")
    print(f"chunks with invalid/missing page information: {invalid_pages}")
    print(f"duplicate chunk_id count: {len(chunk_ids) - len(set(chunk_ids))}")
    print(f"deterministic ordering validation: {chunk_ids == sorted(chunk_ids)}")
    print(
        "source-document coverage: "
        f"{len(chunk_hashes & expected_hashes)}/{len(expected_hashes)}"
    )
    print(f"output: {output_path.as_posix()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic semantic chunks from extracted PDF text."
    )
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_by_sha = load_metadata(args.metadata)
    expected_hashes = load_expected_hashes(args.manifest)
    text_hashes = {
        path.stem for path in args.text_dir.glob("*.txt") if path.is_file()
    }
    if len(expected_hashes) != EXPECTED_UNIQUE_PDF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_PDF_COUNT} manifest hashes, "
            f"found {len(expected_hashes)}"
        )
    if text_hashes != expected_hashes:
        raise ValueError(
            "Extracted-text SHA-256 files do not exactly match manifest hashes"
        )
    if set(metadata_by_sha) != expected_hashes:
        raise ValueError("Scheme metadata hashes do not match manifest hashes")

    chunks, processed_hashes = write_chunks(
        args.text_dir, metadata_by_sha, args.out
    )
    print_summary(chunks, processed_hashes, expected_hashes, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
