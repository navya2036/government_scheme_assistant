"""Build a read-only manifest for a directory of PDF files."""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pymupdf


LOW_TEXT_CHAR_THRESHOLD = 100
CSV_FIELDS = (
    "relative_path",
    "filename",
    "file_size_bytes",
    "sha256",
    "page_count",
    "text_char_count",
    "extraction_status",
)


def sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_paths(pdf_dir: Path) -> Iterable[Path]:
    return sorted(
        (
            path
            for path in pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(pdf_dir).as_posix(),
    )


def scan_pdf(path: Path, pdf_dir: Path) -> dict[str, object]:
    relative_path = path.relative_to(pdf_dir).as_posix()
    row: dict[str, object] = {
        "relative_path": relative_path,
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_for_file(path),
        "page_count": 0,
        "text_char_count": 0,
        "extraction_status": "unreadable",
    }

    try:
        with pymupdf.open(path) as document:
            row["page_count"] = len(document)
            text_char_count = 0
            for page in document:
                text_char_count += len(page.get_text("text") or "")
            row["text_char_count"] = text_char_count
            row["extraction_status"] = (
                "ok" if text_char_count > LOW_TEXT_CHAR_THRESHOLD else "low_text"
            )
    except Exception:
        # Keep the file in the manifest so unreadable PDFs are visible.
        pass

    return row


def write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    statuses = Counter(row["extraction_status"] for row in rows)
    hashes = Counter(row["sha256"] for row in rows)
    duplicate_files = sum(count - 1 for count in hashes.values() if count > 1)

    print(f"total PDFs: {len(rows)}")
    print(f"successfully readable: {statuses['ok']}")
    print(f"low_text: {statuses['low_text']}")
    print(f"unreadable: {statuses['unreadable']}")
    print(f"total pages: {sum(int(row['page_count']) for row in rows)}")
    print(
        "total extracted characters: "
        f"{sum(int(row['text_char_count']) for row in rows)}"
    )
    print(f"unique SHA-256 hashes: {len(hashes)}")
    print(f"duplicate files: {duplicate_files}")


def scan_command(pdf_dir: Path, output_path: Path) -> int:
    if not pdf_dir.is_dir():
        raise ValueError(f"PDF directory does not exist: {pdf_dir}")

    rows = [scan_pdf(path, pdf_dir) for path in pdf_paths(pdf_dir)]
    write_manifest(rows, output_path)
    print_summary(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan PDFs and write a deterministic, read-only dataset manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan", help="Scan a PDF directory and write its CSV manifest."
    )
    scan_parser.add_argument(
        "--pdf-dir", required=True, type=Path, help="Directory containing PDF files."
    )
    scan_parser.add_argument(
        "--out", required=True, type=Path, help="Path for the generated CSV manifest."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            return scan_command(args.pdf_dir, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
