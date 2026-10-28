"""Extract full, page-delimited text for one representative per PDF hash."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pymupdf


DEFAULT_MANIFEST = Path("data/manifest_all.csv")
DEFAULT_PDF_DIR = Path("gov_myscheme/text_data")
DEFAULT_OUTPUT_DIR = Path("data/extracted_text")
DEFAULT_OUTPUT_MANIFEST = Path("data/extracted_text_manifest.csv")
EXPECTED_UNIQUE_PDF_COUNT = 2066

OUTPUT_FIELDS = (
    "sha256",
    "relative_path",
    "filename",
    "page_count",
    "extracted_character_count",
    "extraction_status",
    "output_text_path",
    "error_message",
)


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)

    representatives = [
        min(group, key=lambda row: row["relative_path"])
        for group in groups.values()
    ]
    return sorted(representatives, key=lambda row: row["sha256"])


def page_delimited_text(
    sha256: str,
    relative_path: str,
    filename: str,
    pages: list[str],
) -> str:
    parts = [
        f"# sha256: {sha256}\n",
        f"# relative_path: {relative_path}\n",
        f"# filename: {filename}\n",
    ]
    for page_number, page_text in enumerate(pages, start=1):
        parts.append(f"\n===== PAGE {page_number} =====\n")
        parts.append(page_text.rstrip("\r\n"))
        parts.append("\n")
    return "".join(parts)


def extract_representative(
    row: dict[str, str], pdf_dir: Path, output_dir: Path
) -> dict[str, str]:
    sha256 = row["sha256"]
    relative_path = row["relative_path"]
    output_relative_path = (Path("data/extracted_text") / f"{sha256}.txt").as_posix()
    output_path = output_dir / f"{sha256}.txt"
    result = {
        "sha256": sha256,
        "relative_path": relative_path,
        "filename": row["filename"],
        "page_count": "0",
        "extracted_character_count": "0",
        "extraction_status": "unreadable",
        "output_text_path": "",
        "error_message": "",
    }

    try:
        with pymupdf.open(pdf_dir / Path(relative_path)) as document:
            page_count = len(document)
            pages = []
            page_errors = []
            for page_number in range(page_count):
                try:
                    pages.append(document[page_number].get_text("text") or "")
                except Exception as error:
                    pages.append("")
                    page_errors.append(f"page {page_number + 1}: {error}")

        output_path.write_text(
            page_delimited_text(sha256, relative_path, row["filename"], pages),
            encoding="utf-8",
            newline="\n",
        )
        extracted_character_count = sum(len(page) for page in pages)
        result["page_count"] = str(page_count)
        result["extracted_character_count"] = str(extracted_character_count)
        result["output_text_path"] = output_relative_path
        result["extraction_status"] = (
            "partial" if page_errors else "zero_text" if extracted_character_count == 0 else "ok"
        )
        if page_errors:
            result["error_message"] = "; ".join(page_errors)
    except Exception as error:
        result["error_message"] = f"{type(error).__name__}: {error}"

    return result


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    successful = sum(row["extraction_status"] in {"ok", "zero_text", "partial"} for row in rows)
    failures = sum(row["extraction_status"] == "unreadable" for row in rows)
    zero_text = sum(row["extraction_status"] == "zero_text" for row in rows)
    print(f"unique PDFs processed: {len(rows)}")
    print(f"successfully extracted: {successful}")
    print(f"extraction failures: {failures}")
    print(f"total pages: {sum(int(row['page_count']) for row in rows)}")
    print(
        "total extracted characters: "
        f"{sum(int(row['extracted_character_count']) for row in rows)}"
    )
    print(f"PDFs with zero extracted text: {zero_text}")
    print(f"output directory: {output_dir.as_posix()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract full page-delimited text for unique PDF contents."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--out-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    representatives = representative_rows(rows)
    if len(representatives) != EXPECTED_UNIQUE_PDF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_PDF_COUNT} unique SHA-256 values, "
            f"found {len(representatives)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        extract_representative(row, args.pdf_dir, args.out_dir)
        for row in representatives
    ]
    write_manifest(results, args.out_manifest)
    print_summary(results, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
