"""Generate deterministic local embeddings for semantic chunks.

This script is intentionally limited to the embedding stage.  It validates
``data/chunks.jsonl``, encodes chunks in batches, and writes a NumPy matrix
plus a JSONL row mapping that can be consumed by a future vector-database
stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np


DEFAULT_INPUT = Path("data/chunks.jsonl")
DEFAULT_EMBEDDINGS = Path("data/embeddings.npy")
DEFAULT_METADATA = Path("data/embeddings_metadata.jsonl")
DEFAULT_CONFIG = Path("data/embeddings_config.json")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
EXPECTED_CHUNK_COUNT = 23_928
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 0
DEFAULT_TORCH_THREADS = 1

REQUIRED_FIELDS = ("chunk_id", "sha256", "scheme_name", "text")
SHA256_PATTERN = r"^[0-9a-f]{64}$"
METADATA_FIELDS = (
    "chunk_id",
    "sha256",
    "scheme_name",
    "department",
    "state",
    "beneficiaries",
    "source_file",
    "page_start",
    "page_end",
    "text",
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced or validated completely."""


def validate_record(record: Any, line_number: int) -> list[str]:
    """Return explicit validation errors for one input record."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"line {line_number}: JSON value is not an object"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"line {line_number}: missing required field {field!r}")

    chunk_id = record.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        errors.append(f"line {line_number}: chunk_id must be a non-empty string")

    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or not sha256.strip():
        errors.append(f"line {line_number}: sha256 must be a non-empty string")
    elif len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        errors.append(f"line {line_number}: sha256 is not a lowercase SHA-256 value")

    if not isinstance(record.get("scheme_name"), str):
        errors.append(f"line {line_number}: scheme_name must be a string")

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"line {line_number}: text must be a non-empty string")

    for field in ("page_start", "page_end"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(
                f"line {line_number}: {field} must be an integer greater than zero"
            )
    if not errors and record["page_end"] < record["page_start"]:
        errors.append(f"line {line_number}: page_end is before page_start")

    return errors


def scan_input(path: Path, expected_count: Optional[int]) -> dict[str, Any]:
    """Validate all chunks without retaining their text in memory."""

    errors: list[str] = []
    chunk_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    record_count = 0

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    errors.append(f"line {line_number}: blank line")
                    continue
                record_count += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    errors.append(f"line {line_number}: invalid JSON: {error.msg}")
                    continue

                errors.extend(validate_record(record, line_number))
                chunk_id = record.get("chunk_id") if isinstance(record, dict) else None
                if isinstance(chunk_id, str) and chunk_id:
                    if chunk_id in chunk_ids:
                        duplicate_ids.add(chunk_id)
                    chunk_ids.add(chunk_id)
    except OSError as error:
        raise EmbeddingError(f"Unable to read input chunks: {error}") from error

    if expected_count is not None and record_count != expected_count:
        errors.append(
            f"input record count is {record_count}; expected {expected_count}"
        )
    if duplicate_ids:
        errors.append(f"duplicate chunk_id values found: {len(duplicate_ids)}")

    return {
        "record_count": record_count,
        "chunk_ids": chunk_ids,
        "duplicate_ids": duplicate_ids,
        "errors": errors,
    }


def iter_valid_chunks(path: Path) -> Iterator[dict[str, Any]]:
    """Stream validated chunks in their original order."""

    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                raise EmbeddingError(f"line {line_number}: blank line")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise EmbeddingError(
                    f"line {line_number}: invalid JSON: {error.msg}"
                ) from error
            errors = validate_record(record, line_number)
            if errors:
                raise EmbeddingError("; ".join(errors))
            chunk_id = record["chunk_id"]
            if chunk_id in seen_ids:
                raise EmbeddingError(
                    f"line {line_number}: duplicate chunk_id {chunk_id!r}"
                )
            seen_ids.add(chunk_id)
            yield record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_text(record: dict[str, Any]) -> str:
    """Build semantic input without using the source filename."""

    return f"passage: Scheme: {record['scheme_name']}\n{record['text']}"


def set_deterministic_runtime(seed: int, torch_threads: int) -> Any:
    """Configure deterministic inference before importing torch-backed code."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise EmbeddingError(
            "PyTorch is required. Install sentence-transformers in the active environment."
        ) from error

    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch


def load_model(model_name: str, revision: str, device: str, seed: int, threads: int) -> Any:
    set_deterministic_runtime(seed, threads)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise EmbeddingError(
            "sentence-transformers is required. Install it in the active environment."
        ) from error

    try:
        model = SentenceTransformer(model_name, revision=revision, device=device)
        model.eval()
    except Exception as error:
        raise EmbeddingError(
            f"Unable to load embedding model {model_name!r} at revision {revision!r}: {error}"
        ) from error
    return model


def encode_batch(
    model: Any,
    records: list[dict[str, Any]],
    normalize_embeddings: bool,
) -> np.ndarray:
    texts = [embedding_text(record) for record in records]
    try:
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as error:
        raise EmbeddingError(
            f"model encoding failed for chunk batch beginning "
            f"{records[0]['chunk_id']!r}: {error}"
        ) from error

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise EmbeddingError(
            f"model returned shape {vectors.shape}; expected ({len(records)}, dimension)"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingError("model returned a non-finite embedding value")
    return vectors


def metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in METADATA_FIELDS}


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.as_posix() for path in existing)
        raise EmbeddingError(
            f"output file(s) already exist: {names}; use --overwrite to replace them"
        )


def validate_outputs(
    embeddings_path: Path,
    metadata_path: Path,
    expected_ids: set[str],
    expected_count: int,
    expected_dimension: int,
) -> dict[str, Any]:
    """Validate the matrix and its row mapping after generation."""

    try:
        vectors = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise EmbeddingError(f"unable to read generated embeddings: {error}") from error

    if vectors.ndim != 2 or vectors.shape != (expected_count, expected_dimension):
        raise EmbeddingError(
            f"embedding matrix shape is {vectors.shape}; expected "
            f"({expected_count}, {expected_dimension})"
        )
    if vectors.dtype != np.float32:
        raise EmbeddingError(f"embedding matrix dtype is {vectors.dtype}; expected float32")
    for start in range(0, expected_count, 4096):
        if not np.isfinite(vectors[start : start + 4096]).all():
            raise EmbeddingError("generated embeddings contain non-finite values")

    metadata_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    metadata_count = 0
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise EmbeddingError(
                    f"metadata line {line_number}: invalid JSON: {error.msg}"
                ) from error
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise EmbeddingError(
                    f"metadata line {line_number}: missing chunk_id"
                )
            if chunk_id in metadata_ids:
                duplicate_ids.add(chunk_id)
            metadata_ids.add(chunk_id)
            metadata_count += 1

    missing_ids = expected_ids - metadata_ids
    unexpected_ids = metadata_ids - expected_ids
    if metadata_count != expected_count:
        raise EmbeddingError(
            f"metadata rows are {metadata_count}; expected {expected_count}"
        )
    if duplicate_ids or missing_ids or unexpected_ids:
        raise EmbeddingError(
            "embedding metadata mapping is invalid: "
            f"duplicates={len(duplicate_ids)}, missing={len(missing_ids)}, "
            f"unexpected={len(unexpected_ids)}"
        )

    return {
        "embedding_count": int(vectors.shape[0]),
        "metadata_count": metadata_count,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "dimension": int(vectors.shape[1]),
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    temporary = temporary_path(path)
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input
    output_paths = (args.embeddings, args.metadata, args.config)
    ensure_outputs_available(output_paths, args.overwrite)

    scan = scan_input(input_path, args.expected_count)
    print(
        f"Validated input: {scan['record_count']:,} records, "
        f"{len(scan['chunk_ids']):,} unique chunk IDs"
    )
    if scan["duplicate_ids"]:
        print(f"Duplicate chunk IDs: {len(scan['duplicate_ids']):,}")
    if scan["errors"]:
        print(f"Input validation errors: {len(scan['errors']):,}", file=sys.stderr)
        for error in scan["errors"][:20]:
            print(f"  - {error}", file=sys.stderr)
        if len(scan["errors"]) > 20:
            print("  - ... additional errors omitted", file=sys.stderr)
        raise EmbeddingError("input validation failed; no embeddings were generated")

    input_hash = sha256_file(input_path)
    print(f"Loading model: {args.model} (revision={args.revision}, device={args.device})")
    model = load_model(
        args.model,
        args.revision,
        args.device,
        args.seed,
        args.torch_threads,
    )
    dimension = model.get_sentence_embedding_dimension()
    if not isinstance(dimension, int) or dimension <= 0:
        raise EmbeddingError(f"model reported invalid embedding dimension: {dimension!r}")

    output_tmp = temporary_path(args.embeddings)
    metadata_tmp = temporary_path(args.metadata)
    config_tmp = temporary_path(args.config)
    for path in (output_tmp, metadata_tmp, config_tmp):
        remove_if_exists(path)

    matrix: Any = None
    processed = 0
    failed_ids: list[str] = []
    total_batches = (scan["record_count"] + args.batch_size - 1) // args.batch_size
    try:
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            output_tmp,
            mode="w+",
            dtype=np.float32,
            shape=(scan["record_count"], dimension),
        )
        with metadata_tmp.open("w", encoding="utf-8", newline="\n") as metadata_file:
            batch: list[dict[str, Any]] = []
            for record in iter_valid_chunks(input_path):
                batch.append(record)
                if len(batch) < args.batch_size:
                    continue
                vectors = encode_batch(model, batch, args.normalize_embeddings)
                matrix[processed : processed + len(batch)] = vectors
                for item in batch:
                    metadata_file.write(
                        json.dumps(
                            metadata_record(item),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                processed += len(batch)
                if processed == scan["record_count"] or (processed // args.batch_size) % args.progress_every == 0:
                    print(
                        f"Embedded batch {processed // args.batch_size:,}/{total_batches:,} "
                        f"({processed:,}/{scan['record_count']:,} chunks)"
                    )
                batch = []

            if batch:
                vectors = encode_batch(model, batch, args.normalize_embeddings)
                matrix[processed : processed + len(batch)] = vectors
                for item in batch:
                    metadata_file.write(
                        json.dumps(
                            metadata_record(item),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                processed += len(batch)
                print(
                    f"Embedded batch {processed // args.batch_size:,}/{total_batches:,} "
                    f"({processed:,}/{scan['record_count']:,} chunks)"
                )
        matrix.flush()
        del matrix
        matrix = None

        os.replace(output_tmp, args.embeddings)
        os.replace(metadata_tmp, args.metadata)
        validation = validate_outputs(
            args.embeddings,
            args.metadata,
            scan["chunk_ids"],
            scan["record_count"],
            dimension,
        )
        config = {
            "artifact_type": "chunk_embeddings",
            "model": args.model,
            "model_revision": args.revision,
            "library": "sentence-transformers",
            "embedding_dimension": dimension,
            "dtype": "float32",
            "normalize_embeddings": args.normalize_embeddings,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "input_file": input_path.as_posix(),
            "input_file_sha256": input_hash,
            "input_chunk_count": scan["record_count"],
            "embedded_chunk_count": validation["embedding_count"],
            "failed_embeddings": 0,
            "missing_embeddings": len(validation["missing_ids"]),
            "duplicate_chunk_ids": len(validation["duplicate_ids"]),
            "semantic_input": "passage: Scheme: <scheme_name> + newline + <text>; source filename excluded",
            "row_mapping": {
                "vectors": args.embeddings.as_posix(),
                "metadata": args.metadata.as_posix(),
                "same_zero_based_row_index": True,
            },
        }
        write_config(args.config, config)
        return {
            "dimension": dimension,
            "processed": processed,
            "failed": len(failed_ids),
            "missing": len(validation["missing_ids"]),
            "duplicates": len(validation["duplicate_ids"]),
            "config": config,
        }
    except Exception:
        if matrix is not None:
            matrix.flush()
            del matrix
        for path in (output_tmp, metadata_tmp, config_tmp):
            remove_if_exists(path)
        # Remove promoted files too if validation failed; never leave a partial artifact.
        if args.embeddings.exists() and not args.overwrite:
            remove_if_exists(args.embeddings)
        if args.metadata.exists() and not args.overwrite:
            remove_if_exists(args.metadata)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic local embeddings for data/chunks.jsonl."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default="main",
        help="Model repository revision; pin this for exact model reproducibility.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_CHUNK_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--normalize",
        dest="normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize vectors (default: enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing embedding artifacts after successful generation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 2
    if args.expected_count < 1:
        print("--expected-count must be at least 1", file=sys.stderr)
        return 2
    if args.torch_threads < 1 or args.progress_every < 1:
        print("--torch-threads and --progress-every must be at least 1", file=sys.stderr)
        return 2

    try:
        result = generate(args)
    except Exception as error:
        print(f"Embedding generation failed: {error}", file=sys.stderr)
        print("Failed embeddings: unable to produce a complete artifact", file=sys.stderr)
        return 1

    print("Embedding generation and validation completed successfully.")
    print(f"Embedding model: {result['config']['model']}")
    print(f"Total chunks processed: {result['processed']:,}")
    print(f"Embedding dimension: {result['dimension']}")
    print(f"Batch size: {result['config']['batch_size']}")
    print(f"Total embeddings generated: {result['processed']:,}")
    print(f"Failed embeddings: {result['failed']}")
    print(f"Missing embeddings: {result['missing']}")
    print(f"Duplicate chunk IDs: {result['duplicates']}")
    print(f"Embeddings artifact: {args.embeddings.as_posix()}")
    print(f"Metadata artifact: {args.metadata.as_posix()}")
    print(f"Configuration: {args.config.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
