"""Build a deterministic, persistent FAISS index over the frozen e5 embeddings.

This script is intentionally limited to the vector-index stage.  It treats the
embedding artifacts as read-only inputs, proves that they agree with each other
and with ``data/chunks.jsonl``, and writes three new artifacts:

* ``data/vector_index.faiss``           - exact inner-product FAISS index
* ``data/vector_index_metadata.sqlite`` - row-aligned metadata for filtering
* ``data/vector_index_config.json``     - build manifest + retrieval contract

No embedding is recomputed, no source artifact is modified, and no retrieval
API is exposed.  ``verify_vector_index.py`` re-opens the artifacts in a fresh
process to confirm persistence independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

# Pinned before importing FAISS/BLAS so index construction is single-threaded
# and therefore byte-for-byte reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


DEFAULT_EMBEDDINGS = Path("data/embeddings.npy")
DEFAULT_METADATA = Path("data/embeddings_metadata.jsonl")
DEFAULT_EMBEDDINGS_CONFIG = Path("data/embeddings_config.json")
DEFAULT_CHUNKS = Path("data/chunks.jsonl")
DEFAULT_INDEX = Path("data/vector_index.faiss")
DEFAULT_DATABASE = Path("data/vector_index_metadata.sqlite")
DEFAULT_MANIFEST = Path("data/vector_index_config.json")

EXPECTED_MODEL = "intfloat/multilingual-e5-small"
EXPECTED_COUNT = 23_928
EXPECTED_DIMENSION = 384
EXPECTED_DTYPE = np.float32

DOCUMENT_PREFIX = "passage: "
DOCUMENT_FORMAT = "passage: Scheme: {scheme_name}\n{text}"
QUERY_PREFIX = "query: "
QUERY_FORMAT = "query: {user_question}"

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

SCHEMA = """
CREATE TABLE chunks (
    row_id      INTEGER PRIMARY KEY,
    chunk_id    TEXT    NOT NULL UNIQUE,
    sha256      TEXT    NOT NULL,
    scheme_name TEXT    NOT NULL,
    department  TEXT,
    state       TEXT,
    beneficiaries TEXT,
    source_file TEXT,
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    text        TEXT    NOT NULL
);
CREATE INDEX idx_chunks_scheme_name ON chunks(scheme_name);
CREATE INDEX idx_chunks_department  ON chunks(department);
CREATE INDEX idx_chunks_state       ON chunks(state);
CREATE INDEX idx_chunks_source_file ON chunks(source_file);
"""


class VectorIndexError(RuntimeError):
    """Raised when the index cannot be built or validated completely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def ensure_outputs_available(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.as_posix() for path in existing)
        raise VectorIndexError(
            f"output file(s) already exist: {names}; use --overwrite to replace them"
        )


def load_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - environment dependent
        raise VectorIndexError(
            "faiss is required. Install it with: pip install faiss-cpu"
        ) from error
    faiss.omp_set_num_threads(1)
    return faiss


def read_embeddings_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except OSError as error:
        raise VectorIndexError(f"unable to read embedding config: {error}") from error
    except json.JSONDecodeError as error:
        raise VectorIndexError(f"embedding config is not valid JSON: {error}") from error

    if config.get("model") != EXPECTED_MODEL:
        raise VectorIndexError(
            f"embedding config model is {config.get('model')!r}; expected {EXPECTED_MODEL!r}"
        )
    if config.get("embedding_dimension") != EXPECTED_DIMENSION:
        raise VectorIndexError(
            f"embedding config dimension is {config.get('embedding_dimension')!r}; "
            f"expected {EXPECTED_DIMENSION}"
        )
    if config.get("dtype") != "float32":
        raise VectorIndexError(
            f"embedding config dtype is {config.get('dtype')!r}; expected 'float32'"
        )
    if config.get("normalize_embeddings") is not True:
        raise VectorIndexError(
            "embedding config does not report L2-normalized vectors; cosine "
            "similarity via inner product would be invalid"
        )
    if config.get("input_chunk_count") != EXPECTED_COUNT:
        raise VectorIndexError(
            f"embedding config input_chunk_count is {config.get('input_chunk_count')!r}; "
            f"expected {EXPECTED_COUNT}"
        )
    return config


def load_matrix(path: Path, count: int, dimension: int) -> np.ndarray:
    try:
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise VectorIndexError(f"unable to read embeddings: {error}") from error

    if vectors.ndim != 2:
        raise VectorIndexError(f"embeddings are {vectors.ndim}-dimensional; expected 2")
    if vectors.shape != (count, dimension):
        raise VectorIndexError(
            f"embedding matrix shape is {vectors.shape}; expected ({count}, {dimension})"
        )
    if vectors.dtype != EXPECTED_DTYPE:
        raise VectorIndexError(f"embedding dtype is {vectors.dtype}; expected float32")
    if not vectors.flags["C_CONTIGUOUS"]:
        raise VectorIndexError("embedding matrix is not C-contiguous")

    for start in range(0, count, 4096):
        block = np.asarray(vectors[start : start + 4096], dtype=np.float64)
        if not np.isfinite(block).all():
            raise VectorIndexError(
                f"embeddings contain non-finite values near row {start}"
            )
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise VectorIndexError(
                f"row {start + int(np.argmax(np.abs(norms - 1.0)))} is not L2-normalized "
                f"(norm={float(norms[np.argmax(np.abs(norms - 1.0))]):.6f})"
            )
    return vectors


def iter_metadata(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise VectorIndexError(f"unable to read embedding metadata: {error}") from error
    with handle as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                raise VectorIndexError(f"metadata line {line_number}: blank line")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise VectorIndexError(
                    f"metadata line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise VectorIndexError(f"metadata line {line_number}: not an object")
            for field in METADATA_FIELDS:
                if field not in record:
                    raise VectorIndexError(
                        f"metadata line {line_number}: missing field {field!r}"
                    )
            yield record


def iter_chunk_ids(path: Path) -> Iterator[str]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise VectorIndexError(f"unable to read chunks: {error}") from error
    with handle as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                raise VectorIndexError(f"chunks line {line_number}: blank line")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise VectorIndexError(
                    f"chunks line {line_number}: invalid JSON: {error.msg}"
                ) from error
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise VectorIndexError(
                    f"chunks line {line_number}: missing chunk_id"
                )
            yield chunk_id


def build_sqlite(
    path: Path,
    metadata_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Write row-aligned metadata, keyed by FAISS row id."""

    connection = sqlite3.connect(path)
    chunk_ids: list[str] = []
    empty_state = 0
    empty_department = 0
    logical_digest = hashlib.sha256()
    try:
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("BEGIN")

        rows: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        for row_id, record in enumerate(iter_metadata(metadata_path)):
            chunk_id = record["chunk_id"]
            if chunk_id in seen:
                raise VectorIndexError(
                    f"duplicate chunk_id in metadata: {chunk_id!r}"
                )
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
            if not str(record.get("state") or "").strip():
                empty_state += 1
            if not str(record.get("department") or "").strip():
                empty_department += 1

            row = (
                row_id,
                chunk_id,
                record["sha256"],
                record["scheme_name"],
                record.get("department"),
                record.get("state"),
                record.get("beneficiaries"),
                record.get("source_file"),
                record["page_start"],
                record["page_end"],
                record["text"],
            )
            rows.append(row)
            logical_digest.update(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            if len(rows) >= 1000:
                connection.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
                )
                rows.clear()

        if rows:
            connection.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            rows.clear()

        if len(chunk_ids) != expected_count:
            raise VectorIndexError(
                f"metadata rows are {len(chunk_ids)}; expected {expected_count}"
            )
        connection.commit()

        stored = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if stored != expected_count:
            raise VectorIndexError(
                f"stored metadata rows are {stored}; expected {expected_count}"
            )
        duplicates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        if duplicates:
            raise VectorIndexError(f"stored metadata has {duplicates} duplicate chunk_ids")
    except Exception:
        connection.close()
        raise
    else:
        connection.close()

    return {
        "chunk_ids": chunk_ids,
        "logical_sha256": logical_digest.hexdigest(),
        "empty_state": empty_state,
        "empty_department": empty_department,
    }


def validate_against_chunks(chunks_path: Path, chunk_ids: list[str]) -> str:
    """Prove the metadata order matches data/chunks.jsonl exactly."""

    seen = 0
    for seen, chunk_id in enumerate(iter_chunk_ids(chunks_path), start=1):
        if seen > len(chunk_ids):
            raise VectorIndexError(
                f"chunks file has more than {len(chunk_ids)} records"
            )
        if chunk_id != chunk_ids[seen - 1]:
            raise VectorIndexError(
                f"row {seen - 1} is misaligned: chunks.jsonl has {chunk_id!r}, "
                f"embeddings metadata has {chunk_ids[seen - 1]!r}"
            )
    if seen != len(chunk_ids):
        raise VectorIndexError(
            f"chunks file has {seen} records; metadata has {len(chunk_ids)}"
        )
    return sha256_file(chunks_path)


def build_index(
    faiss: Any,
    vectors: np.ndarray,
    path: Path,
    dimension: int,
    count: int,
) -> Any:
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(vectors))
    if index.ntotal != count:
        raise VectorIndexError(
            f"FAISS index holds {index.ntotal} vectors; expected {count}"
        )
    faiss.write_index(index, str(path))
    return index


def validate_index(
    faiss: Any,
    index_path: Path,
    vectors: np.ndarray,
    count: int,
    dimension: int,
    chunk_ids: list[str],
) -> dict[str, Any]:
    """Re-open the written index and prove vector/row/metadata alignment."""

    index = faiss.read_index(str(index_path))
    if index.ntotal != count:
        raise VectorIndexError(
            f"re-opened index holds {index.ntotal} vectors; expected {count}"
        )
    if index.d != dimension:
        raise VectorIndexError(
            f"re-opened index dimension is {index.d}; expected {dimension}"
        )
    if not isinstance(index, faiss.IndexFlatIP):
        raise VectorIndexError(
            f"re-opened index type is {type(index).__name__}; expected IndexFlatIP"
        )

    restored = np.asarray(index.reconstruct_n(0, count), dtype=np.float32)
    if restored.shape != (count, dimension):
        raise VectorIndexError(
            f"reconstructed matrix shape is {restored.shape}; expected ({count}, {dimension})"
        )
    if not np.array_equal(restored, np.asarray(vectors)):
        mismatch = int(np.argmax(np.any(restored != np.asarray(vectors), axis=1)))
        raise VectorIndexError(
            f"index vectors differ from embeddings.npy starting at row {mismatch}"
        )

    probes = sorted({0, 1, count // 2, count - 2, count - 1})
    for row_id in probes:
        query = np.asarray(vectors[row_id : row_id + 1], dtype=np.float32)
        scores, ids = index.search(query, 1)
        if scores[0][0] < 0.9999:
            raise VectorIndexError(
                f"self-query for row {row_id} scored {float(scores[0][0]):.6f}; "
                "expected ~1.0"
            )
        returned = restored[int(ids[0][0])]
        if not np.array_equal(returned, query[0]):
            raise VectorIndexError(
                f"self-query for row {row_id} returned a different vector"
            )

    return {
        "index_type": type(index).__name__,
        "vector_count": int(index.ntotal),
        "dimension": int(index.d),
        "self_query_probes": probes,
    }


def build_manifest(
    args: argparse.Namespace,
    embeddings_config: dict[str, Any],
    hashes: dict[str, str],
    sqlite_info: dict[str, Any],
    validation: dict[str, Any],
    faiss: Any,
    count: int,
    dimension: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "vector_index",
        "index_backend": "faiss",
        "faiss_version": faiss.__version__,
        "index_type": validation["index_type"],
        "metric": "inner_product",
        "similarity": (
            "cosine: document vectors are L2-normalized, so inner product equals "
            "cosine similarity"
        ),
        "dimension": dimension,
        "vector_count": count,
        "dtype": "float32",
        "embedding_model": embeddings_config["model"],
        "embedding_model_revision": embeddings_config.get("model_revision"),
        "embedding_config_file": args.embeddings_config.as_posix(),
        "embeddings_file": args.embeddings.as_posix(),
        "embeddings_sha256": hashes["embeddings"],
        "embedding_metadata_file": args.metadata.as_posix(),
        "embedding_metadata_sha256": hashes["metadata"],
        "chunks_file": args.chunks.as_posix(),
        "chunks_sha256": hashes["chunks"],
        "index_file": args.index.as_posix(),
        "index_sha256": hashes["index"],
        "metadata_database_file": args.database.as_posix(),
        "metadata_database_logical_sha256": sqlite_info["logical_sha256"],
        "metadata_rows": count,
        "metadata_columns": list(METADATA_FIELDS),
        "filterable_columns": [
            "scheme_name",
            "department",
            "state",
            "source_file",
            "page_start",
            "page_end",
        ],
        "field_coverage": {
            "chunks_with_empty_state": sqlite_info["empty_state"],
            "chunks_with_empty_department": sqlite_info["empty_department"],
        },
        "row_alignment": {
            "faiss_id": "zero-based row index of data/embeddings.npy",
            "metadata_key": "chunks.row_id",
            "embedding_metadata_line": "1-based line number == row_id + 1",
            "chunk_record": "data/chunks.jsonl line row_id + 1",
            "verified": True,
        },
        "retrieval_contract": {
            "document_prefix": DOCUMENT_PREFIX,
            "document_format": DOCUMENT_FORMAT,
            "query_prefix": QUERY_PREFIX,
            "query_format": QUERY_FORMAT,
            "normalize_query_embeddings": True,
            "query_model": embeddings_config["model"],
            "query_model_revision": embeddings_config.get("model_revision"),
            "note": (
                "e5 requires asymmetric prefixes: documents were embedded with "
                "'passage: ' and queries MUST be embedded with 'query: '. "
                "Query vectors must be L2-normalized before inner-product search."
            ),
        },
        "build": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": args.seed,
            "omp_threads": 1,
            "deterministic": True,
            "profile": "exact brute-force search over all vectors (no ANN training)",
        },
        "self_query_probes": validation["self_query_probes"],
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = (args.index, args.database, args.manifest)
    ensure_outputs_available(output_paths, args.overwrite)

    embeddings_config = read_embeddings_config(args.embeddings_config)
    count = args.expected_count
    dimension = EXPECTED_DIMENSION

    print(f"Reading embeddings: {args.embeddings.as_posix()}")
    vectors = load_matrix(args.embeddings, count, dimension)
    print(f"Validated matrix: {count:,} vectors x {dimension} dimensions, normalized")

    index_tmp = temporary_path(args.index)
    database_tmp = temporary_path(args.database)
    manifest_tmp = temporary_path(args.manifest)
    for path in (index_tmp, database_tmp, manifest_tmp):
        remove_if_exists(path)

    try:
        faiss = load_faiss()

        print(f"Building {type(faiss.IndexFlatIP(dimension)).__name__} "
              f"(inner product, exact search)")
        build_index(faiss, vectors, index_tmp, dimension, count)

        print(f"Writing row-aligned metadata: {args.database.as_posix()}")
        sqlite_info = build_sqlite(database_tmp, args.metadata, count)

        chunks_hash = validate_against_chunks(args.chunks, sqlite_info["chunk_ids"])
        print("Verified metadata order matches data/chunks.jsonl exactly")

        print("Validating written index (re-open, full vector comparison)")
        validation = validate_index(
            faiss,
            index_tmp,
            vectors,
            count,
            dimension,
            sqlite_info["chunk_ids"],
        )

        hashes = {
            "embeddings": sha256_file(args.embeddings),
            "metadata": sha256_file(args.metadata),
            "chunks": chunks_hash,
            "index": sha256_file(index_tmp),
        }
        manifest = build_manifest(
            args,
            embeddings_config,
            hashes,
            sqlite_info,
            validation,
            faiss,
            count,
            dimension,
        )
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            file.write("\n")

        del validation
        del vectors

        os.replace(index_tmp, args.index)
        os.replace(database_tmp, args.database)
        os.replace(manifest_tmp, args.manifest)
        return {"manifest": manifest, "hashes": hashes}
    except Exception:
        for path in (index_tmp, database_tmp, manifest_tmp):
            remove_if_exists(path)
        for path in output_paths:
            if path.exists() and not args.overwrite:
                remove_if_exists(path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a FAISS inner-product index plus row-aligned SQLite metadata "
            "over data/embeddings.npy."
        )
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--embeddings-config", type=Path, default=DEFAULT_EMBEDDINGS_CONFIG
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Recorded for provenance; exact flat indexing uses no randomness.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing index artifacts after a successful build.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_count < 1:
        print("--expected-count must be at least 1", file=sys.stderr)
        return 2

    try:
        result = generate(args)
    except Exception as error:
        print(f"Vector index build failed: {error}", file=sys.stderr)
        return 1

    manifest = result["manifest"]
    print("Vector index build and validation completed successfully.")
    print(f"Index backend: {manifest['index_backend']} {manifest['faiss_version']}")
    print(f"Index type: {manifest['index_type']} ({manifest['metric']})")
    print(f"Vectors indexed: {manifest['vector_count']:,}")
    print(f"Dimension: {manifest['dimension']}")
    print(f"Metadata rows: {manifest['metadata_rows']:,}")
    print(f"Index artifact: {args.index.as_posix()}")
    print(f"Metadata database: {args.database.as_posix()}")
    print(f"Manifest: {args.manifest.as_posix()}")
    print(
        "Retrieval contract: documents use 'passage: ', queries MUST use 'query: '"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
