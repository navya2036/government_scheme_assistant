"""Independently verify the persisted vector index artifacts.

This script trusts nothing but the files on disk.  It re-opens the FAISS index,
the SQLite metadata database, the frozen embeddings, and the source chunk file,
then proves count, dimension, row alignment, uniqueness, filter readiness, and
persistence across a fresh process.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_EMBEDDINGS = Path("data/embeddings.npy")
DEFAULT_METADATA = Path("data/embeddings_metadata.jsonl")
DEFAULT_CHUNKS = Path("data/chunks.jsonl")
DEFAULT_INDEX = Path("data/vector_index.faiss")
DEFAULT_DATABASE = Path("data/vector_index_metadata.sqlite")
DEFAULT_MANIFEST = Path("data/vector_index_config.json")

EXPECTED_COUNT = 23_928
EXPECTED_DIMENSION = 384
EXPECTED_MODEL = "intfloat/multilingual-e5-small"


class Checker:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.results.append((name, bool(condition), detail))
        return bool(condition)

    def fail(self, name: str, detail: str) -> None:
        self.results.append((name, False, detail))

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)

    def report(self) -> None:
        width = max(len(name) for name, _, _ in self.results)
        for name, passed, detail in self.results:
            status = "PASS" if passed else "FAIL"
            suffix = f"  {detail}" if detail else ""
            print(f"[{status}] {name.ljust(width)}{suffix}")
        passed_count = sum(1 for _, passed, _ in self.results if passed)
        print(f"\n{passed_count}/{len(self.results)} checks passed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_chunk_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            ids.append(json.loads(line)["chunk_id"])
    return ids


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the FAISS vector index.")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    checker = Checker()
    try:
        import faiss
    except ImportError as error:
        print(f"faiss is required to verify the index: {error}", file=sys.stderr)
        return 2

    # --- artifacts exist -------------------------------------------------
    for label, path in (
        ("index file", args.index),
        ("metadata database", args.database),
        ("manifest", args.manifest),
    ):
        checker.check(f"{label} exists", path.is_file(), path.as_posix())
    if not checker.ok:
        checker.report()
        return 1

    # --- manifest ---------------------------------------------------------
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checker.check(
        "manifest vector_count",
        manifest.get("vector_count") == EXPECTED_COUNT,
        str(manifest.get("vector_count")),
    )
    checker.check(
        "manifest dimension",
        manifest.get("dimension") == EXPECTED_DIMENSION,
        str(manifest.get("dimension")),
    )
    contract = manifest.get("retrieval_contract", {})
    checker.check(
        "manifest document prefix",
        contract.get("document_prefix") == "passage: ",
        repr(contract.get("document_prefix")),
    )
    checker.check(
        "manifest query prefix",
        contract.get("query_prefix") == "query: ",
        repr(contract.get("query_prefix")),
    )
    checker.check(
        "manifest embedding model",
        contract.get("query_model") == EXPECTED_MODEL,
        str(contract.get("query_model")),
    )
    checker.check(
        "manifest query normalization",
        contract.get("normalize_query_embeddings") is True,
        str(contract.get("normalize_query_embeddings")),
    )

    # --- hashes still match what was built --------------------------------
    if manifest.get("index_sha256"):
        checker.check(
            "index file unchanged since build",
            sha256_file(args.index) == manifest["index_sha256"],
            args.index.name,
        )
    if manifest.get("embeddings_sha256"):
        checker.check(
            "embeddings.npy unchanged since build",
            sha256_file(args.embeddings) == manifest["embeddings_sha256"],
            args.embeddings.name,
        )

    # --- FAISS index ------------------------------------------------------
    index = faiss.read_index(str(args.index))
    checker.check(
        "index persisted and re-opened",
        index.ntotal == EXPECTED_COUNT,
        f"{index.ntotal:,} vectors",
    )
    checker.check("index dimension", index.d == EXPECTED_DIMENSION, str(index.d))
    checker.check(
        "index type",
        isinstance(index, faiss.IndexFlatIP),
        type(index).__name__,
    )
    checker.check(
        "exact search profile",
        index.metric_type == faiss.METRIC_INNER_PRODUCT,
        f"metric_type={index.metric_type}",
    )

    # --- frozen embeddings ------------------------------------------------
    vectors = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    checker.check(
        "embeddings.npy shape",
        vectors.shape == (EXPECTED_COUNT, EXPECTED_DIMENSION),
        str(vectors.shape),
    )
    checker.check("embeddings.npy dtype", vectors.dtype == np.float32, str(vectors.dtype))

    # --- vector-level alignment ------------------------------------------
    restored = np.asarray(
        index.reconstruct_n(0, EXPECTED_COUNT), dtype=np.float32
    )
    checker.check(
        "index vectors == embeddings.npy row-for-row",
        bool(np.array_equal(restored, np.asarray(vectors))),
        "exact float32 equality",
    )
    norms = np.linalg.norm(np.asarray(vectors[:2000], dtype=np.float64), axis=1)
    checker.check(
        "vectors are L2-normalized",
        bool(np.allclose(norms, 1.0, atol=1e-3)),
        f"first 2000 rows, max deviation {float(np.max(np.abs(norms - 1.0))):.2e}",
    )

    probes = [0, 1, EXPECTED_COUNT // 2, EXPECTED_COUNT - 2, EXPECTED_COUNT - 1]
    probe_ok = True
    for row_id in probes:
        query = np.asarray(vectors[row_id : row_id + 1], dtype=np.float32)
        scores, ids = index.search(query, 1)
        if scores[0][0] < 0.9999 or not np.array_equal(
            restored[int(ids[0][0])], query[0]
        ):
            probe_ok = False
    checker.check(
        "self-query returns identical vector (cosine ~1.0)",
        probe_ok,
        f"probes {probes}",
    )

    # --- SQLite metadata --------------------------------------------------
    connection = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
    try:
        row_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        checker.check(
            "metadata row count",
            row_count == EXPECTED_COUNT,
            f"{row_count:,} rows",
        )
        duplicates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        checker.check("no duplicate chunk IDs", duplicates == 0, f"{duplicates} duplicates")

        min_id, max_id, distinct_ids = connection.execute(
            "SELECT MIN(row_id), MAX(row_id), COUNT(DISTINCT row_id) FROM chunks"
        ).fetchone()
        checker.check(
            "row_id is contiguous zero-based",
            (min_id, max_id, distinct_ids) == (0, EXPECTED_COUNT - 1, EXPECTED_COUNT),
            f"min={min_id}, max={max_id}, distinct={distinct_ids}",
        )

        sqlite_ids = [
            row[0]
            for row in connection.execute("SELECT chunk_id FROM chunks ORDER BY row_id")
        ]
    finally:
        connection.close()

    metadata_ids = [
        json.loads(line)["chunk_id"]
        for line in args.metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_ids = stream_chunk_ids(args.chunks)

    checker.check(
        "metadata rows == embeddings rows",
        len(metadata_ids) == EXPECTED_COUNT,
        f"{len(metadata_ids):,}",
    )
    checker.check(
        "sqlite row order == embeddings_metadata.jsonl order",
        sqlite_ids == metadata_ids,
        "chunk_id sequence identical",
    )
    checker.check(
        "sqlite row order == chunks.jsonl order",
        sqlite_ids == chunk_ids,
        "chunk_id sequence identical",
    )

    # --- filter readiness -------------------------------------------------
    connection = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
    try:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        for column in ("scheme_name", "department", "state"):
            checker.check(
                f"filter index on {column}",
                f"idx_chunks_{column}" in indexes,
                f"idx_chunks_{column}",
            )
        schemes = connection.execute(
            "SELECT COUNT(DISTINCT scheme_name) FROM chunks"
        ).fetchone()[0]
        with_state = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE TRIM(COALESCE(state,'')) <> ''"
        ).fetchone()[0]
        checker.check("scheme_name populated", schemes > 0, f"{schemes:,} distinct schemes")
        checker.check(
            "state filterable where available",
            with_state > 0,
            f"{with_state:,}/{EXPECTED_COUNT:,} rows have a state",
        )
        sample = connection.execute(
            "SELECT row_id, chunk_id, scheme_name, page_start, page_end FROM chunks "
            "WHERE row_id = 0"
        ).fetchone()
        checker.check(
            "row 0 metadata readable",
            sample is not None and sample[0] == 0,
            f"{sample[1] if sample else None}",
        )
    finally:
        connection.close()

    checker.report()
    return 0 if checker.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
