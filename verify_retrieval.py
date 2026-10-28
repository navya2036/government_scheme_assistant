"""Verification tests for the retrieval stage.

Every assertion is checked against the frozen artifacts on disk rather than
against the retriever's own internal state:

* the query vector is re-derived from the pinned e5 model and checked for
  dimension 384 and unit norm
* scores are recomputed as ``dot(query, embeddings[row])``
* the top-1 result is checked against an exhaustive brute-force search over all
  23,928 vectors
* FAISS row IDs are cross-checked against SQLite rows, ``embeddings_metadata``
  lines, and ``chunks.jsonl`` lines
* repeat runs must be identical

Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from retrieval import (
    DEFAULT_DATABASE,
    DEFAULT_INDEX,
    DEFAULT_MANIFEST,
    DOCUMENT_PREFIX,
    EXPECTED_COUNT,
    EXPECTED_DIMENSION,
    QUERY_PREFIX,
    RetrievalError,
    Retriever,
    build_query_text,
)

EXAMPLE_QUERY = "What financial assistance is available for old age pension?"


class Checker:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: Any, detail: str = "") -> bool:
        passed = bool(condition)
        self.results.append((name, passed, detail))
        return passed

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)

    def report(self) -> None:
        width = max(len(name) for name, _, _ in self.results)
        for name, passed, detail in self.results:
            status = "PASS" if passed else "FAIL"
            suffix = f"  {detail}" if detail else ""
            print(f"[{status}] {name.ljust(width)}{suffix}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{passed}/{len(self.results)} checks passed")


def expect_error(checker: Checker, name: str, function: Any) -> None:
    try:
        function()
    except RetrievalError:
        checker.check(name, True, "raised RetrievalError")
    except Exception as error:  # pragma: no cover - diagnostic path
        checker.check(name, False, f"raised {type(error).__name__}: {error}")
    else:
        checker.check(name, False, "no error raised")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify the retrieval stage.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=Path("data/embeddings_metadata.jsonl"),
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    args = parser.parse_args(argv)

    checker = Checker()

    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    embedding_metadata = [
        json.loads(line)
        for line in args.embedding_metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [
        json.loads(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    connection = sqlite3.connect(
        f"file:{args.database.as_posix()}?mode=ro", uri=True
    )

    retriever = Retriever(
        index_path=args.index,
        database_path=args.database,
        manifest_path=args.manifest,
        top_k=5,
    )
    try:
        # --- prefix contract ---------------------------------------------
        checker.check(
            "query prefix applied",
            build_query_text("hello") == f"{QUERY_PREFIX}hello",
            repr(build_query_text("hello")),
        )
        checker.check(
            "prefixes are asymmetric",
            QUERY_PREFIX != DOCUMENT_PREFIX,
            f"{DOCUMENT_PREFIX!r} vs {QUERY_PREFIX!r}",
        )
        expect_error(checker, "empty question rejected", lambda: build_query_text("   "))
        expect_error(checker, "non-string question rejected", lambda: build_query_text(None))

        # --- query encoding ----------------------------------------------
        query_vector = retriever.encoder.encode(EXAMPLE_QUERY)
        checker.check(
            "query vector dimension",
            query_vector.shape == (EXPECTED_DIMENSION,),
            str(query_vector.shape),
        )
        norm = float(np.linalg.norm(query_vector.astype(np.float64)))
        checker.check(
            "query vector normalized",
            abs(norm - 1.0) <= 1e-3,
            f"L2 norm={norm:.6f}",
        )
        checker.check(
            "query vector dtype",
            query_vector.dtype == np.float32,
            str(query_vector.dtype),
        )

        # --- top-K behaviour ---------------------------------------------
        one = retriever.search(EXAMPLE_QUERY, top_k=1)
        ten = retriever.search(EXAMPLE_QUERY, top_k=10)
        checker.check("top_k=1 returns 1 result", len(one) == 1, f"{len(one)}")
        checker.check("top_k=10 returns 10 results", len(ten) == 10, f"{len(ten)}")
        checker.check(
            "results ranked contiguously",
            [item.rank for item in ten] == list(range(1, 11)),
            str([item.rank for item in ten]),
        )
        checker.check(
            "row ids unique within results",
            len({item.row_id for item in ten}) == len(ten),
            f"{len({item.row_id for item in ten})} unique",
        )
        checker.check(
            "scores non-increasing",
            all(
                ten[i].similarity >= ten[i + 1].similarity
                for i in range(len(ten) - 1)
            ),
            f"{ten[0].similarity:.4f} .. {ten[-1].similarity:.4f}",
        )
        checker.check(
            "scores within cosine range",
            all(-1.0 <= item.similarity <= 1.0 for item in ten),
            "all in [-1, 1]",
        )

        # --- row id validity and metadata alignment ----------------------
        valid_ids = all(0 <= item.row_id < EXPECTED_COUNT for item in ten)
        checker.check(
            "faiss row ids in range",
            valid_ids,
            f"0 <= id < {EXPECTED_COUNT:,}",
        )

        sqlite_ok = True
        metadata_ok = True
        chunk_ok = True
        vector_ok = True
        for item in ten:
            row = connection.execute(
                "SELECT chunk_id FROM chunks WHERE row_id = ?", (item.row_id,)
            ).fetchone()
            if row is None or row[0] != item.chunk_id:
                sqlite_ok = False
            meta = embedding_metadata[item.row_id]
            if (
                meta["chunk_id"] != item.chunk_id
                or meta["text"] != item.text
                or meta["scheme_name"] != item.scheme_name
            ):
                metadata_ok = False
            chunk = chunks[item.row_id]
            if chunk["chunk_id"] != item.chunk_id or chunk["text"] != item.text:
                chunk_ok = False
            reconstructed = np.asarray(
                retriever.index.index.reconstruct_n(item.row_id, 1), dtype=np.float32
            )[0]
            if not np.array_equal(reconstructed, np.asarray(embeddings[item.row_id])):
                vector_ok = False

        checker.check("faiss id maps to sqlite row", sqlite_ok, "row_id -> chunk_id")
        checker.check(
            "returned chunk_id matches embeddings_metadata row",
            metadata_ok,
            "chunk_id + text + scheme_name",
        )
        checker.check(
            "returned chunk_id matches chunks.jsonl row", chunk_ok, "chunk_id + text"
        )
        checker.check(
            "faiss vector matches embeddings.npy row", vector_ok, "exact float32"
        )

        # --- score correctness -------------------------------------------
        recomputed_ok = True
        worst = 0.0
        for item in ten:
            expected = float(
                np.dot(
                    np.asarray(embeddings[item.row_id], dtype=np.float64),
                    query_vector.astype(np.float64),
                )
            )
            worst = max(worst, abs(expected - item.similarity))
            if abs(expected - item.similarity) > 1e-4:
                recomputed_ok = False
        checker.check(
            "reported score == dot(query, row vector)",
            recomputed_ok,
            f"max deviation {worst:.2e}",
        )

        # --- exhaustive agreement ----------------------------------------
        all_scores = np.asarray(embeddings, dtype=np.float32) @ query_vector
        best_id = int(np.argmax(all_scores))
        best_score = float(all_scores[best_id])
        checker.check(
            "top-1 matches brute-force over all 23,928 vectors",
            best_id == ten[0].row_id and abs(best_score - ten[0].similarity) <= 1e-4,
            f"argmax id={best_id}, score={best_score:.4f}",
        )

        # --- determinism --------------------------------------------------
        repeat_a = retriever.search(EXAMPLE_QUERY, top_k=5)
        repeat_b = retriever.search(EXAMPLE_QUERY, top_k=5)
        checker.check(
            "repeat query returns identical ids",
            [r.chunk_id for r in repeat_a] == [r.chunk_id for r in repeat_b],
            "same chunk_id sequence",
        )
        checker.check(
            "repeat query returns identical scores",
            np.array_equal(
                np.array([r.similarity for r in repeat_a], dtype=np.float32),
                np.array([r.similarity for r in repeat_b], dtype=np.float32),
            ),
            "bitwise equal float32",
        )

        fresh = Retriever(
            index_path=args.index,
            database_path=args.database,
            manifest_path=args.manifest,
            top_k=5,
        )
        try:
            fresh_results = fresh.search(EXAMPLE_QUERY, top_k=5)
        finally:
            fresh.close()
        checker.check(
            "fresh instance returns identical ids",
            [r.chunk_id for r in fresh_results] == [r.chunk_id for r in repeat_a],
            "no hidden state",
        )
        checker.check(
            "fresh instance returns identical scores",
            np.array_equal(
                np.array([r.similarity for r in fresh_results], dtype=np.float32),
                np.array([r.similarity for r in repeat_a], dtype=np.float32),
            ),
            "bitwise equal float32",
        )

        # --- payload completeness ----------------------------------------
        required = (
            "chunk_id",
            "scheme_name",
            "department",
            "state",
            "beneficiaries",
            "source_file",
            "page_start",
            "page_end",
            "text",
            "similarity",
        )
        payload = ten[0].to_dict()
        checker.check(
            "all required fields returned",
            all(field in payload for field in required),
            ", ".join(required),
        )
        checker.check(
            "field types valid",
            isinstance(payload["scheme_name"], str)
            and isinstance(payload["page_start"], int)
            and isinstance(payload["page_end"], int)
            and isinstance(payload["similarity"], float)
            and payload["text"].strip() != "",
            "str/int/float/text non-empty",
        )
        checker.check(
            "page range valid",
            all(item.page_start >= 1 and item.page_end >= item.page_start for item in ten),
            "page_end >= page_start >= 1",
        )

        # --- input validation ---------------------------------------------
        expect_error(checker, "top_k=0 rejected", lambda: retriever.search(EXAMPLE_QUERY, 0))
        expect_error(checker, "top_k=-1 rejected", lambda: retriever.search(EXAMPLE_QUERY, -1))
        expect_error(
            checker,
            "top_k beyond index size rejected",
            lambda: retriever.search(EXAMPLE_QUERY, EXPECTED_COUNT + 1),
        )
    finally:
        retriever.close()
        connection.close()

    checker.report()
    return 0 if checker.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
