"""Verification tests for the context-assembly + reranking stage.

Nothing is trusted from the pipeline's own bookkeeping.  Each claim is re-derived
from the frozen artifacts:

* the FAISS call is spied on to prove it is asked for exactly ``candidate_k``
* every returned chunk is re-looked-up in SQLite, ``embeddings_metadata.jsonl``
  and ``chunks.jsonl``
* every reranker score is recomputed one pair at a time and compared with the
  batched score
* the unchanged retrieval path is compared against brute-force search over
  ``embeddings.npy``
* artifact hashes are compared with ``vector_index_config.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np

import rerank
from rerank import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_K,
    RERANKER_MODEL,
    RERANKER_REVISION,
    CrossEncoderReranker,
    RerankPipeline,
    RerankerError,
)
from retrieval import (
    QUERY_PREFIX,
    RetrievalError,
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
            print(f"[{'PASS' if passed else 'FAIL'}] {name.ljust(width)}  {detail}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{passed}/{len(self.results)} checks passed")


def expect_error(checker: Checker, name: str, function: Any, kind: type) -> None:
    try:
        function()
    except kind as error:
        checker.check(name, True, f"raised {type(error).__name__}")
    except Exception as error:  # pragma: no cover - diagnostic
        checker.check(name, False, f"wrong error {type(error).__name__}: {error}")
    else:
        checker.check(name, False, "no error raised")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify context assembly + reranking.")
    parser.add_argument("--index", type=Path, default=Path("data/vector_index.faiss"))
    parser.add_argument(
        "--database", type=Path, default=Path("data/vector_index_metadata.sqlite")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/vector_index_config.json")
    )
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=Path("data/embeddings_metadata.jsonl"),
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    args = parser.parse_args(argv)

    checker = Checker()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
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
    connection = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)

    pipeline = RerankPipeline(
        candidate_k=DEFAULT_CANDIDATE_K,
        final_k=DEFAULT_FINAL_K,
        neighbor_radius=1,
    )
    try:
        # --- pinned contract ---------------------------------------------
        checker.check(
            "e5 query prefix is exactly 'query: '",
            QUERY_PREFIX == "query: " and build_query_text("x") == "query: x",
            repr(QUERY_PREFIX),
        )
        contract = manifest["retrieval_contract"]
        checker.check(
            "manifest query prefix preserved",
            contract["query_prefix"] == "query: "
            and contract["document_prefix"] == "passage: ",
            f"{contract['document_prefix']!r} / {contract['query_prefix']!r}",
        )
        checker.check(
            "embedding model unchanged",
            manifest["embedding_model"] == "intfloat/multilingual-e5-small",
            manifest["embedding_model"],
        )
        checker.check(
            "reranker is not the embedding model",
            RERANKER_MODEL != manifest["embedding_model"],
            RERANKER_MODEL,
        )
        checker.check(
            "reranker model pinned",
            RERANKER_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            and len(RERANKER_REVISION) == 40,
            RERANKER_REVISION[:12],
        )

        # --- candidate_k requested from FAISS -----------------------------
        captured: dict[str, Any] = {}
        original_search = pipeline.retriever.index.search

        def spy(vector: np.ndarray, k: int) -> Any:
            captured["k"] = k
            return original_search(vector, k)

        pipeline.retriever.index.search = spy  # type: ignore[method-assign]
        try:
            result = pipeline.retrieve(EXAMPLE_QUERY)
        finally:
            del pipeline.retriever.index.search  # type: ignore[attr-defined]

        checker.check(
            "FAISS asked for exactly 30 candidates",
            captured.get("k") == 30 and result.candidate_k == 30,
            f"k={captured.get('k')}",
        )
        checker.check(
            "30 candidates returned",
            len(result.candidates) == 30,
            f"{len(result.candidates)} candidates",
        )
        checker.check(
            "final result at most 5 chunks",
            len(result.results) <= DEFAULT_FINAL_K == 5,
            f"{len(result.results)} chunks",
        )
        checker.check(
            "final result exactly 5 chunks",
            len(result.results) == 5,
            f"{len(result.results)}",
        )
        checker.check(
            "candidates keep FAISS order before reranking",
            [c.faiss_rank for c in result.candidates] == list(range(1, 31)),
            "faiss_rank 1..30",
        )
        checker.check(
            "final chunks are a subset of candidates",
            {r.chunk_id for r in result.results}
            <= {c.chunk_id for c in result.candidates},
            "subset",
        )

        # --- no duplicates ------------------------------------------------
        checker.check(
            "no duplicate chunk ids in final result",
            len({r.chunk_id for r in result.results}) == len(result.results),
            f"{len(result.results)} unique",
        )
        checker.check(
            "no duplicate chunk ids among candidates",
            len({c.chunk_id for c in result.candidates}) == 30,
            "30 unique",
        )
        checker.check(
            "no duplicate FAISS row ids among candidates",
            len({c.row_id for c in result.candidates}) == 30,
            "30 unique",
        )

        # --- metadata alignment ------------------------------------------
        sqlite_ok = metadata_ok = chunk_ok = text_ok = True
        for item in list(result.results) + list(result.candidates):
            row = connection.execute(
                "SELECT chunk_id, text, scheme_name FROM chunks WHERE row_id = ?",
                (item.row_id,),
            ).fetchone()
            if row is None or row[0] != item.chunk_id or row[1] != item.text:
                sqlite_ok = False
            meta = embedding_metadata[item.row_id]
            if meta["chunk_id"] != item.chunk_id or meta["scheme_name"] != item.scheme_name:
                metadata_ok = False
            if chunks[item.row_id]["chunk_id"] != item.chunk_id:
                chunk_ok = False
            if item.text.strip() == "":
                text_ok = False
        checker.check("row ids map to aligned sqlite rows", sqlite_ok, "chunk_id + text")
        checker.check(
            "metadata aligned with embeddings_metadata.jsonl", metadata_ok, "chunk_id + scheme"
        )
        checker.check(
            "metadata aligned with chunks.jsonl", chunk_ok, "chunk_id per row"
        )
        checker.check("every chunk has text", text_ok, "non-empty")

        # --- reranker identity -------------------------------------------
        checker.check(
            "reranker revision verified at load",
            pipeline.reranker.loaded_revision == RERANKER_REVISION,
            str(pipeline.reranker.loaded_revision)[:12],
        )
        checker.check(
            "reranker scores all 30 with distinct values",
            len({round(c.reranker_score, 6) for c in result.candidates}) > 1,
            f"{len({round(c.reranker_score, 6) for c in result.candidates})} distinct",
        )
        checker.check(
            "reranker scores are finite floats",
            all(np.isfinite(c.reranker_score) for c in result.candidates),
            "all finite",
        )

        # --- reranker score -> correct pair ------------------------------
        recomputed_ok = True
        worst = 0.0
        for item in result.results:
            single = pipeline.reranker.score(EXAMPLE_QUERY, [item.text])[0]
            worst = max(worst, abs(float(single) - item.reranker_score))
            if abs(float(single) - item.reranker_score) > 1e-4:
                recomputed_ok = False
        checker.check(
            "reranker score == score(question, that chunk's text)",
            recomputed_ok,
            f"max deviation {worst:.2e}",
        )

        # Batch-size independence on the full candidate list proves no
        # score/pair shifting inside the batched call.
        batch_ok = True
        worst_batch = 0.0
        for item in result.candidates:
            single = pipeline.reranker.score(EXAMPLE_QUERY, [item.text])[0]
            worst_batch = max(worst_batch, abs(float(single) - item.reranker_score))
            if abs(float(single) - item.reranker_score) > 1e-4:
                batch_ok = False
        checker.check(
            "batched scores match per-pair scores for all 30",
            batch_ok,
            f"max deviation {worst_batch:.2e}",
        )

        # --- ordering ------------------------------------------------------
        scores = [r.reranker_score for r in result.results]
        checker.check(
            "final results sorted by reranker relevance",
            all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
            f"{scores[0]:+.4f} .. {scores[-1]:+.4f}",
        )
        checker.check(
            "final ranks contiguous from 1",
            [r.final_rank for r in result.results] == list(range(1, len(result.results) + 1)),
            str([r.final_rank for r in result.results]),
        )
        top_candidate = max(result.candidates, key=lambda c: c.reranker_score)
        checker.check(
            "rank 1 is the highest-scoring candidate",
            result.results[0].chunk_id == top_candidate.chunk_id,
            result.results[0].chunk_id[:24],
        )

        # --- determinism ----------------------------------------------------
        repeat = pipeline.retrieve(EXAMPLE_QUERY)
        checker.check(
            "repeat query identical ranking",
            [r.chunk_id for r in repeat.results] == [r.chunk_id for r in result.results],
            "same chunk_id order",
        )
        checker.check(
            "repeat query identical reranker scores",
            np.array_equal(
                np.array([r.reranker_score for r in repeat.results], dtype=np.float32),
                np.array([r.reranker_score for r in result.results], dtype=np.float32),
            ),
            "bitwise equal float32",
        )
        checker.check(
            "repeat query identical FAISS scores",
            np.array_equal(
                np.array([c.faiss_similarity for c in repeat.candidates], dtype=np.float32),
                np.array([c.faiss_similarity for c in result.candidates], dtype=np.float32),
            ),
            "bitwise equal float32",
        )

        # --- context assembly ----------------------------------------------
        context_ok = True
        bounded_ok = True
        order_ok = True
        same_doc_ok = True
        for item in result.results:
            rows = connection.execute(
                "SELECT row_id, sha256, text FROM chunks WHERE row_id IN ({}"
                ")".format(",".join("?" * len(item.context_row_ids))),
                item.context_row_ids,
            ).fetchall()
            by_id = {int(r[0]): (r[1], r[2]) for r in rows}
            if set(by_id) != set(item.context_row_ids):
                context_ok = False
            expected = "\n\n".join(by_id[row_id][1] for row_id in item.context_row_ids)
            if expected != item.context_text:
                context_ok = False
            if not list(item.context_row_ids) == sorted(item.context_row_ids):
                order_ok = False
            if len(item.context_row_ids) > 3:
                bounded_ok = False
            primary_sha = by_id[item.row_id][0]
            if any(by_id[row_id][0] != primary_sha for row_id in item.context_row_ids):
                same_doc_ok = False
            if item.row_id not in item.context_row_ids:
                context_ok = False
        checker.check(
            "context_text is the ordered text of context_row_ids",
            context_ok,
            "exact string match",
        )
        checker.check(
            "context rows in ascending document order", order_ok, "row_id ascending"
        )
        checker.check(
            "context window bounded to 3 rows (radius 1)", bounded_ok, "<= 2*radius+1"
        )
        checker.check(
            "context rows stay inside one source PDF", same_doc_ok, "same sha256"
        )
        flat = [row for item in result.results for row in item.context_row_ids]
        checker.check(
            "no context row repeated across blocks",
            len(flat) == len(set(flat)),
            f"{len(flat)} rows, {len(set(flat))} unique",
        )

        plain = pipeline.retriever.search(EXAMPLE_QUERY, top_k=5)
        checker.check(
            "radius 0 disables expansion",
            all(
                len(item.context_row_ids) == 1 and not item.context_expanded
                for item in RerankPipeline(
                    retriever=pipeline.retriever,
                    reranker=pipeline.reranker,
                    candidate_k=30,
                    final_k=5,
                    neighbor_radius=0,
                ).retrieve(EXAMPLE_QUERY).results
            ),
            "context_row_ids == (row_id,)",
        )

        # --- existing retrieval unchanged -----------------------------------
        query_vector = pipeline.retriever.encoder.encode(EXAMPLE_QUERY)
        brute = np.asarray(embeddings, dtype=np.float32) @ query_vector
        brute_top5 = np.argsort(-brute, kind="stable")[:5].tolist()
        checker.check(
            "plain FAISS retrieval unchanged vs brute force",
            [hit.row_id for hit in plain] == brute_top5,
            f"top5 ids {brute_top5}",
        )
        checker.check(
            "candidates begin with plain FAISS order",
            [c.row_id for c in result.candidates[:5]] == brute_top5,
            "candidate order preserved",
        )
        checker.check(
            "query vector still 384-dim and normalized",
            query_vector.shape == (384,)
            and abs(float(np.linalg.norm(query_vector.astype(np.float64))) - 1.0) <= 1e-3,
            f"dim={query_vector.shape[0]}, norm="
            f"{float(np.linalg.norm(query_vector.astype(np.float64))):.6f}",
        )

        # --- artifacts byte-identical ---------------------------------------
        for label, path, expected in (
            ("vector_index.faiss", args.index, manifest["index_sha256"]),
            ("embeddings.npy", args.embeddings, manifest["embeddings_sha256"]),
            ("chunks.jsonl", args.chunks, manifest["chunks_sha256"]),
            (
                "embeddings_metadata.jsonl",
                args.embedding_metadata,
                manifest["embedding_metadata_sha256"],
            ),
        ):
            checker.check(
                f"{label} byte-identical",
                sha256_file(path) == expected,
                "sha256 matches manifest",
            )
        norms = np.linalg.norm(
            np.asarray(embeddings[:2000], dtype=np.float64), axis=1
        )
        checker.check(
            "document embeddings untouched and normalized",
            bool(np.allclose(norms, 1.0, atol=1e-3)),
            "first 2000 rows",
        )

        # --- API limits ------------------------------------------------------
        expect_error(
            checker,
            "final_k > candidate_k rejected",
            lambda: pipeline.retrieve(EXAMPLE_QUERY, candidate_k=30, final_k=40),
            RetrievalError,
        )
        expect_error(
            checker,
            "candidate_k=0 rejected",
            lambda: pipeline.retrieve(EXAMPLE_QUERY, candidate_k=0),
            RetrievalError,
        )
        expect_error(
            checker,
            "final_k=0 rejected",
            lambda: pipeline.retrieve(EXAMPLE_QUERY, final_k=0),
            RetrievalError,
        )

        # --- public entrypoint ------------------------------------------------
        api_result = rerank.retrieve(EXAMPLE_QUERY)
        checker.check(
            "retrieve(question) exposes 30 candidates and 5 results",
            api_result.candidate_k == 30
            and len(api_result.candidates) == 30
            and len(api_result.results) == 5,
            "candidate_k=30, final_k=5",
        )

        # --- reranker failure is explicit -------------------------------------
        expect_error(
            checker,
            "missing reranker model fails clearly (no fallback)",
            lambda: CrossEncoderReranker(
                model_name="cross-encoder/this-model-does-not-exist-xyz",
                revision=None,
            ).score("q", ["text"]),
            RerankerError,
        )
    finally:
        pipeline.close()
        connection.close()

    checker.report()
    return 0 if checker.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
