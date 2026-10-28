"""Deterministic e5 retrieval over the frozen FAISS vector index.

This module is limited to retrieval: encode one question, search the existing
``data/vector_index.faiss``, and map FAISS row IDs back to row-aligned metadata.
It never re-embeds documents, never rebuilds the index, and never generates an
answer.

The e5 asymmetry is enforced rather than assumed:

* documents were embedded as ``passage: Scheme: {scheme_name}\\n{text}``
* questions are embedded as ``query: {user_question}`` and L2-normalized

Pinned model and revision are read from ``data/vector_index_config.json`` so the
query encoder cannot silently drift from the document encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Single-threaded BLAS/FAISS keeps scores and tie order reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


DEFAULT_INDEX = Path("data/vector_index.faiss")
DEFAULT_DATABASE = Path("data/vector_index_metadata.sqlite")
DEFAULT_MANIFEST = Path("data/vector_index_config.json")

EXPECTED_MODEL = "intfloat/multilingual-e5-small"
EXPECTED_DIMENSION = 384
EXPECTED_COUNT = 23_928

DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "
# Candidate count retrieved from FAISS before the cross-encoder reranking stage
# (see rerank.py).  This is NOT the number of chunks sent to a generator.
DEFAULT_TOP_K = 30
DEFAULT_SEED = 0


class RetrievalError(RuntimeError):
    """Raised when a query cannot be answered from the existing index."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk with its metadata and cosine similarity."""

    chunk_id: str
    scheme_name: str
    department: str
    state: str
    beneficiaries: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    similarity: float
    row_id: int
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query_text(question: str) -> str:
    """Apply the e5 query prefix; documents used a different prefix."""

    if not isinstance(question, str):
        raise RetrievalError("question must be a string")
    stripped = question.strip()
    if not stripped:
        raise RetrievalError("question must not be empty or whitespace")
    return f"{QUERY_PREFIX}{stripped}"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except OSError as error:
        raise RetrievalError(f"unable to read index manifest: {error}") from error
    except json.JSONDecodeError as error:
        raise RetrievalError(f"index manifest is not valid JSON: {error}") from error

    contract = manifest.get("retrieval_contract")
    if not isinstance(contract, dict):
        raise RetrievalError("index manifest is missing retrieval_contract")
    if contract.get("query_prefix") != QUERY_PREFIX:
        raise RetrievalError(
            f"index contract query_prefix is {contract.get('query_prefix')!r}; "
            f"expected {QUERY_PREFIX!r}"
        )
    if contract.get("document_prefix") != DOCUMENT_PREFIX:
        raise RetrievalError(
            f"index contract document_prefix is {contract.get('document_prefix')!r}; "
            f"expected {DOCUMENT_PREFIX!r}"
        )
    if contract.get("normalize_query_embeddings") is not True:
        raise RetrievalError(
            "index contract does not require normalized query embeddings"
        )
    if manifest.get("dimension") != EXPECTED_DIMENSION:
        raise RetrievalError(
            f"index dimension is {manifest.get('dimension')!r}; "
            f"expected {EXPECTED_DIMENSION}"
        )
    return manifest


class E5QueryEncoder:
    """Lazy, deterministic encoder for ``query: {question}`` strings."""

    def __init__(
        self,
        model_name: str = EXPECTED_MODEL,
        revision: Optional[str] = None,
        device: str = "cpu",
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.seed = seed
        self._model: Any = None

    def _load(self) -> Any:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Keep stderr clean: weight-loading progress bars would otherwise
        # corrupt piped/JSON output.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RetrievalError(
                "sentence-transformers and PyTorch are required for query encoding"
            ) from error

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:  # pragma: no cover - optional cosmetic step
            pass

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Compatibility with older PyTorch versions.
            torch.use_deterministic_algorithms(True)

        try:
            model = SentenceTransformer(
                self.model_name, revision=self.revision, device=self.device
            )
            model.eval()
        except Exception as error:
            raise RetrievalError(
                f"unable to load query model {self.model_name!r} "
                f"at revision {self.revision!r}: {error}"
            ) from error
        return model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def encode(self, question: str) -> np.ndarray:
        vector = self.encode_many([question])
        return vector[0]

    def encode_many(self, questions: list[str]) -> np.ndarray:
        texts = [build_query_text(question) for question in questions]
        try:
            with_grad = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as error:
            raise RetrievalError(f"query encoding failed: {error}") from error

        vectors = np.ascontiguousarray(with_grad, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RetrievalError(
                f"query encoder returned shape {vectors.shape}; "
                f"expected ({len(texts)}, dimension)"
            )
        if vectors.shape[1] != EXPECTED_DIMENSION:
            raise RetrievalError(
                f"query embedding dimension is {vectors.shape[1]}; "
                f"expected {EXPECTED_DIMENSION}"
            )
        if not np.isfinite(vectors).all():
            raise RetrievalError("query embedding contains non-finite values")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RetrievalError(
                f"query embeddings are not L2-normalized (norms={norms.tolist()})"
            )
        return vectors


class VectorIndex:
    """Read-only view of the FAISS index and its row-aligned metadata."""

    def __init__(self, index_path: Path, database_path: Path, manifest_path: Path):
        self.index_path = index_path
        self.database_path = database_path
        self.manifest_path = manifest_path
        self.manifest = load_manifest(manifest_path)

        try:
            import faiss
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RetrievalError(
                "faiss is required. Install it with: pip install faiss-cpu"
            ) from error
        faiss.omp_set_num_threads(1)
        self._faiss = faiss

        try:
            self.index = faiss.read_index(str(index_path))
        except Exception as error:
            raise RetrievalError(f"unable to read FAISS index: {error}") from error

        if self.index.d != EXPECTED_DIMENSION:
            raise RetrievalError(
                f"index dimension is {self.index.d}; expected {EXPECTED_DIMENSION}"
            )
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise RetrievalError(
                f"index type is {type(self.index).__name__}; expected IndexFlatIP"
            )
        if self.index.ntotal != self.manifest.get("vector_count"):
            raise RetrievalError(
                f"index holds {self.index.ntotal} vectors but the manifest declares "
                f"{self.manifest.get('vector_count')}"
            )

        uri = f"file:{database_path.as_posix()}?mode=ro"
        try:
            self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        except sqlite3.Error as error:
            raise RetrievalError(f"unable to open metadata database: {error}") from error

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def fetch(self, row_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT chunk_id, scheme_name, department, state, beneficiaries, "
            "source_file, page_start, page_end, text FROM chunks WHERE row_id = ?",
            (int(row_id),),
        ).fetchone()
        if row is None:
            raise RetrievalError(f"no metadata row for FAISS id {row_id}")
        return {
            "chunk_id": row["chunk_id"],
            "scheme_name": row["scheme_name"],
            "department": row["department"] or "",
            "state": row["state"] or "",
            "beneficiaries": row["beneficiaries"] or "",
            "source_file": row["source_file"] or "",
            "page_start": int(row["page_start"]),
            "page_end": int(row["page_end"]),
            "text": row["text"],
        }

    def search(self, vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(query, top_k)
        return np.asarray(scores[0], dtype=np.float32), np.asarray(ids[0], dtype=np.int64)

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class Retriever:
    """Encode a question, search the index, and return aligned chunks."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX,
        database_path: Path = DEFAULT_DATABASE,
        manifest_path: Path = DEFAULT_MANIFEST,
        model_name: Optional[str] = None,
        revision: Optional[str] = None,
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.index = VectorIndex(index_path, database_path, manifest_path)
        contract = self.index.manifest["retrieval_contract"]

        pinned_model = model_name or contract.get("query_model") or EXPECTED_MODEL
        if pinned_model != self.index.manifest.get("embedding_model"):
            raise RetrievalError(
                f"query model {pinned_model!r} does not match the indexed model "
                f"{self.index.manifest.get('embedding_model')!r}"
            )
        pinned_revision = revision or contract.get("query_model_revision")

        self.top_k = self._validate_top_k(top_k)
        self.encoder = E5QueryEncoder(
            model_name=pinned_model,
            revision=pinned_revision,
            device=device,
            seed=seed,
        )

    def _validate_top_k(self, top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrievalError("top_k must be an integer")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1")
        if top_k > self.index.count:
            raise RetrievalError(
                f"top_k is {top_k} but the index holds only {self.index.count} vectors"
            )
        return top_k

    def search(
        self, question: str, top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        k = self.top_k if top_k is None else self._validate_top_k(top_k)
        vector = self.encoder.encode(question)
        return self.search_vector(vector, top_k=k)

    def search_vector(
        self, vector: np.ndarray, top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        k = self.top_k if top_k is None else self._validate_top_k(top_k)
        scores, ids = self.index.search(vector, k)

        results: list[RetrievedChunk] = []
        for score, row_id in zip(scores, ids):
            if int(row_id) < 0:
                continue
            record = self.index.fetch(int(row_id))
            results.append(
                RetrievedChunk(
                    similarity=float(score),
                    row_id=int(row_id),
                    rank=0,
                    **record,
                )
            )

        # Stable tie-breaking: higher score first, then lower FAISS id.  FAISS
        # already returns this order for flat search; re-sorting makes the
        # guarantee explicit and independent of backend internals.
        results.sort(key=lambda item: (-item.similarity, item.row_id))
        ranked = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                scheme_name=item.scheme_name,
                department=item.department,
                state=item.state,
                beneficiaries=item.beneficiaries,
                source_file=item.source_file,
                page_start=item.page_start,
                page_end=item.page_end,
                text=item.text,
                similarity=item.similarity,
                row_id=item.row_id,
                rank=rank,
            )
            for rank, item in enumerate(results, start=1)
        ]
        return ranked

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top-K chunks for a question from the FAISS index."
    )
    parser.add_argument("question", help="user question to search for")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a readable summary",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=400,
        help="truncate displayed text in the readable summary (0 = full text)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        print("--top-k must be at least 1", file=sys.stderr)
        return 2
    if args.max_text_chars < 0:
        print("--max-text-chars must not be negative", file=sys.stderr)
        return 2

    try:
        with Retriever(
            index_path=args.index,
            database_path=args.database,
            manifest_path=args.manifest,
            device=args.device,
            top_k=args.top_k,
        ) as retriever:
            results = retriever.search(args.question)
    except Exception as error:
        print(f"Retrieval failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {"question": args.question, "results": [r.to_dict() for r in results]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Question: {args.question}")
    print(f"Results: {len(results)} (top-K = {args.top_k}, metric = cosine)\n")
    for result in results:
        text = result.text
        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[: args.max_text_chars].rstrip() + "..."
        print(f"[{result.rank}] score={result.similarity:.4f}  row_id={result.row_id}")
        print(f"    scheme:     {result.scheme_name}")
        print(f"    department: {result.department or '(none)'}")
        print(f"    state:      {result.state or '(none)'}")
        print(f"    source:     {result.source_file} (pages {result.page_start}-{result.page_end})")
        print(f"    chunk_id:   {result.chunk_id}")
        print(f"    text:       {text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
